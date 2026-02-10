"""
src/pipeline/state.py

Holds all mutable state for the SfM reconstruction.
Provides unified helpers that work for both multicam and singlecam.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from src.features import Features
from src.tracks import Track
from data_io.camera import DecomposedCamera


@dataclass
class PointMeta:
    """Metadata for a reconstructed 3D point."""
    track_id: int
    obs_views: List[int]
    track_len: int
    max_triang_angle_deg: float
    median_triang_angle_deg: float
    median_reproj_error: float


@dataclass
class SfMResult:
    """Final output of SfM reconstruction."""
    cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]]  # image_id -> (R, t)
    X: np.ndarray                                         # (P, 3) point cloud
    track_to_point: Dict[int, int]                        # track_id -> point_index
    registered_images: List[int]
    tracks: List[Track]
    feats: List[Features]
    point_meta: Dict[int, PointMeta]


class SfMState:
    """
    Mutable state container for incremental SfM.
    
    Centralizes all state management and provides unified helpers
    that automatically handle multicam vs singlecam differences.
    
    Usage:
        state = SfMState(images, K, cams=cams, silhouettes=silhouettes)
        
        # Unified helpers
        K_i = state.K_for_view(img_id)
        C = state.camera_center_for_view(img_id)
        state.add_point(track_id, X)
    """
    
    def __init__(
        self,
        images: List[np.ndarray],
        K: Optional[np.ndarray] = None,
        cams: Optional[List[DecomposedCamera]] = None,
        silhouettes: Optional[List[np.ndarray]] = None,
    ):
        self.images = images
        self.n_images = len(images)
        self.silhouettes = silhouettes
        
        # Multicam detection
        self.multicam = cams is not None
        self.cams = cams
        
        # Intrinsics
        if self.multicam:
            self.K_global = None  # Not used in multicam
        else:
            if K is None:
                raise ValueError("K required for singlecam mode")
            self.K_global = np.asarray(K, dtype=np.float64)
        
        # Features (populated by matching stage)
        self.feats: List[Features] = []
        self.num_kp: List[int] = []
        
        # Pairwise matches (populated by matching stage)
        self.pairwise: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.pair_scores: Dict[Tuple[int, int], int] = {}
        
        # Tracks (populated by track building stage)
        self.tracks: List[Track] = []
        
        # Camera poses: img_id -> (R, t)
        self.cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Registration state
        self.registered: Set[int] = set()
        self.registered_order: List[int] = []
        
        # 3D points
        self.X_list: List[np.ndarray] = []
        self.track_to_point: Dict[int, int] = {}
        self.point_meta: Dict[int, PointMeta] = {}
    
    # =========================================================
    # Unified helpers (handle multicam vs singlecam automatically)
    # =========================================================
    
    def K_for_view(self, img_id: int) -> np.ndarray:
        """Get intrinsic matrix for a specific view."""
        if self.multicam:
            return np.asarray(self.cams[img_id].K, np.float64)
        return self.K_global
    
    def camera_center_for_view(self, img_id: int) -> np.ndarray:
        """
        Get camera center in world coordinates.
        
        Multicam: Uses pre-computed C from calibration.
        Singlecam: Computes C = -R^T @ t from cam_poses.
        """
        if self.multicam:
            return np.asarray(self.cams[img_id].C, np.float64).reshape(3)
        else:
            if img_id not in self.cam_poses:
                raise ValueError(f"Camera {img_id} not registered")
            R, t = self.cam_poses[img_id]
            R = np.asarray(R, np.float64)
            t = np.asarray(t, np.float64).reshape(3, 1)
            return (-R.T @ t).reshape(3)
    
    def get_pose(self, img_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (R, t) for a view. Works for both multicam and singlecam."""
        if img_id in self.cam_poses:
            return self.cam_poses[img_id]
        if self.multicam:
            cam = self.cams[img_id]
            return (
                np.asarray(cam.R, np.float64),
                np.asarray(cam.t, np.float64).reshape(3, 1)
            )
        raise ValueError(f"Camera {img_id} not registered")
    
    # =========================================================
    # Point management
    # =========================================================
    
    def add_point(self, track_id: int, Xp: np.ndarray) -> int:
        """
        Add a new 3D point to the reconstruction.
        
        Args:
            track_id: Track ID this point corresponds to
            Xp: 3D point coordinates (3,)
            
        Returns:
            point_id: Index of the new point in X_list
        """
        pid = len(self.X_list)
        self.X_list.append(np.asarray(Xp, np.float64).reshape(3))
        self.track_to_point[track_id] = pid
        
        # Compute metadata
        tr = self.tracks[track_id]
        obs_views = [vid for vid in tr.obs.keys() if vid in self.cam_poses]
        
        cam_centers = [self.camera_center_for_view(vid) for vid in obs_views]
        angles = self._triangulation_angles_deg(np.asarray(Xp, np.float64), cam_centers)
        
        median_err = self._point_median_reproj_error(track_id, Xp)
        
        meta = PointMeta(
            track_id=int(track_id),
            obs_views=obs_views,
            track_len=int(len(obs_views)),
            max_triang_angle_deg=float(max(angles)) if angles else 0.0,
            median_triang_angle_deg=float(np.median(angles)) if angles else 0.0,
            median_reproj_error=median_err,
        )
        self.point_meta[pid] = meta
        
        return pid
    
    def _point_median_reproj_error(self, track_id: int, Xp: np.ndarray) -> float:
        """Compute median reprojection error for a point."""
        tr = self.tracks[track_id]
        X = np.asarray(Xp, np.float64).reshape(3, 1)
        errs = []
        
        for img_id, kp_id in tr.obs.items():
            if img_id not in self.cam_poses:
                continue
            
            R, t = self.cam_poses[img_id]
            R = np.asarray(R, np.float64)
            t = np.asarray(t, np.float64).reshape(3, 1)
            K_i = self.K_for_view(img_id)
            
            Xc = (R @ X) + t
            z = float(Xc[2, 0])
            if z <= 1e-9:
                continue
            
            x = Xc[:2, 0] / z
            uv = (K_i[:2, :2] @ x) + K_i[:2, 2]
            uv_obs = self.feats[img_id].kpts_xy[kp_id].astype(np.float64)
            
            errs.append(float(np.linalg.norm(uv - uv_obs)))
        
        return float(np.median(errs)) if errs else float("nan")
    
    @staticmethod
    def _triangulation_angles_deg(X: np.ndarray, cam_centers: List[np.ndarray]) -> List[float]:
        """Compute pairwise triangulation angles between camera centers."""
        X = X.reshape(3)
        dirs = []
        for C in cam_centers:
            v = X - C
            v /= (np.linalg.norm(v) + 1e-12)
            dirs.append(v)
        
        angles = []
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                dot = np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(dot)))
        return angles
    
    # =========================================================
    # Registration helpers
    # =========================================================
    
    def register_image(self, img_id: int, R: np.ndarray = None, t: np.ndarray = None):
        """
        Register an image (mark as having known pose).
        
        For multicam: Pose is already known, just marks as registered.
        For singlecam: Must provide R, t.
        """
        if not self.multicam:
            if R is None or t is None:
                raise ValueError("Must provide R, t for singlecam registration")
            self.cam_poses[img_id] = (
                np.asarray(R, np.float64),
                np.asarray(t, np.float64).reshape(3, 1)
            )
        
        self.registered.add(img_id)
        if img_id not in self.registered_order:
            self.registered_order.append(img_id)
    
    def count_2d3d_correspondences(self, img_id: int) -> int:
        """Count 2D-3D correspondences for an image."""
        cnt = 0
        for tid, tr in enumerate(self.tracks):
            if img_id in tr.obs and tid in self.track_to_point:
                cnt += 1
        return cnt
    
    # =========================================================
    # Reprojection error computation
    # =========================================================
    
    def reprojection_error_stats(self, max_obs: int = 20000) -> Dict[str, float]:
        """
        Compute reprojection error statistics over all observations.
        
        Returns dict with: n, mean, median, p90, p95
        """
        errs: List[float] = []
        count = 0
        
        for tid, pid in self.track_to_point.items():
            tr = self.tracks[tid]
            X = np.asarray(self.X_list[pid], dtype=np.float64).reshape(3, 1)
            
            for img_id, kp_id in tr.obs.items():
                if img_id not in self.registered:
                    continue
                
                R, t = self.cam_poses[img_id]
                R = np.asarray(R, dtype=np.float64)
                t = np.asarray(t, dtype=np.float64).reshape(3, 1)
                K_i = self.K_for_view(img_id)
                
                Xc = (R @ X) + t
                z = float(Xc[2, 0])
                if z <= 1e-9:
                    continue
                
                x = Xc[:2, 0] / z
                uv = (K_i[:2, :2] @ x) + K_i[:2, 2]
                uv_obs = self.feats[img_id].kpts_xy[kp_id].astype(np.float64)
                
                e = float(np.linalg.norm(uv - uv_obs))
                errs.append(e)
                
                count += 1
                if count >= max_obs:
                    break
            if count >= max_obs:
                break
        
        if len(errs) == 0:
            return {"n": 0.0, "mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan}
        
        arr = np.asarray(errs, dtype=np.float64)
        return {
            "n": float(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }
    
    # =========================================================
    # Build final result
    # =========================================================
    
    def build_result(self) -> SfMResult:
        """Build the final SfMResult from current state."""
        # Recompute point_meta for final points (after cleanup)
        final_point_meta = {}
        for tid, pid in self.track_to_point.items():
            Xp = self.X_list[pid]
            tr = self.tracks[tid]
            obs_views = [vid for vid in tr.obs.keys() if vid in self.cam_poses]
            
            cam_centers = [self.camera_center_for_view(vid) for vid in obs_views]
            angles = self._triangulation_angles_deg(np.asarray(Xp, np.float64), cam_centers)
            
            # Compute reprojection errors
            errs = []
            X = np.asarray(Xp, np.float64).reshape(3, 1)
            for img_id, kp_id in tr.obs.items():
                if img_id not in self.cam_poses:
                    continue
                R, t = self.cam_poses[img_id]
                R = np.asarray(R, np.float64)
                t = np.asarray(t, np.float64).reshape(3, 1)
                K_i = self.K_for_view(img_id)
                
                Xc = (R @ X) + t
                z = float(Xc[2, 0])
                if z <= 1e-9:
                    continue
                x = Xc[:2, 0] / z
                uv = (K_i[:2, :2] @ x) + K_i[:2, 2]
                uv_obs = self.feats[img_id].kpts_xy[kp_id].astype(np.float64)
                errs.append(float(np.linalg.norm(uv - uv_obs)))
            
            final_point_meta[pid] = PointMeta(
                track_id=int(tid),
                obs_views=obs_views,
                track_len=int(len(obs_views)),
                max_triang_angle_deg=float(max(angles)) if angles else 0.0,
                median_triang_angle_deg=float(np.median(angles)) if angles else 0.0,
                median_reproj_error=float(np.median(errs)) if errs else float("nan"),
            )
        
        return SfMResult(
            cam_poses=self.cam_poses,
            X=np.asarray(self.X_list, dtype=np.float64),
            track_to_point=self.track_to_point,
            registered_images=self.registered_order,
            tracks=self.tracks,
            feats=self.feats,
            point_meta=final_point_meta,
        )
    
    # =========================================================
    # Initialize poses (for multicam, set all at once)
    # =========================================================
    
    def initialize_multicam_poses(self):
        """Initialize all camera poses from calibration data (multicam only)."""
        if not self.multicam:
            raise RuntimeError("initialize_multicam_poses only for multicam mode")
        
        for k in range(self.n_images):
            cam = self.cams[k]
            self.cam_poses[k] = (
                np.asarray(cam.R, np.float64),
                np.asarray(cam.t, np.float64).reshape(3, 1),
            )
