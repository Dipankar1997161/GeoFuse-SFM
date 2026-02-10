"""
src/pipeline/triangulation.py

All triangulation operations for SfM.
Includes two-view, multi-view DLT, and point validation.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import cv2

from src.features import Features, triangulation_angle_deg, projection_matrix
from data_io.camera import DecomposedCamera

from .config import TriangulationConfig
from .state import SfMState


def triangulate_pair_with_known_cams(
    feats_i: Features,
    feats_j: Features,
    inliers: List[Tuple[int, int]],
    cam_i: DecomposedCamera,
    cam_j: DecomposedCamera,
    max_reproj_px: float,
    min_triang_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate points from a pair of cameras with known poses.
    
    Args:
        feats_i, feats_j: Features for each image
        inliers: List of (kp_idx_i, kp_idx_j) matches
        cam_i, cam_j: DecomposedCamera objects
        max_reproj_px: Maximum reprojection error
        min_triang_angle_deg: Minimum triangulation angle
        
    Returns:
        X: (N, 3) triangulated points
        keep: (N,) boolean mask of valid points
    """
    Ki, Ri, ti, ci = cam_i.K, cam_i.R, cam_i.t, cam_i.C
    Kj, Rj, tj, cj = cam_j.K, cam_j.R, cam_j.t, cam_j.C
    
    Pi = projection_matrix(Ki, Ri, ti)
    Pj = projection_matrix(Kj, Rj, tj)
    
    pts_i = np.float32([feats_i.kpts_xy[a] for a, _ in inliers]).T  # 2xN
    pts_j = np.float32([feats_j.kpts_xy[b] for _, b in inliers]).T  # 2xN
    
    Xh = cv2.triangulatePoints(Pi, Pj, pts_i, pts_j)
    X = (Xh[:3] / (Xh[3:4] + 1e-12)).T  # Nx3
    
    Ci = ci.reshape(3) if ci.ndim > 1 else ci
    Cj = cj.reshape(3) if cj.ndim > 1 else cj
    
    keep = np.zeros((X.shape[0],), dtype=bool)
    
    for k in range(X.shape[0]):
        Xk = X[k].reshape(3, 1)
        
        # Depth check
        zi = float((Ri @ Xk + ti)[2, 0])
        zj = float((Rj @ Xk + tj)[2, 0])
        if zi <= 1e-9 or zj <= 1e-9:
            continue
        
        a, b = inliers[k]
        
        # Reprojection error in image i
        ui = (Ki @ (Ri @ Xk + ti))
        ui = (ui[:2, 0] / ui[2, 0])
        e_i = float(np.linalg.norm(ui - feats_i.kpts_xy[a]))
        
        # Reprojection error in image j
        uj = (Kj @ (Rj @ Xk + tj))
        uj = (uj[:2, 0] / uj[2, 0])
        e_j = float(np.linalg.norm(uj - feats_j.kpts_xy[b]))
        
        if e_i > max_reproj_px or e_j > max_reproj_px:
            continue
        
        # Triangulation angle
        ang = triangulation_angle_deg(X[k], Ci, Cj)
        if ang < min_triang_angle_deg:
            continue
        
        keep[k] = True
    
    return X, keep


def triangulate_multiview_dlt(
    state: SfMState,
    track_id: int,
    max_reproj_px: float,
) -> Optional[np.ndarray]:
    """
    Triangulate a point using DLT from all registered views.
    
    Args:
        state: SfMState with registered cameras
        track_id: Track to triangulate
        max_reproj_px: Maximum reprojection error for validation
        
    Returns:
        X: (3,) point coordinates, or None if failed
    """
    tr = state.tracks[track_id]
    
    # Collect observations from registered cameras
    obs_list = []
    for img_id, kp_id in tr.obs.items():
        if img_id not in state.registered:
            continue
        
        K_i = state.K_for_view(img_id)
        R_i, t_i = state.get_pose(img_id)
        R_i = np.asarray(R_i, np.float64)
        t_i = np.asarray(t_i, np.float64).reshape(3, 1)
        
        uv = state.feats[img_id].kpts_xy[kp_id]
        obs_list.append((img_id, K_i, R_i, t_i, uv))
    
    if len(obs_list) < 2:
        return None
    
    # Build DLT system
    A = []
    for img_id, K_i, R_i, t_i, uv in obs_list:
        P = K_i @ np.hstack([R_i, t_i])  # 3x4
        u, v = uv[0], uv[1]
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A = np.array(A)
    
    # SVD solution
    try:
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        if abs(X_h[3]) < 1e-12:
            return None
        X = X_h[:3] / X_h[3]
    except:
        return None
    
    # Validate reprojection
    errors = []
    for img_id, K_i, R_i, t_i, uv_obs in obs_list:
        Xc = R_i @ X.reshape(3, 1) + t_i
        z = float(Xc[2, 0])
        if z <= 1e-6:
            return None
        uv_proj = (K_i[:2, :2] @ (Xc[:2] / z) + K_i[:2, 2:3]).flatten()
        errors.append(np.linalg.norm(uv_proj - uv_obs))
    
    if np.median(errors) > max_reproj_px * 2.0:
        return None
    
    return X


def validate_point_multiview(
    state: SfMState,
    track_id: int,
    Xp: np.ndarray,
    config: TriangulationConfig,
) -> bool:
    """
    Validate a triangulated point using majority voting.
    
    Checks:
    1. Sufficient number of views
    2. Majority of views have good reprojection error
    3. Depth consistency across good views
    
    Args:
        state: SfMState with registered cameras
        track_id: Track ID
        Xp: (3,) candidate point
        config: TriangulationConfig with thresholds
        
    Returns:
        True if point is valid
    """
    tr = state.tracks[track_id]
    Xp = np.asarray(Xp, dtype=np.float64).reshape(3, 1)
    
    errs = []
    depths = []
    
    # Evaluate across all registered cameras that observe this track
    for img_id, kp_id in tr.obs.items():
        if img_id not in state.cam_poses:
            continue
        
        R, t = state.cam_poses[img_id]
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        
        # Camera coords
        Xc = (R @ Xp) + t
        z = float(Xc[2, 0])
        
        # Soft depth gating
        if not np.isfinite(z):
            continue
        if z <= config.z_min:
            continue
        if config.z_max is not None and z > config.z_max:
            continue
        
        K_i = state.K_for_view(img_id)
        x = Xc[:2, 0] / z
        uv = (K_i[:2, :2] @ x) + K_i[:2, 2]
        
        uv_obs = state.feats[img_id].kpts_xy[kp_id].astype(np.float64)
        e = float(np.linalg.norm(uv - uv_obs))
        
        if not np.isfinite(e):
            continue
        
        errs.append(e)
        depths.append(z)
    
    n_views = len(errs)
    if n_views < config.min_views_accept:
        return False
    
    errs_np = np.asarray(errs, dtype=np.float64)
    depths_np = np.asarray(depths, dtype=np.float64)
    
    # Adaptive threshold
    thr = float(config.max_reproj_px)
    
    if config.adaptive and n_views >= 3:
        med = float(np.median(errs_np))
        mad = float(np.median(np.abs(errs_np - med)))
        sigma = 1.4826 * mad
        thr_adapt = med + config.mad_k * sigma
        
        cap = config.thresh_cap_mult * config.max_reproj_px
        thr = float(np.clip(thr_adapt, 0.75 * config.max_reproj_px, cap))
    
    # Majority vote
    good = errs_np <= thr
    n_good = int(np.sum(good))
    
    req = int(np.ceil(config.good_ratio * n_views))
    req = max(req, config.min_views_accept)
    
    if n_good < req:
        return False
    
    # Depth consistency on good views
    good_depths = depths_np[good]
    if good_depths.size >= 3:
        d_mean = float(np.mean(good_depths))
        d_std = float(np.std(good_depths))
        cov = d_std / max(d_mean, 1e-9)
        if cov > config.max_depth_cov:
            return False
    
    return True


def triangulate_new_points_multiview(
    state: SfMState,
    new_img: int,
    config: TriangulationConfig,
) -> int:
    """
    Triangulate new points that become visible with addition of new_img.
    
    For multicam: Uses DLT with all registered views.
    Requires min_views_accept (typically 3) registered views.
    
    Args:
        state: SfMState
        new_img: Newly registered image
        config: TriangulationConfig
        
    Returns:
        Number of points added
    """
    added = 0
    
    for tid, tr in enumerate(state.tracks):
        # Skip if already triangulated
        if tid in state.track_to_point:
            continue
        
        # Skip if not observed in new image
        if new_img not in tr.obs:
            continue
        
        # Count registered views
        views = [img_id for img_id in tr.obs.keys() if img_id in state.registered]
        
        # KEY: Require minimum views (typically 3 for multicam)
        if len(views) < config.min_views_accept:
            continue
        
        # Triangulate using DLT
        Xp = triangulate_multiview_dlt(state, tid, config.max_reproj_px)
        if Xp is None:
            continue
        
        # Validate
        if validate_point_multiview(state, tid, Xp, config):
            state.add_point(tid, Xp)
            added += 1
    
    return added


def triangulate_new_points_singlecam(
    state: SfMState,
    new_img: int,
    config: TriangulationConfig,
) -> int:
    """
    Triangulate new points for singlecam setup.
    
    Uses two-view triangulation with best baseline pair.
    
    Args:
        state: SfMState
        new_img: Newly registered image
        config: TriangulationConfig
        
    Returns:
        Number of points added
    """
    from src.geometry import triangulate_and_filter
    from src.geometry_utils.projective import camera_center
    
    added = 0
    
    for tid, tr in enumerate(state.tracks):
        if tid in state.track_to_point:
            continue
        if new_img not in tr.obs:
            continue
        
        # Find best other view (largest baseline)
        Rn, tn = state.cam_poses[new_img]
        Cn = camera_center(Rn, tn)
        
        best_other = None
        best_score = -1.0
        
        for img_id in tr.obs.keys():
            if img_id == new_img or img_id not in state.registered:
                continue
            
            Ro, to = state.cam_poses[img_id]
            Co = camera_center(Ro, to)
            
            score = float(np.linalg.norm(Co - Cn))
            if score > best_score:
                best_score = score
                best_other = img_id
        
        if best_other is None:
            continue
        
        Ro, to = state.cam_poses[best_other]
        p_new = state.feats[new_img].kpts_xy[tr.obs[new_img]]
        p_oth = state.feats[best_other].kpts_xy[tr.obs[best_other]]
        
        # Adaptive minimum angle
        min_angle = max(1.0, 3.0 - 0.3 * len(state.registered))
        
        Xcand, keep = triangulate_and_filter(
            np.float32([p_oth]),
            np.float32([p_new]),
            state.K_global,
            Ro, to,
            Rn, tn,
            max_reproj_px=config.max_reproj_px,
            min_triang_angle_deg=min_angle,
        )
        
        if Xcand.shape[0] == 1 and bool(keep[0]):
            Xp = Xcand[0]
            
            # Depth check
            Xc = (Rn @ np.asarray(Xp, np.float64).reshape(3, 1)) + np.asarray(tn, np.float64).reshape(3, 1)
            z = float(Xc[2, 0])
            if not np.isfinite(z) or z <= 1e-6:
                continue
            
            # Validate with multiview check
            min_views = 2 if len(state.registered) < 5 else 3
            config_copy = TriangulationConfig(
                max_reproj_px=config.max_reproj_px * 1.5,
                min_views_accept=min_views,
                good_ratio=0.60,
                adaptive=True,
            )
            
            if validate_point_multiview(state, tid, Xp, config_copy):
                state.add_point(tid, Xp)
                added += 1
    
    return added


def triangulate_new_points(
    state: SfMState,
    new_img: int,
    config: TriangulationConfig,
) -> int:
    """
    Main entry point for triangulating new points.
    
    Dispatches to multicam or singlecam implementation.
    """
    if state.multicam:
        return triangulate_new_points_multiview(state, new_img, config)
    else:
        return triangulate_new_points_singlecam(state, new_img, config)
