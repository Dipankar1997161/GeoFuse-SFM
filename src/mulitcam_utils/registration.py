"""
multicam_registration.py - Helpers for multi-camera SfM with known geometry

For calibrated multi-camera rigs (like BBM), we have KNOWN camera poses.
The challenge is finding enough feature correspondences.

KEY INSIGHT: 
- We don't need to ESTIMATE poses (they're known!)
- We just need to find which features correspond across views
- Use epipolar constraint from known geometry to validate matches
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from src.geometry_utils.epipolar import compute_fundamental_matrix, epipolar_distance
from data_io.camera import DecomposedCamera
from src.features import triangulation_angle_deg



def match_with_epipolar_filter(
    feats_i,  # Features object with kpts_xy, desc
    feats_j,
    cam_i: DecomposedCamera,
    cam_j: DecomposedCamera,
    ratio: float = 0.9,
    max_epipolar_dist: float = 3.0,
    mutual: bool = False,
) -> List[Tuple[int, int]]:
    """
    Match features with epipolar constraint from known cameras.
    
    This is more reliable than triangulation verification because
    it doesn't require the match to triangulate well - just that
    the points lie on corresponding epipolar lines.
    """
    from src.features import match_descriptors
    
    # Get raw descriptor matches
    raw = match_descriptors(feats_i, feats_j, method="sift", ratio=ratio, mutual=mutual)
    
    if len(raw) == 0:
        return []
    
    # Compute F matrix
    F = compute_fundamental_matrix(cam_i, cam_j)
    
    # Filter by epipolar constraint
    inliers = []
    for a, b in raw:
        p1 = feats_i.kpts_xy[a]
        p2 = feats_j.kpts_xy[b]
        
        dist = epipolar_distance(p1, p2, F)
        
        if dist < max_epipolar_dist:
            inliers.append((a, b))
    
    return inliers


def find_all_valid_pairs_multicam(
    feats: List,  # List of Features
    cams: List[DecomposedCamera],  # per-view intrinsics+pose
    ratio: float = 0.9,
    max_epipolar_dist: float = 3.0,
    min_inliers: int = 8,
    verbose: bool = True,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Find all valid feature correspondences between all pairs of cameras.
    
    Uses epipolar constraint from known geometry instead of 
    triangulation verification.
    """
    n = len(feats)
    pairwise = {}
    
    if verbose:
        print(f"[Multicam] Finding matches between {n} cameras...")
        print(f"  ratio={ratio}, max_epipolar_dist={max_epipolar_dist}px")
    
    total_pairs = 0
    valid_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            
            inliers = match_with_epipolar_filter(
                feats[i], feats[j],
                cams[i], cams[j],
                ratio=ratio,
                max_epipolar_dist=max_epipolar_dist,
                mutual=False,
            )
            
            if len(inliers) >= min_inliers:
                pairwise[(i, j)] = inliers
                valid_pairs += 1
                if verbose:
                    print(f"  ({i:2d},{j:2d}): {len(inliers):3d} inliers ✓")
    
    if verbose:
        print(f"[Multicam] Valid pairs: {valid_pairs}/{total_pairs}")
        total_matches = sum(len(m) for m in pairwise.values())
        print(f"[Multicam] Total matches: {total_matches}")
    
    return pairwise


def register_all_cameras_multicam(
    tracks: List,  # List of Track objects
    feats: List,   # List of Features
    cams: List[DecomposedCamera],  # Known camera parameters
    X_list: List[np.ndarray],  # Current 3D points
    track_to_point: Dict[int, int],  # track_id -> point_id
    registered: Set[int],  # Currently registered cameras
    max_reproj_px: float = 2.0,
    min_visible_points: int = 5,
    verbose: bool = True,
) -> Set[int]:
    """
    Try to register all cameras by checking visibility of existing 3D points.
    
    For calibrated rigs, we don't need PnP - poses are known!
    We just need to verify that enough 3D points project correctly.
    """
    n = len(cams)
    newly_registered = set()
    
    for img_id in range(n):
        if img_id in registered:
            continue
        
        cam = cams[img_id]
        K, R, t, C = cam.K, cam.R, cam.t, cam.C

        K = np.asarray(K, np.float64)
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64).reshape(3, 1)
        
        # Count how many existing 3D points are visible and have observations
        visible_with_obs = 0
        reproj_errors = []
        
        for tid, pid in track_to_point.items():
            tr = tracks[tid]
            
            # Does this track have an observation in this image?
            if img_id not in tr.obs:
                continue
            
            X = np.asarray(X_list[pid], np.float64).reshape(3, 1)
            
            # Project to this camera
            Xc = R @ X + t
            z = float(Xc[2, 0])
            
            if z <= 1e-6:
                continue
            
            uv_proj = (K[:2, :2] @ (Xc[:2] / z)) + K[:2, 2:3]
            uv_proj = uv_proj.flatten()
            
            uv_obs = feats[img_id].kpts_xy[tr.obs[img_id]]
            
            err = np.linalg.norm(uv_proj - uv_obs)
            reproj_errors.append(err)
            
            if err < max_reproj_px:
                visible_with_obs += 1
        
        if visible_with_obs >= min_visible_points:
            registered.add(img_id)
            newly_registered.add(img_id)
            
            if verbose:
                med_err = np.median(reproj_errors) if reproj_errors else float('nan')
                print(f"  Registered camera {img_id}: {visible_with_obs} visible points, "
                      f"median reproj={med_err:.1f}px")
    
    return newly_registered


def triangulate_multiview(
    track: 'Track',
    feats: List,
    cams: List[DecomposedCamera],
    registered: Set[int],
    min_views: int = 3,
    max_reproj_px: float = 2.0,
) -> Optional[np.ndarray]:
    """
    Triangulate a track using multiple views with known cameras.
    
    Uses linear triangulation (DLT) with all available views,
    then validates with reprojection error.
    """
    # Collect observations from registered cameras
    obs_list = []
    for img_id, kp_id in track.obs.items():
        if img_id not in registered:
            continue
        
        cam = cams[img_id]
        K, R, t = cam.K, cam.R, cam.t

        uv = feats[img_id].kpts_xy[kp_id]
        
        obs_list.append((img_id, K, R, t, uv))
    
    if len(obs_list) < min_views:
        return None
    
    # Build DLT system: each observation gives 2 equations
    A = []
    for img_id, K, R, t, uv in obs_list:
        K = np.asarray(K, np.float64)
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64).reshape(3, 1)
        
        P = K @ np.hstack([R, t])  # 3x4 projection matrix
        
        u, v = uv[0], uv[1]
        
        # Two equations per observation
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    
    A = np.array(A)
    
    # SVD solution
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]  # Last row of V^T
    
    if abs(X_h[3]) < 1e-12:
        return None
    
    X = X_h[:3] / X_h[3]
    
    # Validate with reprojection error
    errors = []
    for img_id, K, R, t, uv_obs in obs_list:
        K = np.asarray(K, np.float64)
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64).reshape(3, 1)
        
        Xc = R @ X.reshape(3, 1) + t
        z = float(Xc[2, 0])
        
        if z <= 1e-6:
            return None
        
        uv_proj = (K[:2, :2] @ (Xc[:2] / z) + K[:2, 2:3]).flatten()
        err = np.linalg.norm(uv_proj - uv_obs)
        errors.append(err)
    
    # Check median error
    if np.median(errors) > max_reproj_px:
        return None
    
    return X


def incremental_sfm_multicam_known_poses(
    feats: List,
    tracks: List,
    cams: List[DecomposedCamera],
    pairwise: Dict[Tuple[int, int], List[Tuple[int, int]]],
    init_pair: Tuple[int, int],
    max_reproj_px: float = 2.0,
    min_triang_angle_deg: float = 1.0,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict[int, int], Set[int]]:
    """
    Incremental SfM for multi-camera rig with KNOWN poses.
    
    Since poses are known, we:
    1. Initialize with best pair (already verified)
    2. Register all cameras that see existing points
    3. Triangulate new points from newly registered cameras
    4. Repeat until no more cameras can be registered
    """
    
    n = len(cams)
    
    X_list = []
    track_to_point = {}
    registered = set()
    
    # Helper to add a point
    def add_point(tid, X):
        pid = len(X_list)
        X_list.append(np.asarray(X, np.float64))
        track_to_point[tid] = pid
    
    # Initialize with the best pair
    i0, j0 = init_pair
    registered.add(i0)
    registered.add(j0)
    
    if verbose:
        print(f"[Multicam SfM] Initializing with pair ({i0}, {j0})")
    
    # Triangulate initial points
    ci = cams[i0]
    cj = cams[j0]

    Ki, Ri, ti, Ci = ci.K, ci.R, ci.t, ci.C
    Kj, Rj, tj, Cj = cj.K, cj.R, cj.t, cj.C

    
    inliers = pairwise.get((i0, j0), pairwise.get((j0, i0), []))
    if (j0, i0) in pairwise:
        inliers = [(b, a) for a, b in inliers]  # Swap order
    
    Pi = np.asarray(Ki) @ np.hstack([np.asarray(Ri), np.asarray(ti).reshape(3, 1)])
    Pj = np.asarray(Kj) @ np.hstack([np.asarray(Rj), np.asarray(tj).reshape(3, 1)])
    
    pts_i = np.float32([feats[i0].kpts_xy[a] for a, _ in inliers]).T
    pts_j = np.float32([feats[j0].kpts_xy[b] for _, b in inliers]).T
    
    Xh = cv2.triangulatePoints(Pi, Pj, pts_i, pts_j)
    X_init = (Xh[:3] / (Xh[3:4] + 1e-12)).T
    
    # Filter and add initial points
    Ci = np.asarray(Ci).reshape(3)
    Cj = np.asarray(Cj).reshape(3)
    
    init_added = 0
    for k, ((a, b), Xp) in enumerate(zip(inliers, X_init)):
        # Check cheirality
        Xp = np.asarray(Xp, np.float64)
        zi = float((np.asarray(Ri) @ Xp.reshape(3, 1) + np.asarray(ti).reshape(3, 1))[2, 0])
        zj = float((np.asarray(Rj) @ Xp.reshape(3, 1) + np.asarray(tj).reshape(3, 1))[2, 0])
        
        if zi <= 0 or zj <= 0:
            continue
        
        # Check triangulation angle
        ang = triangulation_angle_deg(Xp, Ci, Cj)
        if ang < min_triang_angle_deg:
            continue
        
        # Find track ID
        for tid, tr in enumerate(tracks):
            if tr.obs.get(i0) == a and tr.obs.get(j0) == b:
                if tid not in track_to_point:
                    add_point(tid, Xp)
                    init_added += 1
                break
    
    if verbose:
        print(f"  Initial points: {init_added}")
    
    # Iterative registration
    max_iters = n
    for iteration in range(max_iters):
        # Try to register more cameras
        newly_registered = register_all_cameras_multicam(
            tracks, feats, cams, X_list, track_to_point,
            registered, max_reproj_px=max_reproj_px,
            min_visible_points=3, verbose=verbose,
        )
        
        if len(newly_registered) == 0:
            if verbose:
                print(f"[Multicam SfM] No more cameras can be registered")
            break
        
        # Triangulate new points using newly registered cameras
        new_points = 0
        for tid, tr in enumerate(tracks):
            if tid in track_to_point:
                continue
            
            # Try multi-view triangulation
            X = triangulate_multiview(
                tr, feats, cams, registered,
                min_views=2, max_reproj_px=max_reproj_px,
            )
            
            if X is not None:
                add_point(tid, X)
                new_points += 1
        
        if verbose:
            print(f"  Triangulated {new_points} new points (total: {len(X_list)})")
        
        if new_points == 0 and len(newly_registered) == 0:
            break
    
    if verbose:
        print(f"[Multicam SfM] Final: {len(registered)}/{n} cameras, {len(X_list)} points")
    
    return X_list, track_to_point, registered


if __name__ == "__main__":
    print("Multi-camera Registration Module")
    print("="*50)
    print()
    print("For calibrated multi-camera rigs:")
    print("  1. Use match_with_epipolar_filter() instead of triangulation verification")
    print("  2. Use find_all_valid_pairs_multicam() for exhaustive pair matching")
    print("  3. Use incremental_sfm_multicam_known_poses() for reconstruction")