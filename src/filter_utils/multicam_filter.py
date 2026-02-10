"""
FIXED STRICT OUTLIER REMOVAL FOR MULTICAM

Bug fix: The original version kept orphan points (points in X_list but not in track_to_point).
This version removes ALL points that don't pass the strict filter.
"""

import numpy as np
from typing import List, Dict, Tuple, Set


def remove_outliers_strict_multicam(
    K,
    cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    X_list: List[np.ndarray],
    track_to_point: Dict[int, int],
    tracks,
    feats,
    cams=None,  # For multicam: list of (K, R, t, C)
    max_reproj_any_view: float = 5.0,   # Max error in ANY single view
    max_reproj_median: float = 2.0,     # Max median error across views
    min_good_views: int = 2,            # Minimum views with good reprojection
    verbose: bool = True,
):
    """
    Strict outlier removal that checks reprojection error in ALL views.
    
    A point is KEPT only if:
    1. It has a valid track (exists in track_to_point)
    2. Its max error across all views <= max_reproj_any_view
    3. Its median error <= max_reproj_median
    4. It has at least min_good_views with error <= max_reproj_any_view
    
    Returns:
        X_list_clean: Filtered points (NEW indices)
        track_to_point_clean: Updated mapping (tid -> NEW pid)
        removed_pids: List of removed point IDs (OLD indices)
    """
    
    multicam = cams is not None
    
    def K_for_view(img_id):
        if multicam:
            cam = cams[img_id]
            return np.asarray(cam.K, np.float64)
        return np.asarray(K, np.float64)
    
    # Invert track_to_point: pid -> tid
    point_to_track = {pid: tid for tid, pid in track_to_point.items()}
    
    # Track which points to KEEP (not remove)
    keep_pids = set()
    removed_pids = []
    reasons = {"no_track": 0, "any_view": 0, "median": 0, "few_good": 0}
    
    for pid, X in enumerate(X_list):
        # Check if point has a track
        if pid not in point_to_track:
            removed_pids.append(pid)
            reasons["no_track"] += 1
            continue
            
        tid = point_to_track[pid]
        tr = tracks[tid]
        
        X_pt = np.asarray(X, np.float64).reshape(3, 1)
        
        errors = []
        good_views = 0
        max_err = 0.0
        
        for img_id, kp_id in tr.obs.items():
            if img_id not in cam_poses:
                continue
            
            R, t = cam_poses[img_id]
            R = np.asarray(R, np.float64)
            t = np.asarray(t, np.float64).reshape(3, 1)
            K_i = K_for_view(img_id)
            
            # Project
            Xc = R @ X_pt + t
            z = float(Xc[2, 0])
            
            if z <= 1e-6:
                errors.append(1000.0)
                continue
            
            x = Xc[:2, 0] / z
            uv_proj = (K_i[:2, :2] @ x) + K_i[:2, 2]

            
            uv_obs = feats[img_id].kpts_xy[kp_id].astype(np.float64)
            err = float(np.linalg.norm(uv_proj - uv_obs))
            
            errors.append(err)
            max_err = max(max_err, err)
            
            if err <= max_reproj_any_view:
                good_views += 1
        
        if not errors:
            removed_pids.append(pid)
            reasons["few_good"] += 1
            continue
        
        median_err = float(np.median(errors))
        
        # Check removal conditions
        remove = False
        reason = None
        
        if max_err > max_reproj_any_view:
            remove = True
            reason = "any_view"
        elif median_err > max_reproj_median:
            remove = True
            reason = "median"
        elif good_views < min_good_views:
            remove = True
            reason = "few_good"
        
        if remove:
            removed_pids.append(pid)
            if reason:
                reasons[reason] += 1
        else:
            keep_pids.add(pid)
    
    # Build clean lists - only include points we explicitly decided to keep
    X_list_clean = []
    track_to_point_clean = {}
    old_to_new = {}
    
    for pid in sorted(keep_pids):  # Maintain order
        new_pid = len(X_list_clean)
        old_to_new[pid] = new_pid
        X_list_clean.append(X_list[pid])
        
        tid = point_to_track[pid]
        track_to_point_clean[tid] = new_pid
    
    if verbose:
        print(f"[strict filter] Input: {len(X_list)} points")
        print(f"[strict filter] Removed {len(removed_pids)} points:")
        print(f"  - no_track (orphan): {reasons['no_track']}")
        print(f"  - any_view > {max_reproj_any_view}px: {reasons['any_view']}")
        print(f"  - median > {max_reproj_median}px: {reasons['median']}")
        print(f"  - good_views < {min_good_views}: {reasons['few_good']}")
        print(f"[strict filter] Kept: {len(X_list_clean)} points")
    
    return X_list_clean, track_to_point_clean, removed_pids


def remove_outliers_spatial(
    X_list: List[np.ndarray],
    track_to_point: Dict[int, int],
    percentile: float = 95.0,
    multiplier: float = 3.0,
    verbose: bool = True,
):
    """
    Remove spatial outliers based on distance from centroid.
    
    Points further than (multiplier * percentile_distance) are removed.
    """
    if len(X_list) < 10:
        return X_list, track_to_point, []
    
    X_arr = np.array(X_list)
    centroid = np.median(X_arr, axis=0)  # Use median for robustness
    
    distances = np.linalg.norm(X_arr - centroid, axis=1)
    threshold = np.percentile(distances, percentile) * multiplier
    
    point_to_track = {pid: tid for tid, pid in track_to_point.items()}
    
    keep_pids = set()
    removed_pids = []
    
    for pid, dist in enumerate(distances):
        if dist <= threshold:
            keep_pids.add(pid)
        else:
            removed_pids.append(pid)
    
    # Build clean lists
    X_list_clean = []
    track_to_point_clean = {}
    
    for pid in sorted(keep_pids):
        new_pid = len(X_list_clean)
        X_list_clean.append(X_list[pid])
        
        if pid in point_to_track:
            tid = point_to_track[pid]
            track_to_point_clean[tid] = new_pid
    
    if verbose:
        print(f"[spatial filter] Removed {len(removed_pids)} points beyond {threshold:.3f} units")
        print(f"[spatial filter] Kept: {len(X_list_clean)} points")
    
    return X_list_clean, track_to_point_clean, removed_pids
