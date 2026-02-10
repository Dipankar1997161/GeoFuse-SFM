"""
src/diagnostics/sfm_diagnostics.py

Diagnostic functions for SfM (works for BOTH multicam and singlecam).

1. project_all_points_to_all_cameras - Project EVERY point to EVERY camera
2. analyze_point_statistics - Get detailed stats on points including outliers
3. find_points_outside_silhouettes - Find points projecting outside masks
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def project_all_points_to_all_cameras(
    out_dir: str,
    images: List[np.ndarray],
    sfm_result,
    cams=None,                              # Optional: for multicam
    K: Optional[np.ndarray] = None,         # Optional: for singlecam
    max_draw_points: int = 500,
    err_thresh_px: float = 5.0,
    silhouettes: Optional[List[np.ndarray]] = None,
):
    """
    Project ALL points to ALL cameras, not just the ones that contributed.
    
    Works for both MULTICAM and SINGLECAM:
    - Multicam: pass cams (list of DecomposedCamera)
    - Singlecam: pass K (shared intrinsic matrix)
    
    Color coding:
    - GREEN dot: observed keypoint (if this camera contributed to this point)
    - RED dot: projected 3D point location
    - WHITE line: good reprojection (< err_thresh_px)
    - YELLOW line: bad reprojection (>= err_thresh_px)
    - CYAN dot: projected point with NO observation in this view
    """
    from cv2 import circle, line, putText, imwrite, cvtColor
    from cv2 import COLOR_RGB2BGR, COLOR_GRAY2BGR, FONT_HERSHEY_SIMPLEX, LINE_AA
    
    # Validate inputs
    if cams is None and K is None:
        raise ValueError("Must provide either cams (multicam) or K (singlecam)")
    
    is_multicam = cams is not None
    
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    
    cam_poses = sfm_result.cam_poses
    X = np.asarray(sfm_result.X, np.float64)
    tracks = sfm_result.tracks
    feats = sfm_result.feats
    track_to_point = sfm_result.track_to_point
    
    # Invert: pid -> tid
    point_to_track = {pid: tid for tid, pid in track_to_point.items()}
    
    n_points = X.shape[0]
    print(f"[project_all] Projecting {n_points} points to {len(sfm_result.registered_images)} cameras...")
    print(f"[project_all] Mode: {'MULTICAM' if is_multicam else 'SINGLECAM'}")
    
    for img_id in sfm_result.registered_images:
        img = images[img_id].copy()
        if img.ndim == 2:
            img = cvtColor(img, COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cvtColor(img, COLOR_RGB2BGR)
        
        h, w = img.shape[:2]
        
        R, t = cam_poses[img_id]
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64).reshape(3, 1)
        
        # Get K for this view
        if is_multicam:
            if hasattr(cams[img_id], "K"):
                K_view = np.asarray(cams[img_id].K, np.float64)
            else:
                K_view = np.asarray(cams[img_id][0], np.float64)
        else:
            K_view = np.asarray(K, np.float64)
        
        # Get silhouette for this view (if available)
        sil = None
        if silhouettes is not None and img_id < len(silhouettes):
            sil = silhouettes[img_id]
            if sil.ndim == 3:
                sil = sil[:, :, 0]
        
        observed_count = 0
        unobserved_count = 0
        outside_silhouette = 0
        outside_image = 0
        
        proj_data = []  # (uv_proj, uv_obs or None, error or None, is_observed)
        
        for pid in range(n_points):
            X_pt = X[pid].reshape(3, 1)
            
            # Project
            Xc = R @ X_pt + t
            z = float(Xc[2, 0])
            
            if z <= 1e-6:
                continue
            
            uv_proj = K_view @ Xc
            uv_proj = uv_proj[:2, 0] / uv_proj[2, 0]
            
            u, v = int(round(uv_proj[0])), int(round(uv_proj[1]))
            
            # Check if inside image
            if u < 0 or u >= w or v < 0 or v >= h:
                outside_image += 1
                continue
            
            # Check if inside silhouette
            if sil is not None:
                if sil[v, u] < 128:  # Outside silhouette (background)
                    outside_silhouette += 1
            
            # Check if this camera observed this point
            is_observed = False
            uv_obs = None
            error = None
            
            if pid in point_to_track:
                tid = point_to_track[pid]
                tr = tracks[tid]
                if img_id in tr.obs:
                    is_observed = True
                    observed_count += 1
                    kp_id = tr.obs[img_id]
                    uv_obs = feats[img_id].kpts_xy[kp_id].astype(np.float64)
                    error = float(np.linalg.norm(uv_proj - uv_obs))
                else:
                    unobserved_count += 1
            else:
                unobserved_count += 1
            
            proj_data.append((uv_proj, uv_obs, error, is_observed))
        
        # Sort: draw unobserved first, then observed (so observed are on top)
        proj_data.sort(key=lambda x: (x[3], x[2] if x[2] is not None else 0))
        
        # Draw
        for uv_proj, uv_obs, error, is_observed in proj_data[:max_draw_points]:
            p_i = (int(round(uv_proj[0])), int(round(uv_proj[1])))
            
            if is_observed and uv_obs is not None:
                o_i = (int(round(uv_obs[0])), int(round(uv_obs[1])))
                
                # Draw observation (green)
                circle(img, o_i, 4, (0, 255, 0), -1)
                
                # Draw projection (red)
                circle(img, p_i, 3, (0, 0, 255), -1)
                
                # Draw line
                col = (255, 255, 255) if error <= err_thresh_px else (0, 255, 255)
                line(img, o_i, p_i, col, 1)
            else:
                # Unobserved: draw cyan dot
                circle(img, p_i, 3, (255, 255, 0), -1)  # Cyan in BGR
        
        # Summary text
        txt1 = f"observed={observed_count} unobserved={unobserved_count}"
        txt2 = f"outside_img={outside_image}"
        if sil is not None:
            txt2 += f" outside_sil={outside_silhouette}"
        putText(img, txt1, (20, 30), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, LINE_AA)
        putText(img, txt2, (20, 60), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, LINE_AA)
        
        imwrite(str(outp / f"all_proj_{img_id:03d}.png"), img)
    
    print(f"[project_all] Saved to {out_dir}")


def analyze_point_statistics(
    sfm_result,
    cams=None,                              # Optional: for multicam
    K: Optional[np.ndarray] = None,         # Optional: for singlecam  
    percentile_threshold: float = 95.0,
):
    """
    Analyze point cloud statistics to identify outliers.
    
    Works for both MULTICAM and SINGLECAM.
    """
    X = np.asarray(sfm_result.X, np.float64)
    track_to_point = sfm_result.track_to_point
    tracks = sfm_result.tracks
    point_meta = getattr(sfm_result, 'point_meta', {})
    
    point_to_track = {pid: tid for tid, pid in track_to_point.items()}
    
    n = X.shape[0]
    
    print("\n" + "="*60)
    print("POINT CLOUD STATISTICS")
    print("="*60)
    
    # Basic stats
    centroid = np.median(X, axis=0)
    mean_pt = np.mean(X, axis=0)
    
    print(f"\nTotal points: {n}")
    print(f"Centroid (median): [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    print(f"Mean: [{mean_pt[0]:.4f}, {mean_pt[1]:.4f}, {mean_pt[2]:.4f}]")
    
    # Bounding box
    bb_min = X.min(axis=0)
    bb_max = X.max(axis=0)
    bb_size = bb_max - bb_min
    
    print(f"\nBounding box:")
    print(f"  X: [{bb_min[0]:.4f}, {bb_max[0]:.4f}] (size={bb_size[0]:.4f})")
    print(f"  Y: [{bb_min[1]:.4f}, {bb_max[1]:.4f}] (size={bb_size[1]:.4f})")
    print(f"  Z: [{bb_min[2]:.4f}, {bb_max[2]:.4f}] (size={bb_size[2]:.4f})")
    
    # Distance from centroid
    distances = np.linalg.norm(X - centroid, axis=1)
    
    print(f"\nDistance from centroid:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  Std: {distances.std():.4f}")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")
    print(f"  P90: {np.percentile(distances, 90):.4f}")
    print(f"  P95: {np.percentile(distances, 95):.4f}")
    print(f"  P99: {np.percentile(distances, 99):.4f}")
    
    # Identify outliers
    threshold = np.percentile(distances, percentile_threshold)
    outlier_mask = distances > threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    print(f"\nOutliers (beyond P{percentile_threshold}):")
    print(f"  Threshold distance: {threshold:.4f}")
    print(f"  Count: {len(outlier_indices)}")
    
    if len(outlier_indices) > 0:
        print(f"\n  Top 10 outliers:")
        sorted_outliers = sorted(outlier_indices, key=lambda i: distances[i], reverse=True)
        for i, pid in enumerate(sorted_outliers[:10]):
            X_pt = X[pid]
            dist = distances[pid]
            
            # Get track info
            track_len = "?"
            obs_views = []
            if pid in point_to_track:
                tid = point_to_track[pid]
                tr = tracks[tid]
                track_len = len(tr.obs)
                obs_views = list(tr.obs.keys())[:5]
            
            print(f"    pid={pid}: dist={dist:.4f} pos=[{X_pt[0]:.3f}, {X_pt[1]:.3f}, {X_pt[2]:.3f}] "
                  f"track_len={track_len} views={obs_views}")
    
    # Track length distribution
    track_lengths = []
    for tid, pid in track_to_point.items():
        tr = tracks[tid]
        track_lengths.append(len(tr.obs))
    
    track_lengths = np.array(track_lengths)
    
    print(f"\nTrack length distribution:")
    print(f"  Mean: {track_lengths.mean():.2f}")
    print(f"  Median: {np.median(track_lengths):.2f}")
    print(f"  Min: {track_lengths.min()}")
    print(f"  Max: {track_lengths.max()}")
    
    # Count by track length
    for length in range(2, min(8, track_lengths.max() + 1)):
        count = np.sum(track_lengths == length)
        pct = 100 * count / len(track_lengths)
        print(f"  Length {length}: {count} points ({pct:.1f}%)")
    
    if track_lengths.max() >= 8:
        count = np.sum(track_lengths >= 8)
        pct = 100 * count / len(track_lengths)
        print(f"  Length 8+: {count} points ({pct:.1f}%)")
    
    return {
        'centroid': centroid,
        'distances': distances,
        'outlier_indices': outlier_indices,
        'track_lengths': track_lengths,
    }


def find_points_outside_silhouettes(
    sfm_result,
    cams,
    silhouettes: List[np.ndarray],
    K: Optional[np.ndarray] = None,         # For singlecam
    margin_px: int = 3,
    min_outside_ratio: float = 0.5,
):
    """
    Find points that project outside the silhouette in too many views.
    
    Works for both MULTICAM and SINGLECAM.
    """
    is_multicam = cams is not None
    
    X = np.asarray(sfm_result.X, np.float64)
    cam_poses = sfm_result.cam_poses
    track_to_point = sfm_result.track_to_point
    tracks = sfm_result.tracks
    
    point_to_track = {pid: tid for tid, pid in track_to_point.items()}
    
    bad_points = []
    
    for pid in range(X.shape[0]):
        if pid not in point_to_track:
            continue
        
        tid = point_to_track[pid]
        tr = tracks[tid]
        X_pt = X[pid].reshape(3, 1)
        
        inside_count = 0
        outside_count = 0
        
        for img_id in tr.obs.keys():
            if img_id not in cam_poses:
                continue
            if img_id >= len(silhouettes):
                continue
            
            R, t = cam_poses[img_id]
            R = np.asarray(R, np.float64)
            t = np.asarray(t, np.float64).reshape(3, 1)
            
            # Get K
            if is_multicam:
                if hasattr(cams[img_id], "K"):
                    K_view = np.asarray(cams[img_id].K, np.float64)
                else:
                    K_view = np.asarray(cams[img_id][0], np.float64)
            else:
                K_view = np.asarray(K, np.float64)
            
            # Project
            Xc = R @ X_pt + t
            z = float(Xc[2, 0])
            
            if z <= 1e-6:
                continue
            
            uv_proj = K_view @ Xc
            uv_proj = uv_proj[:2, 0] / uv_proj[2, 0]
            
            u, v = int(round(uv_proj[0])), int(round(uv_proj[1]))
            
            sil = silhouettes[img_id]
            if sil.ndim == 3:
                sil = sil[:, :, 0]
            
            h, w = sil.shape
            
            # Check with margin
            is_inside = False
            for du in range(-margin_px, margin_px + 1):
                for dv in range(-margin_px, margin_px + 1):
                    uu, vv = u + du, v + dv
                    if 0 <= uu < w and 0 <= vv < h:
                        if sil[vv, uu] > 127:
                            is_inside = True
                            break
                if is_inside:
                    break
            
            if is_inside:
                inside_count += 1
            else:
                outside_count += 1
        
        total = inside_count + outside_count
        if total > 0:
            outside_ratio = outside_count / total
            if outside_ratio >= min_outside_ratio:
                bad_points.append((pid, outside_ratio, total))
    
    bad_points.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n[silhouette check] Found {len(bad_points)} points with >{min_outside_ratio*100:.0f}% outside silhouette")
    if bad_points:
        print("Top 10:")
        for pid, ratio, total in bad_points[:10]:
            print(f"  pid={pid}: {ratio*100:.1f}% outside ({total} views checked)")
    
    return bad_points
