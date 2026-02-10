# ---------------------------------------------------------
# Adaptive / majority-vote outlier removal
# ---------------------------------------------------------
import numpy as np
from typing import List, Dict, Tuple, Set

def remove_outlier_points_adaptive(
    K,
    cam_poses,
    X_list,
    track_to_point,
    tracks,
    feats,
    min_observations=3,      # "trusted" points: used to estimate threshold
    good_view_ratio=0.6,     # for >=3-view points
    use_spatial_filtering=True,

    # --- new/adaptive knobs ---
    min_obs_floor=2,         # allow 2-view points to be evaluated
    ratio_2view=1.0,         # require both views good for 2-view points
    ratio_3view=2/3,         # require 2/3 good for 3-view points (more sensible than 0.6)
    mad_k=3.5,               # threshold = median + mad_k * (1.4826*MAD)
    thresh_floor=0.75,       # never below this many pixels
    thresh_cap_mult=4.0,     # never above cap_mult * median_err (rough cap)
    depth_cov_max=1.0,       # allow depth variability (1.0 is forgiving)
    depth_check_good_only=True,
    cams = None,            # Optional: precomputed camera centers for depth checks
):
    """
    Adaptive outlier filtering:
    - considers 2-view points (but with stricter acceptance)
    - estimates reprojection threshold from well-supported points (>=min_observations)
    - majority voting across views
    - optional spatial MAD filtering
    """

    if len(X_list) == 0 or len(track_to_point) == 0:
        return X_list, track_to_point

    # --------------------------------------------------
    # Pass 1: compute reprojection errors for points
    # --------------------------------------------------
    per_point_errors = {}   # pid -> (errors, depths)
    errors_for_thresh = []  # build threshold from >=min_observations (preferred)
    errors_fallback = []    # fallback if none meet min_observations

    for tid, pid in track_to_point.items():
        tr = tracks[tid]
        X = np.asarray(X_list[pid], dtype=np.float64).reshape(3, 1)

        errors = []
        depths = []

        for img_id, kp_id in tr.obs.items():
            if img_id not in cam_poses:
                continue

            R, t = cam_poses[img_id]
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64).reshape(3, 1)

            Xc = (R @ X) + t
            z = float(Xc[2, 0])
            if not np.isfinite(z) or z <= 1e-9:
                continue

            x = Xc[:2, 0] / z
            uv = (K[:2, :2] @ x) + K[:2, 2]


            uv_obs = feats[img_id].kpts_xy[kp_id].astype(np.float64)

            err = float(np.linalg.norm(uv - uv_obs))
            if not np.isfinite(err):
                continue

            errors.append(err)
            depths.append(z)

        # --- adaptive: keep >=2 view points for evaluation ---
        if len(errors) >= min_obs_floor:
            per_point_errors[pid] = (errors, depths)
            errors_fallback.extend(errors)

            # use stronger points to set threshold
            if len(errors) >= min_observations:
                errors_for_thresh.extend(errors)

    if not per_point_errors:
        return X_list, track_to_point

    # --------------------------------------------------
    # Adaptive threshold using robust statistics
    # Prefer >=min_observations points; fallback to all
    # --------------------------------------------------
    errs = np.asarray(errors_for_thresh if len(errors_for_thresh) else errors_fallback, dtype=np.float64)

    if errs.size == 0:
        return X_list, track_to_point

    median_err = float(np.median(errs))
    mad = float(np.median(np.abs(errs - median_err))) + 1e-9
    sigma = 1.4826 * mad

    adaptive_thresh = median_err + mad_k * sigma

    # keep threshold sane
    adaptive_thresh = max(adaptive_thresh, float(thresh_floor))
    # cap to avoid absurd thresholds if distribution is wide
    cap = max(2.0, thresh_cap_mult * max(median_err, 1e-6))
    adaptive_thresh = min(adaptive_thresh, cap)

    print(f"[adaptive filter] median={median_err:.2f} MAD={mad:.2f} thr={adaptive_thresh:.2f} (cap={cap:.2f})")

    # --------------------------------------------------
    # Pass 2: majority-vote filtering (with adaptive ratios)
    # --------------------------------------------------
    kept_pids = []
    new_X_list = []

    # optional debug counters
    drop_ratio = 0
    drop_depth = 0

    for pid, (errors, depths) in per_point_errors.items():
        errors = np.asarray(errors, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64)

        n = int(errors.size)
        if n == 0:
            continue

        good_mask = errors < adaptive_thresh
        n_good = int(np.sum(good_mask))
        ratio = n_good / n

        # adaptive required ratio by support
        if n == 2:
            req_ratio = float(ratio_2view)     # typically 1.0
        elif n == 3:
            req_ratio = float(ratio_3view)     # typically 0.67
        else:
            req_ratio = float(good_view_ratio) # your standard rule

        if ratio < req_ratio:
            drop_ratio += 1
            continue

        # depth consistency check (prefer using good views only)
        if depths.size >= 2:
            if depth_check_good_only and np.any(good_mask):
                d = depths[good_mask]
            else:
                d = depths

            if d.size >= 2:
                cov = float(np.std(d) / max(np.mean(d), 1e-9))
                if cov > float(depth_cov_max):
                    drop_depth += 1
                    continue

        kept_pids.append(pid)
        new_X_list.append(X_list[pid])

    # --------------------------------------------------
    # Optional spatial filtering (MAD in 3D)
    # --------------------------------------------------
    if use_spatial_filtering and len(new_X_list) > 20:
        pts = np.asarray(new_X_list, dtype=np.float64)
        centroid = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)

        med = float(np.median(dists))
        mad3 = float(np.median(np.abs(dists - med))) + 1e-9
        spatial_thresh = med + 4.0 * (1.4826 * mad3)

        keep_mask = dists <= spatial_thresh

        new_X_list = [p for p, k in zip(new_X_list, keep_mask) if k]
        kept_pids  = [pid for pid, k in zip(kept_pids,  keep_mask) if k]

        print(f"[spatial filter] kept {len(new_X_list)} points (thr={spatial_thresh:.3f})")

    # Build pid remap: old_pid -> new_pid
    good_points = {old_pid: new_pid for new_pid, old_pid in enumerate(kept_pids)}

    # rebuild track_to_point
    new_track_to_point = {}
    for tid, old_pid in track_to_point.items():
        if old_pid in good_points:
            new_track_to_point[tid] = good_points[old_pid]

    removed = len(X_list) - len(new_X_list)
    print(f"[adaptive filter] removed {removed}/{len(X_list)} "
          f"(drop_ratio={drop_ratio}, drop_depth={drop_depth})")
    
    kept_set = set(kept_pids)
    all_pids_considered = set(per_point_errors.keys())  # or set(track_to_point.values())
    removed_pids = sorted(list(all_pids_considered - kept_set))
    return new_X_list, new_track_to_point, removed_pids


