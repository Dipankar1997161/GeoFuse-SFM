"""
src/pipeline/ba_runner.py

Bundle adjustment operations.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import time

from src.ba import BAObservation, bundle_adjustment

from .config import BAConfig, DiagnosticsConfig
from .state import SfMState


def build_ba_problem(
    state: SfMState,
    selected_cam_order: List[int],
    points_cap: int,
) -> Tuple[List[int], np.ndarray, List[BAObservation], Dict[int, int]]:
    """
    Build a bundle adjustment problem.
    
    Args:
        state: SfMState
        selected_cam_order: List of camera indices to include
        points_cap: Maximum number of points to include
        
    Returns:
        cam_order: Cameras actually used
        X_sub: (N, 3) points subset
        obs_sub: List of observations
        oldpid_to_newpid: Mapping from old to new point indices
    """
    cam_index = {img_id: ci for ci, img_id in enumerate(selected_cam_order)}
    
    # Count observations per point
    point_hits: Dict[int, int] = {}
    for tid, old_pid in state.track_to_point.items():
        tr = state.tracks[tid]
        cnt = 0
        for img_id in tr.obs.keys():
            if img_id in cam_index:
                cnt += 1
        if cnt >= 2:
            point_hits[old_pid] = cnt
    
    if not point_hits:
        return selected_cam_order, np.zeros((0, 3), np.float64), [], {}
    
    # Select top points by observation count
    chosen_old_pids = [
        pid for pid, _ in sorted(point_hits.items(), key=lambda kv: kv[1], reverse=True)[:points_cap]
    ]
    oldpid_to_newpid = {old: new for new, old in enumerate(chosen_old_pids)}
    
    X_sub = np.asarray([state.X_list[old] for old in chosen_old_pids], dtype=np.float64)
    
    # Build observations
    obs_sub: List[BAObservation] = []
    for tid, old_pid in state.track_to_point.items():
        if old_pid not in oldpid_to_newpid:
            continue
        tr = state.tracks[tid]
        for img_id, kp_id in tr.obs.items():
            if img_id not in cam_index:
                continue
            obs_sub.append(BAObservation(
                cam_idx=cam_index[img_id],
                pt_idx=oldpid_to_newpid[old_pid],
                xy=state.feats[img_id].kpts_xy[kp_id].astype(np.float64),
            ))
    
    return selected_cam_order, X_sub, obs_sub, oldpid_to_newpid


def apply_ba_updates(
    state: SfMState,
    cam_refined: Dict[int, Tuple[np.ndarray, np.ndarray]],
    X_refined_sub: np.ndarray,
    oldpid_to_newpid: Dict[int, int],
    cam_order_used: List[int],
) -> None:
    """
    Apply bundle adjustment results back to state.
    
    Args:
        state: SfMState to update
        cam_refined: Refined camera poses
        X_refined_sub: Refined points
        oldpid_to_newpid: Point index mapping
        cam_order_used: Cameras that were optimized
    """
    # Update camera poses
    for img_id in cam_order_used:
        state.cam_poses[img_id] = cam_refined[img_id]
    
    # Update points
    inv = {new: old for old, new in oldpid_to_newpid.items()}
    for new_pid, Xp in enumerate(X_refined_sub):
        old_pid = inv[new_pid]
        state.X_list[old_pid] = Xp


def run_local_ba(
    state: SfMState,
    config: BAConfig,
    diag_config: DiagnosticsConfig,
    logger=None,
    iteration: int = 0,
) -> None:
    """
    Run local bundle adjustment.
    
    For multicam: Skips (poses are calibrated).
    For singlecam: Optimizes recent cameras and their points.
    """
    if len(state.registered_order) < 3:
        return
    
    # Report stats before
    if diag_config.reproj_report:
        s0 = state.reprojection_error_stats(diag_config.reproj_max_obs)
        if logger:
            logger.info(
                f"Reproj BEFORE local BA: mean={s0['mean']:.2f}px median={s0['median']:.2f}px "
                f"p95={s0['p95']:.2f}px (n={int(s0['n'])})"
            )
    
    # Skip BA for multicam (poses are calibrated)
    if state.multicam and config.fix_poses_multicam:
        if logger:
            logger.info("Local BA skipped (multicam - poses are calibrated)")
    else:
        # Select recent cameras
        cam_order_local = state.registered_order[-config.local_window:]
        cam_order_used, X_sub, obs_sub, old2new = build_ba_problem(
            state, cam_order_local, config.local_points_cap
        )
        
        if X_sub.shape[0] >= 50 and len(obs_sub) >= 200:
            if logger:
                logger.info(
                    f"Local BA starting: cams={len(cam_order_used)} "
                    f"points={X_sub.shape[0]} obs={len(obs_sub)} max_nfev={config.local_max_nfev}"
                )
            
            t0 = time.time()
            
            # Get K for first camera (assume same K for local BA)
            K = state.K_for_view(cam_order_used[0])
            
            cam_refined, X_refined_sub = bundle_adjustment(
                K=K,
                cam_order=cam_order_used,
                cam_poses=state.cam_poses,
                X=X_sub,
                observations=obs_sub,
                loss=config.loss,
                f_scale=config.f_scale,
                max_nfev=config.local_max_nfev,
                verbose=config.verbose,
            )
            
            if logger:
                logger.info(f"Local BA finished in {time.time()-t0:.2f}s")
            
            apply_ba_updates(state, cam_refined, X_refined_sub, old2new, cam_order_used)
        else:
            if logger:
                logger.info("Local BA skipped (not enough points/observations).")
    
    # Report stats after
    if diag_config.reproj_report:
        s1 = state.reprojection_error_stats(diag_config.reproj_max_obs)
        if logger:
            logger.info(
                f"Reproj AFTER  local BA: mean={s1['mean']:.2f}px median={s1['median']:.2f}px "
                f"p95={s1['p95']:.2f}px (n={int(s1['n'])})"
            )


def run_global_ba(
    state: SfMState,
    config: BAConfig,
    diag_config: DiagnosticsConfig,
    logger=None,
) -> None:
    """
    Run global bundle adjustment at the end.
    
    For multicam: Skips (poses are calibrated).
    For singlecam: Optimizes all cameras and points.
    """
    if len(state.registered_order) < 3:
        return
    
    # Report stats before
    if diag_config.reproj_report:
        s0 = state.reprojection_error_stats(diag_config.reproj_max_obs)
        if logger:
            logger.info(
                f"Reproj BEFORE global_end BA: mean={s0['mean']:.2f}px median={s0['median']:.2f}px "
                f"p95={s0['p95']:.2f}px (n={int(s0['n'])})"
            )
    
    # Skip BA for multicam (poses are calibrated)
    if state.multicam and config.fix_poses_multicam:
        if logger:
            logger.info("Global-end BA skipped (multicam - poses are calibrated)")
    else:
        cam_order_all = list(state.registered_order)
        cam_order_used, X_sub, obs_sub, old2new = build_ba_problem(
            state, cam_order_all, config.global_points_cap
        )
        
        if X_sub.shape[0] >= config.global_min_points and len(obs_sub) >= config.global_min_obs:
            if logger:
                logger.info(
                    f"Global-end BA starting: cams={len(cam_order_used)} "
                    f"points={X_sub.shape[0]} obs={len(obs_sub)} max_nfev={config.global_max_nfev}"
                )
            
            t0 = time.time()
            
            # Get K for first camera
            K = state.K_for_view(cam_order_used[0])
            
            cam_refined, X_refined_sub = bundle_adjustment(
                K=K,
                cam_order=cam_order_used,
                cam_poses=state.cam_poses,
                X=X_sub,
                observations=obs_sub,
                loss=config.loss,
                f_scale=config.f_scale,
                max_nfev=config.global_max_nfev,
                verbose=config.verbose,
            )
            
            if logger:
                logger.info(f"Global-end BA finished in {time.time()-t0:.2f}s")
            
            apply_ba_updates(state, cam_refined, X_refined_sub, old2new, cam_order_used)
        else:
            if logger:
                logger.info("Global-end BA skipped (not enough points/observations).")
    
    # Report stats after
    if diag_config.reproj_report:
        s1 = state.reprojection_error_stats(diag_config.reproj_max_obs)
        if logger:
            logger.info(
                f"Reproj AFTER  global_end BA: mean={s1['mean']:.2f}px median={s1['median']:.2f}px "
                f"p95={s1['p95']:.2f}px (n={int(s1['n'])})"
            )
