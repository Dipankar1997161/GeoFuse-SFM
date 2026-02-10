"""
src/pipeline/registration.py

Incremental image registration loop.
Handles next image selection, pose estimation, and triangulation.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

from src.geometry_utils.projective import camera_center

from .config import SfMConfig, has_mode
from .state import SfMState
from .triangulation import triangulate_new_points
from .ba_runner import run_local_ba


def _pnp_pose(
    K: np.ndarray,
    X3d: np.ndarray,
    x2d: np.ndarray,
    reproj_thresh: float = 3.0,
) -> Optional[tuple]:
    """
    PnP RANSAC pose estimation.
    
    Returns:
        (R, t, inlier_indices) or None if failed
    """
    if X3d.shape[0] < 6:
        return None
    
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X3d.astype(np.float64),
        x2d.astype(np.float64),
        K.astype(np.float64),
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=reproj_thresh,
        iterationsCount=2000,
        confidence=0.999,
    )
    
    if (not ok) or (inliers is None) or (len(inliers) < 6):
        return None
    
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1), inliers.ravel()


def candidate_score(state: SfMState, img_id: int) -> float:
    """
    Score a candidate image for registration.
    
    Score = 2D-3D correspondences + 0.1 * max_baseline
    """
    corr = state.count_2d3d_correspondences(img_id)
    
    if img_id not in state.cam_poses:
        return corr
    
    try:
        Cn = state.camera_center_for_view(img_id)
    except:
        return corr
    
    baseline = 0.0
    for rid in state.registered:
        try:
            Cr = state.camera_center_for_view(rid)
            baseline = max(baseline, np.linalg.norm(Cn - Cr))
        except:
            pass
    
    return corr + 0.1 * baseline


def select_next_image(state: SfMState) -> int:
    """
    Select the next image to register.
    
    Returns image with highest candidate score.
    """
    candidates = [img for img in range(state.n_images) if img not in state.registered]
    candidates.sort(key=lambda x: candidate_score(state, x), reverse=True)
    return candidates[0]


def register_next_image(
    state: SfMState,
    next_img: int,
    config: SfMConfig,
    logger=None,
) -> bool:
    """
    Register a single image.
    
    For multicam: Pose is already known.
    For singlecam: Runs PnP.
    
    Returns:
        True if registration succeeded
    """
    corr = state.count_2d3d_correspondences(next_img)
    
    if logger:
        logger.info(f"Next image: {next_img} 2D-3D corr={corr}")
    
    # Check minimum correspondences
    if state.multicam:
        min_corr_required = 0
    else:
        min_corr_required = max(4, int(0.01 * len(state.track_to_point)))
    
    if corr < min_corr_required:
        if state.multicam:
            # Check if image has ANY tracks
            has_tracks = any(next_img in tr.obs for tr in state.tracks)
            if not has_tracks:
                if logger:
                    logger.info(f"Stopping: img{next_img} has no tracks at all")
                return False
            if logger:
                logger.info(f"[multicam] img{next_img} has {corr} corr but pose known, continuing...")
        else:
            if logger:
                logger.info(f"Stopping: too few correspondences for img{next_img} (corr={corr})")
            return False
    
    # Gather 2D-3D correspondences
    X3d = []
    x2d = []
    for tid, tr in enumerate(state.tracks):
        if tid not in state.track_to_point:
            continue
        if next_img not in tr.obs:
            continue
        pid = state.track_to_point[tid]
        X3d.append(state.X_list[pid])
        x2d.append(state.feats[next_img].kpts_xy[tr.obs[next_img]])
    
    X3d = np.asarray(X3d, dtype=np.float64)
    x2d = np.asarray(x2d, dtype=np.float32)
    
    # Pose estimation
    if state.multicam:
        # Pose is already known
        if logger:
            logger.info(f"[multicam] Pose known: img{next_img} using decomposed (R,t)")
    else:
        # Run PnP
        pnp_result = _pnp_pose(
            state.K_global, X3d, x2d,
            reproj_thresh=config.matching.ransac_thresh_px
        )
        
        if pnp_result is None:
            if logger:
                logger.info(f"Stopping: PnP failed for img{next_img}")
            return False
        
        R, t, inl_idx = pnp_result
        state.cam_poses[next_img] = (R, t)
        
        if logger:
            logger.info(f"PnP success: img{next_img} inliers={len(inl_idx)}/{X3d.shape[0]}")
    
    # Mark as registered
    state.registered.add(next_img)
    state.registered_order.append(next_img)
    
    # Triangulate new points
    added = triangulate_new_points(state, next_img, config.triangulation)
    
    if logger:
        logger.info(f"Triangulated new points from img{next_img}: +{added} (total={len(state.X_list)})")
    
    return True


def run_incremental_registration(
    state: SfMState,
    config: SfMConfig,
    logger=None,
) -> None:
    """
    Main incremental registration loop.
    
    Iteratively:
    1. Select next best image
    2. Register it (PnP for singlecam, known pose for multicam)
    3. Triangulate new points
    4. Run local BA (if enabled)
    """
    iters = 0
    
    while len(state.registered) < state.n_images:
        if logger:
            logger.info(
                f"--- Incremental step {iters+1}: "
                f"registered {len(state.registered)}/{state.n_images} | "
                f"points={len(state.X_list)} ---"
            )
        
        # Select next image
        next_img = select_next_image(state)
        
        # Register it
        success = register_next_image(state, next_img, config, logger)
        if not success:
            break
        
        iters += 1
        
        # Local BA
        if has_mode(config.ba.mode, "local") and (iters % config.ba.local_every == 0):
            run_local_ba(state, config.ba, config.diagnostics, logger, iters)
