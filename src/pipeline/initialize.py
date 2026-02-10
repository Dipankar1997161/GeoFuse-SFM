"""
src/pipeline/initialize.py

Initial pair selection and reconstruction seeding.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from src.geometry import recover_pose_from_points, triangulate_and_filter

from .config import TriangulationConfig
from .state import SfMState
from .triangulation import triangulate_pair_with_known_cams


def select_initial_pair(
    state: SfMState,
    config: TriangulationConfig,
    logger=None,
) -> Tuple[int, int]:
    """
    Select the best initial pair for reconstruction.
    
    For multicam: Selects pair with most inliers.
    For singlecam: Selects pair with most good triangulated points.
    
    Args:
        state: SfMState with pairwise matches
        config: TriangulationConfig
        logger: Optional logger
        
    Returns:
        (i0, j0): Initial pair indices
    """
    best_pair = None
    best_good = -1
    
    for (i, j), inliers in state.pairwise.items():
        if len(inliers) < 8:
            continue
        
        if state.multicam:
            # For multicam: score by number of inliers (already verified)
            good = len(inliers)
            
            if logger:
                logger.info(f"  init candidate ({i},{j}): inliers={len(inliers)} good3d={good}")
            
            if good > best_good:
                best_good = good
                best_pair = (i, j)
        
        else:
            # For singlecam: triangulate and count good points
            pts1 = np.float32([state.feats[i].kpts_xy[a] for a, _ in inliers])
            pts2 = np.float32([state.feats[j].kpts_xy[b] for _, b in inliers])
            
            tw = recover_pose_from_points(pts1, pts2, state.K_global, E=None)
            Xij, _keep = triangulate_and_filter(
                pts1, pts2, state.K_global,
                np.eye(3), np.zeros((3, 1)),
                tw.R, tw.t,
                max_reproj_px=config.max_reproj_px,
                min_triang_angle_deg=config.min_triang_angle_deg,
            )
            
            good = int(Xij.shape[0])
            
            if logger:
                logger.info(f"  init candidate ({i},{j}): inliers={len(inliers)} good3d={good}")
            
            if good > best_good:
                best_good = good
                best_pair = (i, j)
    
    if best_pair is None:
        raise RuntimeError("Could not select an initial pair.")
    
    if logger:
        logger.info(f"Chosen initial pair: {best_pair} good3d={best_good}")
    
    return best_pair


def initialize_reconstruction(
    state: SfMState,
    i0: int,
    j0: int,
    config: TriangulationConfig,
    logger=None,
) -> None:
    """
    Initialize the reconstruction with the initial pair.
    
    1. Set up camera poses
    2. Triangulate initial points
    3. Mark initial cameras as registered
    
    Args:
        state: SfMState to initialize
        i0, j0: Initial pair indices
        config: TriangulationConfig
        logger: Optional logger
    """
    if state.multicam:
        _initialize_multicam(state, i0, j0, config, logger)
    else:
        _initialize_singlecam(state, i0, j0, config, logger)


def _initialize_multicam(
    state: SfMState,
    i0: int,
    j0: int,
    config: TriangulationConfig,
    logger=None,
) -> None:
    """Initialize reconstruction for multicam setup."""
    
    # Set all poses from calibration
    state.initialize_multicam_poses()
    
    # Mark initial pair as registered
    state.registered = {i0, j0}
    state.registered_order = [i0, j0]
    
    # Get inliers for initial pair
    inliers0 = state.pairwise[(i0, j0)]
    if logger:
        logger.info(f"Init pair inliers: {len(inliers0)}")
    
    if len(inliers0) < 8:
        raise RuntimeError("Too few inliers for initial triangulation in multicam.")
    
    # Triangulate initial points
    X0, keep0 = triangulate_pair_with_known_cams(
        state.feats[i0], state.feats[j0], inliers0,
        state.cams[i0], state.cams[j0],
        max_reproj_px=config.max_reproj_px,
        min_triang_angle_deg=config.min_triang_angle_deg,
    )
    
    # Add points that belong to tracks
    kept = 0
    for (a, b), ok, Xp in zip(inliers0, keep0, X0):
        if not ok:
            continue
        
        # Find track that contains this match
        for tid, tr in enumerate(state.tracks):
            if tr.obs.get(i0, None) == a and tr.obs.get(j0, None) == b:
                if tid not in state.track_to_point:
                    state.add_point(tid, Xp)
                    kept += 1
                break
    
    if logger:
        logger.info(f"Initialized points (multicam): {kept}")


def _initialize_singlecam(
    state: SfMState,
    i0: int,
    j0: int,
    config: TriangulationConfig,
    logger=None,
) -> None:
    """Initialize reconstruction for singlecam setup."""
    
    # First camera at origin
    state.cam_poses[i0] = (np.eye(3), np.zeros((3, 1)))
    
    # Recover pose for second camera
    inliers0 = state.pairwise[(i0, j0)]
    pts1 = np.float32([state.feats[i0].kpts_xy[a] for a, _ in inliers0])
    pts2 = np.float32([state.feats[j0].kpts_xy[b] for _, b in inliers0])
    tw = recover_pose_from_points(pts1, pts2, state.K_global, E=None)
    state.cam_poses[j0] = (tw.R, tw.t)
    
    # Mark as registered
    state.registered = {i0, j0}
    state.registered_order = [i0, j0]
    
    # Find tracks shared by initial pair
    init_track_ids = []
    init_pts1 = []
    init_pts2 = []
    
    for tid, tr in enumerate(state.tracks):
        if i0 in tr.obs and j0 in tr.obs:
            init_track_ids.append(tid)
            init_pts1.append(state.feats[i0].kpts_xy[tr.obs[i0]])
            init_pts2.append(state.feats[j0].kpts_xy[tr.obs[j0]])
    
    if logger:
        logger.info(f"Tracks shared by init pair: {len(init_track_ids)}")
    
    if len(init_track_ids) < 30:
        raise RuntimeError("Too few shared tracks for initialization.")
    
    # Triangulate
    R1, t1 = state.cam_poses[i0]
    R2, t2 = state.cam_poses[j0]
    
    X0, keep0 = triangulate_and_filter(
        np.float32(init_pts1),
        np.float32(init_pts2),
        state.K_global,
        R1, t1,
        R2, t2,
        max_reproj_px=config.max_reproj_px,
        min_triang_angle_deg=config.min_triang_angle_deg,
    )
    
    # Add points
    kept_ids = [tid for tid, k in zip(init_track_ids, keep0) if k]
    for tid, Xp in zip(kept_ids, X0):
        state.add_point(tid, Xp)
    
    if logger:
        logger.info(f"Initialized points: {len(state.X_list)}")
