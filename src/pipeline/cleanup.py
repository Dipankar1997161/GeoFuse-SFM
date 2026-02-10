"""
src/pipeline/cleanup.py

Outlier removal and point cloud cleanup.
"""

from __future__ import annotations
from pathlib import Path

from src.filter_utils.multicam_filter import remove_outliers_strict_multicam, remove_outliers_spatial
from src.filter_utils.filter import remove_outlier_points_adaptive
from src.visualization.reproject_overlays import save_removed_points_overlays

from .config import CleanupConfig, DiagnosticsConfig
from .state import SfMState


def run_cleanup(
    state: SfMState,
    config: CleanupConfig,
    diag_config: DiagnosticsConfig,
    logger=None,
) -> None:
    """
    Run outlier removal on the point cloud.
    
    For multicam: Strict filter + spatial filter
    For singlecam: Adaptive filter
    
    Updates state.X_list and state.track_to_point in place.
    """
    # Snapshot before cleanup
    track_to_point_before = dict(state.track_to_point)
    removed_pts = []
    
    if state.multicam:
        removed_pts = _cleanup_multicam(state, config, logger)
    else:
        removed_pts = _cleanup_singlecam(state, config, logger)
    
    # Save removed points overlays
    if diag_config.enabled and diag_config.out_dir and diag_config.save_removed_overlays:
        out_path = Path(diag_config.out_dir) / "diagnostics" / "removed_points_overlays"
        save_removed_points_overlays(
            out_dir=str(out_path),
            images=state.images,
            feats=state.feats,
            tracks=state.tracks,
            track_to_point_before=track_to_point_before,
            removed_pids=removed_pts,
        )


def _cleanup_multicam(
    state: SfMState,
    config: CleanupConfig,
    logger=None,
) -> list:
    """
    Cleanup for multicam: strict filter + spatial filter.
    """
    removed_pts = []
    
    # Step 1: Strict reprojection-based filter
    print("\n[final cleanup] Running STRICT multicam outlier removal...")
    
    # Need to pass K (use first camera's K)
    K = state.K_for_view(0)
    
    state.X_list, state.track_to_point, removed_strict = remove_outliers_strict_multicam(
        K=K,
        cam_poses=state.cam_poses,
        X_list=state.X_list,
        track_to_point=state.track_to_point,
        tracks=state.tracks,
        feats=state.feats,
        cams=state.cams,
        max_reproj_any_view=config.max_reproj_any_view,
        max_reproj_median=config.max_reproj_median,
        min_good_views=config.min_good_views,
        verbose=True,
    )
    removed_pts.extend(removed_strict)
    
    # Step 2: Spatial filter
    print("\n[final cleanup] Running SPATIAL outlier removal...")
    
    state.X_list, state.track_to_point, removed_spatial = remove_outliers_spatial(
        X_list=state.X_list,
        track_to_point=state.track_to_point,
        percentile=config.spatial_percentile,
        multiplier=config.spatial_multiplier,
        verbose=True,
    )
    removed_pts.extend(removed_spatial)
    
    return removed_pts


def _cleanup_singlecam(
    state: SfMState,
    config: CleanupConfig,
    logger=None,
) -> list:
    """
    Cleanup for singlecam: adaptive filter.
    """
    print("\n[final cleanup] Running ADAPTIVE outlier removal...")
    
    state.X_list, state.track_to_point, removed_pts = remove_outlier_points_adaptive(
        state.K_global,
        state.cam_poses,
        state.X_list,
        state.track_to_point,
        state.tracks,
        state.feats,
        min_observations=config.adaptive_min_observations,
        good_view_ratio=config.adaptive_good_view_ratio,
        use_spatial_filtering=config.adaptive_spatial_filtering,
        cams=None,  # singlecam
    )
    
    return removed_pts
