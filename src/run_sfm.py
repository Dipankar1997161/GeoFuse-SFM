"""
src/pipeline/run_sfm.py

Main entry point for the SfM pipeline.
This is a thin orchestrator that calls the modular components.

ALL numeric defaults come from config.py - no hardcoded values here.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from data_io.camera import DecomposedCamera
from utils.logging_utils import make_logger, timed

from .pipeline.config import (
    SfMConfig,
    MatchingConfig,
    TriangulationConfig,
    BAConfig,
    CleanupConfig,
    DiagnosticsConfig,
    has_mode,
    get_multicam_config,
    get_singlecam_config,
)
from .pipeline.state import SfMState, SfMResult
from .pipeline.matching import run_matching
from .pipeline.tracks_builder import build_tracks
from .pipeline.initialize import select_initial_pair, initialize_reconstruction
from .pipeline.triangulation import triangulate_new_points
from .pipeline.registration import run_incremental_registration
from .pipeline.ba_runner import run_local_ba, run_global_ba
from .pipeline.cleanup import run_cleanup


def run_incremental_sfm(
    images: List[np.ndarray],
    K: Optional[np.ndarray] = None,
    silhouettes: Optional[List[np.ndarray]] = None,
    cams: Optional[List[DecomposedCamera]] = None,
    config: Optional[SfMConfig] = None,
    out_dir: Optional[str] = None,
) -> SfMResult:
    """
    Run incremental Structure from Motion.
    
    Args:
        images: List of input images
        K: Intrinsic matrix (required for singlecam, ignored for multicam)
        silhouettes: Optional silhouette masks
        cams: Optional list of DecomposedCamera for multicam mode
        config: SfMConfig object (if None, uses appropriate default)
        out_dir: Output directory for diagnostics (overrides config.diagnostics.out_dir)
        
    Returns:
        SfMResult with camera poses, 3D points, and metadata
        
    Example:
        # Basic singlecam usage
        result = run_incremental_sfm(images, K=K)
        
        # Multicam with silhouettes
        result = run_incremental_sfm(images, cams=cams, silhouettes=silhouettes)
        
        # With custom config
        config = SfMConfig()
        config.triangulation.min_views_accept = 3
        config.ba.mode = "local+global_end"
        result = run_incremental_sfm(images, K=K, config=config)
    """
    
    # Determine if multicam
    multicam = cams is not None
    
    # Use appropriate default config if not provided
    if config is None:
        if multicam:
            config = get_multicam_config()
        else:
            config = get_singlecam_config()
    
    # Override out_dir if provided
    if out_dir is not None:
        config.diagnostics.enabled = True
        config.diagnostics.out_dir = out_dir
    
    # Apply adaptive parameters (fills in None values based on image size)
    config.apply_adaptive_params(images)
    
    # Setup logger
    logger = make_logger("sfm", level=(20 if config.verbose else 40))
    logger.info(f"Images: {len(images)} | multicam={multicam} | ba_mode={config.ba.mode}")
    
    # Print configuration summary
    if config.verbose:
        print("\nConfiguration:")
        print(f"  Matching:")
        print(f"    ratio={config.matching.ratio}")
        print(f"    ransac_thresh_px={config.matching.ransac_thresh_px}")
        print(f"    pair_window={config.matching.pair_window}")
        print(f"  Triangulation:")
        print(f"    max_reproj_px={config.triangulation.max_reproj_px}")
        print(f"    min_views_accept={config.triangulation.min_views_accept}")
        print(f"  BA: mode={config.ba.mode}")
        print(f"  Cleanup:")
        print(f"    min_good_views={config.cleanup.min_good_views}")
        print()
    
    # Initialize state
    state = SfMState(images, K=K, cams=cams, silhouettes=silhouettes)
    
    # =========================================================
    # STAGE 1: Feature extraction and matching
    # =========================================================
    with timed(logger, "Feature extraction and matching"):
        run_matching(state, config.matching, config.diagnostics, logger)
    
    # =========================================================
    # STAGE 2: Track building
    # =========================================================
    with timed(logger, "Building tracks"):
        build_tracks(state, logger)
    
    # =========================================================
    # STAGE 3: Initialization
    # =========================================================
    with timed(logger, "Initialization"):
        # Select initial pair
        i0, j0 = select_initial_pair(state, config.triangulation, logger)
        
        # Initialize poses and seed points
        initialize_reconstruction(state, i0, j0, config.triangulation, logger)
    
    # Report initial reprojection stats
    if config.diagnostics.reproj_report:
        stats = state.reprojection_error_stats(config.diagnostics.reproj_max_obs)
        logger.info(
            f"Reproj init: n={int(stats['n'])} mean={stats['mean']:.2f}px "
            f"median={stats['median']:.2f}px p90={stats['p90']:.2f}px p95={stats['p95']:.2f}px"
        )
    
    # =========================================================
    # STAGE 4: Incremental registration
    # =========================================================
    with timed(logger, "Incremental registration"):
        run_incremental_registration(state, config, logger)
    
    # =========================================================
    # STAGE 5: Global BA (if enabled)
    # =========================================================
    if has_mode(config.ba.mode, "global_end"):
        with timed(logger, "Global bundle adjustment"):
            run_global_ba(state, config.ba, config.diagnostics, logger)
    
    # =========================================================
    # STAGE 6: Cleanup
    # =========================================================
    with timed(logger, "Outlier removal"):
        run_cleanup(state, config.cleanup, config.diagnostics, logger)
    
    # =========================================================
    # Build and return result
    # =========================================================
    logger.info(f"Finished. Registered={len(state.registered_order)}/{state.n_images} Points={len(state.X_list)}")
    
    return state.build_result()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def run_sfm(
    images: List[np.ndarray],
    K: Optional[np.ndarray] = None,
    cams: Optional[List[DecomposedCamera]] = None,
    silhouettes: Optional[List[np.ndarray]] = None,
    out_dir: Optional[str] = None,
    **config_overrides,
) -> SfMResult:
    """
    Simplified entry point with keyword overrides.
    
    Args:
        images: Input images
        K: Intrinsic matrix (for singlecam)
        cams: DecomposedCamera list (for multicam)
        silhouettes: Optional masks
        out_dir: Output directory
        **config_overrides: Override specific config values
        
    Example:
        # Override specific values
        result = run_sfm(
            images, K=K,
            ba_mode="local+global_end",
            min_views_accept=3,
            max_reproj_px=2.0,
        )
    """
    multicam = cams is not None
    
    if multicam:
        config = get_multicam_config()
    else:
        config = get_singlecam_config()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config.matching, key):
            setattr(config.matching, key, value)
        elif hasattr(config.triangulation, key):
            setattr(config.triangulation, key, value)
        elif hasattr(config.ba, key):
            setattr(config.ba, key, value)
        elif hasattr(config.cleanup, key):
            setattr(config.cleanup, key, value)
        elif hasattr(config.diagnostics, key):
            setattr(config.diagnostics, key, value)
        elif key == "ba_mode":
            config.ba.mode = value
        elif key == "verbose":
            config.verbose = value
        else:
            raise ValueError(f"Unknown config key: {key}")
    
    return run_incremental_sfm(
        images=images,
        K=K,
        cams=cams,
        silhouettes=silhouettes,
        config=config,
        out_dir=out_dir,
    )


def run_multicam_sfm(
    images: List[np.ndarray],
    cams: List[DecomposedCamera],
    silhouettes: Optional[List[np.ndarray]] = None,
    out_dir: Optional[str] = None,
    config: Optional[SfMConfig] = None,
) -> SfMResult:
    """
    Entry point specifically for multicam reconstruction.
    
    Uses multicam-optimized defaults.
    """
    if config is None:
        config = get_multicam_config()
    
    return run_incremental_sfm(
        images=images,
        K=None,
        cams=cams,
        silhouettes=silhouettes,
        config=config,
        out_dir=out_dir,
    )


def run_singlecam_sfm(
    images: List[np.ndarray],
    K: np.ndarray,
    out_dir: Optional[str] = None,
    config: Optional[SfMConfig] = None,
) -> SfMResult:
    """
    Entry point specifically for singlecam reconstruction.
    
    Uses singlecam-optimized defaults.
    """
    if config is None:
        config = get_singlecam_config()
    
    return run_incremental_sfm(
        images=images,
        K=K,
        cams=None,
        silhouettes=None,
        config=config,
        out_dir=out_dir,
    )
