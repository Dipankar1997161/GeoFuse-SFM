"""
src/pipeline/__init__.py

Modular SfM pipeline.

Usage:
    from src.pipeline import run_incremental_sfm, SfMConfig, SfMResult
    
    # Simple usage
    result = run_incremental_sfm(images, K=K, out_dir="output")
    
    # With config
    config = SfMConfig()
    config.triangulation.min_views_accept = 3
    config.ba.mode = "local+global_end"
    result = run_incremental_sfm(images, K=K, config=config)
    
    # For multicam
    result = run_incremental_sfm(images, cams=cams, silhouettes=silhouettes)
"""

from .config import (
    SfMConfig,
    MatchingConfig,
    TriangulationConfig,
    BAConfig,
    CleanupConfig,
    DiagnosticsConfig,
    compute_adaptive_params,
    get_default_config,
    get_multicam_config,
    get_singlecam_config,
    get_turntable_config,
)

from .state import (
    SfMState,
    SfMResult,
    PointMeta,
)

# Import run functions from run_sfm module (not from this package to avoid circular import)
from ..run_sfm import (
    run_incremental_sfm,
    run_sfm,
    run_multicam_sfm,
    run_singlecam_sfm,
)

__all__ = [
    # Config
    "SfMConfig",
    "MatchingConfig",
    "TriangulationConfig",
    "BAConfig",
    "CleanupConfig",
    "DiagnosticsConfig",
    "compute_adaptive_params",
    "get_default_config",
    "get_multicam_config",
    "get_singlecam_config",
    "get_turntable_config",
    # State
    "SfMState",
    "SfMResult",
    "PointMeta",
    # Entry points
    "run_incremental_sfm",
    "run_sfm",
    "run_multicam_sfm",
    "run_singlecam_sfm",
]
