"""
src/pipeline/config.py

All configuration dataclasses for the SfM pipeline.
ALL default values live here - no hardcoded numbers elsewhere.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class FeatureConfig:
    """
    Parameters for feature detection.
    
    SIFT tuning guide:
    - For TEXTURED surfaces (buildings, outdoor): Use defaults
    - For TEXTURELESS surfaces (plastic, skin, smooth objects): 
      Lower contrastThreshold to detect subtle features
    """
    method: str = "sift"                   # Feature detector: "sift", "orb", "superpoint"
    
    # SIFT parameters
    sift_nfeatures: int = 0                # Max features (0 = no limit)
    sift_nOctaveLayers: int = 3            # Layers per octave
    sift_contrastThreshold: float = 0.04   # Default SIFT value
    sift_edgeThreshold: float = 10         # Default SIFT value  
    sift_sigma: float = 1.6                # Default SIFT value
    
    # Convenience presets applied via set_preset()
    # "default"     - Standard SIFT for textured scenes
    # "textureless" - Lowered contrast for smooth/plastic surfaces
    # "high_detail" - Maximum feature extraction


@dataclass
class MatchingConfig:
    """Parameters for feature extraction and matching."""
    # Feature detection (reference to FeatureConfig for SIFT params)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Descriptor matching
    ratio: Optional[float] = None          # Lowe's ratio test (None = use adaptive)
    mutual: bool = True                    # Mutual nearest neighbor
    
    # RANSAC / Epipolar
    ransac_thresh_px: Optional[float] = None  # Essential matrix RANSAC (None = use adaptive)
    max_epipolar_dist: float = 3.0         # Epipolar filter for multicam
    min_pair_inliers: Optional[int] = None # Minimum inliers to accept pair (None = use adaptive)
    
    # Pair selection
    pair_window: Optional[int] = None      # Sequential pair window (None = use adaptive)
    match_all_pairs: bool = True           # Match all pairs (multicam)
    
    # PnP
    pnp_thresh_px: Optional[float] = None  # PnP RANSAC threshold (None = use adaptive)
    
    # Silhouette/contour matching
    num_contour_points: int = 150
    contour_max_epipolar_dist: float = 5.0
    contour_min_matches: int = 8
    proximity_threshold: float = 5.0       # For merging SIFT + contour


@dataclass
class TriangulationConfig:
    """Parameters for triangulation and point validation."""
    max_reproj_px: float = 1.5             # Max reprojection error
    min_triang_angle_deg: float = 3.0      # Minimum triangulation angle
    
    # Multi-view validation
    min_views_seed: int = 2                # Min views for initial triangulation
    min_views_accept: int = 3              # Min views to accept point (multicam)
    good_ratio: float = 0.60               # Fraction of views that must be good
    
    # Adaptive thresholds
    adaptive: bool = True
    mad_k: float = 3.5                     # MAD multiplier for adaptive threshold
    thresh_cap_mult: float = 4.0           # Cap on adaptive threshold
    
    # Depth validation
    z_min: float = 1e-6
    z_max: Optional[float] = None
    max_depth_cov: float = 1.0             # Max depth coefficient of variation


@dataclass 
class BAConfig:
    """Parameters for bundle adjustment."""
    mode: str = "none"                     # "none" | "local" | "global_end" | "local+global_end"
    
    # Local BA
    local_every: int = 3                   # Run local BA every N images
    local_max_nfev: int = 50               # Max function evaluations
    local_window: int = 4                  # Camera window size
    local_points_cap: int = 5000           # Max points in local BA
    local_min_points: int = 50             # Min points to run local BA
    local_min_obs: int = 200               # Min observations to run local BA
    
    # Global BA
    global_max_nfev: int = 25
    global_points_cap: int = 20000
    global_min_points: int = 200           # Min points to run global BA
    global_min_obs: int = 800              # Min observations to run global BA
    
    # Optimization
    loss: str = "cauchy"                   # Loss function: "linear", "huber", "cauchy"
    f_scale: float = 2.0                   # Loss function scale
    verbose: int = 0                       # Scipy verbosity
    
    # Multicam behavior
    fix_poses_multicam: bool = True        # Skip BA for multicam (poses calibrated)


@dataclass
class CleanupConfig:
    """Parameters for outlier removal."""
    # Strict multicam filter
    max_reproj_any_view: float = 5.0       # Remove if error > this in ANY view
    max_reproj_median: float = 1.0         # Remove if median > this
    min_good_views: int = 3                # Require at least this many good views
    
    # Spatial filter
    spatial_percentile: float = 95.0
    spatial_multiplier: float = 2.0
    
    # Adaptive singlecam filter
    adaptive_good_view_ratio: float = 0.60
    adaptive_min_observations: int = 2
    adaptive_spatial_filtering: bool = True


@dataclass
class DiagnosticsConfig:
    """Parameters for diagnostics and visualization."""
    enabled: bool = True
    out_dir: Optional[str] = None
    
    # Reprojection reporting
    reproj_report: bool = True
    reproj_max_obs: int = 20000
    
    # Overlay saving
    save_kp_overlays: bool = True
    save_match_overlays: bool = False
    save_reproj_overlays: bool = True
    save_removed_overlays: bool = True
    
    # Debug pairs for match overlays
    debug_match_pairs: List[tuple] = field(
        default_factory=lambda: [(0, 1), (0, 2), (10, 11), (20, 21)]
    )


@dataclass
class SfMConfig:
    """
    Master configuration for the entire SfM pipeline.
    
    ALL numeric defaults live in this file. No hardcoded values elsewhere.
    
    Usage:
        # Default config
        config = SfMConfig()
        
        # For textureless objects (plastic, skin, smooth surfaces)
        config = get_textureless_config()
        
        # Modify specific values
        config.matching.ratio = 0.75
        config.triangulation.min_views_accept = 3
    """
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    ba: BAConfig = field(default_factory=BAConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    
    # Logging
    verbose: bool = True
    log_every_pair: int = 50
    
    def set_feature_preset(self, preset: str) -> None:
        """
        Set SIFT parameters based on scene type.
        
        Presets:
            "default"     - Standard SIFT for textured scenes (buildings, outdoor)
            "textureless" - For smooth/plastic/skin surfaces
            "high_detail" - Maximum features for complex scenes
        """
        if preset == "default":
            self.matching.feature.sift_nfeatures = 0
            self.matching.feature.sift_contrastThreshold = 0.04
            self.matching.feature.sift_edgeThreshold = 10
            self.matching.feature.sift_sigma = 1.6
        elif preset == "textureless":
            self.matching.feature.sift_nfeatures = 150000
            self.matching.feature.sift_contrastThreshold = 0.004  # 10x lower!
            self.matching.feature.sift_edgeThreshold = 10
            self.matching.feature.sift_sigma = 1.2
        elif preset == "high_detail":
            self.matching.feature.sift_nfeatures = 0
            self.matching.feature.sift_contrastThreshold = 0.02
            self.matching.feature.sift_edgeThreshold = 15
            self.matching.feature.sift_sigma = 1.6
        else:
            raise ValueError(f"Unknown preset: {preset}. Use 'default', 'textureless', or 'high_detail'")
    
    @classmethod
    def from_dict(cls, d: dict) -> "SfMConfig":
        """Create config from dictionary (e.g., loaded from YAML/JSON)."""
        feature_dict = d.get("matching", {}).get("feature", {})
        matching_dict = d.get("matching", {})
        matching_dict["feature"] = FeatureConfig(**feature_dict)
        
        return cls(
            matching=MatchingConfig(**matching_dict),
            triangulation=TriangulationConfig(**d.get("triangulation", {})),
            ba=BAConfig(**d.get("ba", {})),
            cleanup=CleanupConfig(**d.get("cleanup", {})),
            diagnostics=DiagnosticsConfig(**d.get("diagnostics", {})),
            verbose=d.get("verbose", True),
            log_every_pair=d.get("log_every_pair", 50),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (for saving to YAML/JSON)."""
        from dataclasses import asdict
        return asdict(self)
    
    def apply_adaptive_params(self, images: List[np.ndarray]) -> None:
        """
        Apply adaptive parameters based on image size.
        Only fills in None values - explicit values are preserved.
        """
        adaptive = compute_adaptive_params(images)
        
        if self.matching.ratio is None:
            self.matching.ratio = adaptive["ratio"]
        if self.matching.ransac_thresh_px is None:
            self.matching.ransac_thresh_px = adaptive["ransac_thresh_px"]
        if self.matching.pair_window is None:
            self.matching.pair_window = adaptive["pair_window"]
        if self.matching.pnp_thresh_px is None:
            self.matching.pnp_thresh_px = adaptive["pnp_thresh_px"]
        if self.matching.min_pair_inliers is None:
            self.matching.min_pair_inliers = adaptive["min_pair_inliers"]


def compute_adaptive_params(images: List[np.ndarray]) -> dict:
    """
    Compute dataset-dependent thresholds based on image size.
    """
    h, w = images[0].shape[:2]
    n = len(images)
    max_dim = max(h, w)

    params = {}

    # Descriptor ratio - looser for higher resolution
    if max_dim < 1500:
        params["ratio"] = 0.75
    elif max_dim < 3000:
        params["ratio"] = 0.80
    else:
        params["ratio"] = 0.85

    # RANSAC threshold for Essential matrix
    params["ransac_thresh_px"] = max(1.0, 0.002 * max_dim)

    # Pair window
    params["pair_window"] = min(max(8, n // 4), 20)

    # PnP threshold
    params["pnp_thresh_px"] = max(3.0, 0.002 * max_dim)

    # Minimum inliers
    params["min_pair_inliers"] = max(20, int(0.01 * 2000))

    return params


def has_mode(ba_mode: str, token: str) -> bool:
    """Check if BA mode string contains a specific mode."""
    if not ba_mode:
        return False
    s = ba_mode.replace(",", "+").replace(" ", "")
    parts = [p for p in s.split("+") if p]
    return token in parts


# ============================================================
# PRESET CONFIGURATIONS
# ============================================================

def get_default_config() -> SfMConfig:
    """Get default configuration for textured scenes."""
    config = SfMConfig()
    config.set_feature_preset("default")
    return config


def get_multicam_config() -> SfMConfig:
    """Get configuration for multicam (calibrated rig) with textured objects."""
    config = SfMConfig(
        matching=MatchingConfig(
            feature=FeatureConfig(
                sift_nfeatures=0,
                sift_contrastThreshold=0.04,  # Default for textured
            ),
            ratio=0.80,
            max_epipolar_dist=3.0,
        ),
        triangulation=TriangulationConfig(
            max_reproj_px=1.5,
            min_views_accept=3,
        ),
        ba=BAConfig(
            mode="none",
            fix_poses_multicam=True,
        ),
        cleanup=CleanupConfig(
            max_reproj_any_view=5.0,
            max_reproj_median=1.0,
            min_good_views=3,
        ),
    )
    return config


def get_textureless_config() -> SfMConfig:
    """
    Get configuration for TEXTURELESS objects (plastic, skin, smooth surfaces).
    
    Key differences from default:
    - Much lower SIFT contrastThreshold (0.004 vs 0.04)
    - Higher nfeatures limit
    - Tuned for detecting subtle gradients
    """
    config = SfMConfig(
        matching=MatchingConfig(
            feature=FeatureConfig(
                sift_nfeatures=150000,
                sift_contrastThreshold=0.004,  # 10x lower for textureless!
                sift_edgeThreshold=10,
                sift_sigma=1.2,
            ),
            ratio=0.80,
            max_epipolar_dist=3.0,
        ),
        triangulation=TriangulationConfig(
            max_reproj_px=1.5,
            min_views_accept=3,
        ),
        ba=BAConfig(
            mode="none",
            fix_poses_multicam=True,
        ),
        cleanup=CleanupConfig(
            max_reproj_any_view=5.0,
            max_reproj_median=1.0,
            min_good_views=3,
        ),
    )
    return config


def get_singlecam_config() -> SfMConfig:
    """Get configuration for singlecam (sequential images)."""
    config = SfMConfig(
        matching=MatchingConfig(
            feature=FeatureConfig(
                sift_nfeatures=0,
                sift_contrastThreshold=0.04,
            ),
            mutual=True,
        ),
        triangulation=TriangulationConfig(
            max_reproj_px=2.0,
            min_views_accept=2,
            min_triang_angle_deg=2.0,
        ),
        ba=BAConfig(
            mode="local+global_end",
            fix_poses_multicam=False,
        ),
        cleanup=CleanupConfig(
            adaptive_good_view_ratio=0.60,
            adaptive_min_observations=2,
            adaptive_spatial_filtering=True,
        ),
    )
    return config


def get_turntable_config() -> SfMConfig:
    """Get configuration for turntable capture."""
    config = SfMConfig(
        matching=MatchingConfig(
            ratio=0.85,
        ),
        triangulation=TriangulationConfig(
            max_reproj_px=2.0,
            min_triang_angle_deg=1.0,
            min_views_accept=2,
        ),
        ba=BAConfig(
            mode="local+global_end",
        ),
    )
    return config