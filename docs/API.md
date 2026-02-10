# GeoFuse-SFM API Reference

Complete reference for configuration classes, functions, and data structures.

---

## Configuration Classes

### SfMConfig

Master configuration container.

```python
from src.pipeline import SfMConfig

@dataclass
class SfMConfig:
    matching: MatchingConfig
    triangulation: TriangulationConfig
    ba: BAConfig
    cleanup: CleanupConfig
    diagnostics: DiagnosticsConfig
    verbose: bool = True
    log_every_pair: int = 50
```

**Methods:**

| Method | Description |
|--------|-------------|
| `set_feature_preset(preset)` | Set SIFT params: `"default"`, `"textureless"`, `"high_detail"` |
| `apply_adaptive_params(images)` | Fill `None` values based on image size |
| `from_dict(d)` | Create from dictionary (YAML/JSON) |
| `to_dict()` | Export to dictionary |

---

### FeatureConfig

SIFT/ORB feature detection parameters.

```python
@dataclass
class FeatureConfig:
    method: str = "sift"              # "sift" or "orb"
    sift_nfeatures: int = 0           # Max features (0 = unlimited)
    sift_nOctaveLayers: int = 3
    sift_contrastThreshold: float = 0.04   # Lower for textureless (0.004)
    sift_edgeThreshold: float = 10
    sift_sigma: float = 1.6
```

**Presets:**

| Preset | contrastThreshold | nfeatures | Use Case |
|--------|-------------------|-----------|----------|
| `default` | 0.04 | 0 | Textured scenes |
| `textureless` | 0.004 | 150000 | Plastic, skin, smooth |
| `high_detail` | 0.02 | 0 | Maximum features |

---

### MatchingConfig

Feature matching parameters.

```python
@dataclass
class MatchingConfig:
    feature: FeatureConfig              # Nested feature config
    
    # Descriptor matching
    ratio: Optional[float] = None       # Lowe's ratio (None = adaptive)
    mutual: bool = True                 # Mutual nearest neighbor
    
    # RANSAC / Epipolar
    ransac_thresh_px: Optional[float] = None
    max_epipolar_dist: float = 3.0      # Multicam epipolar filter
    min_pair_inliers: Optional[int] = None
    
    # Pair selection
    pair_window: Optional[int] = None   # Singlecam sequential window
    match_all_pairs: bool = True        # Multicam: match all pairs
    
    # PnP
    pnp_thresh_px: Optional[float] = None
    
    # Silhouette matching
    num_contour_points: int = 150
    contour_max_epipolar_dist: float = 5.0
    contour_min_matches: int = 8
    proximity_threshold: float = 5.0
```

---

### TriangulationConfig

Triangulation and point validation.

```python
@dataclass
class TriangulationConfig:
    max_reproj_px: float = 1.5          # Max reprojection error
    min_triang_angle_deg: float = 3.0   # Min triangulation angle
    
    # Multi-view validation
    min_views_seed: int = 2             # Min views for seeding
    min_views_accept: int = 3           # Min views to accept (multicam: 3)
    good_ratio: float = 0.60            # Fraction of good views required
    
    # Adaptive thresholds
    adaptive: bool = True
    mad_k: float = 3.5                  # MAD multiplier
    thresh_cap_mult: float = 4.0
    
    # Depth validation
    z_min: float = 1e-6
    z_max: Optional[float] = None
    max_depth_cov: float = 1.0          # Max depth coefficient of variation
```

---

### BAConfig

Bundle adjustment parameters.

```python
@dataclass
class BAConfig:
    mode: str = "none"                  # "none", "local", "global_end", "local+global_end"
    
    # Local BA
    local_every: int = 3                # Run every N images
    local_max_nfev: int = 50
    local_window: int = 4               # Camera window size
    local_points_cap: int = 5000
    local_min_points: int = 50
    local_min_obs: int = 200
    
    # Global BA
    global_max_nfev: int = 25
    global_points_cap: int = 20000
    global_min_points: int = 200
    global_min_obs: int = 800
    
    # Optimization
    loss: str = "cauchy"                # "linear", "huber", "cauchy"
    f_scale: float = 2.0
    verbose: int = 0
    
    # Multicam
    fix_poses_multicam: bool = True     # Skip BA for calibrated poses
```

---

### CleanupConfig

Outlier removal parameters.

```python
@dataclass
class CleanupConfig:
    # Strict filter (multicam)
    max_reproj_any_view: float = 5.0    # Fail if ANY view > this
    max_reproj_median: float = 1.0      # Fail if median > this
    min_good_views: int = 3
    
    # Spatial filter
    spatial_percentile: float = 95.0
    spatial_multiplier: float = 2.0
    
    # Adaptive filter (singlecam)
    adaptive_good_view_ratio: float = 0.60
    adaptive_min_observations: int = 2
    adaptive_spatial_filtering: bool = True
```

---

### DiagnosticsConfig

Output and visualization settings.

```python
@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    out_dir: Optional[str] = None
    
    reproj_report: bool = True
    reproj_max_obs: int = 20000
    
    save_kp_overlays: bool = True
    save_match_overlays: bool = False
    save_reproj_overlays: bool = True
    save_removed_overlays: bool = True
```

---

## Preset Functions

```python
from src.pipeline import (
    get_default_config,
    get_multicam_config,
    get_singlecam_config,
    get_textureless_config,
    get_turntable_config,
)
```

| Function | Description |
|----------|-------------|
| `get_default_config()` | Standard SIFT, textured scenes |
| `get_multicam_config()` | Calibrated rig, BA disabled, min_views=3 |
| `get_singlecam_config()` | Sequential, BA enabled, min_views=2 |
| `get_textureless_config()` | Low contrast SIFT for smooth surfaces |
| `get_turntable_config()` | Small angles, turntable capture |

---

## Main Functions

### run_incremental_sfm

Main entry point for sparse reconstruction.

```python
from src.pipeline import run_incremental_sfm

result = run_incremental_sfm(
    images: List[np.ndarray],           # Input images
    K: Optional[np.ndarray] = None,     # Intrinsics (singlecam)
    silhouettes: Optional[List[np.ndarray]] = None,
    cams: Optional[List[DecomposedCamera]] = None,  # Calibration (multicam)
    config: Optional[SfMConfig] = None,
    out_dir: Optional[str] = None,
) -> SfMResult
```

**Returns:** `SfMResult` containing reconstruction data.

---

## Data Structures

### SfMResult

Final reconstruction output.

```python
@dataclass
class SfMResult:
    cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]]  # img_id -> (R, t)
    X: np.ndarray                        # (N, 3) point cloud
    track_to_point: Dict[int, int]       # track_id -> point_index
    registered_images: List[int]         # Ordered list of registered images
    tracks: List[Track]                  # Track objects
    feats: List[Features]                # Per-image features
    point_meta: Dict[int, PointMeta]     # Per-point metadata
```

### PointMeta

Metadata for each 3D point.

```python
@dataclass
class PointMeta:
    track_id: int
    obs_views: List[int]                 # Camera IDs observing this point
    track_len: int                       # Number of observations
    max_triang_angle_deg: float
    median_triang_angle_deg: float
    median_reproj_error: float
```

### Features

Per-image feature data.

```python
@dataclass
class Features:
    kpts_xy: np.ndarray    # (N, 2) keypoint coordinates
    desc: np.ndarray       # (N, D) descriptors
```

### Track

Multi-view feature track.

```python
@dataclass
class Track:
    obs: Dict[int, int]    # img_id -> keypoint_id
```

### DecomposedCamera

Decomposed camera calibration.

```python
@dataclass
class DecomposedCamera:
    K: np.ndarray    # (3, 3) intrinsic matrix
    R: np.ndarray    # (3, 3) rotation (world to camera)
    t: np.ndarray    # (3, 1) translation
    C: np.ndarray    # (3, 1) camera center in world
```

---

## Dataset Loaders

### load_multicam_dataset

Load calibrated multi-camera dataset.

```python
from datasets.multicam import load_multicam_dataset

ds = load_multicam_dataset(
    image_dir: Path,
    camera_dir: Path,
    mask_dir: Optional[Path] = None,
    resize_max: Optional[int] = None,
    image_exts: Tuple[str, ...] = (".ppm",),
    mask_exts: Tuple[str, ...] = (".pgm",),
) -> MultiCamDataset
```

**Returns:**
```python
@dataclass
class MultiCamDataset:
    images: List[np.ndarray]
    image_paths: List[Path]
    cams: List[DecomposedCamera]
    cam_paths: List[Path]
    silhouettes: Optional[List[np.ndarray]]
    silhouette_paths: Optional[List[Path]]
    
    @property
    def K_shared(self) -> np.ndarray  # First camera's K
```

### load_singlecam_dataset

Load single-camera sequential dataset.

```python
from datasets.singlecam import load_singlecam_dataset

ds = load_singlecam_dataset(
    image_dir: Path,
    K_file: Optional[Path] = None,      # If None, estimates from FOV
    resize_max: Optional[int] = None,
    fov_deg: float = 60.0,              # Assumed FOV if no K_file
) -> SingleCamDataset
```

**Returns:**
```python
@dataclass
class SingleCamDataset:
    images: List[np.ndarray]
    image_paths: List[Path]
    K: np.ndarray
```

---

## Diagnostic Functions

```python
from src.diagnostics.multicam_diagnostics import (
    project_all_points_to_all_cameras,
    analyze_point_statistics,
    find_points_outside_silhouettes,
)
```

### project_all_points_to_all_cameras

Generate reprojection overlay images.

```python
project_all_points_to_all_cameras(
    out_dir: str,
    images: List[np.ndarray],
    sfm_result: SfMResult,
    cams=None,                  # Multicam
    K: Optional[np.ndarray] = None,  # Singlecam
    max_draw_points: int = 500,
    err_thresh_px: float = 5.0,
    silhouettes: Optional[List[np.ndarray]] = None,
)
```

### analyze_point_statistics

Print point cloud statistics.

```python
stats = analyze_point_statistics(
    sfm_result: SfMResult,
    cams=None,
    K: Optional[np.ndarray] = None,
    percentile_threshold: float = 95.0,
) -> dict
```

**Returns:** `{'centroid', 'distances', 'outlier_indices', 'track_lengths'}`

---

## Utility Functions

### Checkpoint Management

```python
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists

save_checkpoint(sfm_result, "checkpoint.pkl")
sfm_result = load_checkpoint("checkpoint.pkl")
exists = checkpoint_exists("checkpoint.pkl")
```

### Point Cloud Export

```python
from data_io.pointcloud_io import write_ply

write_ply("points.ply", sfm_result.X)
```
