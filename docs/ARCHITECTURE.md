# GeoFuse-SFM Architecture

This document describes the internal architecture of GeoFuse-SFM, including module responsibilities, data flow, and design decisions.

---

## Overview

GeoFuse-SFM follows a **modular pipeline architecture** where each stage is implemented as a separate module. The pipeline is orchestrated by `run_sfm.py`, which calls each module in sequence while maintaining shared state in `SfMState`.

```
┌─────────────────────────────────────────────────────────────┐
│                      run_sfm.py                             │
│                   (Orchestrator)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │matching │→ │ tracks  │→ │  init   │→ │register │      │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                                               │             │
│                                               ▼             │
│                              ┌─────────┐  ┌─────────┐      │
│                              │   BA    │→ │ cleanup │      │
│                              └─────────┘  └─────────┘      │
│                                                             │
│   ┌─────────────────────────────────────────────────┐      │
│   │                  SfMState                        │      │
│   │  (Shared mutable state across all modules)       │      │
│   └─────────────────────────────────────────────────┘      │
│                                                             │
│   ┌─────────────────────────────────────────────────┐      │
│   │                  SfMConfig                       │      │
│   │  (Immutable configuration for all modules)       │      │
│   └─────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
src/
├── pipeline/                  # Core reconstruction pipeline
│   ├── __init__.py           # Public API exports
│   ├── config.py             # Configuration dataclasses
│   ├── state.py              # Mutable reconstruction state
│   ├── matching.py           # Feature extraction & matching
│   ├── tracks_builder.py     # Track building
│   ├── initialize.py         # Seed point triangulation
│   ├── triangulation.py      # Multi-view triangulation
│   ├── registration.py       # Incremental registration loop
│   ├── ba_runner.py          # Bundle adjustment
│   ├── cleanup.py            # Outlier removal
│   └── run_sfm.py            # Main entry point
│
├── features.py               # Low-level feature detection
├── tracks.py                 # Union-find track building
├── geometry.py               # Pose recovery, triangulation
├── ba.py                     # Bundle adjustment solver
│
├── diagnostics/              # Analysis tools
│   ├── sfm_diagnostics.py    # Point cloud statistics
│   └── multicam_diagnostics.py
│
├── filter_utils/             # Outlier filtering
│   ├── filter.py             # Adaptive filter (singlecam)
│   └── multicam_filter.py    # Strict filter (multicam)
│
├── geometry_utils/           # Geometric primitives
│   ├── epipolar.py           # Epipolar geometry
│   ├── projective.py         # Projection utilities
│   ├── reprojection.py       # Error computation
│   ├── triangulation.py      # Triangulation methods
│   └── twoview.py            # Two-view geometry
│
├── mulitcam_utils/           # Multicam-specific
│   ├── registration.py       # Epipolar filtering
│   └── silhouette_matching.py # Contour-based matching
│
└── visualization/            # Output generation
    └── reproject_overlays.py # Overlay images
```

---

## Core Modules

### 1. Configuration (`config.py`)

Centralized configuration using Python dataclasses.

```python
@dataclass
class SfMConfig:
    matching: MatchingConfig
    triangulation: TriangulationConfig
    ba: BAConfig
    cleanup: CleanupConfig
    diagnostics: DiagnosticsConfig
```

**Key Design Decisions:**
- All numeric defaults in one place
- `None` values trigger adaptive computation
- Preset functions for common scenarios
- YAML/JSON serialization support

**Presets:**
- `get_multicam_config()` - Calibrated rigs, BA disabled
- `get_singlecam_config()` - Sequential capture, BA enabled
- `get_textureless_config()` - Low contrast threshold SIFT

### 2. State Management (`state.py`)

Mutable state container that unifies multicam/singlecam handling.

```python
class SfMState:
    # Data
    images: List[np.ndarray]
    feats: List[Features]
    tracks: List[Track]
    
    # Reconstruction
    cam_poses: Dict[int, Tuple[R, t]]
    X_list: List[np.ndarray]
    track_to_point: Dict[int, int]
    
    # Helpers (abstract multicam vs singlecam)
    def K_for_view(img_id) -> np.ndarray
    def camera_center_for_view(img_id) -> np.ndarray
    def add_point(track_id, X) -> int
```

**Key Design Decisions:**
- Single class handles both modes
- Unified helpers hide implementation differences
- `add_point()` centralizes point metadata computation

### 3. Matching (`matching.py`)

Feature extraction and pairwise matching.

**Multicam Mode:**
1. SIFT detection with configurable parameters
2. Descriptor matching with ratio test
3. Epipolar filtering using known F matrix
4. Silhouette contour matching (optional)
5. Merge SIFT + contour matches

**Singlecam Mode:**
1. SIFT detection
2. Descriptor matching
3. Essential matrix RANSAC verification

### 4. Triangulation (`triangulation.py`)

Multi-view point triangulation with validation.

```python
def triangulate_multiview_dlt(state, track_id, max_reproj_px) -> Optional[np.ndarray]
def validate_point_multiview(state, track_id, X, config) -> bool
def triangulate_new_points(state, new_img, config) -> int
```

**Key Policy: min_views=3 for Multicam**

Two-view triangulation is unreliable for wide-baseline multicam rigs:
- Distant cameras can satisfy epipolar constraint
- But triangulated point may be far from true position
- Requiring 3+ views ensures geometric consistency

### 5. Registration (`registration.py`)

Incremental image addition loop.

```python
while len(registered) < n_images:
    next_img = select_next_image()      # Best candidate by 2D-3D correspondences
    register_next_image(next_img)       # PnP (singlecam) or known pose (multicam)
    triangulate_new_points(next_img)    # Add new 3D points
    run_local_ba()                      # Optional refinement
```

**Candidate Scoring:**
```python
score = count_2d3d_correspondences + 0.1 * max_baseline
```

### 6. Bundle Adjustment (`ba_runner.py`)

Optional refinement using scipy least_squares.

**Multicam:** BA is **skipped** because poses are calibrated. Only point positions could be refined, but this risks degrading the reconstruction.

**Singlecam:** Both local (windowed) and global BA are supported.

```python
config.ba.mode = "none"              # Skip BA
config.ba.mode = "local"             # Every N images
config.ba.mode = "global_end"        # Once at end
config.ba.mode = "local+global_end"  # Both
```

### 7. Cleanup (`cleanup.py`)

Outlier removal with mode-specific strategies.

**Multicam (Strict Filter):**
```python
remove_outliers_strict_multicam(
    max_reproj_any_view=5.0,   # Fail if ANY view > 5px
    max_reproj_median=1.0,     # Fail if median > 1px
    min_good_views=3,          # Require 3+ good views
)
```

**Singlecam (Adaptive Filter):**
```python
remove_outlier_points_adaptive(
    good_view_ratio=0.60,
    min_observations=2,
    use_spatial_filtering=True,
)
```

---

## Data Flow

```
Input                    Processing                      Output
─────                    ──────────                      ──────

images[]     ──────►  Feature Extraction  ──────►  feats[]
                              │
K / cams[]   ──────►  Pairwise Matching   ──────►  pairwise{}
                              │
masks[]      ──────►  Track Building      ──────►  tracks[]
(optional)                    │
                      Initialization       ──────►  cam_poses{0,1}
                              │                     X_list (seed)
                      ┌───────┴───────┐
                      │  Registration  │◄────────── Loop until
                      │     Loop       │            all registered
                      └───────┬───────┘
                              │
                      Bundle Adjustment    ──────►  Refined poses/points
                              │
                      Outlier Removal      ──────►  Clean point cloud
                              │
                              ▼
                         SfMResult
                      ┌─────────────┐
                      │ cam_poses   │
                      │ X (N×3)     │
                      │ tracks      │
                      │ feats       │
                      │ point_meta  │
                      └─────────────┘
```

---

## Multicam vs Singlecam Differences

| Aspect | Multicam | Singlecam |
|--------|----------|-----------|
| **Input** | Per-view K, R, t from calibration | Shared K, poses unknown |
| **Matching** | Epipolar filter + contour | E-RANSAC |
| **Pose estimation** | Use known pose | PnP RANSAC |
| **Min views** | 3 (strict) | 2 (flexible) |
| **Bundle adjustment** | Disabled (poses calibrated) | Enabled |
| **Cleanup** | Strict reprojection filter | Adaptive filter |

---

## Extension Points

### Adding Dense Fusion

The sparse reconstruction provides seed geometry for dense fusion:

```python
sfm_result = run_incremental_sfm(...)

# Dense fusion uses:
# - sfm_result.X as seed points
# - sfm_result.cam_poses for depth map computation
# - sfm_result.tracks for visibility information

dense_points = run_dense_fusion(
    images=images,
    sfm_result=sfm_result,
    cams=cams,
)
```

### Adding New Feature Detectors

1. Add parameters to `FeatureConfig`
2. Update `detect_and_describe()` in `features.py`
3. Add preset in `config.py`

```python
@dataclass
class FeatureConfig:
    method: str = "sift"  # "sift", "orb", "superpoint"
    
    # SuperPoint parameters (future)
    superpoint_threshold: float = 0.015
    superpoint_nms_radius: int = 4
```

---

## Performance Considerations

- **Feature extraction:** Parallelizable across images
- **Pairwise matching:** O(n²) pairs, but can limit to window
- **Track building:** O(n·m) with union-find
- **Triangulation:** O(tracks × views) per new image
- **BA:** Most expensive; cap points/observations

Typical runtime (27 images, 1024×768):
- Feature extraction: ~10s
- Matching: ~15s
- Registration loop: ~20s
- Total: ~45s (without BA)
