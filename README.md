# GeoFuse-SFM

**Geometry-Aware Dense Reconstruction from Sparse Structure-from-Motion**

<p align="center">
  <img src="docs/assets/pipeline_overview.svg" alt="GeoFuse-SFM Pipeline" width="800">
</p>

GeoFuse-SFM is a modular Structure-from-Motion pipeline that produces sparse 3D reconstructions from multi-view images. It supports both **calibrated multi-camera rigs** (like the BBM dataset) and **sequential single-camera captures** (like the Fountain dataset). The sparse reconstruction serves as seed geometry for downstream dense fusion.

---

## ✨ Key Features

- **Unified Pipeline** - Single codebase handles both multicam and singlecam setups
- **Textureless Surface Support** - Adaptive SIFT parameters for smooth/plastic objects
- **Calibrated Rig Support** - Leverages known camera poses for multicam datasets
- **Modular Architecture** - Clean separation of matching, triangulation, BA, and cleanup
- **Configurable** - All parameters in dataclass configs with presets
- **Silhouette Integration** - Contour matching + filtering for challenging objects

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/geofuse-sfm.git
cd geofuse-sfm
pip install -r requirements.txt
```

### Run Sparse Reconstruction

**Multicam (calibrated rig):**
```bash
python -m scripts.run_multicam \
    --images_dir Data/bird/images \
    --cameras_dir Data/bird/calib \
    --masks_dir Data/bird/silhouettes \
    --feature_preset textureless \
    --visualize
```

**Singlecam (sequential images):**
```bash
python -m scripts.run_multicam \
    --images_dir Data/fountain/images \
    --K_file Data/fountain/K.txt \
    --visualize
```

### Output
```
output/<scene_name>/
├── sparse_points.ply          # Sparse point cloud
├── checkpoint_sfm.pkl         # Full reconstruction state
├── sfm_diagnostics/           # Per-stage diagnostics
└── all_proj_overlays/         # Reprojection visualizations
```

---

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GeoFuse-SFM Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                                 │
│  │ Images  │    │ K / Cams│    │ Masks   │  (optional)                     │
│  └────┬────┘    └────┬────┘    └────┬────┘                                 │
│       │              │              │                                       │
│       ▼              ▼              ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                 FEATURE EXTRACTION                        │              │
│  │  SIFT with configurable contrast (textureless preset)     │              │
│  └─────────────────────────┬────────────────────────────────┘              │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                  PAIRWISE MATCHING                        │              │
│  │  Multicam: Epipolar filter + Contour matching             │              │
│  │  Singlecam: E-RANSAC verification                         │              │
│  └─────────────────────────┬────────────────────────────────┘              │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                   TRACK BUILDING                          │              │
│  │  Union-find to link matches across views                  │              │
│  └─────────────────────────┬────────────────────────────────┘              │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                  INITIALIZATION                           │              │
│  │  Select best pair → Triangulate seed points               │              │
│  └─────────────────────────┬────────────────────────────────┘              │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              INCREMENTAL REGISTRATION                     │              │
│  │  For each image:                                          │              │
│  │    • PnP pose estimation (singlecam)                      │              │
│  │    • Use known pose (multicam)                            │              │
│  │    • Triangulate new points (min 3 views for multicam)    │              │
│  │    • Local bundle adjustment (singlecam only)             │              │
│  └─────────────────────────┬────────────────────────────────┘              │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                 OUTLIER REMOVAL                           │              │
│  │  Strict reprojection filter + Spatial filter              │              │
│  └─────────────────────────┬────────────────────────────────┘              │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │                    OUTPUT                                 │              │
│  │  Sparse point cloud + Camera poses + Tracks               │              │
│  └──────────────────────────────────────────────────────────┘              │
│                            │                                                │
│                            ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │              DENSE FUSION (Coming Soon)                   │              │
│  │  Seed-based depth propagation using sparse geometry       │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
geofuse-sfm/
├── src/
│   ├── pipeline/              # Core SfM pipeline modules
│   │   ├── config.py          # Configuration dataclasses
│   │   ├── state.py           # Reconstruction state management
│   │   ├── matching.py        # Feature extraction & matching
│   │   ├── triangulation.py   # Multi-view triangulation
│   │   ├── initialize.py      # Initial pair selection
│   │   ├── registration.py    # Incremental image registration
│   │   ├── ba_runner.py       # Bundle adjustment
│   │   └── cleanup.py         # Outlier removal
│   ├── features.py            # SIFT/ORB feature detection
│   ├── tracks.py              # Track building (union-find)
│   ├── geometry.py            # Geometric utilities
│   ├── ba.py                  # Bundle adjustment solver
│   ├── diagnostics/           # Analysis & statistics
│   ├── filter_utils/          # Outlier filtering
│   ├── geometry_utils/        # Projection, triangulation helpers
│   ├── mulitcam_utils/        # Multicam-specific utilities
│   └── visualization/         # Overlay generation
├── datasets/
│   ├── multicam.py            # Multicam dataset loader
│   └── singlecam.py           # Singlecam dataset loader
├── data_io/
│   ├── camera.py              # Camera I/O & decomposition
│   ├── images.py              # Image loading utilities
│   └── pointcloud_io.py       # PLY export
├── scripts/
│   └── run_multicam.py        # Main runner script
├── utils/
│   ├── checkpoint.py          # Checkpoint save/load
│   └── logging_utils.py       # Logging utilities
└── docs/
    └── assets/                # Documentation images
```

---

## ⚙️ Configuration

All parameters are organized in dataclass configs:

```python
from src.pipeline import SfMConfig, get_textureless_config

# Use a preset
config = get_textureless_config()

# Or customize
config = SfMConfig()
config.matching.feature.sift_contrastThreshold = 0.004  # For textureless
config.triangulation.min_views_accept = 3               # Require 3 views
config.ba.mode = "local+global_end"                     # Enable BA
```

### Feature Presets

| Preset | Use Case | SIFT Contrast |
|--------|----------|---------------|
| `default` | Textured scenes (buildings, outdoor) | 0.04 |
| `textureless` | Smooth surfaces (plastic, skin) | 0.004 |
| `high_detail` | Maximum feature extraction | 0.02 |

```bash
python -m scripts.run_multicam --feature_preset textureless ...
```

---

## 🧪 Tested Datasets

| Dataset | Type | Cameras | Status |
|---------|------|---------|--------|
| Fountain | Singlecam | 11 | ✅ Working |
| TempleRing | Singlecam | 47 | ✅ Working |
| Bird (BBM) | Multicam | 27 | ✅ Working |
| Pig (BBM) | Multicam | 27 | ✅ Working |

---

## 🔮 Roadmap

- [x] Sparse SfM pipeline
- [x] Multicam support with calibrated poses
- [x] Textureless surface handling
- [x] Modular config system
- [ ] Dense fusion from seed points
- [ ] Multi-view stereo integration
- [ ] Mesh reconstruction

---

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed module documentation
- [Pipeline Details](docs/PIPELINE.md) - Step-by-step algorithm explanation
- [API Reference](docs/API.md) - Config classes and functions
- [Getting Started](docs/GETTING_STARTED.md) - Installation and usage guide

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- BBM Multi-camera dataset
- OpenCV SIFT implementation
- SciPy optimization for bundle adjustment
