# Getting Started with GeoFuse-SFM

This guide walks you through installation, running your first reconstruction, and understanding the outputs.

---

## Installation

### Prerequisites

- Python 3.9+
- OpenCV with contrib modules
- NumPy, SciPy, scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/geofuse-sfm.git
cd geofuse-sfm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "from src.pipeline import run_incremental_sfm; print('OK')"
```

---

## Quick Start

### Option 1: Singlecam (Sequential Images)

For standard photo collections where camera poses are unknown.

**Required inputs:**
- `images/` - Directory of images (JPG, PNG, etc.)
- `K.txt` - 3×3 intrinsic matrix

```bash
python -m scripts.run_multicam \
    --images_dir Data/fountain/images \
    --K_file Data/fountain/K.txt \
    --visualize \
    --verbose
```

**K.txt format:**
```
2759.48    0    1520.69
   0    2764.16 1006.81
   0       0       1
```

### Option 2: Multicam (Calibrated Rig)

For multi-camera setups where camera poses are pre-calibrated.

**Required inputs:**
- `images/` - Directory of images
- `calib/` - Directory of camera files (one per image, same filename stem)
- `silhouettes/` - Optional masks

```bash
python -m scripts.run_multicam \
    --images_dir Data/bird/images \
    --cameras_dir Data/bird/calib \
    --masks_dir Data/bird/silhouettes \
    --feature_preset textureless \
    --visualize \
    --verbose
```

**Camera file format (3×4 projection matrix):**
```
p11 p12 p13 p14
p21 p22 p23 p24
p31 p32 p33 p34
```

---

## Understanding the Outputs

After running, you'll find these outputs:

```
output/<scene_name>/
├── sparse_points.ply          # 3D point cloud (view in MeshLab/CloudCompare)
├── checkpoint_sfm.pkl         # Full reconstruction state (for reloading)
├── sfm_diagnostics/           # Per-stage debug info
│   ├── kpts/                  # Keypoint overlays
│   ├── pre_ba/                # Track statistics before BA
│   └── removed_points_overlays/
└── all_proj_overlays/         # Reprojection visualization
    ├── all_proj_000.png
    ├── all_proj_001.png
    └── ...
```

### Viewing the Point Cloud

```bash
# MeshLab
meshlab output/fountain/sparse_points.ply

# CloudCompare
cloudcompare output/fountain/sparse_points.ply

# Open3D (Python)
import open3d as o3d
pcd = o3d.io.read_point_cloud("output/fountain/sparse_points.ply")
o3d.visualization.draw_geometries([pcd])
```

### Reprojection Overlays

The overlay images show:
- 🟢 **Green dots** - Observed keypoints
- 🔴 **Red dots** - Projected 3D point locations
- 🟡 **Yellow lines** - Bad reprojection (≥ 5px error)
- 🔵 **Cyan dots** - Points visible but not tracked in this view

---

## Common Scenarios

### Textured Scenes (Buildings, Outdoor)

Use default settings:

```bash
python -m scripts.run_multicam \
    --images_dir Data/temple/images \
    --K_file Data/temple/K.txt
```

### Textureless Objects (Plastic, Skin, Smooth Surfaces)

Use the textureless preset for lower SIFT contrast threshold:

```bash
python -m scripts.run_multicam \
    --images_dir Data/bird/images \
    --cameras_dir Data/bird/calib \
    --feature_preset textureless
```

Or manually tune SIFT:

```bash
python -m scripts.run_multicam \
    --images_dir Data/bird/images \
    --cameras_dir Data/bird/calib \
    --sift_contrast 0.004 \
    --sift_nfeatures 150000
```

### Resume from Checkpoint

Skip the reconstruction and just visualize:

```bash
python -m scripts.viz_reconstruction \
    --images_dir Data/fountain/images \
    --K_file Data/fountain/K.txt \
    --checkpoint output/TempleRing/checkpoint_sfm.pkl \
    --all_points
    --out_dir output/TempleRing/

# Will load from checkpoint if it exists

note: Use --all_points to display every visible point cloud on the image.
default shows only the observed points per view
```

Force re-run:

```bash
python -m scripts.run_multicam \
    --images_dir Data/fountain/images \
    --K_file Data/fountain/K.txt \
    --force_rerun
```

---

## Configuration Options

### Feature Detection

| Flag | Description | Default |
|------|-------------|---------|
| `--feature_preset` | `default`, `textureless`, `high_detail` | auto |
| `--sift_contrast` | SIFT contrastThreshold | 0.04 |
| `--sift_nfeatures` | Max features (0 = unlimited) | 0 |

### Triangulation

| Flag | Description | Default |
|------|-------------|---------|
| `--max_reproj_px` | Max reprojection error | 1.5 |
| `--min_triang_angle_deg` | Min triangulation angle | 3.0 |
| `--min_views` | Min views to accept point | 3 (multicam), 2 (singlecam) |

### Bundle Adjustment

| Flag | Description | Default |
|------|-------------|---------|
| `--ba_mode` | `none`, `local`, `global_end`, `local+global_end` | varies |

### Cleanup

| Flag | Description | Default |
|------|-------------|---------|
| `--min_good_views` | Min good views for strict filter | 3 |
| `--max_reproj_any_view` | Max error in any view | 5.0 |

---

## Troubleshooting

### "No verified pairs found"

**Cause:** Feature matching failed.

**Solutions:**
1. Check image quality and overlap
2. Try `--feature_preset textureless` for smooth objects
3. Lower `--sift_contrast 0.01`
4. Verify K matrix is correct

### "Too few shared tracks for initialization"

**Cause:** Not enough common features between initial pair.

**Solutions:**
1. Ensure sufficient image overlap (30%+)
2. Try different images (better textured)
3. Lower matching thresholds

### High reprojection errors

**Cause:** Bad triangulation or incorrect calibration.

**Solutions:**
1. Check camera calibration
2. Increase `--min_views 3`
3. Lower `--max_reproj_px`

### Points outside object (multicam)

**Cause:** Two-view triangulation ambiguity.

**Solution:** Ensure `--min_views 3` is set (default for multicam).

---

## Python API

For programmatic usage:

```python
from src.pipeline import (
    run_incremental_sfm,
    SfMConfig,
    get_textureless_config,
)
from datasets.multicam import load_multicam_dataset

# Load data
ds = load_multicam_dataset(
    image_dir="Data/bird/images",
    camera_dir="Data/bird/calib",
    mask_dir="Data/bird/silhouettes",
)

# Configure
config = get_textureless_config()
config.triangulation.min_views_accept = 3

# Run
result = run_incremental_sfm(
    images=ds.images,
    cams=ds.cams,
    silhouettes=ds.silhouettes,
    config=config,
    out_dir="output/bird",
)

# Access results
print(f"Points: {result.X.shape[0]}")
print(f"Cameras: {len(result.registered_images)}")
```

---

## Next Steps

- [Architecture Guide](ARCHITECTURE.md) - Understand the internals
- [API Reference](API.md) - Full configuration options
- [Pipeline Details](PIPELINE.md) - Algorithm deep dive
