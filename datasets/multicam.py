# sfm/datasets/multicam.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_io.images import collect_image_paths, load_images
from data_io.masks import load_mask
from data_io.camera import DecomposedCamera, decompose_projection_matrix, read_projection_matrix


@dataclass(frozen=True)
class MultiCamDataset:
    images: List[np.ndarray]
    image_paths: List[Path]

    cams: List[DecomposedCamera]
    cam_paths: List[Path]

    silhouettes: Optional[List[np.ndarray]] = None
    silhouette_paths: Optional[List[Path]] = None

    @property
    def K_shared(self) -> np.ndarray:
        # convenience only
        return self.cams[0].K


def _index_by_stem(paths: List[Path]) -> Dict[str, Path]:
    return {p.stem: p for p in paths}


def load_multicam_dataset(
    image_dir: Path,
    camera_dir: Path,
    mask_dir: Path | None = None,
    *,
    resize_max: int | None = None,
    image_exts: Tuple[str, ...] = (".ppm",),
    mask_exts: Tuple[str, ...] = (".pgm",),
) -> MultiCamDataset:
    image_dir = Path(image_dir)
    camera_dir = Path(camera_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else None

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not camera_dir.exists():
        raise FileNotFoundError(f"camera_dir not found: {camera_dir}")
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    image_paths_all = collect_image_paths(image_dir, exts=image_exts)
    cam_paths_all = sorted(camera_dir.glob("*"))  # allow txt/json/yaml etc; reader will validate
    cam_paths_all = [p for p in cam_paths_all if p.is_file()]

    if len(image_paths_all) == 0:
        raise FileNotFoundError(f"No images found in {image_dir} with {image_exts}")
    if len(cam_paths_all) == 0:
        raise FileNotFoundError(f"No camera files found in {camera_dir}")

    images_by_id = _index_by_stem(image_paths_all)
    cams_by_id = _index_by_stem(cam_paths_all)

    common_ids = sorted(set(images_by_id.keys()) & set(cams_by_id.keys()))
    if not common_ids:
        raise RuntimeError(
            "No matching image/camera pairs by filename stem.\n"
            f"Example image stems: {sorted(list(images_by_id.keys()))[:5]}\n"
            f"Example camera stems: {sorted(list(cams_by_id.keys()))[:5]}"
        )

    missing_cams = sorted(set(images_by_id.keys()) - set(cams_by_id.keys()))
    missing_imgs = sorted(set(cams_by_id.keys()) - set(images_by_id.keys()))
    if missing_cams:
        print(f"[warn] images without camera files (showing up to 10): {missing_cams[:10]}")
    if missing_imgs:
        print(f"[warn] camera files without images (showing up to 10): {missing_imgs[:10]}")

    image_paths = [images_by_id[i] for i in common_ids]
    cam_paths = [cams_by_id[i] for i in common_ids]

    print("\n[Loading] Multi-camera dataset...")
    print(f"  image_dir:   {image_dir}")
    print(f"  camera_dir:  {camera_dir}")
    if mask_dir is not None:
        print(f"  mask_dir:    {mask_dir}")
    print(f"  pairs:       {len(common_ids)}")

    images = load_images(image_paths, color=True, resize_max=resize_max)

    cams: List[DecomposedCamera] = []
    for p in cam_paths:
        P = read_projection_matrix(p)        # format-agnostic
        cams.append(decompose_projection_matrix(P))

    silhouettes: Optional[List[np.ndarray]] = None
    silhouette_paths: Optional[List[Path]] = None

    if mask_dir is not None:
        mask_paths_all = []
        for ext in mask_exts:
            mask_paths_all.extend(mask_dir.glob(f"*{ext}"))
        mask_paths_all = sorted([p for p in mask_paths_all if p.is_file()])

        if len(mask_paths_all) == 0:
            print("  [warn] mask_dir provided but no masks found; continuing without masks.")
        else:
            masks_by_id = _index_by_stem(mask_paths_all)
            # only keep masks if all are present to avoid subtle misalignment
            if all(i in masks_by_id for i in common_ids):
                silhouette_paths = [masks_by_id[i] for i in common_ids]
                silhouettes = [load_mask(p) for p in silhouette_paths]
            else:
                missing = [i for i in common_ids if i not in masks_by_id]
                print(f"  [warn] missing masks for {len(missing)}/{len(common_ids)} ids; disabling masks.")
                silhouettes = None
                silhouette_paths = None

    print(f"  Loaded images:      {len(images)}")
    print(f"  Loaded cameras:     {len(cams)}")
    if silhouettes is not None:
        print(f"  Loaded silhouettes: {len(silhouettes)}")

    return MultiCamDataset(
        images=images,
        image_paths=image_paths,
        cams=cams,
        cam_paths=cam_paths,
        silhouettes=silhouettes,
        silhouette_paths=silhouette_paths,
    )
