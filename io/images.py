from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from cv2 import IMREAD_COLOR, imread, IMREAD_GRAYSCALE, resize, INTER_AREA, INTER_LINEAR
import numpy as np

IMG_EXTS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"
)


def collect_image_paths(
    image_dir: Union[str, Path],
    exts: Tuple[str, ...] = IMG_EXTS,
) -> List[Path]:
    """
    Collect image paths from a directory and return them sorted by filename.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")

    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: p.name)

    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in: {image_dir} with extensions {exts}")

    return paths


def load_image(
    path: Union[str, Path],
    *,
    color: bool = True,
    resize_max: Optional[int] = None,
) -> np.ndarray:
    """
    Load a single image with OpenCV.

    Args:
      color: True -> BGR. False -> grayscale.
      resize_max: if set, resizes so max(H,W)==resize_max while preserving aspect ratio.
    """
    path = Path(path)
    flag = IMREAD_COLOR if color else IMREAD_GRAYSCALE

    img = imread(str(path), flag)
    if img is None:
        raise IOError(f"Failed to load image: {path}")

    if resize_max is not None:
        h, w = img.shape[:2]
        m = max(h, w)
        if m > resize_max:
            scale = resize_max / float(m)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    return img


def load_images(
    image_paths: Iterable[Union[str, Path]],
    *,
    color: bool = True,
    resize_max: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Load multiple images with OpenCV.
    """
    images: List[np.ndarray] = []
    for p in image_paths:
        images.append(load_image(p, color=color, resize_max=resize_max))
    return images
