from __future__ import annotations

from pathlib import Path
from typing import Union

from cv2 import imread, IMREAD_GRAYSCALE
import numpy as np


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Load a single-channel mask (silhouette) image.
    """
    path = Path(path)
    m = imread(str(path), IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"Failed to load mask {path}")
    return m