# datasets/singlecam.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from data_io.camera import _k_from_obj
from data_io.images import collect_image_paths, load_images
from data_io.parsing import load_data


@dataclass(frozen=True)
class SingleCamDataset:
    """Dataset for single-camera (sequential) image capture."""
    images: List[np.ndarray]
    image_paths: List[Path]
    K: np.ndarray  # (3, 3) intrinsic matrix


def read_intrinsic_matrix(K_file: Path) -> np.ndarray:
    """
    Read intrinsic matrix from file.
    
    Supports formats:
    - Plain text: 3x3 matrix (space/comma separated)
    - NumPy .npy file
    - JSON with "K" key
    """
    K_file = Path(K_file)
    
    if not K_file.exists():
        raise FileNotFoundError(f"Intrinsic matrix file not found: {K_file}")
    
    suffix = K_file.suffix.lower()
    
    if suffix == ".npy":
        K = np.load(K_file)
    elif suffix == ".json":
        import json
        with open(K_file, "r") as f:
            data = json.load(f)
        if "K" in data:
            K = np.array(data["K"], dtype=np.float64)
        elif "intrinsic" in data:
            K = np.array(data["intrinsic"], dtype=np.float64)
        else:
            raise ValueError(f"JSON file must contain 'K' or 'intrinsic' key: {K_file}")
    else:
        # Plain text - try to parse as matrix
        K = _k_from_obj(load_data(K_file))
    
    if K.shape != (3, 3):
        # Maybe it's flattened?
        if K.size == 9:
            K = K.reshape(3, 3)
        else:
            raise ValueError(f"Expected 3x3 matrix, got shape {K.shape} from {K_file}")
    
    return K


def estimate_intrinsic_matrix(
    image_shape: Tuple[int, int],
    fov_deg: float = 60.0,
) -> np.ndarray:
    """
    Estimate intrinsic matrix from image dimensions and assumed FOV.
    
    Args:
        image_shape: (H, W) of images
        fov_deg: Assumed horizontal field of view in degrees
        
    Returns:
        K: (3, 3) estimated intrinsic matrix
    """
    h, w = image_shape
    
    # Focal length from FOV
    fov_rad = np.deg2rad(fov_deg)
    fx = w / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    
    # Principal point at image center
    cx = w / 2
    cy = h / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    
    return K


def load_singlecam_dataset(
    image_dir: Path,
    K_file: Optional[Path] = None,
    *,
    resize_max: Optional[int] = None,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".ppm", ".tif", ".tiff"),
    fov_deg: float = 60.0,
) -> SingleCamDataset:
    """
    Load a single-camera dataset (sequential images with shared intrinsics).
    
    Args:
        image_dir: Directory containing images
        K_file: Optional path to intrinsic matrix file
                If None, estimates K from image size and assumed FOV
        resize_max: Optional max dimension for resizing
        image_exts: Image file extensions to search for
        fov_deg: Assumed FOV if K_file not provided
        
    Returns:
        SingleCamDataset with images and intrinsic matrix
        
    Example:
        # With known intrinsics
        ds = load_singlecam_dataset(
            image_dir=Path("data/fountain/images"),
            K_file=Path("data/fountain/K.txt"),
        )
        
        # With estimated intrinsics
        ds = load_singlecam_dataset(
            image_dir=Path("data/my_photos"),
            fov_deg=55.0,  # Approximate phone camera FOV
        )
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    
    # Collect image paths
    image_paths = collect_image_paths(image_dir, exts=image_exts)
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir} with extensions {image_exts}")
    
    print("\n[Loading] Single-camera dataset...")
    print(f"  image_dir: {image_dir}")
    print(f"  images:    {len(image_paths)}")
    
    # Load images
    images = load_images(image_paths, color=True, resize_max=resize_max)
    
    # Get intrinsic matrix
    if K_file is not None:
        K_file = Path(K_file)
        K = read_intrinsic_matrix(K_file)
        print(f"  K_file:    {K_file}")
        
        # Adjust K if images were resized
        if resize_max is not None:
            # Get original image size from first path
            import cv2
            orig_img = cv2.imread(str(image_paths[0]))
            orig_h, orig_w = orig_img.shape[:2]
            new_h, new_w = images[0].shape[:2]
            
            if (orig_h, orig_w) != (new_h, new_w):
                scale_x = new_w / orig_w
                scale_y = new_h / orig_h
                K = K.copy()
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] *= scale_x  # cx
                K[1, 2] *= scale_y  # cy
                print(f"  [info] Adjusted K for resize: {orig_w}x{orig_h} -> {new_w}x{new_h}")
    else:
        # Estimate from image size
        h, w = images[0].shape[:2]
        K = estimate_intrinsic_matrix((h, w), fov_deg=fov_deg)
        print(f"  K_file:    None (estimated from image size with FOV={fov_deg}°)")
    
    print(f"\n  Intrinsic matrix K:")
    print(f"    fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")
    print(f"    cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    
    return SingleCamDataset(
        images=images,
        image_paths=image_paths,
        K=K,
    )
