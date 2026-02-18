"""
data_io/nerfstudio_export.py

Export GeoFuse-SFM results to Nerfstudio format.
Converts images to JPG for NeRF compatibility.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import cv2


def rotmat_to_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert camera pose (R, t) to camera-to-world 4x4 matrix.
    
    GeoFuse-SFM stores world-to-camera: X_cam = R @ X_world + t
    Nerfstudio uses camera-to-world: X_world = c2w @ X_cam
    
    So: c2w = [R^T | -R^T @ t]
              [0   |    1    ]
    """
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).reshape(3, 1)
    
    C = -R.T @ t
    R_c2w = R.T
    
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = C.flatten()
    
    return c2w


def convert_image_to_jpg(src_path: Path, dst_path: Path, quality: int = 95) -> bool:
    """
    Convert any image format to JPG with correct color handling.
    OpenCV reads all formats as BGR internally, so colors will be correct.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"  [Warning] Could not read: {src_path}")
        return False
    
    cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return True


def export_to_nerfstudio(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,                          # For multicam
    K: Optional[np.ndarray] = None,     # For singlecam
    aabb_scale: int = 16,
    copy_images: bool = True,
    convert_to_jpg: bool = True,
    jpg_quality: int = 95,
):
    """
    Export GeoFuse-SFM result to Nerfstudio format.
    
    Args:
        sfm_result: SfMResult from run_incremental_sfm
        images_dir: Directory containing source images
        output_dir: Output directory for Nerfstudio files
        cams: List of DecomposedCamera (multicam mode)
        K: Shared intrinsic matrix (singlecam mode)
        aabb_scale: Scene bounding box scale (power of 2)
        copy_images: If True, copy/convert images to output_dir/images
        convert_to_jpg: If True, convert PPM/PNG to JPG
        jpg_quality: JPEG quality (0-100)
    
    Creates:
        output_dir/
        ├── transforms.json
        └── images/
            ├── 0000.jpg
            └── ...
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_path = Path(images_dir)
    is_multicam = cams is not None
    
    print(f"[Nerfstudio Export] Mode: {'multicam' if is_multicam else 'singlecam'}")
    print(f"[Nerfstudio Export] Registered images: {len(sfm_result.registered_images)}")
    
    # Collect image paths
    from data_io.images import collect_image_paths
    img_paths = collect_image_paths(images_path)
    
    # Get image dimensions
    first_img_id = sfm_result.registered_images[0]
    if first_img_id < len(img_paths):
        img = cv2.imread(str(img_paths[first_img_id]))
        h, w = img.shape[:2]
    else:
        h, w = 1024, 1024
    
    # Prepare images output directory
    images_out = None
    if copy_images:
        images_out = output_path / "images"
        images_out.mkdir(exist_ok=True)
    
    # =========================================================
    # Build transforms.json
    # =========================================================
    transforms = {
        "aabb_scale": aabb_scale,
        "frames": [],
    }
    
    # Add camera intrinsics (shared for singlecam)
    if not is_multicam:
        K_arr = np.asarray(K, np.float64)
        transforms["fl_x"] = float(K_arr[0, 0])
        transforms["fl_y"] = float(K_arr[1, 1])
        transforms["cx"] = float(K_arr[0, 2])
        transforms["cy"] = float(K_arr[1, 2])
        transforms["w"] = int(w)
        transforms["h"] = int(h)
        transforms["camera_model"] = "OPENCV"
        transforms["k1"] = 0.0
        transforms["k2"] = 0.0
        transforms["p1"] = 0.0
        transforms["p2"] = 0.0
    
    # =========================================================
    # Build frames
    # =========================================================
    for img_id in sfm_result.registered_images:
        R, t = sfm_result.cam_poses[img_id]
        
        # Convert to camera-to-world
        c2w = rotmat_to_c2w(R, t)
        
        # Apply OpenGL/Nerfstudio coordinate convention
        # Nerfstudio uses: X-right, Y-up, Z-back (OpenGL)
        # OpenCV uses: X-right, Y-down, Z-forward
        # Transform: flip Y and Z
        transform_opengl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ], dtype=np.float64)
        
        c2w = c2w @ transform_opengl
        
        # Determine output image name
        if img_id < len(img_paths):
            src_path = img_paths[img_id]
            
            # Convert to JPG if needed
            if convert_to_jpg and src_path.suffix.lower() not in ['.jpg', '.jpeg']:
                new_name = src_path.stem + ".jpg"
            else:
                new_name = src_path.name
            
            if copy_images:
                file_path = f"images/{new_name}"
            else:
                file_path = str(src_path.resolve())
        else:
            new_name = f"image_{img_id:04d}.jpg"
            file_path = f"images/{new_name}"
        
        frame = {
            "file_path": file_path,
            "transform_matrix": c2w.tolist(),
        }
        
        # Add per-frame intrinsics for multicam
        if is_multicam:
            cam = cams[img_id]
            K_i = np.asarray(cam.K, np.float64)
            
            # Get actual image dimensions for this view
            if img_id < len(img_paths):
                img_i = cv2.imread(str(img_paths[img_id]))
                h_i, w_i = img_i.shape[:2]
            else:
                h_i, w_i = h, w
            
            frame["fl_x"] = float(K_i[0, 0])
            frame["fl_y"] = float(K_i[1, 1])
            frame["cx"] = float(K_i[0, 2])
            frame["cy"] = float(K_i[1, 2])
            frame["w"] = int(w_i)
            frame["h"] = int(h_i)
            frame["camera_model"] = "OPENCV"
            frame["k1"] = 0.0
            frame["k2"] = 0.0
            frame["p1"] = 0.0
            frame["p2"] = 0.0
        
        transforms["frames"].append(frame)
        
        # Copy/convert image
        if copy_images and img_id < len(img_paths):
            src_path = img_paths[img_id]
            dst_path = images_out / new_name
            
            if not dst_path.exists():
                needs_conversion = (
                    convert_to_jpg and 
                    src_path.suffix.lower() in ['.ppm', '.pgm', '.pbm', '.png', '.tif', '.tiff']
                )
                
                if needs_conversion:
                    convert_image_to_jpg(src_path, dst_path, jpg_quality)
                else:
                    import shutil
                    shutil.copy(src_path, dst_path)
    
    # =========================================================
    # Write transforms.json
    # =========================================================
    transforms_path = output_path / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)
    
    print(f"[Nerfstudio Export] Written: {transforms_path}")
    if copy_images:
        print(f"[Nerfstudio Export] Converted images to: {images_out}")
    
    print(f"[Nerfstudio Export] Done! Frames: {len(transforms['frames'])}")
    
    return output_path


def export_to_instant_ngp(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,
    K: Optional[np.ndarray] = None,
    aabb_scale: int = 16,
    copy_images: bool = True,
    convert_to_jpg: bool = True,
):
    """Export to Instant-NGP format (same as Nerfstudio)."""
    return export_to_nerfstudio(
        sfm_result=sfm_result,
        images_dir=images_dir,
        output_dir=output_dir,
        cams=cams,
        K=K,
        aabb_scale=aabb_scale,
        copy_images=copy_images,
        convert_to_jpg=convert_to_jpg,
    )


def export_to_3dgs(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,
    K: Optional[np.ndarray] = None,
    copy_images: bool = True,
    convert_to_jpg: bool = True,
):
    """
    Export to 3D Gaussian Splatting format.
    3DGS uses COLMAP format, so we delegate to colmap_export.
    """
    from data_io.colmap_export import export_to_colmap
    
    print("[3DGS Export] 3D Gaussian Splatting uses COLMAP format")
    
    return export_to_colmap(
        sfm_result=sfm_result,
        images_dir=images_dir,
        output_dir=output_dir,
        cams=cams,
        K=K,
        binary=True,
        copy_images=copy_images,
        convert_to_jpg=convert_to_jpg,
    )