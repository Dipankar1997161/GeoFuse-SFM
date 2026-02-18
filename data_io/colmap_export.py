"""
data_io/colmap_export.py

Export GeoFuse-SFM results to COLMAP format.
Supports both binary (.bin) and text (.txt) formats.
Converts images to JPG for 3DGS/NeRF compatibility.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
import cv2


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns (x, y, z, w)
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    quat = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
    rot = Rotation.from_quat(quat)
    return rot.as_matrix()


# ============================================================
# IMAGE CONVERSION
# ============================================================

def convert_image(
    src_path: Path, 
    dst_path: Path, 
    to_jpg: bool = True,
    jpg_quality: int = 95,
):
    """
    Convert image to JPG/PNG with correct color handling.
    
    Handles PPM, PGM, PNG, JPG, etc.
    OpenCV reads all formats as BGR internally, so colors will be correct.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"  [Warning] Could not read: {src_path}")
        return False
    
    if to_jpg:
        cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    else:
        # PNG
        cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    return True


# ============================================================
# BINARY FORMAT WRITERS
# ============================================================

def write_cameras_binary(
    path: Path,
    camera_models: Dict[int, Tuple[int, int, int, np.ndarray]],
):
    """Write cameras.bin file."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(camera_models)))
        
        for cam_id, (model_id, width, height, params) in camera_models.items():
            f.write(struct.pack("<I", cam_id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", width))
            f.write(struct.pack("<Q", height))
            
            for p in params:
                f.write(struct.pack("<d", p))


def write_images_binary(
    path: Path,
    images_data: Dict[int, Tuple[np.ndarray, np.ndarray, int, str, List[Tuple[float, float, int]]]],
):
    """Write images.bin file."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images_data)))
        
        for img_id, (qvec, tvec, cam_id, name, points2D) in images_data.items():
            f.write(struct.pack("<I", img_id))
            
            for q in qvec:
                f.write(struct.pack("<d", q))
            
            for t in tvec:
                f.write(struct.pack("<d", t))
            
            f.write(struct.pack("<I", cam_id))
            
            name_bytes = name.encode("utf-8") + b"\x00"
            f.write(name_bytes)
            
            f.write(struct.pack("<Q", len(points2D)))
            
            for x, y, p3d_id in points2D:
                f.write(struct.pack("<d", x))
                f.write(struct.pack("<d", y))
                f.write(struct.pack("<q", p3d_id))


def write_points3D_binary(
    path: Path,
    points_data: Dict[int, Tuple[np.ndarray, np.ndarray, float, List[Tuple[int, int]]]],
):
    """Write points3D.bin file."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points_data)))
        
        for p3d_id, (xyz, rgb, error, track) in points_data.items():
            f.write(struct.pack("<Q", p3d_id))
            
            for coord in xyz:
                f.write(struct.pack("<d", coord))
            
            for c in rgb:
                f.write(struct.pack("<B", int(c)))
            
            f.write(struct.pack("<d", error))
            
            f.write(struct.pack("<Q", len(track)))
            
            for img_id, p2d_idx in track:
                f.write(struct.pack("<I", img_id))
                f.write(struct.pack("<I", p2d_idx))


# ============================================================
# TEXT FORMAT WRITERS
# ============================================================

def write_cameras_text(
    path: Path,
    camera_models: Dict[int, Tuple[int, int, int, np.ndarray]],
):
    """Write cameras.txt file."""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_models)}\n")
        
        model_names = {
            0: "SIMPLE_PINHOLE",
            1: "PINHOLE",
            2: "SIMPLE_RADIAL",
            3: "RADIAL",
            4: "OPENCV",
        }
        
        for cam_id, (model_id, width, height, params) in camera_models.items():
            model_name = model_names.get(model_id, "PINHOLE")
            params_str = " ".join(f"{p:.6f}" for p in params)
            f.write(f"{cam_id} {model_name} {width} {height} {params_str}\n")


def write_images_text(
    path: Path,
    images_data: Dict[int, Tuple[np.ndarray, np.ndarray, int, str, List[Tuple[float, float, int]]]],
):
    """Write images.txt file."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_data)}\n")
        
        for img_id, (qvec, tvec, cam_id, name, points2D) in images_data.items():
            qw, qx, qy, qz = qvec
            tx, ty, tz = tvec
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} {cam_id} {name}\n")
            
            pts_str = " ".join(f"{x:.6f} {y:.6f} {p3d}" for x, y, p3d in points2D)
            f.write(f"{pts_str}\n")


def write_points3D_text(
    path: Path,
    points_data: Dict[int, Tuple[np.ndarray, np.ndarray, float, List[Tuple[int, int]]]],
):
    """Write points3D.txt file."""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points_data)}\n")
        
        for p3d_id, (xyz, rgb, error, track) in points_data.items():
            x, y, z = xyz
            r, g, b = rgb
            track_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in track)
            f.write(f"{p3d_id} {x:.9f} {y:.9f} {z:.9f} "
                    f"{int(r)} {int(g)} {int(b)} {error:.9f} {track_str}\n")


# ============================================================
# MAIN EXPORT FUNCTION
# ============================================================

def export_to_colmap(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,
    K: Optional[np.ndarray] = None,
    binary: bool = True,
    copy_images: bool = True,
    convert_to_jpg: bool = True,
    jpg_quality: int = 95,
):
    """
    Export GeoFuse-SFM result to COLMAP format.
    
    Args:
        sfm_result: SfMResult from run_incremental_sfm
        images_dir: Directory containing source images
        output_dir: Output directory for COLMAP files
        cams: List of DecomposedCamera (multicam mode)
        K: Shared intrinsic matrix (singlecam mode)
        binary: If True, write binary format; else text
        copy_images: If True, copy/convert images
        convert_to_jpg: If True, convert PPM/PNG to JPG
        jpg_quality: JPEG quality (0-100)
    
    Creates:
        output_dir/
        ├── sparse/0/
        │   ├── cameras.bin
        │   ├── images.bin
        │   └── points3D.bin
        └── images/
            ├── image_0000.jpg
            └── ...
    """
    output_path = Path(output_dir)
    sparse_path = output_path / "sparse" / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    images_path = Path(images_dir)
    is_multicam = cams is not None
    
    print(f"[COLMAP Export] Mode: {'multicam' if is_multicam else 'singlecam'}")
    print(f"[COLMAP Export] Registered images: {len(sfm_result.registered_images)}")
    print(f"[COLMAP Export] 3D points: {sfm_result.X.shape[0]}")
    
    from data_io.images import collect_image_paths
    img_paths = collect_image_paths(images_path)
    
    # Build image name mapping
    img_id_to_name = {}
    img_id_to_path = {}
    img_id_to_new_name = {}  # For converted images
    
    for img_id in sfm_result.registered_images:
        if img_id < len(img_paths):
            src_path = img_paths[img_id]
            img_id_to_path[img_id] = src_path
            
            # Determine output name
            if convert_to_jpg and src_path.suffix.lower() not in ['.jpg', '.jpeg']:
                new_name = src_path.stem + ".jpg"
            else:
                new_name = src_path.name
            
            img_id_to_name[img_id] = new_name
            img_id_to_new_name[img_id] = new_name
    
    # =========================================================
    # 1. Copy/Convert images
    # =========================================================
    if copy_images:
        images_out = output_path / "images"
        images_out.mkdir(exist_ok=True)
        
        print(f"[COLMAP Export] Converting images to {images_out}")
        
        for img_id in sfm_result.registered_images:
            if img_id not in img_id_to_path:
                continue
            
            src_path = img_id_to_path[img_id]
            new_name = img_id_to_new_name[img_id]
            dst_path = images_out / new_name
            
            if not dst_path.exists():
                needs_conversion = (
                    convert_to_jpg and 
                    src_path.suffix.lower() in ['.ppm', '.pgm', '.pbm', '.png', '.tif', '.tiff']
                )
                
                if needs_conversion:
                    convert_image(src_path, dst_path, to_jpg=True, jpg_quality=jpg_quality)
                else:
                    import shutil
                    shutil.copy(src_path, dst_path)
        
        print(f"[COLMAP Export] Images converted: {len(sfm_result.registered_images)}")
    
    # =========================================================
    # 2. Build camera models
    # =========================================================
    camera_models = {}
    img_to_cam_id = {}
    
    if is_multicam:
        for img_id in sfm_result.registered_images:
            cam = cams[img_id]
            K_i = np.asarray(cam.K, np.float64)
            
            fx, fy = K_i[0, 0], K_i[1, 1]
            cx, cy = K_i[0, 2], K_i[1, 2]
            
            if img_id in img_id_to_path:
                img = cv2.imread(str(img_id_to_path[img_id]))
                h, w = img.shape[:2]
            else:
                h, w = 1024, 1024
            
            camera_models[img_id + 1] = (1, w, h, np.array([fx, fy, cx, cy]))
            img_to_cam_id[img_id] = img_id + 1
    else:
        K_arr = np.asarray(K, np.float64)
        fx, fy = K_arr[0, 0], K_arr[1, 1]
        cx, cy = K_arr[0, 2], K_arr[1, 2]
        
        if sfm_result.registered_images:
            first_id = sfm_result.registered_images[0]
            if first_id in img_id_to_path:
                img = cv2.imread(str(img_id_to_path[first_id]))
                h, w = img.shape[:2]
            else:
                h, w = 1024, 1024
        else:
            h, w = 1024, 1024
        
        camera_models[1] = (1, w, h, np.array([fx, fy, cx, cy]))
        for img_id in sfm_result.registered_images:
            img_to_cam_id[img_id] = 1
    
    # =========================================================
    # 3. Build images data
    # =========================================================
    images_data = {}
    
    img_kp_to_p3d = {}
    for tid, pid in sfm_result.track_to_point.items():
        tr = sfm_result.tracks[tid]
        for img_id, kp_id in tr.obs.items():
            if img_id not in sfm_result.cam_poses:
                continue
            if img_id not in img_kp_to_p3d:
                img_kp_to_p3d[img_id] = {}
            img_kp_to_p3d[img_id][kp_id] = pid
    
    for img_id in sfm_result.registered_images:
        R, t = sfm_result.cam_poses[img_id]
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64).flatten()
        
        qvec = rotmat_to_qvec(R)
        tvec = t
        
        cam_id = img_to_cam_id[img_id]
        name = img_id_to_name.get(img_id, f"image_{img_id:04d}.jpg")
        
        feats = sfm_result.feats[img_id]
        points2D = []
        
        kp_to_p3d = img_kp_to_p3d.get(img_id, {})
        
        for kp_idx, xy in enumerate(feats.kpts_xy):
            x, y = float(xy[0]), float(xy[1])
            p3d_id = kp_to_p3d.get(kp_idx, -1)
            points2D.append((x, y, p3d_id))
        
        images_data[img_id + 1] = (qvec, tvec, cam_id, name, points2D)
    
    # =========================================================
    # 4. Build points3D data
    # =========================================================
    points_data = {}
    
    for tid, pid in sfm_result.track_to_point.items():
        xyz = sfm_result.X[pid]
        
        rgb = np.array([128, 128, 128], dtype=np.uint8)
        
        meta = sfm_result.point_meta.get(pid)
        error = meta.median_reproj_error if meta else 0.0
        if np.isnan(error):
            error = 0.0
        
        track = []
        tr = sfm_result.tracks[tid]
        for img_id, kp_id in tr.obs.items():
            if img_id in sfm_result.cam_poses:
                track.append((img_id + 1, kp_id))
        
        points_data[pid + 1] = (xyz, rgb, error, track)
    
    # Update points2D references to 1-indexed
    for img_id, (qvec, tvec, cam_id, name, points2D) in images_data.items():
        updated_points2D = []
        for x, y, p3d_id in points2D:
            if p3d_id >= 0:
                p3d_id += 1
            updated_points2D.append((x, y, p3d_id))
        images_data[img_id] = (qvec, tvec, cam_id, name, updated_points2D)
    
    # =========================================================
    # 5. Write files
    # =========================================================
    if binary:
        write_cameras_binary(sparse_path / "cameras.bin", camera_models)
        write_images_binary(sparse_path / "images.bin", images_data)
        write_points3D_binary(sparse_path / "points3D.bin", points_data)
        print(f"[COLMAP Export] Written binary files to {sparse_path}")
    else:
        write_cameras_text(sparse_path / "cameras.txt", camera_models)
        write_images_text(sparse_path / "images.txt", images_data)
        write_points3D_text(sparse_path / "points3D.txt", points_data)
        print(f"[COLMAP Export] Written text files to {sparse_path}")
    
    # =========================================================
    # 6. Symlink images if not copied
    # =========================================================
    if not copy_images:
        images_link = output_path / "images"
        if not images_link.exists():
            try:
                images_link.symlink_to(images_path.resolve())
                print(f"[COLMAP Export] Symlinked images: {images_link} -> {images_path}")
            except OSError:
                print(f"[COLMAP Export] Note: Manually copy images to {images_link}")
    
    print(f"[COLMAP Export] Done! Output: {output_path}")
    print(f"[COLMAP Export] Cameras: {len(camera_models)}")
    print(f"[COLMAP Export] Images: {len(images_data)}")
    print(f"[COLMAP Export] Points: {len(points_data)}")
    
    return output_path
