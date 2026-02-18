"""
data_io/export.py

Unified export interface for GeoFuse-SFM results.

Supports:
- COLMAP (binary/text) - for OpenMVS, 3DGS, etc.
- Nerfstudio - for NeRF training
- Instant-NGP - for fast NeRF
- 3D Gaussian Splatting - uses COLMAP format

Usage:
    from data_io.export import export_reconstruction
    
    # Export to all formats
    export_reconstruction(
        sfm_result=result,
        images_dir="Data/fountain/images",
        output_dir="output/fountain",
        cams=cams,
        formats=["colmap", "nerfstudio"],
    )
    
    # Or use specific functions
    from data_io.colmap_export import export_to_colmap
    from data_io.nerfstudio_export import export_to_nerfstudio
"""

from pathlib import Path
from typing import List, Optional
import numpy as np


def export_reconstruction(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,                          # For multicam
    K: Optional[np.ndarray] = None,     # For singlecam
    formats: List[str] = ["colmap"],
    copy_images: bool = False,
):
    """
    Export reconstruction to multiple formats.
    
    Args:
        sfm_result: SfMResult from run_incremental_sfm
        images_dir: Directory containing source images
        output_dir: Base output directory
        cams: List of DecomposedCamera (multicam mode)
        K: Shared intrinsic matrix (singlecam mode)
        formats: List of formats to export:
                 ["colmap", "colmap_text", "nerfstudio", "instant_ngp", "3dgs"]
        copy_images: If True, copy images to output directories
    
    Returns:
        Dict mapping format name to output path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for fmt in formats:
        fmt = fmt.lower().replace("-", "_").replace(" ", "_")
        
        if fmt == "colmap" or fmt == "colmap_bin" or fmt == "colmap_binary":
            from data_io.colmap_export import export_to_colmap
            out = export_to_colmap(
                sfm_result=sfm_result,
                images_dir=images_dir,
                output_dir=str(output_path / "colmap"),
                cams=cams,
                K=K,
                binary=True,
            )
            results["colmap"] = out
            
        elif fmt == "colmap_text" or fmt == "colmap_txt":
            from data_io.colmap_export import export_to_colmap
            out = export_to_colmap(
                sfm_result=sfm_result,
                images_dir=images_dir,
                output_dir=str(output_path / "colmap_text"),
                cams=cams,
                K=K,
                binary=False,
            )
            results["colmap_text"] = out
            
        elif fmt == "nerfstudio" or fmt == "nerf":
            from data_io.nerfstudio_export import export_to_nerfstudio
            out = export_to_nerfstudio(
                sfm_result=sfm_result,
                images_dir=images_dir,
                output_dir=str(output_path / "nerfstudio"),
                cams=cams,
                K=K,
                copy_images=copy_images,
            )
            results["nerfstudio"] = out
            
        elif fmt == "instant_ngp" or fmt == "ingp" or fmt == "ngp":
            from data_io.nerfstudio_export import export_to_instant_ngp
            out = export_to_instant_ngp(
                sfm_result=sfm_result,
                images_dir=images_dir,
                output_dir=str(output_path / "instant_ngp"),
                cams=cams,
                K=K,
                copy_images=copy_images,
            )
            results["instant_ngp"] = out
            
        elif fmt == "3dgs" or fmt == "gaussian_splatting" or fmt == "gs":
            from data_io.nerfstudio_export import export_to_3dgs
            out = export_to_3dgs(
                sfm_result=sfm_result,
                images_dir=images_dir,
                output_dir=str(output_path / "3dgs"),
                cams=cams,
                K=K,
            )
            results["3dgs"] = out
            
        else:
            print(f"[Export] Unknown format: {fmt}")
            print(f"[Export] Supported: colmap, colmap_text, nerfstudio, instant_ngp, 3dgs")
    
    return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def export_for_nerf_training(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,
    K=None,
):
    """Export for NeRF training (Nerfstudio + Instant-NGP)."""
    return export_reconstruction(
        sfm_result=sfm_result,
        images_dir=images_dir,
        output_dir=output_dir,
        cams=cams,
        K=K,
        formats=["nerfstudio", "instant_ngp"],
        copy_images=True,
    )


def export_for_3dgs_training(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,
    K=None,
):
    """Export for 3D Gaussian Splatting training."""
    return export_reconstruction(
        sfm_result=sfm_result,
        images_dir=images_dir,
        output_dir=output_dir,
        cams=cams,
        K=K,
        formats=["3dgs"],
    )


def export_for_openmvs(
    sfm_result,
    images_dir: str,
    output_dir: str,
    cams=None,
    K=None,
):
    """Export for OpenMVS dense reconstruction."""
    return export_reconstruction(
        sfm_result=sfm_result,
        images_dir=images_dir,
        output_dir=output_dir,
        cams=cams,
        K=K,
        formats=["colmap"],
    )
