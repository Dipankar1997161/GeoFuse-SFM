"""
scripts/run_multicam.py

Unified runner for both multicam and singlecam SfM.
Uses the new config system - all parameters in one place.
"""

import argparse
from pathlib import Path

from src.diagnostics.multicam_diagnostics import (
    analyze_point_statistics,
    find_points_outside_silhouettes,
    project_all_points_to_all_cameras,
)
from src.pipeline import (
    run_incremental_sfm,
    SfMConfig,
    MatchingConfig,
    TriangulationConfig,
    BAConfig,
    CleanupConfig,
    DiagnosticsConfig,
    get_multicam_config,
    get_singlecam_config,
)
from data_io.pointcloud_io import write_ply
from datasets.multicam import load_multicam_dataset
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


STAGES = ["sparse", "mvs", "propagate", "filter", "diagnosis", "fusion"]

EXPORT_FORMATS = ["colmap", "colmap_text", "nerfstudio", "instant_ngp", "3dgs"]


def build_config_from_args(args, is_multicam: bool) -> SfMConfig:
    """
    Build SfMConfig from command line arguments.
    
    Starts with appropriate preset (multicam or singlecam),
    then overrides with any explicitly provided arguments.
    """
    # Start with preset
    if args.feature_preset == "textureless":
        # Use textureless config as base
        from src.pipeline.config import get_textureless_config
        config = get_textureless_config()
    elif is_multicam:
        config = get_multicam_config()
    else:
        config = get_singlecam_config()
    
    # Override matching params
    if args.ratio is not None:
        config.matching.ratio = args.ratio
    if args.ransac_thresh_px is not None:
        config.matching.ransac_thresh_px = args.ransac_thresh_px
    if args.pair_window is not None:
        config.matching.pair_window = args.pair_window
    
    # Override triangulation params
    if args.max_reproj_px is not None:
        config.triangulation.max_reproj_px = args.max_reproj_px
    if args.min_triang_angle_deg is not None:
        config.triangulation.min_triang_angle_deg = args.min_triang_angle_deg
    if args.min_views is not None:
        config.triangulation.min_views_accept = args.min_views
    
    # Override BA params
    if args.ba_mode is not None:
        config.ba.mode = args.ba_mode
    
    # Override cleanup params
    if args.adaptive_good_view_ratio is not None:
        config.cleanup.adaptive_good_view_ratio = args.adaptive_good_view_ratio
    if args.adaptive_min_observations is not None:
        config.cleanup.adaptive_min_observations = args.adaptive_min_observations
    if args.min_good_views is not None:
        config.cleanup.min_good_views = args.min_good_views
    if args.max_reproj_any_view is not None:
        config.cleanup.max_reproj_any_view = args.max_reproj_any_view
    
    # Spatial filtering
    adaptive_spatial = args.adaptive_spatial_filtering and not args.no_adaptive_spatial_filtering
    config.cleanup.adaptive_spatial_filtering = adaptive_spatial
    
    # Diagnostics
    config.diagnostics.reproj_report = True
    config.verbose = args.verbose

    # Apply feature preset if specified (overrides base)
    if args.feature_preset is not None:
        config.set_feature_preset(args.feature_preset)
    
    # Override individual SIFT params if specified
    if args.sift_contrast is not None:
        config.matching.feature.sift_contrastThreshold = args.sift_contrast
    if args.sift_nfeatures is not None:
        config.matching.feature.sift_nfeatures = args.sift_nfeatures

    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="GeoFuse-SfM runner (supports both multicam and singlecam)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multicam with calibrated cameras
  python -m scripts.run_multicam --images_dir Data/bird/images --cameras_dir Data/bird/calib --masks_dir Data/bird/silhouettes --visualize

  # Singlecam (no --cameras_dir)
  python -m scripts.run_multicam --images_dir Data/fountain/images --K_file Data/fountain/K.txt --visualize

  # Export to COLMAP + Nerfstudio for NeRF training
  python -m scripts.run_multicam --images_dir Data/fountain/images --K_file Data/fountain/K.txt --export colmap nerfstudio
 
  # Export to 3DGS format
  python -m scripts.run_multicam --images_dir Data/fountain/images --K_file Data/fountain/K.txt --export 3dgs
        """
    )

    # =========================================================
    # INPUT PATHS
    # =========================================================
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--cameras_dir", type=str, default=None,
                        help="Directory containing per-image camera files (multicam mode)")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="Optional directory containing silhouettes/masks")
    parser.add_argument("--K_file", type=str, default=None,
                        help="Intrinsic matrix file (singlecam mode)")

    # =========================================================
    # OUTPUT
    # =========================================================
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: output/<scene_name>)")

    # =========================================================
    # PREPROCESSING
    # =========================================================
    parser.add_argument("--resize_max", type=int, default=None,
                        help="Resize so max(H,W)=resize_max")

    # =========================================================
    # STAGE CONTROL
    # =========================================================
    parser.add_argument("--skip_to_stage", type=str, choices=STAGES)
    parser.add_argument("--stop_after_stage", type=str, choices=STAGES)
    parser.add_argument("--force_rerun", action="store_true")

    # =========================================================
    # FEATURE DETECTION CONFIG (add to argparse)
    # =========================================================

    parser.add_argument("--feature_preset", type=str, default=None,
                        choices=["default", "textureless", "high_detail"],
                        help="""Feature detection preset:
                        - default: Standard SIFT for textured scenes (buildings, outdoor)
                        - textureless: For smooth/plastic/skin surfaces (lower contrast threshold)
                        - high_detail: Maximum features for complex scenes""")

    parser.add_argument("--sift_contrast", type=float, default=None,
                        help="SIFT contrastThreshold (default: 0.04, use 0.004 for textureless)")

    parser.add_argument("--sift_nfeatures", type=int, default=None,
                        help="SIFT max features (0 = no limit)")

    # =========================================================
    # MATCHING CONFIG
    # =========================================================
    parser.add_argument("--ratio", type=float, default=None,
                        help="Lowe's ratio test (default: adaptive)")
    parser.add_argument("--ransac_thresh_px", type=float, default=None,
                        help="RANSAC threshold (default: adaptive)")
    parser.add_argument("--pair_window", type=int, default=None,
                        help="Pair window for singlecam (default: adaptive)")

    # =========================================================
    # TRIANGULATION CONFIG
    # =========================================================
    parser.add_argument("--max_reproj_px", type=float, default=None,
                        help="Max reprojection error (default: 1.5)")
    parser.add_argument("--min_triang_angle_deg", type=float, default=None,
                        help="Min triangulation angle (default: 3.0)")
    parser.add_argument("--min_views", type=int, default=None,
                        help="Min views to accept point (default: 3 for multicam, 2 for singlecam)")

    # =========================================================
    # BA CONFIG
    # =========================================================
    parser.add_argument("--ba_mode", type=str, default=None,
                        choices=["none", "local", "global_end", "local+global_end"],
                        help="BA mode (default: none for multicam, local+global_end for singlecam)")

    # =========================================================
    # CLEANUP CONFIG
    # =========================================================
    parser.add_argument("--adaptive_good_view_ratio", type=float, default=None,
                        help="Good view ratio for adaptive filter (default: 0.60)")
    parser.add_argument("--adaptive_min_observations", type=int, default=None,
                        help="Min observations for adaptive filter (default: 2)")
    parser.add_argument("--min_good_views", type=int, default=None,
                        help="Min good views for strict filter (default: 3)")
    parser.add_argument("--max_reproj_any_view", type=float, default=None,
                        help="Max reproj in any view for strict filter (default: 5.0)")
    parser.add_argument("--adaptive_spatial_filtering", action="store_true", default=True)
    parser.add_argument("--no_adaptive_spatial_filtering", action="store_true", default=False)

    # =========================================================
    # EXPORT CONFIG
    # =========================================================
    parser.add_argument("--export", type=str, nargs="+", default=None,
                        choices=EXPORT_FORMATS,
                        help=f"Export formats: {', '.join(EXPORT_FORMATS)}")
    
    parser.add_argument("--export_dir", type=str, default=None,
                        help="Export output directory (default: <output>/exports)")
    
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy images to export directory (for nerfstudio/instant_ngp)")

    # =========================================================
    # DIAGNOSTICS
    # =========================================================
    parser.add_argument("--visualize", action="store_true",
                        help="Write reprojection overlay images")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # =========================================================
    # SETUP
    # =========================================================
    scene_name = Path(args.images_dir).resolve().name
    output_dir = Path(args.output) if args.output else (Path("output") / scene_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Pipeline] Output directory: {output_dir}")

    CP_SFM = output_dir / "checkpoint_sfm.pkl"

    start_stage = STAGES.index(args.skip_to_stage) if args.skip_to_stage else 0
    end_stage = STAGES.index(args.stop_after_stage) if args.stop_after_stage else len(STAGES) - 1
    print(f"[Pipeline] Stages: {STAGES[start_stage]} -> {STAGES[end_stage]}")

    # =========================================================
    # DETERMINE MODE: MULTICAM vs SINGLECAM
    # =========================================================
    is_multicam = args.cameras_dir is not None
    print(f"[Pipeline] Mode: {'MULTICAM' if is_multicam else 'SINGLECAM'}")

    # =========================================================
    # LOAD DATASET
    # =========================================================
    if is_multicam:
        ds = load_multicam_dataset(
            image_dir=Path(args.images_dir),
            camera_dir=Path(args.cameras_dir),
            mask_dir=Path(args.masks_dir) if args.masks_dir else None,
            resize_max=args.resize_max,
        )
        images = ds.images
        cams = ds.cams
        silhouettes = ds.silhouettes
        K = ds.K_shared  # For reference only
    else:
        # Singlecam mode
        from datasets.singlecam import load_singlecam_dataset
        ds = load_singlecam_dataset(
            image_dir=Path(args.images_dir),
            K_file=Path(args.K_file) if args.K_file else None,
            resize_max=args.resize_max,
        )
        images = ds.images
        cams = None
        silhouettes = None
        K = ds.K

    print(f"[Loading] Loaded {len(images)} images")
    if K is not None:
        print(f"[Loading] K:\n{K}")

    # =========================================================
    # BUILD CONFIG
    # =========================================================
    config = build_config_from_args(args, is_multicam)
    config.diagnostics.out_dir = str(output_dir / "sfm_diagnostics")
    config.diagnostics.enabled = True

    # Print config summary
    print(f"\n[Config] Matching:")
    print(f"  ratio={config.matching.ratio}")
    print(f"  ransac_thresh_px={config.matching.ransac_thresh_px}")
    print(f"[Config] Triangulation:")
    print(f"  max_reproj_px={config.triangulation.max_reproj_px}")
    print(f"  min_views_accept={config.triangulation.min_views_accept}")
    print(f"[Config] BA: mode={config.ba.mode}")
    print(f"[Config] Cleanup:")
    print(f"  min_good_views={config.cleanup.min_good_views}")
    print(f"  max_reproj_any_view={config.cleanup.max_reproj_any_view}")

    # =========================================================
    # SPARSE STAGE
    # =========================================================
    sfm_result = None
    if STAGES.index("sparse") >= start_stage and STAGES.index("sparse") <= end_stage:
        print("\n" + "=" * 70)
        print("STAGE: SPARSE SFM")
        print("=" * 70)

        if checkpoint_exists(CP_SFM) and not args.force_rerun:
            print("[Sparse] Loading from checkpoint...")
            sfm_result = load_checkpoint(CP_SFM)
        else:
            sfm_result = run_incremental_sfm(
                images=images,
                K=K,
                silhouettes=silhouettes,
                cams=cams,
                config=config,
                out_dir=str(output_dir / "sfm_diagnostics"),
            )
            save_checkpoint(sfm_result, CP_SFM)
            print(f"  [Checkpoint] Saved: {CP_SFM}")

        # Visualization
        if args.visualize:
            print("\n[Visualize] Generating overlays...")
            
            # 1. Project ALL points to ALL cameras
            project_all_points_to_all_cameras(
                out_dir=str(output_dir / "all_proj_overlays"),
                images=images,
                sfm_result=sfm_result,
                cams=cams,      # None for singlecam
                K=K,  
                silhouettes=silhouettes,
                max_draw_points=500,
            )

            # 2. Analyze point statistics
            stats = analyze_point_statistics(
                sfm_result, cams,
                percentile_threshold=95.0
            )

            # 3. Find points outside silhouettes (multicam only)
            if silhouettes is not None:
                bad_points = find_points_outside_silhouettes(
                    sfm_result, cams, silhouettes,
                    margin_px=3,
                    min_outside_ratio=0.5,
                )

        # Summary
        print(f"\n  Registered: {len(sfm_result.registered_images)}/{len(images)}")
        print(f"  Sparse points: {sfm_result.X.shape[0]:,}")
        
        # Save point cloud
        ply_path = output_dir / "sparse_points.ply"
        write_ply(str(ply_path), sfm_result.X)
        print(f"  Saved: {ply_path}")

    else:
        sfm_result = load_checkpoint(CP_SFM)

    if STAGES.index("sparse") >= end_stage:
        print("\n[Done] Sparse stage complete.")
        return
    
    # =========================================================
    # EXPORT STAGE
    # =========================================================
    if args.export and sfm_result is not None:
        print("\n" + "=" * 70)
        print("STAGE: EXPORT")
        print("=" * 70)
        
        from data_io.export import export_reconstruction        
        export_dir = Path(args.export_dir) if args.export_dir else (output_dir / "exports")
        
        print(f"[Export] Formats: {args.export}")
        print(f"[Export] Output: {export_dir}")
        
        export_results = export_reconstruction(
            sfm_result=sfm_result,
            images_dir=args.images_dir,
            output_dir=str(export_dir),
            cams=cams,
            K=K,
            formats=args.export,
            copy_images=args.copy_images,
        )
        
        print(f"\n[Export] Completed exports:")
        for fmt, path in export_results.items():
            print(f"  {fmt}: {path}")

    if STAGES.index("sparse") >= end_stage:
        print("\n[Done] Sparse stage complete.")
        return

    # =========================================================
    # FUTURE STAGES (MVS, etc.)
    # =========================================================
    # TODO: Add MVS, propagate, filter, fusion stages here
    
    print("\n[Done] Pipeline complete.")


if __name__ == "__main__":
    main()
