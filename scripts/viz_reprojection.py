import argparse
from pathlib import Path

from data_io.images import collect_image_paths, load_images
from datasets.multicam import load_multicam_dataset
from datasets.singlecam import load_singlecam_dataset
from utils.checkpoint import load_checkpoint
from src.visualization.reproject_overlays import save_sparse_reprojection_overlays


def main():
    parser = argparse.ArgumentParser(
        description="Visualize sparse reprojection overlays from an SfM checkpoint (no SfM run)."
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_sfm.pkl")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for overlay images")

    # Choose ONE mode:
    # - multicam: use images_dir + cameras_dir (+ optional masks_dir) to get per-view K
    # - singlecam: use data + K.txt (K_global)
    parser.add_argument("--multicam", action="store_true", help="Use multicam dataset loader")

    # multicam inputs
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--cameras_dir", type=str, default=None)
    parser.add_argument("--masks_dir", type=str, default=None)

    # singlecam inputs
    parser.add_argument("--data", type=str, default=None, help="Directory of images for singlecam visualization")
    parser.add_argument("--k_file", type=str, default=None, help="Camera Intrinsic file")
    parser.add_argument("--resize_max", type=int, default=None)

    # overlay knobs
    parser.add_argument("--max_draw_points", type=int, default=800)
    parser.add_argument("--err_thresh_px", type=float, default=5.0)

    # IMPORTANT: matches your current overlay assumption about image channel order
    parser.add_argument("--images_are_rgb", action="store_true", default=True,
                        help="Treat loaded 3-channel images as RGB (default True).")
    
    parser.add_argument("--all_points", action="store_true",help="Project all 3D points into each image (not just points observed in that view).")

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sfm_result = load_checkpoint(ckpt_path)

    cams = None
    K_global = None

    if args.multicam:
        if args.images_dir is None or args.cameras_dir is None:
            raise SystemExit("--multicam requires --images_dir and --cameras_dir")

        ds = load_multicam_dataset(
            image_dir=Path(args.images_dir),
            camera_dir=Path(args.cameras_dir),
            mask_dir=Path(args.masks_dir) if args.masks_dir else None,
            resize_max=args.resize_max,
        )
        images = ds.images
        cams = ds.cams
        # K_global not needed in multicam mode
    else:
        if args.data is None:
            raise SystemExit("Singlecam mode requires --data (images directory).")
        ds = load_singlecam_dataset(
        image_dir=Path(args.data),
        K_file=Path(args.k_file) if args.k_file else None,
        resize_max=args.resize_max,
        )
        images = ds.images
        K_global = ds.K

    # If your overlay converts RGB->BGR internally (it currently does),
    # this flag should match your actual loader convention.
    # If your loader returns BGR, set --images_are_rgb false and adjust overlay code accordingly.
    # (Right now your overlay always assumes RGB. That’s fine if you stay consistent.)

    save_sparse_reprojection_overlays(
        all_points=args.all_points,
        out_dir=args.out_dir,
        images=images,
        sfm_result=sfm_result,
        cams=cams,
        K_global=K_global,
        max_draw_points=args.max_draw_points,
        err_thresh_px=args.err_thresh_px,
    )

    print(f"[viz] wrote overlays to: {args.out_dir}")


if __name__ == "__main__":
    main()
