# sfm/visualization/reprojection.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Set
import numpy as np
from src.geometry_utils.reprojection import project_points as project_points_geom

from cv2 import (
    cvtColor, circle, drawMatches, line, putText, imwrite,
    COLOR_GRAY2BGR, COLOR_RGB2BGR,
    FONT_HERSHEY_SIMPLEX, LINE_AA,
    DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, DMatch, KeyPoint
)

from src.features import Features
from src.tracks import Track

# def project_points(K, R, t, X):
#     X = np.asarray(X, np.float64)
#     R = np.asarray(R, np.float64)
#     t = np.asarray(t, np.float64).reshape(3, 1)
#     K = np.asarray(K, np.float64)

#     Xc = (R @ X.T) + t  # 3xN
#     z = Xc[2, :]
#     valid = z > 1e-9

#     x = Xc[:2, :] / (z + 1e-12)
#     uv = (K[:2, :2] @ x) + K[:2, 2:3]  # 2xN
#     return uv.T, valid

def project_points(K, R, t, X):
    uv = project_points_geom(X=np.asarray(X), K=K, R=R, t=t)  # (N,2) with NaNs
    valid = np.isfinite(uv).all(axis=1)                       # matches your intent
    return uv, valid

def save_sparse_reprojection_overlays(
    out_dir: str,
    images,
    sfm_result,
    *,
    K_global=None,
    cams=None,
    max_draw_points=2000,
    err_thresh_px=5.0,
    all_points: bool = False,
):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    cam_poses = sfm_result.cam_poses
    X = np.asarray(sfm_result.X, np.float64)
    tracks = sfm_result.tracks
    feats = sfm_result.feats
    track_to_point = sfm_result.track_to_point

    multicam = cams is not None

    for img_id in sfm_result.registered_images:
        img = images[img_id].copy()
        if img.ndim == 2:
            img = cvtColor(img, COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            # assume RGB -> BGR for OpenCV draw
            img = cvtColor(img, COLOR_RGB2BGR)

        h, w = img.shape[:2]

        R, t = cam_poses[img_id]
        if multicam:
            K = cams[img_id].K if hasattr(cams[img_id], "K") else cams[img_id][0]
        else:
            K = K_global

        if K is None:
            raise ValueError("Need K_global (non-multicam) or cams (multicam).")

        # ---------------------------------------------------------
        # If requested: project ALL points into this image and draw them.
        # Uses the canonical projector (project_points_geom) which returns NaNs for invalid depth.
        # ---------------------------------------------------------
        uv_all = None
        valid_all = None
        in_bounds_all = None

        if all_points:
            uv_all = project_points_geom(X=X, K=np.asarray(K), R=np.asarray(R), t=np.asarray(t))  # (N,2) with NaNs
            valid_all = np.isfinite(uv_all).all(axis=1)

            u = uv_all[:, 0]
            v = uv_all[:, 1]
            in_bounds_all = valid_all & (u >= 0) & (u < w) & (v >= 0) & (v < h)

            draw_ids = np.where(in_bounds_all)[0]
            if draw_ids.size > 0:
                # Sample if too many; otherwise the image becomes a solid blob
                if draw_ids.size > max_draw_points:
                    draw_ids = np.random.choice(draw_ids, size=max_draw_points, replace=False)

                for pid in draw_ids:
                    p = uv_all[pid]
                    p_i = (int(round(p[0])), int(round(p[1])))
                    circle(img, p_i, 2, (255, 255, 0), -1)  # cyan-ish (BGR)

        # ---------------------------------------------------------
        # Observed reprojection overlay (always useful; drawn on top)
        # ---------------------------------------------------------
        obs_uv, proj_uv, errs = [], [], []

        for tid, pid in track_to_point.items():
            tr = tracks[tid]
            if img_id not in tr.obs:
                continue

            kp_id = tr.obs[img_id]
            uv_obs = feats[img_id].kpts_xy[kp_id].astype(np.float64)

            if all_points and uv_all is not None and valid_all is not None:
                # Reuse precomputed projection for this point
                if not valid_all[pid]:
                    continue
                uv_proj = uv_all[pid]
            else:
                # Original path (projects only observed points one-by-one)
                uv_proj_all, valid = project_points(K, R, t, X[pid : pid + 1])
                if not valid[0]:
                    continue
                uv_proj = uv_proj_all[0]

            e = float(np.linalg.norm(uv_proj - uv_obs))
            obs_uv.append(uv_obs)
            proj_uv.append(uv_proj)
            errs.append(e)

        if not errs:
            # In all_points mode, you may still have drawn cyan points above
            suffix = "noobs_all" if all_points else "noobs"
            imwrite(str(outp / f"reproj_{img_id:03d}_{suffix}.png"), img)
            continue

        obs_uv = np.asarray(obs_uv)
        proj_uv = np.asarray(proj_uv)
        errs = np.asarray(errs)

        # Draw worst errors first (your current behavior)
        idx = np.argsort(-errs)[:max_draw_points]

        for k in idx:
            o = obs_uv[k]
            p = proj_uv[k]
            e = errs[k]

            o_i = (int(round(o[0])), int(round(o[1])))
            p_i = (int(round(p[0])), int(round(p[1])))

            circle(img, o_i, 3, (0, 255, 0), -1)  # observed
            circle(img, p_i, 3, (0, 0, 255), -1)  # projected
            col = (255, 255, 255) if e <= err_thresh_px else (0, 255, 255)
            line(img, o_i, p_i, col, 1)

        txt = (
            f"n={len(errs)} mean={errs.mean():.1f}px med={np.median(errs):.1f}px "
            f"p95={np.percentile(errs, 95):.1f}px"
        )
        if all_points and in_bounds_all is not None:
            txt += f" | all_in_bounds={int(in_bounds_all.sum())}/{X.shape[0]}"
        putText(img, txt, (20, 30), FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, LINE_AA)

        suffix = "all" if all_points else ""
        name = f"reproj_{img_id:03d}{'_' + suffix if suffix else ''}.png"
        imwrite(str(outp / name), img)



def save_kp_overlay(image_bgr: np.ndarray, feat, out_path: Path, max_draw: int = 3000):
    vis = image_bgr.copy()
    xy = feat.kpts_xy
    if xy is None or len(xy) == 0:
        imwrite(str(out_path), vis)
        return

    if len(xy) > max_draw:
        idx = np.random.choice(len(xy), size=max_draw, replace=False)
        xy = xy[idx]

    for (x, y) in xy:
        circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1, lineType=LINE_AA)

    imwrite(str(out_path), vis)

def save_match_overlay(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    feat1,
    feat2,
    matches,
    out_path: Path,
    max_draw: int = 200,
):
    # Build cv2.KeyPoint lists just for drawing
    kps1 = [KeyPoint(float(x), float(y), 1) for x, y in feat1.kpts_xy]
    kps2 = [KeyPoint(float(x), float(y), 1) for x, y in feat2.kpts_xy]

    dm = []
    for (qi, ti) in matches[:max_draw]:
        dm.append(DMatch(_queryIdx=int(qi), _trainIdx=int(ti), _distance=0.0))

    vis = drawMatches(
        img1_bgr, kps1,
        img2_bgr, kps2,
        dm, None,
        flags=DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    imwrite(str(out_path), vis)

REASON_NAMES = {
    1: "non_finite",
    2: "cheirality_cam1",
    3: "cheirality_cam2",
    4: "reproj_cam1",
    5: "reproj_cam2",
    6: "triang_angle",
    7: "max_depth",
    8: "max_abs_coord",
}

# BGR colors (OpenCV)
REASON_COLORS = {
    1: (0, 0, 255),     # red
    2: (255, 0, 0),     # blue
    3: (255, 0, 0),     # blue
    4: (0, 165, 255),   # orange
    5: (0, 165, 255),   # orange
    6: (255, 0, 255),   # magenta
    7: (0, 255, 255),   # yellow
    8: (0, 255, 0),     # green
}

def save_triang_reject_overlay(
    out_dir: str,
    img_id: int,
    step: int,
    image_bgr: np.ndarray,
    rejects_by_reason: dict,
    max_draw_per_reason: int = 800,
    radius: int = 2,
    thickness: int = -1,
    tag: str | None = None,   # <-- add this
):
    """
    Draw rejected 2D points color-coded by triangulation filter reason.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = image_bgr.copy()

    for reason, pts in sorted(rejects_by_reason.items(), key=lambda kv: kv[0]):
        if not pts:
            continue

        color = REASON_COLORS.get(int(reason), (200, 200, 200))
        name = REASON_NAMES.get(int(reason), f"reason_{reason}")

        if len(pts) > max_draw_per_reason:
            pts = pts[:max_draw_per_reason]

        for x, y in pts:
            circle(vis, (int(round(x)), int(round(y))), radius, color, thickness)

        putText(vis,
            f"{name}: {len(pts)}",
            (10, 30 + 22 * int(reason)),
            FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            LINE_AA,
        )

    putText(
        vis,
        f"triang rejects | img={img_id} step={step}",
        (10, 20),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        LINE_AA,
    )

    # filename logic
    if tag is None:
        filename = f"reject_overlay_img_{img_id:03d}_step_{step:03d}.png"
    else:
        filename = f"{tag}_img_{img_id:03d}.png"

    out_path = out_dir / filename
    imwrite(str(out_path), vis)
    return str(out_path)


def save_removed_points_overlays(
    out_dir: str,
    images,              # list of np arrays (RGB or BGR)
    feats,               # list of Features with kpts_xy
    tracks,              # list[Track]
    track_to_point_before,  # dict[tid -> old_pid]
    removed_pids,        # list[int] old pids removed
    max_draw_per_image=1500,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    removed_set = set(removed_pids)

    # pid -> list[tid] (inverse index)
    pid_to_tids = {}
    for tid, pid in track_to_point_before.items():
        pid_to_tids.setdefault(pid, []).append(tid)

    for img_id, img in enumerate(images):
        # collect 2D points to draw
        pts = []
        for pid in removed_set:
            for tid in pid_to_tids.get(pid, []):
                tr = tracks[tid]
                if img_id in tr.obs:
                    kp_id = tr.obs[img_id]
                    uv = feats[img_id].kpts_xy[kp_id]
                    pts.append((float(uv[0]), float(uv[1])))

        if not pts:
            continue

        # cap for speed/clarity
        if len(pts) > max_draw_per_image:
            pts = pts[:max_draw_per_image]

        # make BGR for cv2.imwrite
        img_bgr = img
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
            # assume RGB -> BGR (if your images are already BGR, remove this)
            img_bgr = cvtColor(img_bgr, COLOR_RGB2BGR)

        vis = img_bgr.copy()
        for x, y in pts:
            circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)  # red

        putText(
            vis, f"removed pts shown: {len(pts)}",
            (10, 30), FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, LINE_AA
        )

        imwrite(str(out_dir / f"removed_pts_img_{img_id:03d}.png"), vis)


def save_triangulation_reprojection_overlay(
    out_dir: str,
    img_id: int,
    step: int,
    image_bgr: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    X_list: List[np.ndarray],
    track_to_point: Dict[int, int],
    tracks: List[Track],
    feats: List[Features],
    registered_set: Set[int],
    max_draw: int = 500,
    err_thresh_px: float = 3.0,
    cams = None,  # Optional: dict of per-image (K,R,t
) -> None:
    """
    Draw observed keypoints vs reprojected points for one image.
    Saves a debug PNG. Draws only points that have an observation in img_id.
    """
    os.makedirs(out_dir, exist_ok=True)

    multicam = cams is not None
    img = image_bgr.copy()
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).reshape(3, 1)
    K = np.asarray(K, np.float64)

    drawn = 0

    for tid, pid in track_to_point.items():
        tr = tracks[tid]
        if img_id not in tr.obs:
            continue

        # 3D point
        X = np.asarray(X_list[pid], np.float64).reshape(3, 1)

        # project
        Xc = (R @ X) + t
        z = float(Xc[2, 0])
        if z <= 1e-9:
            continue

        x = Xc[:2, 0] / z
        K_i = cams[img_id].K if multicam else K
        uv = (K_i[:2, :2] @ x) + K_i[:2, 2]
        uv = uv.reshape(2)

        # observed
        kp_id = tr.obs[img_id]
        uv_obs = feats[img_id].kpts_xy[kp_id].astype(np.float64).reshape(2)

        e = float(np.linalg.norm(uv - uv_obs))
        if e < err_thresh_px:
            continue  # focus on bad ones (flip logic if you want)

        
        # draw
        p_obs = tuple(np.round(uv_obs).astype(int))
        p_prj = tuple(np.round(uv).astype(int))

        circle(img, p_obs, 2, (0, 255, 0), -1)   # green observed
        circle(img, p_prj, 2, (0, 0, 255), -1)   # red projected
        line(img, p_obs, p_prj, (0, 255, 255), 1)  # yellow line

        putText(
            img,
            f"{e:.1f}px",
            p_prj,
            FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            LINE_AA,
        )

        drawn += 1
        if drawn >= max_draw:
            break

    out_path = os.path.join(out_dir, f"img_{img_id:02d}_step{step:03d}.png")
    imwrite(out_path, img)