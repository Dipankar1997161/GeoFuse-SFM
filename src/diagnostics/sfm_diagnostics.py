"""sfm_diagnostics.py

Lightweight SfM diagnostics utilities:
- Keypoint density heatmaps
- Track coverage heatmaps
- Simple overlays + debug panels

Designed to be backend-agnostic: you can pass in
- keypoints as (N,2) arrays, list[cv2.KeyPoint], list[(x,y)], or dicts with 'pt'/'xy'
- tracks as iterable of track objects/dicts with per-image observations

No dependencies beyond numpy + opencv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import cv2


ArrayLikeXY = Union[np.ndarray, Sequence[Tuple[float, float]]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------
# Input normalization
# -------------------------

def _as_xy_array(keypoints: Any) -> np.ndarray:
    """Convert common keypoint containers to (N,2) float32 array in pixel coords."""
    if keypoints is None:
        return np.zeros((0, 2), dtype=np.float32)

    # Already ndarray
    if isinstance(keypoints, np.ndarray):
        kp = keypoints
        if kp.ndim == 2 and kp.shape[1] >= 2:
            return kp[:, :2].astype(np.float32, copy=False)
        if kp.ndim == 1 and kp.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

    # List-like
    if isinstance(keypoints, (list, tuple)):
        if len(keypoints) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        first = keypoints[0]

        # cv2.KeyPoint
        if hasattr(first, "pt"):
            pts = np.array([k.pt for k in keypoints], dtype=np.float32)
            return pts.reshape(-1, 2)

        # tuple/list (x,y)
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            return np.array([(float(p[0]), float(p[1])) for p in keypoints], dtype=np.float32)

        # dict with xy/pt
        if isinstance(first, dict):
            out = []
            for d in keypoints:
                if "xy" in d:
                    x, y = d["xy"]
                elif "pt" in d:
                    x, y = d["pt"]
                else:
                    continue
                out.append((float(x), float(y)))
            return np.array(out, dtype=np.float32).reshape(-1, 2)

    # Mapping that stores xy directly
    if isinstance(keypoints, dict):
        if "xy" in keypoints:
            return _as_xy_array(keypoints["xy"])

    # Fallback: nothing usable
    return np.zeros((0, 2), dtype=np.float32)


def _collect_track_obs_xy_for_view(
    tracks: Iterable[Any],
    view_id: int,
    *,
    min_track_len: int = 2,
) -> np.ndarray:
    """Return (M,2) xy array of all track observations in a given view.

    Supported track formats:
      - object with .obs or .observations mapping {view_id: (x,y)}
      - object with .image_ids/.xys aligned arrays
      - dict with 'obs' mapping or 'observations' mapping
      - dict with 'views' list and 'xys' list

    We only include tracks with >= min_track_len observations overall.
    """
    pts: List[Tuple[float, float]] = []

    for tr in tracks:
        # Track length
        tlen = None
        if hasattr(tr, "track_len"):
            try:
                tlen = int(tr.track_len)
            except Exception:
                tlen = None
        if tlen is None:
            # Try to infer
            obs_map = None
            if hasattr(tr, "obs"):
                obs_map = getattr(tr, "obs")
            elif hasattr(tr, "observations"):
                obs_map = getattr(tr, "observations")
            elif isinstance(tr, dict):
                obs_map = tr.get("obs", None) or tr.get("observations", None)

            if obs_map is not None and hasattr(obs_map, "__len__"):
                try:
                    tlen = int(len(obs_map))
                except Exception:
                    tlen = None

        if tlen is not None and tlen < int(min_track_len):
            continue

        # Case 1: mapping obs
        obs = None
        if hasattr(tr, "obs"):
            obs = getattr(tr, "obs")
        elif hasattr(tr, "observations"):
            obs = getattr(tr, "observations")
        elif isinstance(tr, dict):
            obs = tr.get("obs", None) or tr.get("observations", None)

        if isinstance(obs, Mapping):
            if view_id in obs:
                x, y = obs[view_id]
                pts.append((float(x), float(y)))
            continue

        # Case 2: aligned arrays: image_ids + xys
        if hasattr(tr, "image_ids") and hasattr(tr, "xys"):
            try:
                ids = np.asarray(getattr(tr, "image_ids"))
                xys = np.asarray(getattr(tr, "xys"))
                m = ids == int(view_id)
                if np.any(m):
                    sel = xys[m]
                    for x, y in sel[:, :2]:
                        pts.append((float(x), float(y)))
            except Exception:
                pass
            continue

        # Case 3: dict with views/xys lists
        if isinstance(tr, dict) and ("views" in tr and "xys" in tr):
            try:
                views = tr["views"]
                xys = tr["xys"]
                for vid, xy in zip(views, xys):
                    if int(vid) == int(view_id):
                        x, y = xy
                        pts.append((float(x), float(y)))
            except Exception:
                pass

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


# -------------------------
# Heatmaps
# -------------------------

def _accumulate_points_to_grid(
    H: int,
    W: int,
    xy: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.float32)
    if xy.size == 0:
        return grid

    x = np.round(xy[:, 0]).astype(np.int32)
    y = np.round(xy[:, 1]).astype(np.int32)

    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[m]
    y = y[m]

    # fast bincount into 2D
    if x.size:
        idx = y * W + x
        bc = np.bincount(idx, minlength=H * W).astype(np.float32)
        grid = bc.reshape(H, W)

    return grid


def density_heatmap(
    image_shape_hw: Tuple[int, int],
    points_xy: Any,
    *,
    sigma_px: float = 25.0,
    normalize: bool = True,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    """Create a smooth density heatmap from sparse points.

    Returns float32 (H,W) in [0,1] if normalize else raw.
    """
    H, W = int(image_shape_hw[0]), int(image_shape_hw[1])
    xy = _as_xy_array(points_xy)
    grid = _accumulate_points_to_grid(H, W, xy)

    # Smooth
    if sigma_px and sigma_px > 0:
        k = int(max(3, 2 * round(3 * float(sigma_px)) + 1))
        grid = cv2.GaussianBlur(grid, (k, k), float(sigma_px))

    if not normalize:
        return grid.astype(np.float32)

    hm = grid.astype(np.float32)
    if hm.max() <= 0:
        return np.zeros((H, W), dtype=np.float32)

    # Robust normalization
    hi = np.percentile(hm[hm > 0], float(clip_percentile)) if np.any(hm > 0) else hm.max()
    hi = float(max(hi, 1e-6))
    hm = np.clip(hm / hi, 0.0, 1.0)
    return hm


def keypoint_density_heatmap(
    image_shape_hw: Tuple[int, int],
    keypoints: Any,
    *,
    sigma_px: float = 25.0,
) -> np.ndarray:
    return density_heatmap(image_shape_hw, keypoints, sigma_px=sigma_px)


def track_coverage_heatmap(
    image_shape_hw: Tuple[int, int],
    track_points_xy: Any,
    *,
    sigma_px: float = 35.0,
) -> np.ndarray:
    return density_heatmap(image_shape_hw, track_points_xy, sigma_px=sigma_px)


# -------------------------
# Visualization
# -------------------------

def colorize_heatmap(hm01: np.ndarray, *, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    hm = np.clip(hm01, 0.0, 1.0)
    u8 = (hm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, colormap)


def overlay_heatmap(
    image_bgr: np.ndarray,
    hm01: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay heatmap onto image (expects BGR)."""
    if image_bgr.ndim == 2:
        base = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    else:
        base = image_bgr.copy()

    hmc = colorize_heatmap(hm01, colormap=colormap)
    out = cv2.addWeighted(base, 1.0 - float(alpha), hmc, float(alpha), 0)
    return out


def draw_points(
    image_bgr: np.ndarray,
    points_xy: Any,
    *,
    radius: int = 2,
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
    alpha: float = 1.0,
) -> np.ndarray:
    if image_bgr.ndim == 2:
        vis = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    else:
        vis = image_bgr.copy()

    pts = _as_xy_array(points_xy)
    if pts.size == 0:
        return vis

    overlay = vis.copy()
    H, W = vis.shape[:2]
    x = np.round(pts[:, 0]).astype(np.int32)
    y = np.round(pts[:, 1]).astype(np.int32)
    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[m]
    y = y[m]

    for xx, yy in zip(x, y):
        cv2.circle(overlay, (int(xx), int(yy)), int(radius), color_bgr, -1)

    if alpha >= 1.0:
        return overlay
    return cv2.addWeighted(vis, 1.0 - float(alpha), overlay, float(alpha), 0)


def make_panel_2x2(
    ref_bgr: np.ndarray,
    a_bgr: np.ndarray,
    b_bgr: np.ndarray,
    c_bgr: np.ndarray,
    *,
    pad: int = 6,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Combine 4 same-sized images into a 2x2 panel."""
    imgs = [ref_bgr, a_bgr, b_bgr, c_bgr]
    H, W = ref_bgr.shape[:2]
    for im in imgs:
        if im.shape[:2] != (H, W):
            raise ValueError("All panel images must have same size")

    pad_row = np.full((pad, 2 * W + pad, 3), pad_color, dtype=np.uint8)
    pad_col = np.full((H, pad, 3), pad_color, dtype=np.uint8)

    top = np.hstack([ref_bgr, pad_col, a_bgr])
    bot = np.hstack([b_bgr, pad_col, c_bgr])
    out = np.vstack([top, pad_row, bot])
    return out


@dataclass
class SfMDiagnosticsConfig:
    kp_sigma_px: float = 25.0
    track_sigma_px: float = 35.0
    min_track_len: int = 3
    heat_alpha: float = 0.45


def export_view_diagnostics(
    image_bgr: np.ndarray,
    *,
    view_id: int,
    keypoints: Any,
    tracks: Optional[Iterable[Any]] = None,
    out_dir: str,
    tag: str,
    cfg: Optional[SfMDiagnosticsConfig] = None,
) -> Dict[str, str]:
    """Export per-view diagnostics images.

    Writes:
      - {tag}_view_{id:03d}_kp_heat.png
      - {tag}_view_{id:03d}_trk_heat.png
      - {tag}_view_{id:03d}_panel.png

    Returns dict of written paths.
    """
    cfg = cfg or SfMDiagnosticsConfig()
    ensure_dir(out_dir)

    H, W = image_bgr.shape[:2]

    kp_xy = _as_xy_array(keypoints)
    kp_hm = keypoint_density_heatmap((H, W), kp_xy, sigma_px=cfg.kp_sigma_px)

    if tracks is None:
        trk_xy = np.zeros((0, 2), dtype=np.float32)
    else:
        trk_xy = _collect_track_obs_xy_for_view(tracks, int(view_id), min_track_len=int(cfg.min_track_len))

    trk_hm = track_coverage_heatmap((H, W), trk_xy, sigma_px=cfg.track_sigma_px)

    kp_overlay = overlay_heatmap(image_bgr, kp_hm, alpha=cfg.heat_alpha)
    trk_overlay = overlay_heatmap(image_bgr, trk_hm, alpha=cfg.heat_alpha)

    # point overlays (optional but useful)
    kp_pts = draw_points(image_bgr, kp_xy, radius=1, color_bgr=(0, 255, 255), alpha=1.0)
    trk_pts = draw_points(image_bgr, trk_xy, radius=1, color_bgr=(255, 255, 0), alpha=1.0)

    panel = make_panel_2x2(image_bgr, kp_overlay, trk_overlay, trk_pts)

    paths = {
        "kp_heat": os.path.join(out_dir, f"{tag}_view_{int(view_id):03d}_kp_heat.png"),
        "trk_heat": os.path.join(out_dir, f"{tag}_view_{int(view_id):03d}_trk_heat.png"),
        "panel": os.path.join(out_dir, f"{tag}_view_{int(view_id):03d}_panel.png"),
        "kp_pts": os.path.join(out_dir, f"{tag}_view_{int(view_id):03d}_kp_pts.png"),
        "trk_pts": os.path.join(out_dir, f"{tag}_view_{int(view_id):03d}_trk_pts.png"),
    }

    cv2.imwrite(paths["kp_heat"], kp_overlay)
    cv2.imwrite(paths["trk_heat"], trk_overlay)
    cv2.imwrite(paths["kp_pts"], kp_pts)
    cv2.imwrite(paths["trk_pts"], trk_pts)
    cv2.imwrite(paths["panel"], panel)

    return paths


def export_dataset_diagnostics(
    images: Mapping[int, np.ndarray],
    keypoints_by_view: Mapping[int, Any],
    *,
    tracks: Optional[Iterable[Any]] = None,
    out_dir: str,
    tag: str,
    cfg: Optional[SfMDiagnosticsConfig] = None,
    view_ids: Optional[Sequence[int]] = None,
) -> None:
    """Export diagnostics for multiple views.

    images: dict {view_id: image (BGR or gray)}
    keypoints_by_view: dict {view_id: keypoints}
    tracks: iterable of track objects/dicts (optional)
    """
    cfg = cfg or SfMDiagnosticsConfig()
    ensure_dir(out_dir)

    if view_ids is None:
        view_ids = sorted(images.keys())

    for vid in view_ids:
        if vid not in images:
            continue
        img = images[vid]
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        kps = keypoints_by_view.get(vid, None)
        export_view_diagnostics(
            img_bgr,
            view_id=int(vid),
            keypoints=kps,
            tracks=tracks,
            out_dir=out_dir,
            tag=tag,
            cfg=cfg,
        )


def export_sfm_diagnostics(
    *,
    images,
    feats,
    tracks,
    out_dir: str,
    stage: str,
    min_track_len: int = 3,
    kp_sigma_px: int = 18,
    tr_sigma_px: int = 18,
):
    """
    Adapter so reconstruction.py can call diagnostics without changing its data model.

    Expects:
      - feats[i].kpts_xy: (Ni,2)
      - tracks where tr.obs = {view_id: kp_id}
    """

    ensure_dir(out_dir)

    # ---- Build keypoints_by_view ----
    keypoints_by_view = {}
    for vid in range(len(feats)):
        k = getattr(feats[vid], "kpts_xy", None)
        if k is None:
            continue
        keypoints_by_view[vid] = np.asarray(k, dtype=np.float32)

    # ---- Convert tracks into real (x,y) observations ----
    converted_tracks = []

    for tr in tracks:
        obs = getattr(tr, "obs", None)
        if obs is None:
            continue

        if len(obs) < min_track_len:
            continue

        views = []
        xys = []

        for vid, kp_idx in obs.items():
            if vid not in keypoints_by_view:
                continue

            kpts = keypoints_by_view[vid]
            if kp_idx < 0 or kp_idx >= len(kpts):
                continue

            x, y = kpts[kp_idx]
            views.append(int(vid))
            xys.append((float(x), float(y)))

        if len(views) >= min_track_len:
            converted_tracks.append({
                "views": views,
                "xys": xys
            })

    # ---- Convert images list to dict if needed ----
    if isinstance(images, list):
        images_dict = {i: images[i] for i in range(len(images))}
    else:
        images_dict = images

    # ---- Export ----
    cfg = SfMDiagnosticsConfig(
        kp_sigma_px=float(kp_sigma_px),
        track_sigma_px=float(tr_sigma_px),
        min_track_len=min_track_len,
    )

    export_dataset_diagnostics(
        images=images_dict,
        keypoints_by_view=keypoints_by_view,
        tracks=converted_tracks,
        out_dir=out_dir,
        tag=stage,
        cfg=cfg,
    )
