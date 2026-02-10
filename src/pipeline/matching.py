"""
src/pipeline/matching.py

Feature extraction and pairwise matching.
Handles both multicam (epipolar filter) and singlecam (E-RANSAC).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

from src.features import Features, detect_and_describe, match_descriptors, verify_matches_essential
from src.mulitcam_utils.registration import match_with_epipolar_filter
from src.mulitcam_utils.silhouette_matching import match_silhouettes_multicam, merge_sift_and_contour_matches
from src.visualization.reproject_overlays import save_kp_overlay

from .config import MatchingConfig, DiagnosticsConfig
from .state import SfMState


def _prep_mask(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Prepare silhouette mask for feature filtering.
    
    Assumes: object is black (0), background is white (255).
    Returns: mask where foreground (object) is 255, background is 0.
    """
    if mask is None:
        return None
    
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    
    h, w = img.shape[:2]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Invert: object (black) -> 255, background (white) -> 0
    m = (m < 128).astype(np.uint8) * 255
    
    # Dilate to keep boundary features
    m = cv2.dilate(m, np.ones((15, 15), np.uint8), iterations=1)
    
    return m


def _filter_features_by_mask(feat: Features, mask255: np.ndarray) -> Features:
    """Filter keypoints to keep only those inside the mask."""
    if feat.desc is None or len(feat.kpts_xy) == 0:
        return feat
    
    xy = feat.kpts_xy
    h, w = mask255.shape[:2]
    x = np.clip(xy[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(xy[:, 1].round().astype(int), 0, h - 1)
    keep = mask255[y, x] > 0
    
    return Features(
        xy[keep],
        feat.desc[keep] if feat.desc is not None else None
    )


def extract_features(
    state: SfMState,
    config: MatchingConfig,
    diag_config: DiagnosticsConfig,
    logger=None,
) -> None:
    """
    Extract SIFT features from all images.
    
    Optionally filters by silhouette mask.
    Updates state.feats and state.num_kp in place.
    """
    # Setup diagnostics directory
    kp_dir = None
    if diag_config.enabled and diag_config.out_dir and diag_config.save_kp_overlays:
        kp_dir = Path(diag_config.out_dir) / "diagnostics" / "kpts"
        kp_dir.mkdir(parents=True, exist_ok=True)
    
    feats: List[Features] = []
    
    for idx, img in enumerate(state.images):
        # Prepare mask
        mask = None
        if state.silhouettes is not None:
            mask = _prep_mask(state.silhouettes[idx], img)
            
            if logger:
                u, c = np.unique(mask, return_counts=True)
                logger.info(f"  img{idx}: mask unique={list(zip(u.tolist(), c.tolist()))[:6]}")
                logger.info(f"  img{idx}: mask mean={float(np.mean(mask)):.2f} fg_ratio={(mask>0).mean():.3f}")
        
        feature_config = config.feature

        # Detect features (on full image)
        f = detect_and_describe(img, method="sift", mask=None, feature_config=feature_config,)
        
        # Filter by silhouette
        if mask is not None:
            f = _filter_features_by_mask(f, mask)
        
        feats.append(f)
        
        if logger:
            logger.info(f"  img{idx}: kpts={len(f.kpts_xy)} mask={'yes' if mask is not None else 'no'}")
        
        # Save keypoint overlay
        if kp_dir is not None:
            save_kp_overlay(img, f, kp_dir / f"kpts_{idx:03d}.png", max_draw=3000)
    
    state.feats = feats
    state.num_kp = [len(f.kpts_xy) for f in feats]


def match_pairwise_multicam(
    state: SfMState,
    config: MatchingConfig,
    logger=None,
) -> None:
    """
    Pairwise matching for multicam setup.
    
    1. SIFT matching with epipolar filter (all pairs)
    2. Silhouette contour matching (if silhouettes available)
    3. Merge SIFT + contour matches
    
    Updates state.pairwise and state.pair_scores in place.
    May also update state.feats if contour points are added.
    """
    n = state.n_images
    
    # --- STEP 1: SIFT matching with epipolar filter ---
    sift_pairwise = {}
    
    if logger:
        logger.info("SIFT matching (epipolar filter)...")
    
    for i in range(n):
        for j in range(i + 1, n):
            # Get raw descriptor matches
            m = match_descriptors(
                state.feats[i], state.feats[j],
                method="sift",
                ratio=config.ratio,
                mutual=False
            )
            
            if len(m) < 4:
                continue
            
            # Use epipolar filter
            inliers = match_with_epipolar_filter(
                state.feats[i], state.feats[j],
                state.cams[i], state.cams[j],
                ratio=config.ratio,
                max_epipolar_dist=config.max_epipolar_dist,
                mutual=False,
            )
            
            if len(inliers) >= 4:
                sift_pairwise[(i, j)] = inliers
                if logger and len(inliers) >= 8:
                    logger.info(f"  SIFT ({i:2d},{j:2d}): raw={len(m)} epipolar={len(inliers)}")
    
    if logger:
        logger.info(f"[SIFT] Verified pairs: {len(sift_pairwise)}")
        if sift_pairwise:
            total = sum(len(v) for v in sift_pairwise.values())
            logger.info(f"[SIFT] Total matches: {total}")
    
    # --- STEP 2: Silhouette contour matching ---
    contour_pairwise = {}
    contour_pts = []
    
    if state.silhouettes is not None:
        if logger:
            logger.info("Silhouette contour matching...")
        
        contour_pairwise, contour_pts = match_silhouettes_multicam(
            state.silhouettes,
            state.cams,
            num_contour_points=config.num_contour_points,
            max_epipolar_dist=config.contour_max_epipolar_dist,
            min_matches=config.contour_min_matches,
            verbose=True,
        )
        
        if logger:
            logger.info(f"[Contour] Verified pairs: {len(contour_pairwise)}")
    
    # --- STEP 3: Merge SIFT + Contour ---
    if logger:
        logger.info("Merging SIFT + contour matches...")
    
    if contour_pts:
        merged_pairwise, merged_feats = merge_sift_and_contour_matches(
            sift_pairwise=sift_pairwise,
            contour_pairwise=contour_pairwise,
            sift_feats=state.feats,
            contour_points=contour_pts,
            proximity_threshold=config.proximity_threshold,
        )
        state.feats = merged_feats
        state.num_kp = [len(f.kpts_xy) for f in state.feats]
        state.pairwise = merged_pairwise
    else:
        state.pairwise = sift_pairwise
    
    state.pair_scores = {k: len(v) for k, v in state.pairwise.items()}
    
    # Log merged results
    total_inliers = sum(len(v) for v in state.pairwise.values())
    avg_inliers = total_inliers / max(len(state.pairwise), 1)
    if logger:
        logger.info(
            f"[MERGED] verified_pairs={len(state.pairwise)} "
            f"total_inliers={total_inliers} "
            f"avg_inliers_per_pair={avg_inliers:.1f}"
        )


def match_pairwise_singlecam(
    state: SfMState,
    config: MatchingConfig,
    logger=None,
) -> None:
    """
    Pairwise matching for singlecam setup using E-RANSAC.
    
    Updates state.pairwise and state.pair_scores in place.
    """
    n = state.n_images
    
    if logger:
        logger.info("Matching candidate pairs + E-RANSAC...")
    
    checked = 0
    kept = 0
    
    for i in range(n):
        checked += 1
        for j in range(i + 1, min(n, i + 1 + config.pair_window)):
            m = match_descriptors(
                state.feats[i], state.feats[j],
                method="sift",
                ratio=config.ratio,
                mutual=config.mutual
            )
            
            inliers, E, _ = verify_matches_essential(
                state.feats[i], state.feats[j], m,
                state.K_global,
                ransac_thresh_px=config.ransac_thresh_px
            )
            
            if len(inliers) < config.min_pair_inliers:
                continue
            
            state.pairwise[(i, j)] = inliers
            state.pair_scores[(i, j)] = len(inliers)
            kept += 1
            
            if logger:
                logger.info(f"  pair ({i},{j}): matches={len(m)} inliers={len(inliers)}")
    
    if logger:
        logger.info(f"Checked pairs: {checked}")
        logger.info(f"Verified pairs: {kept}")
    
    total_inliers = sum(len(v) for v in state.pairwise.values())
    avg_inliers = total_inliers / max(len(state.pairwise), 1)
    if logger:
        logger.info(
            f"[PAIRS] verified_pairs={len(state.pairwise)} "
            f"total_inliers={total_inliers} "
            f"avg_inliers_per_pair={avg_inliers:.1f}"
        )


def run_matching(
    state: SfMState,
    config: MatchingConfig,
    diag_config: DiagnosticsConfig,
    logger=None,
) -> None:
    """
    Main entry point for matching stage.
    
    1. Extract features
    2. Run pairwise matching (multicam or singlecam)
    
    Updates state in place.
    """
    # Feature extraction
    if logger:
        logger.info("Extracting features (SIFT)...")
    extract_features(state, config, diag_config, logger)
    
    # Pairwise matching
    if state.multicam:
        match_pairwise_multicam(state, config, logger)
    else:
        match_pairwise_singlecam(state, config, logger)
    
    # Validate
    if not state.pairwise:
        raise RuntimeError("No verified pairs found. Check data / thresholds.")
