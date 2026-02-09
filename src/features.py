from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
from src.geometry_utils.projective import projection_matrix, camera_center

from cv2 import (KeyPoint, DMatch, FlannBasedMatcher, BFMatcher, 
                 NORM_HAMMING, COLOR_BGR2GRAY, RANSAC, 
                 SIFT_create, ORB_create, triangulatePoints, findEssentialMat, cvtColor)
import numpy as np

MatcherType = Literal["sift", "orb"]


@dataclass
class Features:
    kpts_xy: np.ndarray   # (N,2) float32
    desc: np.ndarray      # (N,D) float32 (SIFT) or uint8 (ORB)


def _to_xy(kps: List[KeyPoint]) -> np.ndarray:
    return np.array([kp.pt for kp in kps], dtype=np.float32)


def detect_and_describe(image_bgr: np.ndarray, method: MatcherType = "sift", mask: np.ndarray | None = None) -> Features:
    """
    Detect keypoints + descriptors.
    SIFT gives strong SfM baseline. ORB is faster but less stable.
    """
    gray = cvtColor(image_bgr, COLOR_BGR2GRAY)

    if method == "sift":
        # More keypoints -> more potential tracks/points
        det = SIFT_create(nfeatures=150000,
                              contrastThreshold=0.004,  # default ~0.04, this is huge change
                              edgeThreshold=10,
                              sigma=1.2)
        kps, desc = det.detectAndCompute(gray, mask)
        if desc is None or len(kps) < 2:
            return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))

        if desc is None or len(kps) == 0:
            return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
        return Features(_to_xy(kps), desc.astype(np.float32))

    if method == "orb":
        det = ORB_create(nfeatures=51000)
        kps, desc = det.detectAndCompute(gray, mask)
        if desc is None or len(kps) == 0:
            return Features(np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8))
        return Features(_to_xy(kps), desc)

    raise ValueError(f"Unknown method: {method}")


def match_descriptors(
    f1: Features,
    f2: Features,
    method: MatcherType = "sift",
    ratio: float = 0.8,
    mutual: bool = False,
) -> List[Tuple[int, int]]:

    # existing guard
    if f1.desc is None or f2.desc is None or len(f1.kpts_xy) == 0 or len(f2.kpts_xy) == 0:
        return []

    # ---- NEW: KNN(k=2) safety guards ----
    # (kpts_xy can be non-empty while desc has only 0/1 rows)
    if len(f1.desc) < 2 or len(f2.desc) < 2:
        return []

    # ---- NEW: ensure dtype for FLANN (SIFT expects float32) ----
    if method == "sift":
        d1 = f1.desc.astype(np.float32, copy=False)
        d2 = f2.desc.astype(np.float32, copy=False)
    else:
        d1 = f1.desc
        d2 = f2.desc

    # ---- 1->2 KNN + ratio test (keep distances for uniqueness) ----
    m12_scored: List[Tuple[int, int, float]] = []
    if method == "sift":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        flann = FlannBasedMatcher(index_params, search_params)
        knn12 = flann.knnMatch(d1, d2, k=2)
        for m_n in knn12:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                m12_scored.append((m.queryIdx, m.trainIdx, float(m.distance)))
    else:
        bf = BFMatcher(NORM_HAMMING, crossCheck=False)
        knn12 = bf.knnMatch(d1, d2, k=2)
        for m_n in knn12:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                m12_scored.append((m.queryIdx, m.trainIdx, float(m.distance)))

    m12 = make_unique_matches_by_distance(m12_scored)
    if not mutual:
        return m12

    # ---- 2->1 KNN + ratio test ----
    # NEW: guard again because reverse direction also needs >=2 in both
    if len(d2) < 2 or len(d1) < 2:
        return []

    m21_scored: List[Tuple[int, int, float]] = []
    if method == "sift":
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        flann = FlannBasedMatcher(index_params, search_params)
        knn21 = flann.knnMatch(d2, d1, k=2)
        for m_n in knn21:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                m21_scored.append((m.queryIdx, m.trainIdx, float(m.distance)))
    else:
        bf = BFMatcher(NORM_HAMMING, crossCheck=False)
        knn21 = bf.knnMatch(d2, d1, k=2)
        for m_n in knn21:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                m21_scored.append((m.queryIdx, m.trainIdx, float(m.distance)))

    m21 = make_unique_matches_by_distance(m21_scored)

    m21_set = set((j, i) for (i, j) in m21)
    mutual_matches = [p for p in m12 if p in m21_set]
    return mutual_matches



def make_unique_matches(matches: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Enforce one-to-one mapping: each i and each j can appear at most once.
    Keeps the first occurrence (ordering matters).
    """
    used_i = set()
    used_j = set()
    out: List[Tuple[int, int]] = []
    for i, j in matches:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out

def make_unique_matches_by_distance(
    matches: List[Tuple[int, int, float]]
) -> List[Tuple[int, int]]:
    """
    Enforce one-to-one mapping by keeping the lowest-distance matches first.
    Input: (i, j, dist)
    Output: (i, j)
    """
    matches_sorted = sorted(matches, key=lambda x: x[2])  # smallest distance first
    used_i = set()
    used_j = set()
    out: List[Tuple[int, int]] = []
    for i, j, d in matches_sorted:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return out


def triangulation_angle_deg(X: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Compute the triangulation angle (in degrees) for a 3D point X observed from two camera centers C1 and C2.
    Returns 0 if either camera center coincides with the point (to avoid NaNs).
    """
    
    v1 = C1 - X
    v2 = C2 - X
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cosang = float(np.dot(v1, v2) / (n1*n2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def verify_matches_essential(
    f1: Features,
    f2: Features,
    matches: List[Tuple[int, int]],
    K: np.ndarray,
    ransac_thresh_px: float = 2.0,
    prob: float = 0.999,
) -> Tuple[List[Tuple[int, int]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Geometric verification using Essential matrix RANSAC.

    Returns:
      inlier_matches: subset of matches
      E: (3,3) or None
      inlier_mask: (M,) uint8 mask aligned with input matches, or None
    """
    if len(matches) < 8:
        return [], None, None

    pts1 = np.float32([f1.kpts_xy[i] for i, _ in matches])
    pts2 = np.float32([f2.kpts_xy[j] for _, j in matches])

    E, mask = findEssentialMat(
        pts1, pts2, K,
        method=RANSAC,
        prob=prob,
        threshold=ransac_thresh_px,
    )
    if E is None or mask is None:
        return [], None, None

    mask = mask.ravel().astype(np.uint8)
    inlier_matches = [m for m, inl in zip(matches, mask) if inl]
    return inlier_matches, E, mask
