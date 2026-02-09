import numpy as np
from typing import Optional, Tuple
from cv2 import triangulatePoints
from src.geometry_utils.projective import projection_matrix
from src.geometry_utils.reprojection import reprojection_errors, _is_finite_xyz

# ----------------------------
# Small numeric helpers
# ----------------------------
_EPS_Z = 1e-12
_EPS_W = 1e-12


def cheirality_mask(
    X: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Keep points with positive depth in the given camera coordinates.
    Camera coordinates: Xc = R*X + t
    """
    X = np.asarray(X, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    keep = np.zeros((X.shape[0],), dtype=bool)
    finite = _is_finite_xyz(X) & np.isfinite(R).all() & np.isfinite(t).all()
    if not np.any(finite):
        return keep

    Xc = (R @ X[finite].T) + t
    z = Xc[2, :]
    keep[np.where(finite)[0]] = np.isfinite(z) & (z > _EPS_Z)
    return keep


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate points from two views. No filtering here.

    Returns:
      X: (N,3) float64. Non-finite points are set to NaN (so downstream filters can drop them).
    """
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    if pts1.ndim != 2 or pts2.ndim != 2 or pts1.shape[1] != 2 or pts2.shape[1] != 2:
        raise ValueError(f"pts1/pts2 must be (N,2). Got {pts1.shape} and {pts2.shape}")

    P1 = projection_matrix(K, R1, t1)
    P2 = projection_matrix(K, R2, t2)

    # OpenCV expects 2xN
    x1 = pts1.T
    x2 = pts2.T

    X_h = triangulatePoints(P1, P2, x1, x2)  # (4,N)
    w = X_h[3]
    good_w = np.isfinite(w) & (np.abs(w) > _EPS_W)

    X = np.full((X_h.shape[1], 3), np.nan, dtype=np.float64)
    if np.any(good_w):
        X_good = (X_h[:3, good_w] / w[good_w]).T
        X[good_w] = X_good.astype(np.float64)

    return X

def triangulation_angles_deg(
    X: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """
    Compute triangulation angle between rays from camera centers to point X.

    Returns:
      angles_deg: (N,) with NaN for non-finite points.
    """
    X = np.asarray(X, dtype=np.float64)
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64).reshape(3, 1)
    t2 = np.asarray(t2, dtype=np.float64).reshape(3, 1)

    ang = np.full((X.shape[0],), np.nan, dtype=np.float64)

    finite = _is_finite_xyz(X) & np.isfinite(R1).all() & np.isfinite(R2).all() & np.isfinite(t1).all() & np.isfinite(t2).all()
    if not np.any(finite):
        return ang

    # Camera centers in world coordinates: C = -R^T t
    C1 = (-R1.T @ t1).reshape(3)
    C2 = (-R2.T @ t2).reshape(3)

    Xf = X[finite]
    v1 = Xf - C1[None, :]
    v2 = Xf - C2[None, :]

    v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12)

    cosang = np.sum(v1n * v2n, axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang_f = np.degrees(np.arccos(cosang))

    ang[np.where(finite)[0]] = ang_f
    return ang


def triangulate_and_filter(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    max_reproj_px: float = 2.0,
    min_triang_angle_deg: float = 1.0,
    # extra safety gates
    max_depth: Optional[float] = None,          # depth in each camera (Z) upper bound
    max_abs_coord: Optional[float] = None,      # bound |X|,|Y|,|Z| in world coords
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate and filter points using:
      - finite / numeric sanity
      - cheirality in both cameras
      - reprojection error in both cameras
      - triangulation angle threshold

    Returns:
      X_filt: (M,3)
      keep_mask: (N,) bool mask over original correspondences
    """
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    N = pts1.shape[0]
    if N == 0:
        return np.zeros((0, 3), np.float64), np.zeros((0,), dtype=bool)

    X = triangulate_points(pts1, pts2, K, R1, t1, R2, t2)  # (N,3) with NaNs

    keep = _is_finite_xyz(X)
    if max_abs_coord is not None and np.any(keep):
        m = np.max(np.abs(X[keep]), axis=1) <= float(max_abs_coord)
        idx = np.where(keep)[0]
        keep[idx] &= m

    # Cheirality in both cameras
    keep &= cheirality_mask(X, R1, t1)
    keep &= cheirality_mask(X, R2, t2)

    if not np.any(keep):
        return np.zeros((0, 3), np.float64), keep

    # Optional depth upper bound (helps kill insane far points early)
    if max_depth is not None:
        md = float(max_depth)
        Xc1 = (np.asarray(R1, np.float64) @ X[keep].T) + np.asarray(t1, np.float64).reshape(3, 1)
        Xc2 = (np.asarray(R2, np.float64) @ X[keep].T) + np.asarray(t2, np.float64).reshape(3, 1)
        z1 = Xc1[2, :]
        z2 = Xc2[2, :]
        ok = np.isfinite(z1) & np.isfinite(z2) & (z1 > _EPS_Z) & (z2 > _EPS_Z) & (z1 < md) & (z2 < md)
        idx = np.where(keep)[0]
        keep[idx] &= ok

        if not np.any(keep):
            return np.zeros((0, 3), np.float64), keep

    # Reprojection error filtering
    err1 = reprojection_errors(X, pts1, K, R1, t1)
    err2 = reprojection_errors(X, pts2, K, R2, t2)
    keep &= (err1 <= float(max_reproj_px)) & (err2 <= float(max_reproj_px))

    if not np.any(keep):
        return np.zeros((0, 3), np.float64), keep

    # Triangulation angle filtering (stability)
    ang = triangulation_angles_deg(X, R1, t1, R2, t2)
    keep &= np.isfinite(ang) & (ang >= float(min_triang_angle_deg))

    X_filt = X[keep]
    return X_filt.astype(np.float64), keep



REJ_FINITE = 1
REJ_CHEIRALITY_1 = 2
REJ_CHEIRALITY_2 = 3
REJ_REPROJ_1 = 4
REJ_REPROJ_2 = 5
REJ_ANGLE = 6
REJ_MAX_DEPTH = 7
REJ_MAX_ABS = 8

def triangulate_and_filter_debug(
    pts1, pts2, K, R1, t1, R2, t2,
    max_reproj_px: float = 2.0,
    min_triang_angle_deg: float = 1.0,
    max_depth: Optional[float] = None,
    max_abs_coord: Optional[float] = None,
):
    """
    Same as triangulate_and_filter but with detailed rejection reasons for diagnostics.
    Returns:  X_filt: (M,3) float64
    keep_mask: (N,) bool
    reason: (N,) int32 with codes:
        0: keep
        1: non-finite point
        2: cheirality fail cam 1
        3: cheirality fail cam 2
        4: reprojection fail cam 1
        5: reprojection fail cam 2
        6: triangulation angle fail
        7: max depth fail
        8: max abs coord fail
    """
    
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    N = pts1.shape[0]
    if N == 0:
        return np.zeros((0,3), np.float64), np.zeros((0,), bool), np.zeros((0,), np.int32)

    X = triangulate_points(pts1, pts2, K, R1, t1, R2, t2)  # (N,3) NaNs possible

    keep = _is_finite_xyz(X)
    reason = np.zeros((N,), dtype=np.int32)
    reason[~keep] = REJ_FINITE

    if max_abs_coord is not None and np.any(keep):
        m = np.max(np.abs(X[keep]), axis=1) <= float(max_abs_coord)
        idx = np.where(keep)[0]
        bad = ~m
        reason[idx[bad]] = REJ_MAX_ABS
        keep[idx] &= m

    # cheirality
    c1 = cheirality_mask(X, R1, t1)
    bad = keep & (~c1)
    reason[bad] = REJ_CHEIRALITY_1
    keep &= c1

    c2 = cheirality_mask(X, R2, t2)
    bad = keep & (~c2)
    reason[bad] = REJ_CHEIRALITY_2
    keep &= c2

    if max_depth is not None and np.any(keep):
        md = float(max_depth)
        Xc1 = (np.asarray(R1, np.float64) @ X[keep].T) + np.asarray(t1, np.float64).reshape(3, 1)
        Xc2 = (np.asarray(R2, np.float64) @ X[keep].T) + np.asarray(t2, np.float64).reshape(3, 1)
        z1 = Xc1[2, :]
        z2 = Xc2[2, :]
        ok = np.isfinite(z1) & np.isfinite(z2) & (z1 > _EPS_Z) & (z2 > _EPS_Z) & (z1 < md) & (z2 < md)
        idx = np.where(keep)[0]
        bad = ~ok
        reason[idx[bad]] = REJ_MAX_DEPTH
        keep[idx] &= ok

    # reproj
    if np.any(keep):
        err1 = reprojection_errors(X, pts1, K, R1, t1)
        err2 = reprojection_errors(X, pts2, K, R2, t2)

        bad1 = keep & (err1 > float(max_reproj_px))
        reason[bad1] = REJ_REPROJ_1
        keep &= (err1 <= float(max_reproj_px))

        bad2 = keep & (err2 > float(max_reproj_px))
        reason[bad2] = REJ_REPROJ_2
        keep &= (err2 <= float(max_reproj_px))

    # angle
    if np.any(keep):
        ang = triangulation_angles_deg(X, R1, t1, R2, t2)
        bad = keep & (~(np.isfinite(ang) & (ang >= float(min_triang_angle_deg))))
        reason[bad] = REJ_ANGLE
        keep &= np.isfinite(ang) & (ang >= float(min_triang_angle_deg))

    X_filt = X[keep]

    print("Triangulation stats:")
    print(f"  total points: {len(X)}")
    print(f"  kept points: {np.sum(keep)}")
    print(f"  rejected points: {len(X) - np.sum(keep)}")
    return X_filt.astype(np.float64), keep, reason