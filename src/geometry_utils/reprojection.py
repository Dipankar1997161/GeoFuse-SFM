import numpy as np

# ----------------------------
# Small numeric helpers
# ----------------------------
_EPS_Z = 1e-12
_EPS_W = 1e-12

def _is_finite_xyz(X: np.ndarray) -> np.ndarray:
    """Return boolean mask of rows of X that are finite."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected (N,3) array, got {X.shape}")
    return np.isfinite(X).all(axis=1)



def project_points(
    X: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Project 3D points to pixel coordinates.

    Returns:
      x: (N,2) float64. If a point is non-finite or has invalid depth, its projection is NaN.
    """
    X = np.asarray(X, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"X must be (N,3). Got {X.shape}")

    x_pix = np.full((X.shape[0], 2), np.nan, dtype=np.float64)

    finite = _is_finite_xyz(X) & np.isfinite(R).all() & np.isfinite(t).all() & np.isfinite(K).all()
    if not np.any(finite):
        return x_pix

    Xf = X[finite]
    Xc = (R @ Xf.T) + t  # (3,Nf)
    Xc = np.nan_to_num(Xc, nan=np.nan, posinf=np.nan, neginf=np.nan)

    z = Xc[2, :]
    good_z = np.isfinite(z) & (z > _EPS_Z)

    if np.any(good_z):
        x_norm = Xc[:2, good_z] / z[good_z][None, :]
        x2 = (K[:2, :2] @ x_norm) + K[:2, 2:3]
        x_pix[np.where(finite)[0][good_z]] = x2.T

    return x_pix



def reprojection_errors(
    X: np.ndarray,
    pts_obs: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Pixel reprojection error per point.

    Returns:
      err: (N,) float64. Non-finite projections yield +inf error.
    """
    x_proj = project_points(X, K, R, t)          # (N,2) with NaNs for invalid
    pts_obs = np.asarray(pts_obs, dtype=np.float64)

    err = np.full((X.shape[0],), np.inf, dtype=np.float64)
    good = np.isfinite(x_proj).all(axis=1) & np.isfinite(pts_obs).all(axis=1)
    if np.any(good):
        err[good] = np.linalg.norm(x_proj[good] - pts_obs[good], axis=1)
    return err

