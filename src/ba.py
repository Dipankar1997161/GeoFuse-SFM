from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@dataclass
class BAObservation:
    cam_idx: int
    pt_idx: int
    xy: np.ndarray  # (2,)


def _project(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project points for one camera; robust to NaNs/Infs and invalid depths.

    Returns (N,2). Invalid projections are set to a very large value, which robust losses will down-weight.
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

    if not (np.isfinite(R).all() and np.isfinite(tvec).all() and np.isfinite(X).all() and np.isfinite(K).all()):
        return np.full((X.shape[0], 2), 1e6, dtype=np.float64)

    Xc = (R @ X.T) + tvec.reshape(3, 1)  # (3,N)
    # Keep things finite (avoid warnings/overflow); NaNs become huge later.
    Xc = np.nan_to_num(Xc, nan=np.nan, posinf=np.nan, neginf=np.nan)

    z = Xc[2, :]
    bad = (~np.isfinite(z)) | (z <= 1e-9)

    # Safe divide for the good subset only
    x_pix = np.full((2, X.shape[0]), 1e6, dtype=np.float64)
    good = ~bad
    if np.any(good):
        x = Xc[:2, good] / z[good][None, :]
        x2 = (K[:2, :2] @ x) + K[:2, 2:3]
        x2 = np.nan_to_num(x2, nan=1e6, posinf=1e6, neginf=-1e6)
        x_pix[:, good] = x2

    return x_pix.T  # (N,2)


def _build_jac_sparsity(M: int, C: int, P: int, obs_cam: np.ndarray, obs_pt: np.ndarray):
    """
    Build the sparsity pattern for the Jacobian of the bundle adjustment problem.
    M = number of observations
    C = number of cameras
    P = number of points
    obs_cam = (M,) array of camera indices for each observation
    obs_pt = (M,) array of point indices for each observation
    """

    J = lil_matrix((2 * M, 6 * C + 3 * P), dtype=np.int8)

    for k in range(M):
        ci = int(obs_cam[k])
        pi = int(obs_pt[k])
        r0 = 2 * k

        c0 = 6 * ci
        p0 = 6 * C + 3 * pi

        J[r0:r0 + 2, c0:c0 + 6] = 1
        J[r0:r0 + 2, p0:p0 + 3] = 1

    return J

def bundle_adjustment_fixed_camera(
    K,
    cam_order,
    cam_poses,
    X,
    observations,
    loss="huber",
    f_scale=1.0,
    max_nfev=200,
    verbose=False,
    fix_cameras=False,
):
    """
    Bundle adjustment with optional fixed cameras.
    
    Args:
        fix_cameras: If True, only optimize 3D points, not camera poses.
                     Use this for multicam rigs with known calibration.
    """
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    
    n_cams = len(cam_order)
    n_pts = X.shape[0]
    
    # Camera parameters: 6 per camera (3 rotation + 3 translation)
    # Point parameters: 3 per point
    
    if fix_cameras:
        # Only optimize points
        n_cam_params = 0
        x0 = X.ravel()  # Only points
    else:
        # Optimize both cameras and points
        n_cam_params = n_cams * 6
        
        # Pack initial camera parameters
        cam_params = []
        for img_id in cam_order:
            R, t = cam_poses[img_id]
            rvec, _ = cv2.Rodrigues(R)
            cam_params.extend(rvec.ravel())
            cam_params.extend(t.ravel())
        
        x0 = np.hstack([cam_params, X.ravel()])
    
    # Build observation arrays
    cam_indices = np.array([obs.cam_idx for obs in observations])
    pt_indices = np.array([obs.pt_idx for obs in observations])
    points_2d = np.array([obs.xy for obs in observations])
    
    def residuals(params):
        if fix_cameras:
            # Cameras are fixed, only points vary
            pts_3d = params.reshape(-1, 3)
            
            res = []
            for obs in observations:
                img_id = cam_order[obs.cam_idx]
                R, t = cam_poses[img_id]
                R = np.asarray(R, np.float64)
                t = np.asarray(t, np.float64).reshape(3, 1)
                
                X_pt = pts_3d[obs.pt_idx].reshape(3, 1)
                Xc = R @ X_pt + t
                
                if Xc[2, 0] <= 1e-6:
                    res.extend([1000.0, 1000.0])
                    continue
                
                proj = K @ Xc
                uv = proj[:2, 0] / proj[2, 0]
                
                res.append(uv[0] - obs.xy[0])
                res.append(uv[1] - obs.xy[1])
            
            return np.array(res)
        else:
            # Original: optimize both cameras and points
            cam_params = params[:n_cam_params].reshape(n_cams, 6)
            pts_3d = params[n_cam_params:].reshape(-1, 3)
            
            res = []
            for obs in observations:
                cp = cam_params[obs.cam_idx]
                rvec = cp[:3]
                tvec = cp[3:6]
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.reshape(3, 1)
                
                X_pt = pts_3d[obs.pt_idx].reshape(3, 1)
                Xc = R @ X_pt + t
                
                if Xc[2, 0] <= 1e-6:
                    res.extend([1000.0, 1000.0])
                    continue
                
                proj = K @ Xc
                uv = proj[:2, 0] / proj[2, 0]
                
                res.append(uv[0] - obs.xy[0])
                res.append(uv[1] - obs.xy[1])
            
            return np.array(res)
    
    # Build sparsity pattern
    n_obs = len(observations)
    n_params = len(x0)
    
    if fix_cameras:
        # Only point parameters
        A = lil_matrix((2 * n_obs, n_params), dtype=int)
        for i, obs in enumerate(observations):
            pt_start = obs.pt_idx * 3
            A[2*i, pt_start:pt_start+3] = 1
            A[2*i+1, pt_start:pt_start+3] = 1
    else:
        # Camera + point parameters
        A = lil_matrix((2 * n_obs, n_params), dtype=int)
        for i, obs in enumerate(observations):
            cam_start = obs.cam_idx * 6
            pt_start = n_cam_params + obs.pt_idx * 3
            A[2*i, cam_start:cam_start+6] = 1
            A[2*i, pt_start:pt_start+3] = 1
            A[2*i+1, cam_start:cam_start+6] = 1
            A[2*i+1, pt_start:pt_start+3] = 1
    
    result = least_squares(
        residuals, x0,
        jac_sparsity=A,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
    )
    
    # Unpack results
    if fix_cameras:
        # Cameras unchanged
        cam_refined = {img_id: cam_poses[img_id] for img_id in cam_order}
        X_refined = result.x.reshape(-1, 3)
    else:
        cam_params_opt = result.x[:n_cam_params].reshape(n_cams, 6)
        X_refined = result.x[n_cam_params:].reshape(-1, 3)
        
        cam_refined = {}
        for i, img_id in enumerate(cam_order):
            rvec = cam_params_opt[i, :3]
            tvec = cam_params_opt[i, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            cam_refined[img_id] = (R, t)
    
    return cam_refined, X_refined


def bundle_adjustment(
    K: np.ndarray,
    cam_order: List[int],
    cam_poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    observations: List[BAObservation],
    loss: str = "huber",
    f_scale: float = 3.0,
    max_nfev: int = 50,
    verbose: int = 0,
    # Optional bounds to keep BA from going numerically wild.
    # Leave as None to preserve original behavior.
    max_translation_abs: Optional[float] = None,
    max_point_abs: Optional[float] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Jointly refine camera poses + 3D points for a subset.

    Notes on the warnings you were seeing:
    - They usually come from NaNs/Infs in points/cameras (often produced by triangulation when w~0).
    - This implementation makes projection safe and (optionally) bounds translations/points to reduce solver blow-ups.
    """
    if not _HAS_SCIPY:
        return cam_poses, X

    K = np.asarray(K, dtype=np.float64)
    X0 = np.asarray(X, dtype=np.float64)

    C = len(cam_order)
    P = int(X0.shape[0])
    M = len(observations)

    if C == 0 or P == 0 or M == 0:
        return cam_poses, X

    # ---- Pack initial parameters ----
    x0 = np.zeros(C * 6 + P * 3, dtype=np.float64)

    for ci, img_id in enumerate(cam_order):
        R, t = cam_poses[img_id]
        rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
        x0[ci * 6: ci * 6 + 3] = rvec.reshape(3)
        x0[ci * 6 + 3: ci * 6 + 6] = np.asarray(t, dtype=np.float64).reshape(3)

    x0[C * 6:] = X0.reshape(-1)

    # ---- Observations arrays ----
    obs_cam = np.fromiter((o.cam_idx for o in observations), dtype=np.int64, count=M)
    obs_pt = np.fromiter((o.pt_idx for o in observations), dtype=np.int64, count=M)
    obs_xy = np.stack([o.xy for o in observations], axis=0).astype(np.float64)  # (M,2)

    cam_obs_idx: List[np.ndarray] = [np.where(obs_cam == ci)[0] for ci in range(C)]
    jac_sparsity = _build_jac_sparsity(M, C, P, obs_cam, obs_pt)

    def residuals(p: np.ndarray) -> np.ndarray:
        cam_params = p[:C * 6].reshape(C, 6)
        pts = p[C * 6:].reshape(P, 3)

        pred = np.empty((M, 2), dtype=np.float64)
        for ci in range(C):
            idxs = cam_obs_idx[ci]
            if idxs.size == 0:
                continue
            rvec = cam_params[ci, :3]
            tvec = cam_params[ci, 3:6]
            Xc = pts[obs_pt[idxs]]
            pred[idxs] = _project(K, rvec, tvec, Xc)

        return (pred - obs_xy).reshape(-1)

    # ---- Optional bounds (kept loose on purpose) ----
    bounds = (-np.inf, np.inf)
    if (max_translation_abs is not None) or (max_point_abs is not None):
        lo = np.full_like(x0, -np.inf)
        hi = np.full_like(x0,  np.inf)

        if max_translation_abs is not None:
            mt = float(max_translation_abs)
            for ci in range(C):
                lo[ci * 6 + 3: ci * 6 + 6] = -mt
                hi[ci * 6 + 3: ci * 6 + 6] =  mt

        if max_point_abs is not None:
            mp = float(max_point_abs)
            lo[C * 6:] = -mp
            hi[C * 6:] =  mp

        bounds = (lo, hi)

    res = least_squares(
        residuals,
        x0,
        jac_sparsity=jac_sparsity,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=verbose,
        x_scale="jac",
        bounds=bounds,
    )

    p = res.x
    cam_params = p[:C * 6].reshape(C, 6)
    pts = p[C * 6:].reshape(P, 3)

    refined: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for ci, img_id in enumerate(cam_order):
        rvec = cam_params[ci, :3].reshape(3, 1)
        tvec = cam_params[ci, 3:6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        refined[img_id] = (R, tvec)

    return refined, pts
