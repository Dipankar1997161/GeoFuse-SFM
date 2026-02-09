from attr import dataclass
import numpy as np
from typing import List, Optional, Tuple, Dict, Literal
from cv2 import findEssentialMat, RANSAC, recoverPose


@dataclass
class TwoViewResult:
    R: np.ndarray                 # (3,3)
    t: np.ndarray                 # (3,1) up-to-scale
    inlier_mask: Optional[np.ndarray]  # (N,) uint8 or None



def recover_pose_from_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    E: Optional[np.ndarray] = None,
) -> TwoViewResult:
    """
    Recover relative pose between two views.

    Args:
      pts1, pts2: (N,2) float32 pixel coordinates (inlier-filtered recommended)
      K: (3,3)
      E: optional Essential matrix. If None, we compute it.

    Returns:
      R, t (up to scale), and inlier mask used by recoverPose.
    """
    if pts1.shape[0] < 8:
        raise ValueError("Need at least 8 correspondences to recover pose.")

    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)

    if E is None:
        E, mask_E = findEssentialMat(
            pts1, pts2, K,
            method=RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            raise ValueError("findEssentialMat failed.")
    else:
        mask_E = None

    # recoverPose returns a mask of inliers that satisfy cheirality for the chosen (R,t)
    _, R, t, mask_pose = recoverPose(E, pts1, pts2, K)

    return TwoViewResult(R=R, t=t, inlier_mask=mask_pose.ravel() if mask_pose is not None else mask_E)
