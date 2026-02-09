import numpy as np


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the 3x4 projection matrix P = K [R | t].
    Args:
        K: (3,3) intrinsic matrix
        R: (3,3) rotation matrix
        t: (3,) or (3,1) translation vector
    Returns:
        P: (3,4) projection matrix
    """

    K = np.asarray(K, np.float64)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).reshape(3, 1)

    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3,3), got {R.shape}")
    if t.shape != (3, 1):
        raise ValueError(f"t must be (3,1), got {t.shape}")

    return K @ np.hstack([R, t])  # 3x4


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute camera center in world coordinates from extrinsics R and t.
    Args:
        R: (3,3) rotation matrix
        t: (3,1) translation vector
    Returns:
        C: (3,) camera center in world coordinates"""   
    # world->cam: Xc = R X + t  => C = -R^T t
    return (-R.T @ t).reshape(3)