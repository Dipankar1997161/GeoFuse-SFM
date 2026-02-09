from data_io.camera import DecomposedCamera
import numpy as np

def compute_fundamental_matrix(
    cam_i: DecomposedCamera,
    cam_j: DecomposedCamera,
) -> np.ndarray:
    """Compute the fundamental matrix F such that p2^T F p1 = 0
    for corresponding points p1 in image i and p2 in image j.
    """

    Ki, Ri, ti = cam_i.K, cam_i.R, cam_i.t
    Kj, Rj, tj = cam_j.K, cam_j.R, cam_j.t

    Ki = np.asarray(Ki, np.float64)
    Kj = np.asarray(Kj, np.float64)
    Ri = np.asarray(Ri, np.float64)
    Rj = np.asarray(Rj, np.float64)
    ti = np.asarray(ti, np.float64).reshape(3, 1)
    tj = np.asarray(tj, np.float64).reshape(3, 1)

    R_rel = Rj @ Ri.T
    t_rel = tj - R_rel @ ti

    tx = np.array([
        [0, -t_rel[2,0], t_rel[1,0]],
        [t_rel[2,0], 0, -t_rel[0,0]],
        [-t_rel[1,0], t_rel[0,0], 0]
    ], dtype=np.float64)

    E = tx @ R_rel
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    F /= (np.linalg.norm(F) + 1e-12)

    return F

def epipolar_distance(
    p1: np.ndarray,  # (2,) point in image 1
    p2: np.ndarray,  # (2,) point in image 2
    F: np.ndarray,   # (3,3) fundamental matrix
) -> float:
    """
    Compute symmetric epipolar distance.
    
    Returns average of:
    - Distance from p2 to epipolar line of p1
    - Distance from p1 to epipolar line of p2
    """
    p1_h = np.array([p1[0], p1[1], 1.0])
    p2_h = np.array([p2[0], p2[1], 1.0])
    
    # Line in image 2 from p1
    l2 = F @ p1_h
    d2 = abs(p2_h @ l2) / (np.sqrt(l2[0]**2 + l2[1]**2) + 1e-12)
    
    # Line in image 1 from p2
    l1 = F.T @ p2_h
    d1 = abs(p1_h @ l1) / (np.sqrt(l1[0]**2 + l1[1]**2) + 1e-12)
    
    return (d1 + d2) / 2