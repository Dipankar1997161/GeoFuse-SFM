from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from .parsing import extract_floats, load_data


@dataclass(frozen=True)
class CameraIntrinsics:
    K: np.ndarray                    # (3,3)
    dist: Optional[np.ndarray] = None  # (N,)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


@dataclass(frozen=True)
class DecomposedCamera:
    K: np.ndarray  # (3,3)
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,1)
    C: np.ndarray  # (3,1)


# -------------------------
# Public, format-agnostic API
# -------------------------

def read_intrinsics(path: Union[str, Path]) -> np.ndarray:
    """
    Read a 3x3 intrinsic matrix K from .txt/.json/.yaml.
    """
    obj = load_data(path)
    K = _k_from_obj(obj)
    _validate_K(K)
    return K


def read_camera(path: Union[str, Path], *, read_dist: bool = True) -> CameraIntrinsics:
    """
    Read camera intrinsics K and optional distortion from .txt/.json/.yaml.
    """
    obj = load_data(path)
    K = _k_from_obj(obj)
    _validate_K(K)

    dist = _dist_from_obj(obj) if read_dist else None
    return CameraIntrinsics(K=K, dist=dist)


def read_projection_matrix(path: Union[str, Path]) -> np.ndarray:
    """
    Read a 3x4 projection matrix P from .txt/.json/.yaml.
    """
    obj = load_data(path)
    P = _p_from_obj(obj)
    if P.shape != (3, 4):
        raise ValueError(f"P must be 3x4, got {P.shape} from {path}")
    if not np.isfinite(P).all():
        raise ValueError("P contains non-finite values.")
    return P


def undistort_points(pts_xy: np.ndarray, cam: CameraIntrinsics) -> np.ndarray:
    """
    Undistort 2D points using OpenCV, returning pixel coordinates in same frame as K.
    """
    if cam.dist is None:
        return pts_xy

    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, cam.K, cam.dist, P=cam.K)
    return und.reshape(-1, 2).astype(np.float32)


def decompose_projection_matrix(P: np.ndarray) -> DecomposedCamera:
    """
    Decomposes P into K, R, t (x_cam = R X + t) and camera center C in world coords.
    """
    K, R, Ch, *_ = cv2.decomposeProjectionMatrix(P)

    # normalize K
    K = K / K[2, 2]

    # camera center in world coords
    C = (Ch[:3] / Ch[3]).reshape(3, 1)

    # translation for [R|t]
    t = -R @ C

    # enforce det(R)=+1
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    return DecomposedCamera(K=K, R=R, t=t, C=C)


# -------------------------
# Domain decoding helpers (private)
# -------------------------

def _k_from_obj(obj: Any) -> np.ndarray:
    """
    Extract K from:
      - raw text: expects >=9 floats
      - dict: supports {"K": ...} or {"intrinsics": {"K": ...}} or {fx,fy,cx,cy}
    """
    if isinstance(obj, str):
        vals = extract_floats(obj)
        if len(vals) < 9:
            raise ValueError(f"Expected >=9 numbers for K, got {len(vals)}")
        return np.array(vals[:9], dtype=np.float64).reshape(3, 3)

    if isinstance(obj, Mapping):
        # 1) direct K
        if "K" in obj:
            return _as_3x3(obj["K"])

        # 2) nested intrinsics
        intr = obj.get("intrinsics")
        if isinstance(intr, Mapping) and "K" in intr:
            return _as_3x3(intr["K"])

        # 3) fx/fy/cx/cy form
        if all(k in obj for k in ("fx", "fy", "cx", "cy")):
            fx = float(obj["fx"]); fy = float(obj["fy"])
            cx = float(obj["cx"]); cy = float(obj["cy"])
            return np.array([[fx, 0.0, cx],
                             [0.0, fy, cy],
                             [0.0, 0.0, 1.0]], dtype=np.float64)

        # sometimes intrinsics are under "camera" or similar
        cam = obj.get("camera")
        if isinstance(cam, Mapping):
            if "K" in cam:
                return _as_3x3(cam["K"])
            if all(k in cam for k in ("fx", "fy", "cx", "cy")):
                fx = float(cam["fx"]); fy = float(cam["fy"])
                cx = float(cam["cx"]); cy = float(cam["cy"])
                return np.array([[fx, 0.0, cx],
                                 [0.0, fy, cy],
                                 [0.0, 0.0, 1.0]], dtype=np.float64)

    raise ValueError("Could not extract K from provided data.")


def _dist_from_obj(obj: Any) -> Optional[np.ndarray]:
    """
    Extract distortion from:
      - raw text: numbers after first 9 are dist
      - dict: "dist" or intrinsics.dist or camera.dist
    """
    if isinstance(obj, str):
        vals = extract_floats(obj)
        if len(vals) <= 9:
            return None
        return np.array(vals[9:], dtype=np.float64).reshape(-1)

    if isinstance(obj, Mapping):
        if "dist" in obj and obj["dist"] is not None:
            return _as_1d(obj["dist"])

        intr = obj.get("intrinsics")
        if isinstance(intr, Mapping) and "dist" in intr and intr["dist"] is not None:
            return _as_1d(intr["dist"])

        cam = obj.get("camera")
        if isinstance(cam, Mapping) and "dist" in cam and cam["dist"] is not None:
            return _as_1d(cam["dist"])

    return None


def _p_from_obj(obj: Any) -> np.ndarray:
    """
    Extract P from:
      - raw text: expects >=12 floats -> first 12 make 3x4
      - dict: supports {"P": ...} or {"projection": ...} or nested camera.P
    """
    if isinstance(obj, str):
        vals = extract_floats(obj)
        if len(vals) < 12:
            raise ValueError(f"Expected >=12 numbers for P, got {len(vals)}")
        return np.array(vals[:12], dtype=np.float64).reshape(3, 4)

    if isinstance(obj, Mapping):
        if "P" in obj:
            return _as_3x4(obj["P"])
        if "projection" in obj:
            return _as_3x4(obj["projection"])

        cam = obj.get("camera")
        if isinstance(cam, Mapping):
            if "P" in cam:
                return _as_3x4(cam["P"])
            if "projection" in cam:
                return _as_3x4(cam["projection"])

    raise ValueError("Could not extract P from provided data.")


def _as_3x3(x: Any) -> np.ndarray:
    arr = np.array(x, dtype=np.float64)
    if arr.size != 9:
        raise ValueError(f"Expected 9 values for 3x3, got {arr.size}")
    return arr.reshape(3, 3)


def _as_3x4(x: Any) -> np.ndarray:
    arr = np.array(x, dtype=np.float64)
    if arr.size != 12:
        raise ValueError(f"Expected 12 values for 3x4, got {arr.size}")
    return arr.reshape(3, 4)


def _as_1d(x: Any) -> np.ndarray:
    arr = np.array(x, dtype=np.float64).reshape(-1)
    return arr


def _validate_K(K: np.ndarray) -> None:
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")

    if not np.isfinite(K).all():
        raise ValueError("K contains non-finite values.")

    if abs(K[2, 2] - 1.0) > 1e-6:
        raise ValueError(f"Expected K[2,2] ~ 1, got {K[2,2]}")

    fx, fy = K[0, 0], K[1, 1]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal lengths fx={fx}, fy={fy}")
