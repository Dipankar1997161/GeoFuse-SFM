# src/geometry.py
"""
Public geometry API.

Internals live in src/geometry_utils/.
Import from here in the rest of the codebase to avoid deep-path imports.
"""

from src.geometry_utils.epipolar import compute_fundamental_matrix, epipolar_distance
from src.geometry_utils.projective import projection_matrix, camera_center
from src.geometry_utils.reprojection import project_points, reprojection_errors
from src.geometry_utils.triangulation import (
    triangulate_points,
    triangulate_and_filter,
    triangulate_and_filter_debug,
    triangulation_angles_deg,
    cheirality_mask,
)
from src.geometry_utils.twoview import recover_pose_from_points, TwoViewResult

__all__ = [
    "compute_fundamental_matrix",
    "projection_matrix",
    "camera_center",
    "project_points",
    "reprojection_errors",
    "triangulate_points",
    "triangulate_and_filter",
    "triangulate_and_filter_debug",
    "triangulation_angles_deg",
    "cheirality_mask",
    "recover_pose_from_points",
    "TwoViewResult",
]
