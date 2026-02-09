from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def write_ply(
    path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Write a point cloud to an ASCII PLY file.

    Args:
        path: output file path
        points: (N, 3) float array
        colors: (N, 3) uint8 array in RGB [0,255], optional
    """
    path = Path(path)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")

    has_color = colors is not None
    if has_color:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != (points.shape[0], 3):
            raise ValueError(f"colors must have shape (N,3), got {colors.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if has_color:
            assert colors is not None
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


def read_ply(
    path: Union[str, Path],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read a (simple) ASCII PLY point cloud.

    Supports:
      - vertices with x,y,z
      - optional uchar red,green,blue

    Returns:
      points: (N,3) float64
      colors: (N,3) uint8 in RGB, or None if not present
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # --- parse header ---
        line = f.readline().strip()
        if line != "ply":
            raise ValueError("Not a PLY file (missing 'ply' header).")

        fmt = None
        vertex_count = None
        props = []
        in_vertex_element = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header.")
            line = line.strip()

            if line.startswith("format "):
                fmt = line.split()
                if len(fmt) < 3:
                    raise ValueError(f"Bad format line: {line}")
                # e.g. format ascii 1.0
                if fmt[1] != "ascii":
                    raise ValueError(f"Only ASCII PLY supported, got: {fmt[1]}")

            elif line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Bad element line: {line}")
                name = parts[1]
                count = int(parts[2])
                in_vertex_element = (name == "vertex")
                if in_vertex_element:
                    vertex_count = count
                    props = []

            elif line.startswith("property ") and in_vertex_element:
                # property <type> <name>
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Bad property line: {line}")
                props.append(parts[2])

            elif line == "end_header":
                break

            # ignore comments and other elements

        if fmt is None:
            raise ValueError("Missing PLY format line.")
        if vertex_count is None:
            raise ValueError("PLY file has no vertex element.")

        # Determine indices of required/optional fields
        try:
            ix = props.index("x")
            iy = props.index("y")
            iz = props.index("z")
        except ValueError as e:
            raise ValueError(f"PLY vertex properties must include x,y,z. Found: {props}") from e

        has_rgb = all(p in props for p in ("red", "green", "blue"))
        if has_rgb:
            ir = props.index("red")
            ig = props.index("green")
            ib = props.index("blue")

        # --- read vertex data ---
        points = np.empty((vertex_count, 3), dtype=np.float64)
        colors = np.empty((vertex_count, 3), dtype=np.uint8) if has_rgb else None

        for i in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY vertices.")
            parts = line.strip().split()
            if len(parts) < len(props):
                raise ValueError(f"Vertex line has too few fields: {line}")

            points[i, 0] = float(parts[ix])
            points[i, 1] = float(parts[iy])
            points[i, 2] = float(parts[iz])

            if has_rgb and colors is not None:
                colors[i, 0] = int(float(parts[ir]))
                colors[i, 1] = int(float(parts[ig]))
                colors[i, 2] = int(float(parts[ib]))

        return points, colors
