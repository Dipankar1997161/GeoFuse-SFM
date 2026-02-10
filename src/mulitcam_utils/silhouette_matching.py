"""
silhouette_matching.py - Match silhouette contour points for textureless objects

When SIFT fails on textureless surfaces, we can use silhouette contours.
The idea:
1. Extract contour points from each silhouette
2. Use epipolar geometry to find corresponding contour points
3. Add these as "synthetic" feature matches

This works because:
- Silhouette contours are visible from all views
- Epipolar constraint narrows down correspondences
- Even textureless objects have distinctive silhouette shapes
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from data_io.camera import DecomposedCamera
from src.geometry_utils.epipolar import compute_fundamental_matrix
from src.features import Features



def extract_contour_points_adaptive(silhouette, num_points=200, min_contour_length=50):
    """
    Extract contour points from silhouette, trying both polarities (object bright or dark).
    Returns evenly spaced points along the largest valid contour. If no valid contour is found, returns empty array.

    Args:   
    silhouette: Binary mask (object=255, background=0) or (object=0, background=255)
    num_points: How many points to sample along the contour (maximum)
    min_contour_length: Minimum length of contour to consider valid
    Returns: points: (N, 2) array of contour points
    """
    
    if silhouette.ndim == 3:
        silhouette = silhouette[:, :, 0]

    # try both polarities and pick the one whose largest contour is NOT the image border
    h, w = silhouette.shape[:2]

    def get_best(mask01):
        mask = (mask01.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        return c, area

    # polarity A: object bright
    a = get_best(silhouette > 127)
    # polarity B: object dark
    b = get_best(silhouette < 127)

    if a is None and b is None:
        return np.zeros((0, 2), np.float32)
    if a is None:
        contour = b[0]
    elif b is None:
        contour = a[0]
    else:
        # pick the one with the smaller largest-area contour (image-border tends to be huge)
        contour = a[0] if a[1] < b[1] else b[0]

    if len(contour) < min_contour_length:
        return np.zeros((0, 2), np.float32)

    contour = contour.reshape(-1, 2).astype(np.float32)

    if len(contour) > num_points:
        idx = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        contour = contour[idx]

    return contour


def extract_contour_points(
    silhouette: np.ndarray,
    num_points: int = 200,
    min_contour_length: int = 50,
) -> np.ndarray:
    """
    Extract evenly-spaced points along the silhouette contour.
    
    Args:
        silhouette: Binary mask (object=255, background=0)
        num_points: How many points to sample
        min_contour_length: Minimum contour length to consider
    
    Returns:
        points: (N, 2) array of contour points
    """
    # Ensure binary
    if silhouette.ndim == 3:
        silhouette = silhouette[:, :, 0]
    
    #mask = (silhouette > 127).astype(np.uint8) * 255

    mask = (silhouette < 127).astype(np.uint8) * 255

    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros((0, 2), np.float32)
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    if len(contour) < min_contour_length:
        return np.zeros((0, 2), np.float32)
    
    # Reshape to (N, 2)
    contour = contour.reshape(-1, 2).astype(np.float32)
    
    # Sample evenly spaced points
    if len(contour) > num_points:
        indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        contour = contour[indices]
    
    return contour


def compute_fundamental_from_cams(cam_i: DecomposedCamera, cam_j: DecomposedCamera) -> np.ndarray:
    """Compute F matrix from known cameras (DecomposedCamera)."""
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
        [0, -t_rel[2, 0], t_rel[1, 0]],
        [t_rel[2, 0], 0, -t_rel[0, 0]],
        [-t_rel[1, 0], t_rel[0, 0], 0]
    ], dtype=np.float64)

    E = tx @ R_rel
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    F = F / (np.linalg.norm(F) + 1e-12)
    return F



def match_contour_points_epipolar(
    pts_i: np.ndarray,  # (N, 2) contour points in image i
    pts_j: np.ndarray,  # (M, 2) contour points in image j
    F: np.ndarray,      # Fundamental matrix
    max_epipolar_dist: float = 3.0,
    unique: bool = True,
) -> List[Tuple[int, int]]:
    """
    Match contour points using epipolar constraint.
    
    For each point in image i, find the closest point in image j
    that lies near the epipolar line.
    """
    if len(pts_i) == 0 or len(pts_j) == 0:
        return []
    
    matches = []
    used_j = set()
    
    for idx_i, p_i in enumerate(pts_i):
        # Epipolar line in image j
        p_i_h = np.array([p_i[0], p_i[1], 1.0])
        l_j = F @ p_i_h  # Line coefficients [a, b, c] -> ax + by + c = 0
        
        # Normalize line
        norm = np.sqrt(l_j[0]**2 + l_j[1]**2) + 1e-12
        l_j = l_j / norm
        
        # Distance from each point in j to the epipolar line
        pts_j_h = np.hstack([pts_j, np.ones((len(pts_j), 1))])
        dists = np.abs(pts_j_h @ l_j)
        
        # Find points within threshold
        candidates = np.where(dists < max_epipolar_dist)[0]
        
        if len(candidates) == 0:
            continue
        
        # Among candidates, pick the closest to epipolar line
        best_j = candidates[np.argmin(dists[candidates])]
        
        if unique and best_j in used_j:
            continue
        
        matches.append((idx_i, best_j))
        used_j.add(best_j)
    
    return matches


def match_silhouettes_multicam(
    silhouettes: List[np.ndarray],
    cams: List[DecomposedCamera],
    num_contour_points: int = 150,
    max_epipolar_dist: float = 4.0,
    min_matches: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[Tuple[int,int], List[Tuple[int,int]]], List[np.ndarray]]:
    
    """
    Match silhouette contour points across all camera pairs.
    
    Returns:
        pairwise: Dict[(i,j), List[(idx_i, idx_j)]] of matches
        contour_points: List of contour point arrays per image
    """
    n = len(silhouettes)
    
    # Extract contour points
    if verbose:
        print(f"[Silhouette] Extracting contours from {n} images...")
    
    contour_points = []
    for i, sil in enumerate(silhouettes):
        pts = extract_contour_points_adaptive(sil, num_points=num_contour_points)
        contour_points.append(pts)
        if verbose:
            pass #print(f"  img{i:02d}: {len(pts)} contour points")
    
    # Match all pairs
    if verbose:
        print(f"[Silhouette] Matching contour points (epipolar)...")
    
    pairwise = {}

    def mutual_filter(matches_ij, matches_ji):
        # matches_ij: list of (a,b) where a in i, b in j
        # matches_ji: list of (b,a) where b in j, a in i
        set_ji = set((a, b) for (b, a) in matches_ji)  # invert to (a,b)
        return [m for m in matches_ij if m in set_ji]
    
    def monotonic_filter(matches, max_backtrack=5):
        """
        Keep matches that are roughly monotonic in contour index space.
        max_backtrack allows small local inversions.
        """
        if not matches:
            return matches

        matches = sorted(matches, key=lambda x: x[0])  # sort by idx in i
        kept = []
        last_b = -10**9
        for a, b in matches:
            if b >= last_b - max_backtrack:
                kept.append((a, b))
                last_b = max(last_b, b)
        return kept
    
    def triangulation_filter(
        matches,
        pts_i,
        pts_j,
        cam_i: DecomposedCamera,
        cam_j: DecomposedCamera,
        max_reproj_px: float = 3.0,
    ):
        Ki, Ri, ti = cam_i.K, cam_i.R, cam_i.t
        Kj, Rj, tj = cam_j.K, cam_j.R, cam_j.t

        Ki = np.asarray(Ki, np.float64)
        Kj = np.asarray(Kj, np.float64)
        Ri = np.asarray(Ri, np.float64)
        Rj = np.asarray(Rj, np.float64)
        ti = np.asarray(ti, np.float64).reshape(3, 1)
        tj = np.asarray(tj, np.float64).reshape(3, 1)

        Pi = Ki @ np.hstack([Ri, ti])  # 3x4
        Pj = Kj @ np.hstack([Rj, tj])  # 3x4

        def reproj(P, X):
            x = P @ X
            x = x[:2] / (x[2] + 1e-12)
            return x

        kept = []
        for a, b in matches:
            u1, v1 = pts_i[a]
            u2, v2 = pts_j[b]

            A = np.array([
                u1 * Pi[2] - Pi[0],
                v1 * Pi[2] - Pi[1],
                u2 * Pj[2] - Pj[0],
                v2 * Pj[2] - Pj[1],
            ], dtype=np.float64)

            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            if abs(X[3]) < 1e-12:
                continue
            X = X / X[3]

            p1 = reproj(Pi, X)
            p2 = reproj(Pj, X)

            e1 = np.linalg.norm(p1 - np.array([u1, v1]))
            e2 = np.linalg.norm(p2 - np.array([u2, v2]))

            if e1 <= max_reproj_px and e2 <= max_reproj_px:
                kept.append((a, b))

        return kept

    
    def mutual_filter_tol(matches_ij, matches_ji, tol_i=1, tol_j=1):
        # matches_ji are (b,a)
        back = {}
        for b,a in matches_ji:
            back.setdefault(b, []).append(a)

        kept = []
        for a,b in matches_ij:
            if b not in back:
                continue
            # accept if any reverse-a is close
            if any(abs(a - a2) <= tol_i for a2 in back[b]):
                kept.append((a,b))
        return kept

    
    for i in range(n):
        for j in range(i + 1, n):
            if len(contour_points[i]) == 0 or len(contour_points[j]) == 0:
                continue
            
            # F_ij = compute_fundamental_matrix(cams[i], cams[j])
            # F_ji = F_ij.T  # because your function expects l = F @ p

            F_ij = compute_fundamental_matrix(cams[i], cams[j])
            F_ji = compute_fundamental_matrix(cams[j], cams[i])
            
            # matches = match_contour_points_epipolar(
            #     contour_points[i],
            #     contour_points[j],
            #     F,
            #     max_epipolar_dist=max_epipolar_dist,
            # )

            matches_ij = match_contour_points_epipolar(contour_points[i], contour_points[j], F_ij, max_epipolar_dist=max_epipolar_dist, unique=True)
            matches_ji = match_contour_points_epipolar(contour_points[j], contour_points[i], F_ji, max_epipolar_dist=max_epipolar_dist, unique=True)

            matches = mutual_filter(matches_ij, matches_ji)
            matches = monotonic_filter(matches, max_backtrack=5)
            matches = triangulation_filter(matches, contour_points[i], contour_points[j], cams[i], cams[j], max_reproj_px=3.0)
            
            if len(matches) >= min_matches:
                pairwise[(i, j)] = matches
                if verbose:
                    pass #print(f"    ({i:2d},{j:2d}): {len(matches)} contour matches")
    
    if verbose:
        print(f"[Silhouette] Valid pairs: {len(pairwise)}")
    
    return pairwise, contour_points


def create_synthetic_features_from_contours(
    contour_points: List[np.ndarray],
    images: List[np.ndarray],
) -> List['Features']:
    """
    Create Features objects from contour points.
    Uses image patches as descriptors (optional, can be None).
    """
    from src.features import Features
    
    feats = []
    for i, (pts, img) in enumerate(zip(contour_points, images)):
        if len(pts) == 0:
            feats.append(Features(np.zeros((0, 2), np.float32), None))
            continue
        
        # For now, just use positions without descriptors
        # The epipolar matching doesn't need descriptors
        feats.append(Features(pts.astype(np.float32), None))
    
    return feats


def merge_sift_and_contour_matches(
    sift_pairwise: dict,
    contour_pairwise: dict,
    sift_feats: List,
    contour_points: List[np.ndarray],
    proximity_threshold: float = 5.0,
) -> Tuple[dict, List]:
    """
    Merge SIFT matches with contour matches.
    
    For each image, we combine:
    - Original SIFT keypoints
    - Contour points (appended after SIFT keypoints)
    
    Returns:
        merged_pairwise: Combined matches (with adjusted indices for contour points)
        merged_feats: Features with both SIFT and contour points
    """
    
    n = len(sift_feats)
    
    # Create merged features
    merged_feats = []
    contour_offset = []  # Starting index of contour points in merged features
    
    for i in range(n):
        sift_pts = sift_feats[i].kpts_xy
        cont_pts = contour_points[i] if i < len(contour_points) else np.zeros((0, 2))
        
        offset = len(sift_pts)
        contour_offset.append(offset)
        
        if len(cont_pts) > 0:
            merged_pts = np.vstack([sift_pts, cont_pts.astype(np.float32)])
        else:
            merged_pts = sift_pts
        
        # Descriptors: keep SIFT descriptors, None for contour points
        if sift_feats[i].desc is not None and len(cont_pts) > 0:
            # Pad with zeros for contour points (won't be used for matching)
            desc_dim = sift_feats[i].desc.shape[1]
            merged_desc = np.vstack([
                sift_feats[i].desc,
                np.zeros((len(cont_pts), desc_dim), dtype=sift_feats[i].desc.dtype)
            ])
        else:
            merged_desc = sift_feats[i].desc
        
        merged_feats.append(Features(merged_pts, merged_desc))
    
    # Merge pairwise matches
    merged_pairwise = dict(sift_pairwise)  # Start with SIFT matches
    
    for (i, j), contour_matches in contour_pairwise.items():
        # Adjust indices for contour points
        adjusted = [
            (a + contour_offset[i], b + contour_offset[j])
            for a, b in contour_matches
        ]
        
        if (i, j) in merged_pairwise:
            # Check for duplicates (SIFT match near contour match)
            existing = set(merged_pairwise[(i, j)])
            for match in adjusted:
                # Simple duplicate check by proximity
                is_duplicate = False
                for ex_a, ex_b in existing:
                    if ex_a < contour_offset[i] and ex_b < contour_offset[j]:
                        # Compare positions
                        pos_a = merged_feats[i].kpts_xy[match[0]]
                        pos_ex_a = merged_feats[i].kpts_xy[ex_a]
                        if np.linalg.norm(pos_a - pos_ex_a) < proximity_threshold:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    merged_pairwise[(i, j)].append(match)
        else:
            merged_pairwise[(i, j)] = adjusted
    
    return merged_pairwise, merged_feats


def visualize_contour_matches(
    img_i: np.ndarray,
    img_j: np.ndarray,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    matches: List[Tuple[int, int]],
    save_path: str,
    max_draw: int = 50,
):
    """Save visualization of contour matches."""
    h1, w1 = img_i.shape[:2]
    h2, w2 = img_j.shape[:2]
    
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    if img_i.ndim == 2:
        canvas[:h1, :w1] = cv2.cvtColor(img_i, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:h1, :w1] = img_i
    
    if img_j.ndim == 2:
        canvas[:h2, w1:w1+w2] = cv2.cvtColor(img_j, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:h2, w1:w1+w2] = img_j
    
    # Draw matches
    for idx, (a, b) in enumerate(matches[:max_draw]):
        p1 = pts_i[a]
        p2 = pts_j[b]
        
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]) + w1, int(p2[1]))
        
        color = (0, 255, 0)  # Green
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)
    
    cv2.imwrite(save_path, canvas)


if __name__ == "__main__":
    print("Silhouette Matching Module")
    print("="*50)
    print()
    print("For textureless objects, use silhouette contours instead of SIFT.")
    print()
    print("Usage:")
    print("  from silhouette_matching import match_silhouettes_multicam")
    print("  pairwise, contours = match_silhouettes_multicam(silhouettes, cams)")