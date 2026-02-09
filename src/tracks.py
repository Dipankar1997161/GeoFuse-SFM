from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

@dataclass
class Track:
    """A multi-view track: mapping image_id -> keypoint_id."""
    obs: Dict[int, int]


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_tracks(
    num_keypoints_per_image: List[int],
    pairwise_matches: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> List[Track]:
    """
    Build tracks across images using union-find over (image_id, kp_id) nodes.

    Key change vs old:
      - if a component contains multiple keypoints from the same image,
        we DO NOT drop the entire track anymore.
        We keep the first one and ignore the rest (salvage).
      - we also report how many conflicts happened, which is a great health signal.
    """
    offsets = np.cumsum([0] + num_keypoints_per_image[:-1]).tolist()
    total = int(sum(num_keypoints_per_image))

    uf = UnionFind(total)

    def node_id(img: int, kp: int) -> int:
        return offsets[img] + kp

    # union matched nodes
    for (i, j), matches in pairwise_matches.items():
        for a, b in matches:
            uf.union(node_id(i, a), node_id(j, b))

    # gather components -> nodes
    comp: Dict[int, List[Tuple[int, int]]] = {}
    for img, n_kp in enumerate(num_keypoints_per_image):
        off = offsets[img]
        for kp in range(n_kp):
            root = uf.find(off + kp)
            comp.setdefault(root, []).append((img, kp))

    # stats
    num_components = len(comp)
    multi_view_components = 0
    conflict_components = 0
    conflict_assignments = 0  # total duplicate (img) occurrences inside components
    dropped_components = 0

    tracks: List[Track] = []

    for _, nodes in comp.items():
        if len(nodes) < 2:
            continue

        multi_view_components += 1

        obs: Dict[int, int] = {}
        had_conflict = False

        for img, kp in nodes:
            if img in obs:
                # conflict: multiple keypoints from same image in one component
                had_conflict = True
                conflict_assignments += 1
                # SALVAGE policy: keep the first, ignore the rest
                continue
            obs[img] = kp

        if had_conflict:
            conflict_components += 1

        # still need at least 2 distinct images
        if len(obs) >= 2:
            tracks.append(Track(obs=obs))
        else:
            dropped_components += 1

    print(
        f"[tracks] components={num_components} multi_view_components={multi_view_components} "
        f"tracks_kept={len(tracks)} conflict_components={conflict_components} "
        f"conflict_assignments={conflict_assignments} dropped_components={dropped_components}"
    )

    return tracks

def build_tracks_v1(
    num_keypoints_per_image,
    pairwise_matches,
    min_track_len=2,
):
    """Initial version of build_tracks that simply drops any component that has same-image conflicts.
    This is simpler but can lead to many dropped tracks if there are many conflicts.
    The improved version build_tracks_v2 uses a conflict-aware union operation to salvage non-conflicting
    parts of the components.
    """
    
    offsets = np.cumsum([0] + num_keypoints_per_image[:-1]).tolist()
    total = int(sum(num_keypoints_per_image))

    uf = UnionFind(total)

    def node_id(img, kp):
        return offsets[img] + kp

    # --- compute node degree support + union ---
    deg = np.zeros(total, dtype=np.int32)
    for (i, j), matches in pairwise_matches.items():
        for a, b in matches:
            u = node_id(i, a)
            v = node_id(j, b)
            uf.union(u, v)
            deg[u] += 1
            deg[v] += 1

    # reverse map nid -> (img,kp)
    img_of = np.empty(total, dtype=np.int32)
    kp_of  = np.empty(total, dtype=np.int32)
    for img, n_kp in enumerate(num_keypoints_per_image):
        off = offsets[img]
        img_of[off:off+n_kp] = img
        kp_of[off:off+n_kp]  = np.arange(n_kp, dtype=np.int32)

    # gather UF components
    comp = defaultdict(list)
    for nid in range(total):
        comp[uf.find(nid)].append(nid)

    tracks = []
    conflict_components = 0
    conflict_assignments = 0
    dropped_components = 0
    multi_view_components = 0

    for _, nodes in comp.items():
        if len(nodes) < 2:
            continue

        multi_view_components += 1

        # choose best node per image by degree
        best_node_per_img = {}
        for u in nodes:
            im = int(img_of[u])
            if im in best_node_per_img:
                conflict_assignments += 1
                # keep the higher degree one
                if deg[u] > deg[best_node_per_img[im]]:
                    best_node_per_img[im] = u
            else:
                best_node_per_img[im] = u

        if len(best_node_per_img) < len(nodes):
            conflict_components += 1

        if len(best_node_per_img) >= min_track_len:
            obs = {im: int(kp_of[u]) for im, u in best_node_per_img.items()}
            tracks.append(Track(obs=obs))
        else:
            dropped_components += 1

    print(
        f"[tracks_v1] components={len(comp)} multi_view_components={multi_view_components} "
        f"tracks_kept={len(tracks)} conflict_components={conflict_components} "
        f"conflict_assignments={conflict_assignments} dropped_components={dropped_components}"
    )
    return tracks

def build_tracks_v2(num_keypoints_per_image, pairwise_matches, min_track_len=2):
    """Improved version of build_tracks that uses union-find with conflict-aware union operation.
    This allows us to avoid dropping entire components that have same-image conflicts, 
    and instead salvage the best supported keypoint per image.
    """
    offsets = np.cumsum([0] + num_keypoints_per_image[:-1]).tolist()
    total = int(sum(num_keypoints_per_image))
    uf = UnionFind(total)

    def node_id(img, kp):
        return offsets[img] + kp

    # Map node -> image id (so we can detect conflicts quickly)
    img_of = np.empty(total, dtype=np.int32)
    for img, n_kp in enumerate(num_keypoints_per_image):
        off = offsets[img]
        img_of[off:off+n_kp] = img

    # Maintain, for each UF root, the set of images present in that component
    comp_imgs = {i: {int(img_of[i])} for i in range(total)}

    def find(a):
        return uf.find(a)

    def union_no_conflict(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return True

        # If components already contain the same image -> would create same-image conflict -> skip
        if comp_imgs[ra].intersection(comp_imgs[rb]):
            return False

        uf.union(ra, rb)
        rnew = find(ra)
        rold = rb if rnew == ra else ra

        # merge image sets
        comp_imgs[rnew] = comp_imgs[ra] | comp_imgs[rb]
        comp_imgs.pop(rold, None)
        return True

    deg = np.zeros(total, dtype=np.int32)

    for (i, j), matches in pairwise_matches.items():
        for a, b in matches:
            u = node_id(i, a)
            v = node_id(j, b)
            uf.union(u, v)
            deg[u] += 1
            deg[v] += 1

    # reverse map nid -> (img,kp)
    img_of = np.empty(total, dtype=np.int32)
    kp_of  = np.empty(total, dtype=np.int32)
    for img, n_kp in enumerate(num_keypoints_per_image):
        off = offsets[img]
        img_of[off:off+n_kp] = img
        kp_of[off:off+n_kp]  = np.arange(n_kp, dtype=np.int32)

    # gather UF components
    comp = defaultdict(list)
    for nid in range(total):
        comp[uf.find(nid)].append(nid)

    tracks = []
    conflict_components = 0
    conflict_assignments = 0
    dropped_components = 0
    multi_view_components = 0

    for _, nodes in comp.items():
        if len(nodes) < 2:
            continue

        multi_view_components += 1

        # choose best node per image by degree
        best_node_per_img = {}
        for u in nodes:
            im = int(img_of[u])
            if im in best_node_per_img:
                conflict_assignments += 1
                # keep the higher degree one
                if deg[u] > deg[best_node_per_img[im]]:
                    best_node_per_img[im] = u
            else:
                best_node_per_img[im] = u

        if len(best_node_per_img) < len(nodes):
            conflict_components += 1

        if len(best_node_per_img) >= min_track_len:
            obs = {im: int(kp_of[u]) for im, u in best_node_per_img.items()}
            tracks.append(Track(obs=obs))
        else:
            dropped_components += 1

    print(
        f"[tracks_v1] components={len(comp)} multi_view_components={multi_view_components} "
        f"tracks_kept={len(tracks)} conflict_components={conflict_components} "
        f"conflict_assignments={conflict_assignments} dropped_components={dropped_components}"
    )
    return tracks
