"""
src/pipeline/tracks_builder.py

Track building wrapper.
"""

from __future__ import annotations

from src.tracks import build_tracks_v2

from .state import SfMState


def build_tracks(state: SfMState, logger=None) -> None:
    """
    Build tracks from pairwise matches.
    
    Updates state.tracks in place.
    """
    tracks = build_tracks_v2(state.num_kp, state.pairwise)
    state.tracks = tracks
    
    if logger:
        logger.info(f"Tracks (multi-view): {len(tracks)}")
