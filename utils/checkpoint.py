# sfm/utils/checkpointing.py
from __future__ import annotations

from pathlib import Path
import pickle


def save_checkpoint(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  [Checkpoint] Saved: {path}")


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  [Checkpoint] Loaded: {path}")
    return data


def checkpoint_exists(path: Path) -> bool:
    return path.exists()
