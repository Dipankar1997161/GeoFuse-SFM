from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Union


def extract_floats(text: str) -> List[float]:
    """
    Extract floats/ints/scientific-notation numbers from arbitrary text.
    """
    pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    return [float(x) for x in re.findall(pattern, text)]


def load_data(path: Union[str, Path]) -> Any:
    """
    Load a file into a Python object based on file extension.

    - .json -> parsed dict/list
    - .yaml/.yml -> parsed dict/list (requires PyYAML)
    - otherwise -> raw text (str)

    This function is domain-neutral: it does NOT interpret the contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()

    if suf == ".json":
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    if suf in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            ) from e
        return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))

    # default: return raw text
    return path.read_text(encoding="utf-8", errors="ignore")
