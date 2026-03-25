from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def soft_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def summarize_tokens(text: str) -> int:
    return len([t for t in text.replace("\n", " ").split(" ") if t.strip()])


def avg(lst: Sequence[float]) -> float:
    return sum(lst) / max(len(lst), 1)


def flatten_evidence(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def as_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: as_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [as_serializable(x) for x in obj]
    return obj
