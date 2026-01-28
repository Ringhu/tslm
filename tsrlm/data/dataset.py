from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


JsonDict = Dict[str, Any]


def _as_2d_values(values: Union[List[float], List[List[float]]]) -> np.ndarray:
    """Return time-major array of shape [T, D]."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"values must be 1D or 2D, got shape={arr.shape}")
    return arr


class TSReportDataset(Dataset):
    """Reads SchemaV1 JSONL.

    Each item returns:
      - values: FloatTensor [T, D]
      - stats: dict (optional)
      - claims: list (optional)
      - text: str
      - id: str
    """

    def __init__(
        self,
        jsonl_path: str,
        max_samples: Optional[int] = None,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.rows: List[JsonDict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
                if max_samples is not None and len(self.rows) >= max_samples:
                    break

        # basic validation
        for i, r in enumerate(self.rows[:50]):  # cheap sanity check
            if "id" not in r or "values" not in r or "text" not in r:
                raise ValueError(f"Row {i} missing required fields: {r.keys()}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> JsonDict:
        r = self.rows[idx]
        values = _as_2d_values(r["values"])
        item: JsonDict = {
            "id": r["id"],
            "values": torch.from_numpy(values),  # [T, D]
            "text": r["text"],
        }
        if "stats" in r:
            item["stats"] = r["stats"]
        if "claims" in r:
            item["claims"] = r["claims"]
        if "meta" in r:
            item["meta"] = r["meta"]
        if "segments" in r:
            item["segments"] = r["segments"]
        return item
