"""Convert *ts_cap-style* generated JSON samples into TS-RLM SchemaV1 JSONL.

SchemaV1 (minimum required by training scripts):
  {
    "id": "...",
    "values": [[...], [...], ...],   # shape [T, D]
    "text": "..."                   # supervision text (caption/report)
  }

Your current samples (as you pasted) look like:
  {
    "dataset": "ucr2018",
    "task": "classification",
    "series_key": "MedicalImages/test/000001",
    "time": ["0", "1", ...],
    "timeseries": [[-0.46], [0.69], ...],
    "dense_captions": {
      "global": {"zh": "...", "en": "..."},
      "domain_integrated": {"zh": "...", "en": "..."},
      "local": [...]
    },
    "label": 10,
    ...
  }

This script extracts:
  - id: by default "{dataset}:{task}:{series_key}"
  - values: from "timeseries" (fallback: "values")
  - text: from a configurable source path, default "dense_captions.global.zh"

It can also optionally carry extra fields into "meta" for later analysis.

Usage examples
--------------

1) Convert to SchemaV1 (keep meta, use global Chinese caption):
  python scripts/prepare_jsonl_adapter.py \
    --in_path  raw.jsonl \
    --out_path schema_v1.jsonl \
    --text_source dense_captions.global.zh \
    --keep_meta

2) Use English domain-integrated caption:
  python scripts/prepare_jsonl_adapter.py \
    --in_path raw.jsonl --out_path schema_v1_en.jsonl \
    --text_source dense_captions.domain_integrated.en \
    --keep_meta

3) Filter out samples that fail your upstream claim_check:
  python scripts/prepare_jsonl_adapter.py \
    --in_path raw.jsonl --out_path schema_v1_pass.jsonl \
    --require_claim_pass
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _get_by_path(obj: Any, path: str) -> Any:
    """Get nested value by dot-path.

    Supports:
      - dots for dict keys: a.b.c
      - integer list indexes: local.0.description_zh

    Returns None if not found.
    """

    cur: Any = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur


def _to_2d(values: Any) -> List[List[float]]:
    """Coerce values into list[list[float]] with shape [T, D]."""
    if values is None:
        raise ValueError("values is None")
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("values must be a non-empty list")

    # Already 2D: [[...], [...]]
    if isinstance(values[0], list):
        out: List[List[float]] = []
        for row in values:
            if not isinstance(row, list) or len(row) == 0:
                raise ValueError("each row must be a non-empty list")
            out.append([float(x) for x in row])
        return out

    # 1D -> wrap as univariate
    return [[float(x)] for x in values]


def extract_id(sample: Dict[str, Any], id_source: Optional[str]) -> str:
    if id_source:
        v = _get_by_path(sample, id_source)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v is not None:
            return str(v)

    dataset = sample.get("dataset") or "unknown"
    task = sample.get("task") or "unknown"
    series_key = sample.get("series_key") or sample.get("id") or "unknown"
    return f"{dataset}:{task}:{series_key}"


def extract_values(sample: Dict[str, Any], values_source: Optional[str]) -> List[List[float]]:
    if values_source:
        v = _get_by_path(sample, values_source)
        if v is not None:
            return _to_2d(v)

    # Default: ts_cap samples use "timeseries".
    if "timeseries" in sample:
        return _to_2d(sample["timeseries"])
    if "values" in sample:
        return _to_2d(sample["values"])

    raise KeyError("Cannot find time-series values. Tried: values_source/timeseries/values")


def extract_text(sample: Dict[str, Any], text_source: str, fallback_sources: Iterable[str]) -> str:
    # Try primary
    v = _get_by_path(sample, text_source)
    if isinstance(v, str) and v.strip():
        return v.strip()

    # Try fallbacks
    for p in fallback_sources:
        vv = _get_by_path(sample, p)
        if isinstance(vv, str) and vv.strip():
            return vv.strip()

    raise KeyError(
        "Cannot find supervision text. "
        f"Tried text_source={text_source} and fallbacks={list(fallback_sources)}"
    )


def build_meta(sample: Dict[str, Any], keep_meta: bool) -> Optional[Dict[str, Any]]:
    if not keep_meta:
        return None

    # Keep a *curated* set of fields to avoid exploding JSON size.
    # (You can always change this list.)
    keys = [
        "dataset",
        "task",
        "series_key",
        "indices",
        "time",
        "time_kind",
        "label",
        "variables",
        "features",
        "claims",
        "claim_check",
        "domain_context",
        "descriptions",
        "dense_captions",
    ]
    meta: Dict[str, Any] = {}
    for k in keys:
        if k in sample:
            meta[k] = sample[k]
    return meta


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {ln}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"JSONL must be objects. Got {type(obj)} at line {ln}.")
            yield obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True, help="Raw JSONL path (ts_cap samples)")
    ap.add_argument("--out_path", type=str, required=True, help="Output SchemaV1 JSONL path")
    ap.add_argument(
        "--id_source",
        type=str,
        default=None,
        help="Optional dot-path to use as id (default builds dataset:task:series_key)",
    )
    ap.add_argument(
        "--values_source",
        type=str,
        default=None,
        help="Optional dot-path to the time-series values (default: timeseries -> values)",
    )
    ap.add_argument(
        "--text_source",
        type=str,
        default="dense_captions.global.zh",
        help="Dot-path to the supervision text. Default: dense_captions.global.zh",
    )
    ap.add_argument(
        "--fallback_text",
        type=str,
        nargs="*",
        default=[
            "dense_captions.domain_integrated.zh",
            "dense_captions.global.en",
            "dense_captions.domain_integrated.en",
            "descriptions.0",
            "descriptions.1",
        ],
        help="Fallback dot-paths for text if text_source not found",
    )
    ap.add_argument(
        "--keep_meta",
        action="store_true",
        help="If set, copy curated extra fields into output['meta']",
    )
    ap.add_argument(
        "--require_claim_pass",
        action="store_true",
        help="If set, only keep samples where claim_check.pass == True",
    )
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    n_skipped_claim = 0

    with open(args.out_path, "w", encoding="utf-8") as wf:
        for sample in iter_jsonl(args.in_path):
            n_in += 1

            if args.require_claim_pass:
                cc = sample.get("claim_check")
                passed = bool(cc.get("pass")) if isinstance(cc, dict) else False
                if not passed:
                    n_skipped_claim += 1
                    continue

            out: Dict[str, Any] = {
                "id": extract_id(sample, args.id_source),
                "values": extract_values(sample, args.values_source),
                "text": extract_text(sample, args.text_source, args.fallback_text),
            }
            meta = build_meta(sample, keep_meta=args.keep_meta)
            if meta is not None:
                out["meta"] = meta

            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"Done. in={n_in}, out={n_out}, skipped_claim={n_skipped_claim}. "
        f"Wrote: {args.out_path}"
    )


if __name__ == "__main__":
    main()
