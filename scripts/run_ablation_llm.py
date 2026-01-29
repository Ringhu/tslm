"""Run LLM-backbone ablations for TS-RLM.

This is a small convenience wrapper that sequentially calls:
  1) scripts/train_sft.py
  2) scripts/evaluate.py

It is intentionally simple (no distributed launching). You can still run a single
model manually by calling train_sft.py directly.

Example:
  python scripts/run_ablation_llm.py \
    --train_jsonl data/schema_v1_train.jsonl \
    --eval_jsonl  data/schema_v1_test.jsonl \
    --out_root    runs/ablation_llm \
    --llms Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B-Base \
    --extra "--epochs 2 --batch_size 4 --lr 2e-4 --freeze_llm"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def train_one(
    llm_name_or_path: str,
    train_jsonl: Path,
    eval_jsonl: Path,
    out_root: Path,
    extra: str,
) -> Path:
    out_dir = out_root / llm_name_or_path.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_sft.py"),
        "--train_jsonl",
        str(train_jsonl),
        "--eval_jsonl",
        str(eval_jsonl),
        "--llm_name_or_path",
        llm_name_or_path,
        "--output_dir",
        str(out_dir),
    ]

    if extra.strip():
        cmd += shlex.split(extra)

    run(cmd)
    return out_dir


def eval_one(ckpt_dir: Path, eval_jsonl: Path, out_root: Path) -> Path:
    out_path = out_root / f"eval__{ckpt_dir.name}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate.py"),
        "--eval_jsonl",
        str(eval_jsonl),
        "--checkpoint_dir",
        str(ckpt_dir),
        "--out_path",
        str(out_path),
    ]
    run(cmd)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--llms", nargs="+", required=True, help="List of HF model ids")
    p.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra args forwarded to scripts/train_sft.py (quoted as one string).",
    )
    p.add_argument("--skip_eval", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    for llm in args.llms:
        ckpt_dir = train_one(
            llm_name_or_path=llm,
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            out_root=args.out_root,
            extra=args.extra,
        )
        if not args.skip_eval:
            eval_one(ckpt_dir=ckpt_dir, eval_jsonl=args.eval_jsonl, out_root=args.out_root)


if __name__ == "__main__":
    main()
