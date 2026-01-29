from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--llms", nargs="+", required=True)
    p.add_argument("--extra", type=str, default="", help="Extra args forwarded to train_sft.py (quoted)")
    p.add_argument("--skip_eval", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    train_py = Path(__file__).parent / "train_sft.py"
    eval_py = Path(__file__).parent / "evaluate.py"

    for llm in args.llms:
        out_dir = args.out_root / llm.replace("/", "__")
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(train_py),
            "--train_jsonl",
            str(args.train_jsonl),
            "--eval_jsonl",
            str(args.eval_jsonl),
            "--llm_name_or_path",
            llm,
            "--output_dir",
            str(out_dir),
        ]
        if args.extra.strip():
            cmd += shlex.split(args.extra)
        run(cmd)

        if not args.skip_eval:
            out_eval = out_dir / "eval"
            cmd2 = [
                sys.executable,
                str(eval_py),
                "--eval_jsonl",
                str(args.eval_jsonl),
                "--checkpoint_dir",
                str(out_dir / "final_model"),
                "--out_dir",
                str(out_eval),
            ]
            run(cmd2)


if __name__ == "__main__":
    main()
