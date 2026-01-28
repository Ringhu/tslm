from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import sacrebleu

from tsrlm.data.dataset import TSReportDataset
from tsrlm.data.collator import pad_2d_sequence
from tsrlm.models.ts_llm import TSReportLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--llm_name_or_path", type=str, required=True)

    p.add_argument("--ts_num_vars", type=int, default=1)
    p.add_argument("--ts_patch_len", type=int, default=16)
    p.add_argument("--ts_d_model", type=int, default=256)
    p.add_argument("--ts_layers", type=int, default=4)
    p.add_argument("--ts_heads", type=int, default=8)
    p.add_argument("--prefix_tokens", type=int, default=32)

    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_state(model: TSReportLM, ckpt_dir: Path) -> None:
    # try typical HF filename
    candidates = [
        ckpt_dir / "pytorch_model.bin",
        ckpt_dir / "model.safetensors",
        ckpt_dir / "checkpoint.pt",
    ]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        # Trainer sometimes nests
        for c in ckpt_dir.rglob("pytorch_model.bin"):
            found = c
            break
    if found is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    state = torch.load(found, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TSReportLM(
        llm_name_or_path=args.llm_name_or_path,
        ts_num_vars=args.ts_num_vars,
        ts_patch_len=args.ts_patch_len,
        ts_d_model=args.ts_d_model,
        ts_layers=args.ts_layers,
        ts_heads=args.ts_heads,
        prefix_tokens=args.prefix_tokens,
        freeze_llm=False,
        use_revin=False,
        trust_remote_code=True,
    ).to(device)

    load_state(model, Path(args.checkpoint_dir))
    model.eval()

    ds = TSReportDataset(args.eval_jsonl, max_samples=args.limit)

    refs: List[str] = []
    hyps: List[str] = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rougeL_scores = []

    for item in tqdm(ds, desc="Generating"):
        values = item["values"].unsqueeze(0).to(device)  # [1,T,D]
        mask = torch.ones((1, values.size(1)), dtype=torch.bool, device=device)
        hyp = model.generate(
            values=values,
            ts_attn_mask=mask,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        ref = item["text"]

        hyps.append(hyp)
        refs.append(ref)

        rouge = scorer.score(ref, hyp)["rougeL"].fmeasure
        rougeL_scores.append(rouge)

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    rougeL = float(np.mean(rougeL_scores))

    out = {
        "bleu": bleu,
        "rougeL": rougeL,
        "n": len(ds),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
