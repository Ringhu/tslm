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
from tsrlm.models.ts_llm import TSReportLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--llm_name_or_path", type=str, required=True)
    p.add_argument("--out_path", type=str, default=None, help="Path to save metrics JSON")

    # Architecture Args (Must match train_sft.py)
    p.add_argument("--ts_num_vars", type=int, default=1)
    p.add_argument("--ts_patch_len", type=int, default=16)
    p.add_argument("--ts_d_model", type=int, default=256)
    p.add_argument("--ts_layers", type=int, default=4)
    p.add_argument("--ts_heads", type=int, default=8)
    
    # Ablation Args
    p.add_argument("--encoder_type", type=str, default="patchtst", choices=["patchtst", "chronos"])
    p.add_argument("--bridge_type", type=str, default="prefix", choices=["prefix", "xattn"])
    p.add_argument("--chronos_model_path", type=str, default="amazon/chronos-t5-small")
    p.add_argument("--prefix_tokens", type=int, default=32)

    # Generation Args
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--limit", type=int, default=None)
    
    # Ignore unknown args (like --learning_rate passed by shell script "$@")
    args, unknown = p.parse_known_args()
    return args


def load_state(model: TSReportLM, ckpt_dir: Path) -> None:
    # try typical HF filename
    candidates = [
        ckpt_dir / "pytorch_model.bin",
        ckpt_dir / "model.safetensors",
        ckpt_dir / "adapter_model.bin", # LoRA case
    ]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
            
    # Trainer sometimes nests checks
    if found is None:
        # Search recursively
        files = list(ckpt_dir.rglob("pytorch_model.bin")) + list(ckpt_dir.rglob("model.safetensors"))
        if files:
            found = files[0]

    if found is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    
    print(f"Loading weights from: {found}")
    
    # Handle safetensors
    if str(found).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(found)
    else:
        state = torch.load(found, map_location="cpu")
        
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
        
    # Loose loading
    keys = model.load_state_dict(state, strict=False)
    print(f"Load keys: missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model with correct architecture
    model = TSReportLM(
        llm_name_or_path=args.llm_name_or_path,
        encoder_type=args.encoder_type,
        bridge_type=args.bridge_type,
        chronos_model_path=args.chronos_model_path,
        ts_num_vars=args.ts_num_vars,
        ts_patch_len=args.ts_patch_len,
        ts_d_model=args.ts_d_model,
        ts_layers=args.ts_layers,
        ts_heads=args.ts_heads,
        prefix_tokens=args.prefix_tokens,
        freeze_llm=False, # Eval doesn't care
        use_revin=False,  # Weights already loaded, RevIN layer exists but state dict covers it
        trust_remote_code=True,
    ).to(device)

    load_state(model, Path(args.checkpoint_dir))
    model.eval()

    ds = TSReportDataset(args.eval_jsonl, max_samples=args.limit)

    refs: List[str] = []
    hyps: List[str] = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rougeL_scores = []

    print(f"Generating for {len(ds)} samples...")
    for item in tqdm(ds, desc="Inference"):
        values = item["values"].unsqueeze(0).to(device)  # [1,T,D]
        # Make a mask (all valid)
        mask = torch.ones((1, values.size(1)), dtype=torch.bool, device=device)
        
        try:
            hyp = model.generate(
                values=values,
                ts_attn_mask=mask,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            print(f"Error generating sample {item['id']}: {e}")
            hyp = ""

        ref = item["text"]
        hyps.append(hyp)
        refs.append(ref)

        rouge = scorer.score(ref, hyp)["rougeL"].fmeasure
        rougeL_scores.append(rouge)

    # Metrics
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0

    out = {
        "bleu": bleu,
        "rougeL": rougeL,
        "n": len(ds),
        "args": vars(args)
    }
    
    # Save or Print
    if args.out_path:
        out_p = Path(args.out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.out_path}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()