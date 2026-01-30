from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import sacrebleu

from tsrlm.config import TSRLMConfig
from tsrlm.data import TSSFTDataset
from tsrlm.models import TSReportLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_jsonl", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # generation
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true")

    return p.parse_args()


def maybe_apply_lora_from_ckpt(model: TSReportLM, ckpt_dir: Path) -> None:
    lora_path = ckpt_dir / "lora_config.json"
    if not lora_path.exists():
        return
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise ImportError("peft is required to load LoRA checkpoint. Install: pip install peft") from e

    cfg = json.loads(lora_path.read_text(encoding="utf-8"))
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["r"]),
        lora_alpha=int(cfg["alpha"]),
        lora_dropout=float(cfg["dropout"]),
        target_modules=list(cfg["target_modules"]),
        bias="none",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)


def load_state(model: TSReportLM, ckpt_dir: Path) -> None:
    candidates = [
        ckpt_dir / "pytorch_model.bin",
        ckpt_dir / "model.safetensors",
    ]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        files = list(ckpt_dir.rglob("pytorch_model.bin")) + list(ckpt_dir.rglob("model.safetensors"))
        if files:
            found = files[0]
    if found is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")

    print(f"Loading weights from: {found}")
    if str(found).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(found)
    else:
        state = torch.load(found, map_location="cpu")

    keys = model.load_state_dict(state, strict=False)
    print(f"Load keys: missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)}")


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    # best-effort: find a JSON object substring
    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    sub = text[l : r + 1]
    try:
        return json.loads(sub)
    except Exception:
        return None


def extract_caption(text: str) -> str:
    obj = try_parse_json(text)
    if isinstance(obj, dict) and "caption" in obj and isinstance(obj["caption"], str):
        return obj["caption"].strip()
    return text.strip()


def extract_facts(text: str) -> Optional[Dict[str, Any]]:
    obj = try_parse_json(text)
    if isinstance(obj, dict) and "facts" in obj and isinstance(obj["facts"], dict):
        return obj["facts"]
    return None


def factual_metrics(gt: Optional[Dict[str, Any]], pred: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Simple factual metrics (compute what we can robustly).

    - exact match for categorical keys: trend, volatility, seasonality.has
    - MAE for numeric keys: delta_pct, seasonality.period
    - MAE for peak/valley idx/value if present
    """
    out: Dict[str, float] = {}
    if gt is None or pred is None:
        return out

    def _eq(k: str) -> Optional[float]:
        if k in gt and k in pred:
            return float(gt[k] == pred[k])
        return None

    def _mae(a: Any, b: Any) -> Optional[float]:
        try:
            return float(abs(float(a) - float(b)))
        except Exception:
            return None

    # categorical
    for k in ["trend", "volatility", "net_change_sign"]:
        v = _eq(k)
        if v is not None:
            out[f"fact_acc/{k}"] = v

    # delta pct
    if "delta_pct" in gt and "delta_pct" in pred:
        v = _mae(gt["delta_pct"], pred["delta_pct"])
        if v is not None:
            out["fact_mae/delta_pct"] = v

    # nested seasonality fields
    if isinstance(gt.get("seasonality"), dict) and isinstance(pred.get("seasonality"), dict):
        g = gt["seasonality"]
        p = pred["seasonality"]
        if "has" in g and "has" in p:
            out["fact_acc/seasonality_has"] = float(bool(g["has"]) == bool(p["has"]))
        if "period" in g and "period" in p:
            v = _mae(g["period"], p["period"])
            if v is not None:
                out["fact_mae/seasonality_period"] = v

    # peak/valley
    def _idx_value(prefix: str, key: str) -> None:
        if not isinstance(gt.get(key), dict) or not isinstance(pred.get(key), dict):
            return
        g = gt[key]
        p = pred[key]
        if "idx" in g and "idx" in p:
            v = _mae(g["idx"], p["idx"])
            if v is not None:
                out[f"fact_mae/{key}_idx"] = v
        if "value" in g and "value" in p:
            v = _mae(g["value"], p["value"])
            if v is not None:
                out[f"fact_mae/{key}_value"] = v

    _idx_value("peak", "peak")
    _idx_value("valley", "valley")

    return out


    def _eq(k: str) -> Optional[float]:
        if k in gt and k in pred:
            return float(gt[k] == pred[k])
        return None

    def _mae(k: str) -> Optional[float]:
        if k in gt and k in pred:
            try:
                return float(abs(float(gt[k]) - float(pred[k])))
            except Exception:
                return None
        return None

    for k in ["trend", "volatility"]:
        v = _eq(k)
        if v is not None:
            out[f"fact_acc/{k}"] = v

    # nested seasonality fields
    if isinstance(gt.get("seasonality"), dict) and isinstance(pred.get("seasonality"), dict):
        g = gt["seasonality"]
        p = pred["seasonality"]
        if "has" in g and "has" in p:
            out["fact_acc/seasonality_has"] = float(bool(g["has"]) == bool(p["has"]))
        if "period" in g and "period" in p:
            try:
                out["fact_mae/seasonality_period"] = float(abs(float(g["period"]) - float(p["period"])))
            except Exception:
                pass

    v = _mae("delta_pct")
    if v is not None:
        out["fact_mae/delta_pct"] = v

    return out


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load config saved by training
    cfg_path = ckpt_dir / "tsrlm_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing tsrlm_config.json in {ckpt_dir}")
    cfg = TSRLMConfig.load(cfg_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # dtype for eval
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = TSReportLM(
        config=cfg,
        freeze_llm=False,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    ).to(device)

    # if LoRA checkpoint, apply before loading weights
    maybe_apply_lora_from_ckpt(model, ckpt_dir)
    load_state(model, ckpt_dir)
    model.eval()

    ds = TSSFTDataset(args.eval_jsonl, max_samples=args.limit)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rougeL_scores: List[float] = []
    refs: List[str] = []
    hyps: List[str] = []

    # factual aggregates
    fact_acc: Dict[str, List[float]] = {}

    pred_path = out_dir / "predictions.jsonl"
    wf = pred_path.open("w", encoding="utf-8")

    print(f"Generating for {len(ds)} samples...")
    for item in tqdm(ds, desc="Inference"):
        values = item["values"].unsqueeze(0).to(device)  # [1,T,D]
        mask = torch.ones((1, values.size(1)), dtype=torch.bool, device=device)

        prompt = item.get("prompt") or cfg.default_prompt
        with torch.no_grad():
            hyp = model.generate(
                values=values,
                ts_attn_mask=mask,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                debug=False,
            )

        ref = item["output"]
        hyp_cap = extract_caption(hyp)
        ref_cap = extract_caption(ref)

        hyps.append(hyp_cap)
        refs.append(ref_cap)
        rougeL_scores.append(scorer.score(ref_cap, hyp_cap)["rougeL"].fmeasure)

        # factual
        gt_facts = item.get("facts") or extract_facts(ref)
        pred_facts = extract_facts(hyp)
        fm = factual_metrics(gt_facts, pred_facts)
        for k, v in fm.items():
            fact_acc.setdefault(k, []).append(v)

        wf.write(
            json.dumps(
                {
                    "id": item["id"],
                    "prompt": prompt,
                    "ref": ref,
                    "hyp": hyp,
                    "ref_caption": ref_cap,
                    "hyp_caption": hyp_cap,
                    "gt_facts": gt_facts,
                    "pred_facts": pred_facts,
                    "factual": fm,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    wf.close()

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0
    rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0

    metrics: Dict[str, Any] = {
        "n": len(ds),
        "bleu": bleu,
        "rougeL": rougeL,
    }
    for k, vals in fact_acc.items():
        metrics[k] = float(np.mean(vals)) if vals else 0.0

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:")
    print(" -", pred_path)
    print(" -", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
