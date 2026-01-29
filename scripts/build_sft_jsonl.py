from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _get_by_path(obj: Any, path: str) -> Any:
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
    if values is None:
        raise ValueError("values is None")
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("values must be a non-empty list")
    if isinstance(values[0], list):
        out: List[List[float]] = []
        for row in values:
            out.append([float(x) for x in row])
        return out
    return [[float(x)] for x in values]


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


def build_id(sample: Dict[str, Any]) -> str:
    dataset = sample.get("dataset") or "unknown"
    task = sample.get("task") or "unknown"
    series_key = sample.get("series_key") or sample.get("id") or "unknown"
    return f"{dataset}:{task}:{series_key}"


def extract_values(sample: Dict[str, Any]) -> List[List[float]]:
    if "timeseries" in sample:
        return _to_2d(sample["timeseries"])
    if "values" in sample:
        return _to_2d(sample["values"])
    raise KeyError("Cannot find time-series values. Tried: timeseries/values")


def extract_caption(sample: Dict[str, Any], lang: str) -> Optional[str]:
    # prefer dense_captions.global
    path1 = f"dense_captions.global.{lang}"
    v = _get_by_path(sample, path1)
    if isinstance(v, str) and v.strip():
        return v.strip()

    # fallbacks (domain_integrated / descriptions)
    for p in [f"dense_captions.domain_integrated.{lang}", "descriptions.0", "descriptions.1"]:
        vv = _get_by_path(sample, p)
        if isinstance(vv, str) and vv.strip():
            return vv.strip()
    return None


def find_claim(claims: List[Dict[str, Any]], claim_type: str) -> Optional[Dict[str, Any]]:
    for c in claims:
        if c.get("type") == claim_type:
            return c
    return None


def all_claims(claims: List[Dict[str, Any]], claim_type: str) -> List[Dict[str, Any]]:
    return [c for c in claims if c.get("type") == claim_type]


def extract_facts(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical facts extracted from your existing claims/features.

    We follow your claim schema examples:
      - global_trend_label: data.label_final (preferred) or data.label
      - global_net_change: data.delta_pct
      - peak / valley: idx + data.value
      - seasonality: data.has + data.period
      - volatility: features.volatility (fallback)
    """
    claims = sample.get("claims") or []
    if not isinstance(claims, list):
        claims = []

    facts: Dict[str, Any] = {}

    tr = find_claim(claims, "global_trend_label")
    if isinstance(tr, dict):
        d = tr.get("data") or {}
        label = d.get("label_final") or d.get("label") or d.get("label_raw")
        if isinstance(label, str):
            facts["trend"] = label

    net = find_claim(claims, "global_net_change")
    if isinstance(net, dict):
        d = net.get("data") or {}
        dp = d.get("delta_pct")
        if dp is not None:
            try:
                dp = float(dp)
                facts["delta_pct"] = dp
                facts["net_change_sign"] = "pos" if dp > 0 else ("neg" if dp < 0 else "zero")
            except Exception:
                pass

    # peak/valley: pick global max/min among candidates
    peaks = all_claims(claims, "peak")
    valleys = all_claims(claims, "valley")

    def _pick_max(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best = None
        best_v = None
        for c in cands:
            d = c.get("data") or {}
            v = d.get("value")
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if best is None or vv > float(best_v):  # type: ignore[arg-type]
                best = c
                best_v = vv
        return best

    def _pick_min(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        best = None
        best_v = None
        for c in cands:
            d = c.get("data") or {}
            v = d.get("value")
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if best is None or vv < float(best_v):  # type: ignore[arg-type]
                best = c
                best_v = vv
        return best

    pk = _pick_max(peaks)
    if pk is not None:
        facts["peak"] = {"idx": pk.get("idx"), "value": (pk.get("data") or {}).get("value")}

    vl = _pick_min(valleys)
    if vl is not None:
        facts["valley"] = {"idx": vl.get("idx"), "value": (vl.get("data") or {}).get("value")}

    seas = find_claim(claims, "seasonality")
    if isinstance(seas, dict):
        d = seas.get("data") or {}
        has = d.get("has")
        period = d.get("period")
        facts["seasonality"] = {"has": bool(has), "period": period if period is None else int(period)}

    # volatility from features if available
    feats = sample.get("features") or {}
    if isinstance(feats, dict):
        vol = feats.get("volatility")
        if isinstance(vol, str):
            facts["volatility"] = vol

    return facts


def render_template_caption(facts: Dict[str, Any], lang: str) -> str:
    """
    Deterministic template caption from facts.
    This is your "text expansion" knob.
    """
    trend = facts.get("trend", "unknown")
    delta_pct = facts.get("delta_pct", None)
    vol = facts.get("volatility", "unknown")

    peak = facts.get("peak") if isinstance(facts.get("peak"), dict) else None
    valley = facts.get("valley") if isinstance(facts.get("valley"), dict) else None
    seas = facts.get("seasonality") if isinstance(facts.get("seasonality"), dict) else None

    if lang == "zh":
        parts: List[str] = []
        # trend + delta
        if delta_pct is not None:
            try:
                dp = float(delta_pct)
                sign = "上升" if dp > 0 else ("下降" if dp < 0 else "基本不变")
                parts.append(f"整体趋势为{trend}，净变化{sign}，相对变化约{dp:.2f}%。")
            except Exception:
                parts.append(f"整体趋势为{trend}。")
        else:
            parts.append(f"整体趋势为{trend}。")

        # peaks/valleys
        if peak is not None and peak.get("idx") is not None:
            parts.append(f"序列在索引{peak.get('idx')}附近出现峰值（约{float(peak.get('value')):.3f}）。")
        if valley is not None and valley.get("idx") is not None:
            parts.append(f"序列在索引{valley.get('idx')}附近出现谷值（约{float(valley.get('value')):.3f}）。")

        # seasonality
        if seas is not None:
            if bool(seas.get("has")):
                parts.append(f"存在季节性，周期约为{seas.get('period')}。")
            else:
                parts.append("未检测到明显季节性。")

        # volatility
        parts.append(f"整体波动性为{vol}。")
        return "".join(parts)

    # English
    parts2: List[str] = []
    if delta_pct is not None:
        try:
            dp = float(delta_pct)
            sign = "increase" if dp > 0 else ("decrease" if dp < 0 else "little net change")
            parts2.append(f"Overall trend is {trend} with a net {sign} of about {dp:.2f}%.")
        except Exception:
            parts2.append(f"Overall trend is {trend}.")
    else:
        parts2.append(f"Overall trend is {trend}.")

    if peak is not None and peak.get("idx") is not None:
        parts2.append(f"A peak occurs around index {peak.get('idx')} (value ≈ {float(peak.get('value')):.3f}).")
    if valley is not None and valley.get("idx") is not None:
        parts2.append(f"A valley occurs around index {valley.get('idx')} (value ≈ {float(valley.get('value')):.3f}).")

    if seas is not None:
        if bool(seas.get("has")):
            parts2.append(f"Seasonality is detected with period ≈ {seas.get('period')}.")
        else:
            parts2.append("No clear seasonality is detected.")

    parts2.append(f"Overall volatility is {vol}.")
    return " ".join(parts2)


def prompt_for_style(style: str, lang: str) -> str:
    if lang == "zh":
        if style == "caption_only":
            return (
                "你是一名时间序列分析助手。给定一段时间序列（已通过前缀表示），"
                "请用中文写一段2-4句的客观描述，必须包含整体趋势、波动性，并尽量提到峰值/谷值与季节性（如存在）。"
                "只输出描述文本，不要输出多余内容。"
            )
        if style == "facts_only":
            return (
                "你是一名时间序列分析助手。给定一段时间序列（已通过前缀表示），"
                "请只输出严格JSON，格式为："
                "{\"facts\":{\"trend\":...,\"delta_pct\":...,\"peak\":{\"idx\":...,\"value\":...},\"valley\":{\"idx\":...,\"value\":...},\"volatility\":...,\"seasonality\":{\"has\":...,\"period\":...}}}。"
                "不要输出任何额外文字。"
            )
        # json_caption
        return (
            "你是一名时间序列分析助手。给定一段时间序列（已通过前缀表示），"
            "请只输出严格JSON，包含两个字段：facts 与 caption。\n"
            "facts 必须包含：trend, delta_pct, peak{idx,value}, valley{idx,value}, volatility, seasonality{has,period}。\n"
            "caption 用中文写2-4句，必须与 facts 一致，不要编造。"
        )

    # English
    if style == "caption_only":
        return (
            "You are a time-series analysis assistant. Given a time series (provided implicitly via a prefix), "
            "write a 2-4 sentence objective English description including overall trend and volatility, and mention peak/valley and seasonality if applicable. "
            "Output only the description."
        )
    if style == "facts_only":
        return (
            "You are a time-series analysis assistant. Given a time series (provided implicitly via a prefix), "
            "output STRICT JSON only: "
            "{\"facts\":{\"trend\":...,\"delta_pct\":...,\"peak\":{\"idx\":...,\"value\":...},\"valley\":{\"idx\":...,\"value\":...},\"volatility\":...,\"seasonality\":{\"has\":...,\"period\":...}}}."
        )
    return (
        "You are a time-series analysis assistant. Given a time series (provided implicitly via a prefix), "
        "output STRICT JSON with exactly two keys: facts and caption.\n"
        "facts must include: trend, delta_pct, peak{idx,value}, valley{idx,value}, volatility, seasonality{has,period}.\n"
        "caption is a 2-4 sentence English description consistent with facts."
    )


def build_examples(sample: Dict[str, Any], styles: List[str], lang: str, keep_meta: bool, template_caption: bool, emit_both_captions: bool) -> List[Dict[str, Any]]:
    sid = build_id(sample)
    values = extract_values(sample)
    facts = extract_facts(sample)
    natural_caption = extract_caption(sample, lang)

    out_rows: List[Dict[str, Any]] = []
    for style in styles:
        prompt = prompt_for_style(style, lang)

        if style == "caption_only":
            outputs: List[Tuple[str, str]] = []
            if natural_caption and (not template_caption) and emit_both_captions:
                outputs.append(("natural", natural_caption))
                outputs.append(("template", render_template_caption(facts, lang)))
            else:
                cap = natural_caption if (natural_caption and not template_caption) else render_template_caption(facts, lang)
                outputs.append(("single", cap))

            for tag, cap in outputs:
                row: Dict[str, Any] = {
                    "id": f"{sid}::{style}_{lang}_{tag}",
                    "values": values,
                    "prompt": prompt,
                    "output": cap,
                    "facts": facts,
                }
                if keep_meta:
                    row["meta"] = {
                        "dataset": sample.get("dataset"),
                        "task": sample.get("task"),
                        "series_key": sample.get("series_key"),
                        "label": sample.get("label"),
                        "claim_check": sample.get("claim_check"),
                        "features": sample.get("features"),
                    }
                out_rows.append(row)
            continue

        elif style == "facts_only":
            output = json.dumps({"facts": facts}, ensure_ascii=False)

        elif style == "bullet":
            # a simple non-JSON template, useful as extra supervision
            output = render_template_caption(facts, lang)

        elif style == "json_caption":
            outputs: List[Tuple[str, str]] = []
            if natural_caption and (not template_caption) and emit_both_captions:
                outputs.append(("natural", natural_caption))
                outputs.append(("template", render_template_caption(facts, lang)))
            else:
                cap = natural_caption if (natural_caption and not template_caption) else render_template_caption(facts, lang)
                outputs.append(("single", cap))

            for tag, cap in outputs:
                row: Dict[str, Any] = {
                    "id": f"{sid}::{style}_{lang}_{tag}",
                    "values": values,
                    "prompt": prompt,
                    "output": json.dumps({"facts": facts, "caption": cap}, ensure_ascii=False),
                    "facts": facts,
                }
                if keep_meta:
                    row["meta"] = {
                        "dataset": sample.get("dataset"),
                        "task": sample.get("task"),
                        "series_key": sample.get("series_key"),
                        "label": sample.get("label"),
                        "claim_check": sample.get("claim_check"),
                        "features": sample.get("features"),
                    }
                out_rows.append(row)
            continue

        else:
            raise ValueError(f"Unknown style: {style}")

        row: Dict[str, Any] = {
            "id": f"{sid}::{style}_{lang}",
            "values": values,
            "prompt": prompt,
            "output": output,
            "facts": facts,
        }
        if keep_meta:
            row["meta"] = {
                "dataset": sample.get("dataset"),
                "task": sample.get("task"),
                "series_key": sample.get("series_key"),
                "label": sample.get("label"),
                "claim_check": sample.get("claim_check"),
                "features": sample.get("features"),
            }
        out_rows.append(row)

    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True, help="Raw ts_cap JSONL")
    ap.add_argument("--out_jsonl", type=str, required=True, help="Output SFT JSONL")
    ap.add_argument("--lang", type=str, default="zh", choices=["zh", "en"])
    ap.add_argument("--styles", type=str, nargs="+", default=["json_caption"], choices=["caption_only", "facts_only", "json_caption", "bullet"])
    ap.add_argument("--keep_meta", action="store_true")
    ap.add_argument("--require_claim_pass", action="store_true")
    ap.add_argument("--template_caption", action="store_true", help="If set, ignore natural captions and always use deterministic templates.")
    ap.add_argument("--emit_both_captions", action="store_true", help="If set, and natural captions exist, emit BOTH natural and template variants for caption styles.")
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    n_skipped = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as wf:
        for sample in iter_jsonl(args.in_jsonl):
            n_in += 1

            if args.require_claim_pass:
                cc = sample.get("claim_check")
                passed = bool(cc.get("pass")) if isinstance(cc, dict) else False
                if not passed:
                    n_skipped += 1
                    continue

            rows = build_examples(sample, styles=args.styles, lang=args.lang, keep_meta=args.keep_meta, template_caption=args.template_caption, emit_both_captions=args.emit_both_captions)
            for r in rows:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"Done. in={n_in}, out={n_out}, skipped_claim={n_skipped}. Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
