from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


def pad_2d_sequence(
    seqs: List[torch.Tensor],
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Pad a list of [T, D] to [B, T_max, D]."""
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    t_max = int(lengths.max().item())
    d = int(seqs[0].shape[1])
    out = seqs[0].new_full((len(seqs), t_max, d), fill_value=pad_value)
    mask = torch.zeros((len(seqs), t_max), dtype=torch.bool)
    for i, s in enumerate(seqs):
        t = s.shape[0]
        out[i, :t] = s
        mask[i, :t] = True
    return {"values": out, "ts_attn_mask": mask, "ts_lengths": lengths}


@dataclass
class TSDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_text_length: int = 512
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # time series
        ts = pad_2d_sequence([b["values"] for b in batch], pad_value=0.0)

        # text
        texts = [b["text"] for b in batch]
        tok = self.tokenizer(
            texts,
            max_length=self.max_text_length,
            truncation=True,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # labels = input_ids with pad masked
        labels = tok["input_ids"].clone()
        labels[tok["attention_mask"] == 0] = -100

        out: Dict[str, Any] = {
            **ts,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": labels,
            "ids": [b["id"] for b in batch],
        }

        # passthrough optional fields
        if "stats" in batch[0]:
            out["stats"] = [b.get("stats") for b in batch]
        if "claims" in batch[0]:
            out["claims"] = [b.get("claims") for b in batch]
        if "meta" in batch[0]:
            out["meta"] = [b.get("meta") for b in batch]
        return out
