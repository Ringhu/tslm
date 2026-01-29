from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


def pad_2d_sequence(seqs: List[torch.Tensor], pad_value: float = 0.0) -> Dict[str, torch.Tensor]:
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


def _get_bos_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    # some causal LMs have no BOS; use EOS as a safe start
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id")


def _get_eos_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    # fallback: BOS
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    raise ValueError("Tokenizer has neither eos_token_id nor bos_token_id")


@dataclass
class TSSFTCollator:
    tokenizer: PreTrainedTokenizerBase
    max_text_length: int = 512
    max_prompt_length: int = 256
    pad_to_multiple_of: Optional[int] = 8
    add_eos: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ---- time series ----
        ts = pad_2d_sequence([b["values"] for b in batch], pad_value=0.0)

        # ---- text ----
        bos_id = _get_bos_id(self.tokenizer)
        eos_id = _get_eos_id(self.tokenizer)

        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []
        prompt_lens: List[int] = []

        for b in batch:
            prompt = b["prompt"]
            output = b["output"]

            # tokenize separately (no special tokens; we add BOS/EOS ourselves)
            p = self.tokenizer(prompt, add_special_tokens=False)
            o = self.tokenizer(output, add_special_tokens=False)

            p_ids = p["input_ids"][: self.max_prompt_length]
            # reserve space: BOS + prompt + (output) + (EOS)
            reserve = 1 + len(p_ids) + (1 if self.add_eos else 0)
            max_o = max(self.max_text_length - reserve, 0)
            o_ids = o["input_ids"][: max_o]

            # final sequence
            ids = [bos_id] + p_ids + o_ids
            if self.add_eos:
                ids = ids + [eos_id]

            # labels: mask prompt (and BOS), supervise output (+EOS)
            # For CausalLM, labels are shifted internally; this still works.
            labels = [-100] * (1 + len(p_ids)) + o_ids
            if self.add_eos:
                labels = labels + [eos_id]

            # attention mask
            attn = [1] * len(ids)

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attn_list.append(torch.tensor(attn, dtype=torch.long))
            prompt_lens.append(1 + len(p_ids))

        # pad to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=int(self.tokenizer.pad_token_id)
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)

        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        out: Dict[str, Any] = {
            **ts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ids": [b["id"] for b in batch],
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        }

        # optional passthrough
        if "meta" in batch[0]:
            out["meta"] = [b.get("meta") for b in batch]
        if "facts" in batch[0]:
            out["facts"] = [b.get("facts") for b in batch]
        return out
