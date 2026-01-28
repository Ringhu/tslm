from __future__ import annotations

"""Cross-attention bridge (stub).

Why stub:
- Injecting cross-attention into an arbitrary HF causal LM requires model-specific hooks
  (Llama/Qwen/Mistral/GPT-NeoX differ).
- For your *first* paper iteration, prefix-embedding bridge is usually enough and much simpler.

What to do if you want this ablation:
- pick one target architecture (e.g., Qwen/Llama-like in `transformers`)
- wrap each decoder block with an extra cross-attn module attending to ts tokens
- optionally freeze the base LLM and train only cross-attn + ts encoder

This file keeps the interface so your training scripts can switch bridges cleanly.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CrossAttentionBridge(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Cross-attention bridge is model-architecture-specific. " 
            "Use PrefixBridge first; then implement this for your chosen LLM."
        )
