from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    """Compress a variable-length token sequence into fixed number of latent tokens.

    Inspired by Perceiver / Flamingo-style resampling.
    """

    def __init__(self, d_in: int, num_latents: int = 32, nhead: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_in) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            attn = nn.MultiheadAttention(d_in, nhead, dropout=dropout, batch_first=True)
            ff = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, 4 * d_in),
                nn.GELU(),
                nn.Linear(4 * d_in, d_in),
            )
            ln = nn.LayerNorm(d_in)
            self.layers.append(nn.ModuleDict({"attn": attn, "ff": ff, "ln": ln}))

        self.out_ln = nn.LayerNorm(d_in)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
          x: [B, N, d_in]
          x_mask: [B, N] bool True valid

        Returns:
          latents: [B, L, d_in]
        """
        b = x.size(0)
        latents = self.latents.expand(b, -1, -1)

        key_padding_mask = None
        if x_mask is not None:
            key_padding_mask = ~x_mask  # True means pad

        for layer in self.layers:
            attn = layer["attn"]
            ln = layer["ln"]
            ff = layer["ff"]

            # cross-attn: queries=latents, keys/values=x
            attn_out, _ = attn(query=latents, key=x, value=x, key_padding_mask=key_padding_mask, need_weights=False)
            latents = ln(latents + attn_out)
            latents = latents + ff(latents)

        return self.out_ln(latents)


class PrefixBridge(nn.Module):
    """Project time-series tokens into LLM prefix embeddings."""

    def __init__(
        self,
        ts_dim: int,
        llm_dim: int,
        num_prefix_tokens: int = 32,
        resampler_layers: int = 2,
        resampler_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.resampler = PerceiverResampler(
            d_in=ts_dim,
            num_latents=num_prefix_tokens,
            nhead=resampler_heads,
            num_layers=resampler_layers,
            dropout=dropout,
        )
        self.proj = nn.Linear(ts_dim, llm_dim)
        self.ln = nn.LayerNorm(llm_dim)

    def forward(self, ts_tokens: torch.Tensor, ts_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns prefix embeddings and prefix mask.

        prefix_embeds: [B, K, llm_dim]
        prefix_mask: [B, K] bool True valid
        """
        latents = self.resampler(ts_tokens, ts_mask)  # [B,K,ts_dim]
        prefix = self.ln(self.proj(latents))  # [B,K,llm_dim]
        prefix_mask = torch.ones(prefix.size()[:2], device=prefix.device, dtype=torch.bool)
        return prefix, prefix_mask
