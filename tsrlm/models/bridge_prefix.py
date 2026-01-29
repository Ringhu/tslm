from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Compress variable-length tokens to a fixed number of latents via cross-attention."""

    def __init__(
        self,
        d_in: int,
        num_latents: int = 32,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_in) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(d_in),
                        "ln_kv": nn.LayerNorm(d_in),
                        "attn": nn.MultiheadAttention(d_in, nhead, dropout=dropout, batch_first=True),
                        "ln_ff": nn.LayerNorm(d_in),
                        "ff": nn.Sequential(
                            nn.Linear(d_in, 4 * d_in),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * d_in, d_in),
                        ),
                    }
                )
            )

        self.out_ln = nn.LayerNorm(d_in)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
          x: [B,N,d_in]
          x_mask: [B,N] bool True valid

        Returns:
          latents: [B,L,d_in]
        """
        b = x.size(0)
        latents = self.latents.expand(b, -1, -1)

        key_padding_mask = None
        if x_mask is not None:
            key_padding_mask = ~x_mask  # True=pad

        for layer in self.layers:
            q = layer["ln_q"](latents)
            kv = layer["ln_kv"](x)
            attn_out, _ = layer["attn"](query=q, key=kv, value=kv, key_padding_mask=key_padding_mask, need_weights=False)
            latents = latents + attn_out

            ff_in = layer["ln_ff"](latents)
            latents = latents + layer["ff"](ff_in)

        return self.out_ln(latents)


class PrefixBridge(nn.Module):
    """Project TS tokens into LLM prefix embeddings + a safe learnable scale."""

    def __init__(
        self,
        ts_dim: int,
        llm_dim: int,
        num_prefix_tokens: int = 32,
        resampler_layers: int = 2,
        resampler_heads: int = 8,
        dropout: float = 0.0,
        prefix_alpha_init: float = 0.1,
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
        nn.init.normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        self.ln = nn.LayerNorm(llm_dim)
        # crucial: start small to avoid blowing up the LLM embedding space
        self.prefix_alpha = nn.Parameter(torch.tensor(float(prefix_alpha_init)))

    def forward(self, ts_tokens: torch.Tensor, ts_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          prefix_embeds: [B,K,llm_dim]
          prefix_mask:   [B,K] bool True valid
        """
        latents = self.resampler(ts_tokens, ts_mask)  # [B,K,ts_dim]
        prefix = self.ln(self.proj(latents))  # [B,K,llm_dim]
        prefix = prefix * self.prefix_alpha
        prefix_mask = torch.ones(prefix.size()[:2], device=prefix.device, dtype=torch.bool)
        return prefix, prefix_mask
