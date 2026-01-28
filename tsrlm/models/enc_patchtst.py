from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchify import patchify


class PatchTSTEncoder(nn.Module):
    """A lightweight PatchTST-style encoder (for representation, not forecasting).

    - Patchify: split [T,D] into N patches of length P.
    - Linear patch embedding -> Transformer encoder.
    - Output token embeddings [B, N(+1), d_model].

    This is intentionally minimal. You can swap in your full PatchTST implementation later.
    """

    def __init__(
        self,
        num_vars: int = 1,
        patch_len: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_patches: int = 2048,
        use_stat_token: bool = True,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.patch_len = patch_len
        self.d_model = d_model
        self.use_stat_token = use_stat_token

        self.patch_embed = nn.Linear(patch_len * num_vars, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        if use_stat_token:
            # stats: mean,std,min,max,length => 5*num_vars + 1 (length) if per-var, but
            # we just feed a fixed-size vector:
            stat_dim = 5 * num_vars + 1
            self.stat_mlp = nn.Sequential(
                nn.Linear(stat_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.stat_ln = nn.LayerNorm(d_model)

    def _masked_stats(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute (mean,std,min,max,delta_pct,length) per variable.
        Returns a vector [B, 5*D + 1]."""
        b, t, d = x.shape
        if mask is None:
            m = torch.ones((b, t, 1), device=x.device, dtype=x.dtype)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)

        denom = m.sum(dim=1).clamp_min(1.0)  # [B,1]
        mean = (x * m).sum(dim=1) / denom  # [B,D]
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-6)

        # for min/max with mask: set invalid to +inf/-inf
        x_min = x.clone()
        x_max = x.clone()
        if mask is not None:
            invalid = (~mask).unsqueeze(-1)
            x_min = x_min.masked_fill(invalid, float("inf"))
            x_max = x_max.masked_fill(invalid, float("-inf"))
        vmin = x_min.amin(dim=1)  # [B,D]
        vmax = x_max.amax(dim=1)  # [B,D]

        # delta pct: (end-start)/(|start|+eps)
        if mask is None:
            start = x[:, 0]
            end = x[:, -1]
            length = torch.full((b, 1), t, device=x.device, dtype=x.dtype)
        else:
            lengths = mask.sum(dim=1).clamp_min(1)  # [B]
            # gather start: first valid
            # assume left-aligned padding (our collator does)
            start = x[:, 0]
            end_idx = (lengths - 1).view(b, 1, 1).expand(-1, 1, d)
            end = x.gather(dim=1, index=end_idx).squeeze(1)
            length = lengths.to(x.dtype).view(b, 1)
        delta_pct = (end - start) / (start.abs() + 1e-6)  # [B,D]

        feats = torch.cat([mean, std, vmin, vmax, delta_pct, length], dim=1)  # [B, 5D+1]
        return feats

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
          x: [B,T,D]
          mask: [B,T] bool, True valid

        Returns:
          tokens: [B, N(+1), d_model]
          token_mask: [B, N(+1)] bool, True valid
        """
        b, t, d = x.shape
        assert d == self.num_vars, f"Expected D={self.num_vars}, got {d}"

        patches, pad_len = patchify(x, self.patch_len)  # [B,N,P*D]
        n = patches.size(1)
        tok = self.patch_embed(patches)  # [B,N,d_model]
        tok = tok + self.pos_embed[:, :n, :]
        tok = self.ln(tok)

        # token mask: all valid patches except those made entirely from padding
        if mask is None:
            tok_mask = torch.ones((b, n), device=x.device, dtype=torch.bool)
        else:
            # patch-level validity: a patch is valid if it has >=1 valid point
            m = mask
            # pad mask to multiple of patch_len
            if pad_len > 0:
                m = F.pad(m, (0, pad_len), value=False)
            m = m.view(b, n, self.patch_len)
            tok_mask = m.any(dim=-1)

        # TransformerEncoder uses src_key_padding_mask where True means PAD
        src_key_padding_mask = ~tok_mask
        tok = self.encoder(tok, src_key_padding_mask=src_key_padding_mask)
        tok = self.ln(tok)

        if self.use_stat_token:
            stat_vec = self._masked_stats(x, mask)  # [B, 5D+1]
            stat_tok = self.stat_ln(self.stat_mlp(stat_vec)).unsqueeze(1)  # [B,1,d_model]
            tok = torch.cat([stat_tok, tok], dim=1)
            tok_mask = torch.cat([torch.ones((b, 1), device=x.device, dtype=torch.bool), tok_mask], dim=1)

        return tok, tok_mask
