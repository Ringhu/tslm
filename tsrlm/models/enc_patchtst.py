from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchify import patchify


class PatchTSTEncoder(nn.Module):
    """
    Lightweight PatchTST-style encoder for representation learning (not forecasting).

    Input:  x [B,T,D]
    Output: tokens [B,N,d_model], token_mask [B,N]
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
        max_patches: int = 4096,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.patch_len = patch_len
        self.d_model = d_model

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
        self.in_ln = nn.LayerNorm(d_model)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B,T,D]
          mask: [B,T] bool, True valid

        Returns:
          tokens: [B,N,d_model]
          tok_mask: [B,N] bool, True valid
        """
        b, t, d = x.shape
        if d != self.num_vars:
            raise ValueError(f"Expected D={self.num_vars}, got {d}")

        patches, pad_len = patchify(x, self.patch_len)  # [B,N,P*D]
        n = patches.size(1)

        tok = self.patch_embed(patches)  # [B,N,d_model]
        tok = tok + self.pos_embed[:, :n, :]
        tok = self.in_ln(tok)

        if mask is None:
            tok_mask = torch.ones((b, n), device=x.device, dtype=torch.bool)
        else:
            m = mask
            if pad_len > 0:
                m = F.pad(m, (0, pad_len), value=False)
            m = m.view(b, n, self.patch_len)
            tok_mask = m.any(dim=-1)

        # TransformerEncoder uses src_key_padding_mask where True means PAD
        tok = self.encoder(tok, src_key_padding_mask=~tok_mask)
        tok = self.out_ln(tok)
        return tok, tok_mask
