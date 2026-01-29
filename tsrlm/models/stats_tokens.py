from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def masked_stats(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute per-variable stats:
      mean, std, min, max, delta_pct, length
    Returns: [B, 5*D + 1]
    """
    b, t, d = x.shape
    if mask is None:
        m = torch.ones((b, t, 1), device=x.device, dtype=x.dtype)
    else:
        m = mask.unsqueeze(-1).to(x.dtype)

    denom = m.sum(dim=1).clamp_min(1.0)  # [B,1]
    mean = (x * m).sum(dim=1) / denom  # [B,D]
    var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
    std = torch.sqrt(var + 1e-6)

    x_min = x.clone()
    x_max = x.clone()
    if mask is not None:
        invalid = (~mask).unsqueeze(-1)
        x_min = x_min.masked_fill(invalid, float("inf"))
        x_max = x_max.masked_fill(invalid, float("-inf"))
    vmin = x_min.amin(dim=1)  # [B,D]
    vmax = x_max.amax(dim=1)  # [B,D]

    if mask is None:
        start = x[:, 0]
        end = x[:, -1]
        length = torch.full((b, 1), t, device=x.device, dtype=x.dtype)
    else:
        lengths = mask.sum(dim=1).clamp_min(1)
        start = x[:, 0]
        end_idx = (lengths - 1).view(b, 1, 1).expand(-1, 1, d)
        end = x.gather(dim=1, index=end_idx).squeeze(1)
        length = lengths.to(x.dtype).view(b, 1)

    delta_pct = (end - start) / (start.abs() + 1e-6)  # [B,D]
    feats = torch.cat([mean, std, vmin, vmax, delta_pct, length], dim=1)  # [B,5D+1]
    return feats


class StatsTokenizer(nn.Module):
    """
    Convert numeric global stats into a small set of learned tokens.

    Output: [B, n_tokens, d_model]
    """

    def __init__(self, num_vars: int, d_model: int, n_tokens: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.n_tokens = n_tokens
        stat_dim = 5 * num_vars + 1

        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_tokens * d_model),
        )
        # important: keep the scale small initially
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        feats = masked_stats(x, mask)  # [B,stat_dim]
        tok = self.mlp(feats)  # [B, n_tokens*d_model]
        tok = tok.view(x.size(0), self.n_tokens, self.d_model)
        tok = self.ln(tok)
        tok = self.drop(tok)
        return tok
