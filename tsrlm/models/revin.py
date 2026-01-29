from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B,T,D]
          mask: [B,T] bool True valid

        Returns:
          x_norm, mean [B,1,D], std [B,1,D]
        """
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (x * m).sum(dim=1, keepdim=True) / denom
            var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True) / denom

        std = torch.sqrt(var + self.eps)
        x_norm = (x - mean) / std

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm, mean, std

    def denorm(self, x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x_norm = (x_norm - self.beta) / (self.gamma + self.eps)
        return x_norm * std + mean
