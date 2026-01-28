from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Reference: Kim et al., ICLR 2022.
    This module normalizes each sample independently:
      x_norm = (x - mean) / (std + eps)
    and provides a reversible denorm.

    In this repo we mainly use it as:
      - stabilize encoder training
      - but we also *expose* (mean,std,...) as extra tokens/stats for the LLM
        to avoid losing scale information.
    """

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
        """Normalize.

        Args:
          x: [B, T, D]
          mask: [B, T] bool, True for valid points.

        Returns:
          x_norm, mean [B,1,D], std [B,1,D]
        """
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)  # [B,T,1]
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
