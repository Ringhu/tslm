from __future__ import annotations

"""Chronos-2 encoder wrapper (optional / best-effort).

Chronos-2 is a time-series foundation model; there are public checkpoints and a python package
(see Chronos-2 docs). APIs may change, so treat this as an *adapter point*.

Recommended use in this repo:
- Keep Chronos-2 frozen.
- Extract embeddings for the context series and feed into PrefixBridge.

If you prefer a fully stable baseline, use PatchTSTEncoder instead.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Chronos2Encoder(nn.Module):
    def __init__(self, model_name_or_path: str = "amazon/chronos-2", device: Optional[str] = None) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device
        self._pipeline = None

    def _lazy_init(self):
        if self._pipeline is not None:
            return
        try:
            from chronos import Chronos2Pipeline  # pip install chronos-forecasting
        except Exception as e:
            raise ImportError(
                "chronos-forecasting not installed or Chronos2Pipeline missing. " 
                "Try: pip install chronos-forecasting"
            ) from e

        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name_or_path,
            device_map=self.device or "auto",
            torch_dtype=torch.bfloat16,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
          x: [B,T,D] (D must be 1 for this stub)
          mask: [B,T] bool

        Returns:
          tokens: [B,N,ts_dim]
          token_mask: [B,N] bool

        Notes:
        - This stub assumes Chronos2Pipeline exposes an embedding method.
          If not, you can directly use the underlying HF encoder model.
        """
        self._lazy_init()
        if x.size(-1) != 1:
            raise ValueError("Chronos2Encoder stub only supports univariate inputs (D=1)")

        # Convert to list[Tensor] as many Chronos pipelines expect 1D sequences.
        series_list = []
        for i in range(x.size(0)):
            if mask is None:
                series_list.append(x[i, :, 0].detach().cpu())
            else:
                t = int(mask[i].sum().item())
                series_list.append(x[i, :t, 0].detach().cpu())

        # Best-effort: try `embed` then fallback to `encode`.
        pipe = self._pipeline
        if hasattr(pipe, "embed"):
            emb = pipe.embed(series_list)  # expected [B,N,ts_dim]
        elif hasattr(pipe, "encode"):
            emb = pipe.encode(series_list)
        else:
            raise AttributeError("Chronos2Pipeline has no embed/encode method. Please adapt this wrapper.")

        if isinstance(emb, torch.Tensor):
            tokens = emb.to(x.device)
        else:
            tokens = torch.tensor(emb, device=x.device)

        token_mask = torch.ones(tokens.size()[:2], device=tokens.device, dtype=torch.bool)
        return tokens, token_mask
