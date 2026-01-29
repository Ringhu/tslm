from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class Chronos2Encoder(nn.Module):
    """
    Best-effort wrapper for Amazon Chronos-2 encoder embeddings.

    Requirements:
      pip install "chronos-forecasting>=2.0.0"

    Recommended usage in this repo:
      - keep it frozen
      - use its embeddings as TS tokens for the PrefixBridge
    """

    def __init__(self, model_name_or_path: str = "amazon/chronos-2", device: Optional[str] = None, freeze: bool = True):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device_map = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze = freeze
        self._pipeline = None
        self._output_dim: Optional[int] = None

    def _lazy_init(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from chronos import Chronos2Pipeline
        except Exception as e:
            raise ImportError(
                "chronos-forecasting is required for encoder_type='chronos2'. "
                "Install with: pip install 'chronos-forecasting>=2.0.0'"
            ) from e

        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if self.freeze:
            self._pipeline.model.requires_grad_(False)

        # Chronos-2 is based on an encoder model with d_model
        self._output_dim = int(self._pipeline.model.config.d_model)

    @property
    def output_dim(self) -> int:
        self._lazy_init()
        assert self._output_dim is not None
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B,T,D] (Chronos is primarily univariate; we flatten D into batch)
          mask: [B,T] bool True valid (assumes right padding)

        Returns:
          tokens: [B, N_total, output_dim]
          tok_mask: [B, N_total] bool (currently all-True as a safe fallback)
        """
        self._lazy_init()
        B, T, D = x.shape

        # flatten (B,D) as separate series
        x_flat = x.transpose(1, 2).reshape(B * D, T)
        mask_flat = None
        if mask is not None:
            mask_flat = mask.unsqueeze(1).repeat(1, D, 1).reshape(B * D, T)

        context_list: List[torch.Tensor] = []
        for i in range(B * D):
            series = x_flat[i]
            if mask_flat is not None:
                valid_len = int(mask_flat[i].sum().item())
                valid_len = max(valid_len, 1)
                series = series[:valid_len]
            context_list.append(series.detach().cpu())

        with torch.no_grad() if self.freeze else torch.enable_grad():
            emb, _ = self._pipeline.embed(context_list)  # [B*D, N, H]

        BD, N, H = emb.shape
        emb = emb.view(B, D, N, H).reshape(B, D * N, H)
        tok_mask = torch.ones((B, D * N), device=emb.device, dtype=torch.bool)
        return emb, tok_mask
