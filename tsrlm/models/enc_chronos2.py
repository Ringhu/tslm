from __future__ import annotations

"""Chronos-2 encoder wrapper (optional / best-effort).

Chronos-2 is a time-series foundation model; there are public checkpoints and a python package
(see Chronos-2 docs). APIs may change, so treat this as an *adapter point*.

Recommended use in this repo:
- Keep Chronos-2 frozen.
- Extract embeddings for the context series and feed into PrefixBridge.

If you prefer a fully stable baseline, use PatchTSTEncoder instead.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union

class Chronos2Encoder(nn.Module):
    """
    Wrapper for Amazon Chronos-2 Foundation Model Encoder.
    Extracts embeddings from the pre-trained model to be used as prefix features for LLM.
    
    Requires: pip install chronos-forecasting>=2.0.0
    """
    def __init__(
        self, 
        model_name_or_path: str = "amazon/chronos-2", 
        device: Optional[str] = None,
        freeze: bool = True
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device_map = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = None
        self._output_dim = None
        self.freeze = freeze

    def _lazy_init(self):
        if self._pipeline is not None:
            return
        
        try:
            # Chronos-2 uses the new Chronos2Pipeline
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting package not found or version too old. "
                "Please run: pip install 'chronos-forecasting>=2.0.0'"
            ) from e

        # Load pipeline (defaulting to bfloat16 for efficiency on Ampere+ GPUs)
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
        )
        
        if self.freeze:
            self._pipeline.model.requires_grad_(False)
            
        # Determine output dimension from the model config
        # Chronos-2 is based on T5-encoder architecture
        self._output_dim = self._pipeline.model.config.d_model

    @property
    def output_dim(self) -> int:
        self._lazy_init()
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: [B, T, D] Input time series values. 
             Note: Chronos-2 natively handles univariate. If D > 1, 
             we currently treat them as batch_size * D univariate series (independent encoding),
             or you can adapt to use Chronos-2's multivariate support if aligned.
             For this implementation, we assume D=1 (Univariate) as per UCR task.
          mask: [B, T] Boolean mask (1=valid, 0=padding).

        Returns:
          embeddings: [B, N, hidden_dim] 
          padding_mask: [B, N] (1=valid, 0=padding)
        """
        self._lazy_init()
        
        B, T, D = x.shape
        
        # 1. Prepare Data for Chronos
        # Chronos expects a list of 1D tensors (or torch.nan for missing values)
        # We will flatten B and D to process everything as univariate series
        # (B, T, D) -> (B*D, T)
        x_flat = x.transpose(1, 2).reshape(B * D, T)
        
        context_list = []
        # Convert to list of tensors, respecting the mask to handle variable lengths
        # Note: Chronos pipeline handles padding automatically if we pass tensors with NaNs 
        # or list of different lengths. We use the mask to slice valid data.
        
        mask_flat = None
        if mask is not None:
            # mask: (B, T) -> replicate for D
            mask_flat = mask.unsqueeze(1).repeat(1, D, 1).reshape(B * D, T)

        for i in range(B * D):
            series = x_flat[i]
            if mask_flat is not None:
                # Keep only valid parts. 
                # Note: This assumes padding is at the end. 
                # If padding is in the middle, Chronos handles NaNs, but here we slice length.
                valid_len = int(mask_flat[i].sum().item())
                if valid_len == 0: 
                    valid_len = 1 # Avoid empty tensor crash
                series = series[:valid_len]
            
            # Ensure CPU for list construction (pipeline handles move to device)
            context_list.append(series.detach().cpu())

        # 2. Extract Embeddings
        # pipeline.embed returns a single tensor of embeddings [Batch, Seq_Len, Dim]
        # It handles tokenization and scaling internally.
        # Note: 'tokenizer_state' is optional/handled internally.
        with torch.no_grad() if self.freeze else torch.enable_grad():
            # embed() returns embeddings corresponding to the tokens
            # The length N depends on T (and tokenization strategy).
            embeddings, tokenizer_state = self._pipeline.embed(context_list)
            
        # embeddings is [B*D, N, out_dim]
        # We need to reshape back to [B, D, N, out_dim] then merge D?
        # Since TSReportLM expects [B, Total_Tokens, Dim], we can flatten D into Sequence.
        
        BD, N, Dim = embeddings.shape
        
        # Reshape to [B, D*N, Dim] so that Perceiver can attend to all variables' features
        embeddings = embeddings.view(B, D, N, Dim)
        embeddings = embeddings.reshape(B, D * N, Dim)
        
        # Construct mask for the embeddings
        # Since Chronos handles padding internally and returns a padded tensor batch,
        # we need to know which are padding.
        # However, pipeline.embed usually returns right-padded tensors.
        # We'll assume the output `embeddings` are valid up to the sequence length produced by the tokenizer.
        # Ideally we'd get a mask from the pipeline, but `embed` currently just returns the tensor.
        # For now, we assume all produced tokens are valid (the Perceiver can handle some padding noise).
        # A stricter implementation would check tokenizer_state or reconstruction.
        
        emb_mask = torch.ones((B, D * N), device=embeddings.device, dtype=torch.bool)
        
        return embeddings, emb_mask