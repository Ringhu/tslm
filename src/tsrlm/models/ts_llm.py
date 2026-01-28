from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .enc_patchtst import PatchTSTEncoder
from .bridge_prefix import PrefixBridge
from .revin import RevIN


class TSReportLM(nn.Module):
    """Time-series -> text model (prefix-bridge baseline)."""

    def __init__(
        self,
        llm_name_or_path: str,
        ts_num_vars: int = 1,
        ts_patch_len: int = 16,
        ts_d_model: int = 256,
        ts_layers: int = 4,
        ts_heads: int = 8,
        prefix_tokens: int = 32,
        freeze_llm: bool = False,
        use_revin: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            # common for causal LMs: use eos as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

        llm_dim = self.llm.config.hidden_size

        self.ts_encoder = PatchTSTEncoder(
            num_vars=ts_num_vars,
            patch_len=ts_patch_len,
            d_model=ts_d_model,
            nhead=ts_heads,
            num_layers=ts_layers,
            use_stat_token=True,
        )
        self.bridge = PrefixBridge(ts_dim=ts_d_model, llm_dim=llm_dim, num_prefix_tokens=prefix_tokens)

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=ts_num_vars, affine=True)

        # Optional extra scale token to preserve original scale when using RevIN
        # vector: mean,std,min,max,delta_pct,length => 5*D+1
        stat_dim = 5 * ts_num_vars + 1
        self.scale_mlp = nn.Sequential(
            nn.Linear(stat_dim, ts_d_model),
            nn.GELU(),
            nn.Linear(ts_d_model, ts_d_model),
            nn.LayerNorm(ts_d_model),
        )

    def _masked_stats(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # same as PatchTSTEncoder._masked_stats but duplicated to avoid tight coupling
        b, t, d = x.shape
        if mask is None:
            m = torch.ones((b, t, 1), device=x.device, dtype=x.dtype)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)

        denom = m.sum(dim=1).clamp_min(1.0)
        mean = (x * m).sum(dim=1) / denom
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-6)

        x_min = x.clone()
        x_max = x.clone()
        if mask is not None:
            invalid = (~mask).unsqueeze(-1)
            x_min = x_min.masked_fill(invalid, float("inf"))
            x_max = x_max.masked_fill(invalid, float("-inf"))
        vmin = x_min.amin(dim=1)
        vmax = x_max.amax(dim=1)

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

        delta_pct = (end - start) / (start.abs() + 1e-6)

        feats = torch.cat([mean, std, vmin, vmax, delta_pct, length], dim=1)
        return feats

    def encode_ts(self, values: torch.Tensor, ts_attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode time series into token embeddings (ts space)."""
        # values: [B,T,D]
        x = values
        # preserve original stats for scale token
        stat_vec = self._masked_stats(x, ts_attn_mask)  # [B,5D+1]
        scale_tok = self.scale_mlp(stat_vec).unsqueeze(1)  # [B,1,ts_d_model]

        if self.use_revin:
            x, _, _ = self.revin(x, ts_attn_mask)

        ts_tokens, ts_mask = self.ts_encoder(x, ts_attn_mask)  # [B,N,ts_d], [B,N]
        # prepend scale token
        ts_tokens = torch.cat([scale_tok, ts_tokens], dim=1)
        ts_mask = torch.cat([torch.ones((ts_mask.size(0), 1), device=ts_mask.device, dtype=torch.bool), ts_mask], dim=1)
        return ts_tokens, ts_mask

    def forward(
        self,
        values: torch.Tensor,
        ts_attn_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Encode TS
        ts_tokens, ts_mask = self.encode_ts(values, ts_attn_mask)
        prefix_embeds, prefix_mask = self.bridge(ts_tokens, ts_mask)  # [B,K,H], [B,K]

        # Token embeddings for text
        tok_embeds = self.llm.get_input_embeddings()(input_ids)  # [B,L,H]
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)  # [B,K+L,H]
        attn = torch.cat([prefix_mask.to(attention_mask.dtype), attention_mask], dim=1)

        if labels is not None:
            # pad labels with -100 for prefix positions
            prefix_labels = torch.full((labels.size(0), prefix_embeds.size(1)), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([prefix_labels, labels], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        return {"loss": out.loss, "logits": out.logits}

    @torch.no_grad()
    def generate(
        self,
        values: torch.Tensor,
        ts_attn_mask: Optional[torch.Tensor],
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        self.eval()
        device = next(self.parameters()).device
        values = values.to(device)
        ts_attn_mask = ts_attn_mask.to(device) if ts_attn_mask is not None else None

        ts_tokens, ts_mask = self.encode_ts(values, ts_attn_mask)
        prefix_embeds, prefix_mask = self.bridge(ts_tokens, ts_mask)

        if prompt == "":
            # some tokenizers cannot handle empty input; use BOS/EOS as a start token
            prompt = self.tokenizer.bos_token or self.tokenizer.eos_token or ""
        tok = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        tok_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        attn = torch.cat([prefix_mask.to(attention_mask.dtype), attention_mask], dim=1)

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # remove the prompt part
        decoded = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return decoded
