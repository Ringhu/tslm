from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .enc_patchtst import PatchTSTEncoder
from .enc_chronos2 import Chronos2Encoder
from .bridge_prefix import PrefixBridge
from .bridge_xattn import CrossAttentionBridge
from .revin import RevIN


class TSReportLM(nn.Module):
    """Time-series (supports PatchTST and Chronos-2 encoders) -> text model (prefix-bridge baseline)."""

    def __init__(
        self,
        llm_name_or_path: str,
        # 新增/修改参数
        encoder_type: str = "patchtst",      # patchtst | chronos2
        bridge_type: str = "prefix",         # prefix | xattn
        chronos_model_path: str = "amazon/chronos-2",
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

        # 1. Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
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

        # 2. Select Encoder
        self.encoder_type = encoder_type
        if self.encoder_type == "patchtst":
            self.ts_encoder = PatchTSTEncoder(
                num_vars=ts_num_vars,
                patch_len=ts_patch_len,
                d_model=ts_d_model,
                nhead=ts_heads,
                num_layers=ts_layers,
                use_stat_token=True,
            )
            # PatchTST 输出维度就是 ts_d_model
            enc_out_dim = ts_d_model
            
        elif self.encoder_type == "chronos2":
            self.ts_encoder = Chronos2Encoder(
                model_name_or_path="amazon/chronos-2", # Can be parameterized if needed
                freeze=True # Usually we freeze the foundation model encoder
            )
            # We need to initialize lazy to get dim, or hardcode/infer. 
            # Accessing output_dim property triggers lazy_init.
            enc_out_dim = self.ts_encoder.output_dim
            
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        # 3. Select Bridge
        if bridge_type == "prefix":
            self.bridge = PrefixBridge(
                ts_dim=enc_out_dim,  # 使用 Encoder 的实际输出维度
                llm_dim=llm_dim,
                num_prefix_tokens=prefix_tokens
            )
        elif bridge_type == "xattn":
            self.bridge = CrossAttentionBridge(
                ts_dim=enc_out_dim,
                llm_dim=llm_dim
            )
        else:
            raise ValueError(f"Unknown bridge_type: {bridge_type}")

        # 4. RevIN & Scale Token
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=ts_num_vars, affine=True)

        # Scale MLP (always init to avoid load_state errors if feasible, or condition it)
        # 为了兼容性，我们只在 encoder_type='patchtst' 时强依赖这个统计特征
        # Chronos 自带缩放处理，但为了 Baseline 统一，我们保留这个逻辑
        stat_dim = 5 * ts_num_vars + 1
        self.scale_mlp = nn.Sequential(
            nn.Linear(stat_dim, enc_out_dim), # 映射到 Encoder 输出空间以便 concat
            nn.GELU(),
            nn.Linear(enc_out_dim, enc_out_dim),
            nn.LayerNorm(enc_out_dim),
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
        x = values
        
        # 1. 提取统计特征用于 Scale Token (无论用什么 Encoder，这都是额外的显式信息)
        # 注意：这里复用了 PatchTST 的 _masked_stats 逻辑，如果类里没定义需要把那个 helper 函数搬进来
        # 或者从外部 import。为简洁，假设你已经在类里保留了 _masked_stats
        stat_vec = self._masked_stats(x, ts_attn_mask)
        scale_tok = self.scale_mlp(stat_vec).unsqueeze(1) 

        # 2. RevIN Normalize (Optional)
        if self.use_revin:
            x, _, _ = self.revin(x, ts_attn_mask)

        # 3. Encoder Forward
        # 不同的 Encoder 接口可能略有不同，这里统一为 (x, mask) -> (tokens, mask)
        ts_tokens, ts_mask = self.ts_encoder(x, ts_attn_mask)

        # 4. Prepend Scale Token
        # 确保 device 和 dtype 一致
        scale_tok = scale_tok.to(ts_tokens.dtype)
        ts_tokens = torch.cat([scale_tok, ts_tokens], dim=1)
        
        ts_mask_pad = torch.ones((ts_mask.size(0), 1), device=ts_mask.device, dtype=torch.bool)
        ts_mask = torch.cat([ts_mask_pad, ts_mask], dim=1)
        
        return ts_tokens, ts_mask

    def state_dict(self, *args, **kwargs):
        # 1. 获取标准的 state_dict
        state_dict = super().state_dict(*args, **kwargs)
        
        # 2. 定义需要检查的共享权重对（根据你的报错信息）
        # 报错指出的键是 'llm.lm_head.weight' 和 'llm.model.embed_tokens.weight'
        # 我们通常保留 embed_tokens (作为 source)，移除 lm_head (作为副本)
        key_to_remove = "llm.lm_head.weight"
        key_source = "llm.model.embed_tokens.weight"

        # 3. 检查并移除重复引用的权重
        if key_to_remove in state_dict and key_source in state_dict:
            # 确认它们确实指向同一个内存对象
            if state_dict[key_to_remove] is state_dict[key_source]:
                del state_dict[key_to_remove]
        
        return state_dict

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
        
        # --- 新增：确保类型一致 (float32 -> bfloat16) ---
        prefix_embeds = prefix_embeds.to(tok_embeds.dtype)
        # ---------------------------------------------
        
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
            # Qwen3 tokenizer_config: bos_token is null, eos_token is <|im_end|>
            # so DO NOT fall back to eos_token here.
            if self.tokenizer.bos_token_id is not None:
                input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
                attention_mask = torch.ones_like(input_ids)
                tok_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                tok = self.tokenizer(" ", return_tensors="pt", add_special_tokens=False)
                input_ids = tok["input_ids"].to(device)
                attention_mask = tok["attention_mask"].to(device)
                tok_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
            tok = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            tok_embeds = self.llm.get_input_embeddings()(input_ids)
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        tok_embeds = self.llm.get_input_embeddings()(input_ids)
        
        prefix_embeds = prefix_embeds.to(tok_embeds.dtype)
        prefix_embeds = 0.03 * prefix_embeds   # 先用 0.03（≈ 1/35）

        with torch.no_grad():
            pe = prefix_embeds
            print("prefix dtype", pe.dtype)
            print("prefix nan", torch.isnan(pe).any().item(), "inf", torch.isinf(pe).any().item())
            print("prefix mean", pe.mean().item(), "std", pe.std().item(),
                "max", pe.abs().max().item(), "norm", pe.norm(dim=-1).mean().item())

            te = tok_embeds
            print("tok   mean", te.mean().item(), "std", te.std().item(),
                "max", te.abs().max().item(), "norm", te.norm(dim=-1).mean().item())

        # inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        # attn = torch.cat([prefix_mask.to(attention_mask.dtype), attention_mask], dim=1)
        inputs_embeds = tok_embeds
        attn = attention_mask
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
        # decoded = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # return decoded
        new_ids = gen_ids[0, input_ids.size(1):]
        decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return decoded.strip()
