from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tsrlm.config import TSRLMConfig
from .enc_patchtst import PatchTSTEncoder
from .enc_chronos2 import Chronos2Encoder
from .bridge_prefix import PrefixBridge
from .bridge_xattn import CrossAttentionBridge
from .revin import RevIN
from .stats_tokens import StatsTokenizer


class TSReportLM(nn.Module):
    """
    Time series -> tokens (encoder) -> fixed prefix (bridge) -> causal LM.

    This is **not** a HF PreTrainedModel; we use HF Trainer with `remove_unused_columns=False`.
    """

    def __init__(
        self,
        config: TSRLMConfig,
        freeze_llm: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # --- LLM ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            # make padding safe
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # last resort
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # right padding for causal LM
        self.tokenizer.padding_side = "right"

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        # if we added pad token above, resize embeddings
        if len(self.tokenizer) != self.llm.get_input_embeddings().weight.size(0):
            self.llm.resize_token_embeddings(len(self.tokenizer))

        if freeze_llm:
            self.llm.requires_grad_(False)

        # most causal LMs use hidden_size
        if hasattr(self.llm.config, "hidden_size"):
            llm_dim = int(self.llm.config.hidden_size)
        else:
            # Fallback for models that might use hidden_sizes (e.g. some encoder-decoders or non-standard archs)
            llm_dim = int(self.llm.config.hidden_sizes[0])

        # --- TS encoder ---
        self.encoder_type = config.encoder_type
        if self.encoder_type == "patchtst":
            self.ts_encoder = PatchTSTEncoder(
                num_vars=config.ts_num_vars,
                patch_len=config.ts_patch_len,
                d_model=config.ts_d_model,
                nhead=config.ts_heads,
                num_layers=config.ts_layers,
            )
            enc_out_dim = config.ts_d_model
        elif self.encoder_type == "chronos2":
            self.ts_encoder = Chronos2Encoder(
                model_name_or_path=config.chronos_model_path,
                freeze=True,
            )
            enc_out_dim = self.ts_encoder.output_dim
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        # global stats -> tokens (same dim as encoder output)
        self.stats_tok = StatsTokenizer(
            num_vars=config.ts_num_vars,
            d_model=enc_out_dim,
            n_tokens=config.n_stat_tokens,
            dropout=0.0,
        )

        # optional RevIN (applied BEFORE encoder; stats are computed on raw series)
        self.use_revin = bool(config.use_revin)
        if self.use_revin:
            self.revin = RevIN(num_features=config.ts_num_vars, affine=True)

        # --- Bridge ---
        if config.bridge_type == "prefix":
            self.bridge = PrefixBridge(
                ts_dim=enc_out_dim,
                llm_dim=llm_dim,
                num_prefix_tokens=config.prefix_tokens,
                resampler_layers=config.resampler_layers,
                resampler_heads=config.resampler_heads,
                prefix_alpha_init=config.prefix_alpha_init,
            )
        elif config.bridge_type == "xattn":
            self.bridge = CrossAttentionBridge(ts_dim=enc_out_dim, llm_dim=llm_dim)
        else:
            raise ValueError(f"Unknown bridge_type: {config.bridge_type}")

        # important for training stability
        self.llm.config.use_cache = False

    def encode_ts(self, values: torch.Tensor, ts_attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          ts_tokens: [B, N_total, enc_out_dim]
          ts_mask:   [B, N_total] bool
        """
        # stats from raw
        stat_tokens = self.stats_tok(values, ts_attn_mask)  # [B,S,dim]
        stat_mask = torch.ones(stat_tokens.size()[:2], device=stat_tokens.device, dtype=torch.bool)

        x = values
        if self.use_revin:
            x, _, _ = self.revin(x, ts_attn_mask)

        ts_tokens, ts_mask = self.ts_encoder(x, ts_attn_mask)
        # prepend stats tokens
        ts_tokens = torch.cat([stat_tokens.to(ts_tokens.dtype), ts_tokens], dim=1)
        ts_mask = torch.cat([stat_mask, ts_mask], dim=1)
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
        # TS -> prefix
        ts_tokens, ts_mask = self.encode_ts(values, ts_attn_mask)
        prefix_embeds, prefix_mask = self.bridge(ts_tokens, ts_mask)  # [B,K,H]

        # text embeddings
        tok_embeds = self.llm.get_input_embeddings()(input_ids)  # [B,L,H]
        prefix_embeds = prefix_embeds.to(tok_embeds.dtype)

        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)  # [B,K+L,H]
        attn = torch.cat([prefix_mask.to(attention_mask.dtype), attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), prefix_embeds.size(1)),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
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
        prompt: Optional[Union[str, List[str]]] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 3,
        debug: bool = False,
    ) -> Union[str, List[str]]:
        self.eval()
        device = next(self.parameters()).device

        # ---- normalize batch ----
        if values.dim() == 2:
            values = values.unsqueeze(0)  # [1,T,D]
        B = values.size(0)

        values = values.to(device)
        if ts_attn_mask is not None:
            if ts_attn_mask.dim() == 1:
                ts_attn_mask = ts_attn_mask.unsqueeze(0)
            ts_attn_mask = ts_attn_mask.to(device)

        # ---- normalize prompt(s) ----
        if prompt is None or prompt == "":
            prompt_list = [self.config.default_prompt] * B
            return_list = (B > 1)
        elif isinstance(prompt, list):
            if len(prompt) != B:
                raise ValueError(f"len(prompt)={len(prompt)} but batch size B={B}")
            prompt_list = prompt
            return_list = True
        else:
            prompt_list = [prompt] * B
            return_list = (B > 1)

        # ---- TS -> prefix (already supports batch) ----
        ts_tokens, ts_mask = self.encode_ts(values, ts_attn_mask)
        prefix_embeds, prefix_mask = self.bridge(ts_tokens, ts_mask)  # [B,K,H]

        # ---- tokenize prompts as a batch ----
        tok = self.tokenizer(
            prompt_list,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        input_ids = tok["input_ids"].to(device)              # [B,L]
        attention_mask = tok["attention_mask"].to(device)    # [B,L]

        tok_embeds = self.llm.get_input_embeddings()(input_ids)  # [B,L,H]
        prefix_embeds = prefix_embeds.to(tok_embeds.dtype)

        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)  # [B,K+L,H]
        attn = torch.cat([prefix_mask.to(attention_mask.dtype), attention_mask], dim=1)  # [B,K+L]

        if debug:
            pe = prefix_embeds.float()
            te = tok_embeds.float()
            print("B", B, "prompt_len(max)", input_ids.size(1), "prefix_len", prefix_embeds.size(1))
            print("prefix dtype", prefix_embeds.dtype)
            print("prefix nan", torch.isnan(pe).any().item(), "inf", torch.isinf(pe).any().item())
            print("prefix mean", pe.mean().item(), "std", pe.std().item(), "max", pe.abs().max().item(), "norm", pe.norm().item())
            print("tok   mean", te.mean().item(), "std", te.std().item(), "max", te.abs().max().item(), "norm", te.norm().item())
            if hasattr(self.bridge, "prefix_alpha"):
                print("prefix_alpha", float(self.bridge.prefix_alpha.detach().cpu()))

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6) if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            generation_config=gen_cfg,
        )  # typically [B, seq_len]

        decoded_list = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # ---- strip each prompt prefix ----
        outs: List[str] = []
        for dec, pr in zip(decoded_list, prompt_list):
            text = dec
            if isinstance(pr, str) and text.startswith(pr):
                text = text[len(pr):].lstrip()
            outs.append(text)

        return outs if return_list else outs[0]

    # ---------- saving helpers ----------

    def safe_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        safetensors cannot store shared tensors (tied weights).
        This removes obvious tied duplicates to make saving safe if needed.
        """
        sd = super().state_dict()
        # common tied keys
        # - lm_head.weight
        # - model.embed_tokens.weight (or llm.model.embed_tokens.weight in wrapped modules)
        lm_keys = [k for k in sd.keys() if k.endswith("lm_head.weight")]
        emb_keys = [k for k in sd.keys() if k.endswith("embed_tokens.weight")]
        if lm_keys and emb_keys:
            lm_k = lm_keys[0]
            emb_k = emb_keys[0]
            try:
                if sd[lm_k].data_ptr() == sd[emb_k].data_ptr():
                    del sd[lm_k]
            except Exception:
                pass
        return sd
