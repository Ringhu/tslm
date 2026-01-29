from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TSRLMConfig:
    # LLM
    llm_name_or_path: str

    # TS encoder
    encoder_type: str = "patchtst"      # patchtst | chronos2
    ts_num_vars: int = 1
    ts_patch_len: int = 16
    ts_d_model: int = 256
    ts_layers: int = 4
    ts_heads: int = 8

    # Bridge
    bridge_type: str = "prefix"         # prefix | xattn (stub)
    prefix_tokens: int = 32
    resampler_layers: int = 2
    resampler_heads: int = 8

    # Extra TS tokens
    n_stat_tokens: int = 2              # how many learned tokens to represent global stats
    use_revin: bool = False

    # Prefix scaling
    prefix_alpha_init: float = 0.1      # start small to avoid destabilizing the LLM

    # Chronos
    chronos_model_path: str = "amazon/chronos-2"

    # Prompting
    default_prompt: str = "You are a time-series captioning assistant. Describe the given time series."

    # Text
    max_text_length: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TSRLMConfig":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
