from __future__ import annotations

import torch.nn as nn


class CrossAttentionBridge(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Cross-attention bridge is architecture-specific. "
            "Keep prefix-bridge as your first stable baseline."
        )
