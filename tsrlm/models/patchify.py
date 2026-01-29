from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = 1, value: float = 0.0) -> Tuple[torch.Tensor, int]:
    """Pad tensor along dim to length multiple of `multiple`. Returns padded tensor and pad_len."""
    length = x.size(dim)
    rem = length % multiple
    pad_len = (multiple - rem) % multiple
    if pad_len == 0:
        return x, 0
    if dim != 1:
        raise NotImplementedError("Only dim=1 supported for now.")
    pad = (0, 0, 0, pad_len)  # pad T on the right for [B,T,D]
    return F.pad(x, pad, value=value), pad_len


def patchify(x: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, int]:
    """Patchify [B,T,D] -> [B,N,patch_len*D]."""
    x, pad_len = pad_to_multiple(x, multiple=patch_len, dim=1, value=0.0)
    b, t, d = x.shape
    n = t // patch_len
    patches = x.view(b, n, patch_len, d).reshape(b, n, patch_len * d)
    return patches, pad_len
