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
    pad_shape = [0, 0] * x.dim()
    # F.pad uses last dims first; we handle only dim=1 (time) for [B,T,D]
    if dim != 1:
        raise NotImplementedError("Only dim=1 supported for now.")
    # pad format: (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right, ...)
    pad = (0, 0, 0, pad_len)  # pad T on the right
    return F.pad(x, pad, value=value), pad_len


def patchify(x: torch.Tensor, patch_len: int) -> Tuple[torch.Tensor, int]:
    """Patchify a time series.

    Args:
      x: [B, T, D]
      patch_len: int

    Returns:
      patches: [B, N, patch_len*D]
      pad_len: how many padded points were appended
    """
    x, pad_len = pad_to_multiple(x, multiple=patch_len, dim=1, value=0.0)
    b, t, d = x.shape
    n = t // patch_len
    patches = x.view(b, n, patch_len, d).reshape(b, n, patch_len * d)
    return patches, pad_len
