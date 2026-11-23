"""Small dependency-free segment reductions used by safe CUDA paths."""

from __future__ import annotations

import torch
from typing import Optional


def _expanded_index(index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return index.view(
        (index.size(0),) + (1,) * (value.dim() - 1)
    ).expand_as(value)


def segment_sum(
        value: torch.Tensor,
        index: torch.Tensor,
        dim_size: Optional[int] = None,
        dim: int = 0) -> torch.Tensor:
    """Sum values sharing a first-dimension segment id."""
    if int(dim) != 0:
        raise ValueError('segment_sum only supports dim=0')
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    output = value.new_zeros((int(dim_size),) + tuple(value.shape[1:]))
    output.scatter_add_(0, _expanded_index(index, value), value)
    return output


def segment_mean(value, index, dim=0, dim_size=None):
    """Average values sharing a first-dimension segment id."""
    total = segment_sum(value, index, dim_size=dim_size, dim=dim)
    count = segment_sum(torch.ones_like(value), index, dim_size=dim_size, dim=dim)
    return total / count.clamp(min=1)


def segment_softmax(
        value: torch.Tensor,
        index: torch.Tensor,
        dim_size: Optional[int] = None,
        dim: int = 0) -> torch.Tensor:
    """Numerically stable first-dimension segment softmax."""
    if int(dim) != 0:
        raise ValueError('segment_softmax only supports dim=0')
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    output_shape = (int(dim_size),) + tuple(value.shape[1:])
    maximum = value.new_full(output_shape, float('-inf'))
    maximum.scatter_reduce_(
        0,
        _expanded_index(index, value),
        value,
        reduce='amax',
        include_self=True,
    )
    exponent = torch.exp(value - maximum[index])
    denominator = segment_sum(exponent, index, dim_size)
    return exponent / denominator[index].clamp(min=1.0e-12)
