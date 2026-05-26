import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.weight


class LayerNorm(nn.LayerNorm):
    """LayerNorm with same constructor signature as RMSNorm for easy swapping."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__(dim, eps=eps)
