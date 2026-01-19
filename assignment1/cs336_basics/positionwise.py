import torch
import torch.nn as nn
from einops import einsum
from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_input = self.w1(x)
        silu = first_input*torch.sigmoid(first_input)
        h = self.w3(x)
        h = silu * h
        ffn = self.w2(h)
        return ffn
