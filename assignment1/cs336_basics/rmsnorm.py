import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, device = device, dtype = dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x.to(torch.float32)
        std = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / std * self.weight
        return result.to(in_dtype)
        
