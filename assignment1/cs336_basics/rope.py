import torch
import torch.nn as nn
from einops import einsum, rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        half_dim = d_k // 2
        self.theta = 1.0 / (theta ** (torch.arange(0, half_dim, device=device) / half_dim)) # (half_dim, )
        positions = torch.arange(max_seq_len, device=device).float()
        angles = einsum(positions, self.theta, "seq_len, half_dim -> seq_len half_dim")
        
        cos = torch.cos(angles) # (seq_len, half_dim)
        sin = torch.sin(angles) # (seq_len, half_dim)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
            x : shape (..., seq_len, d_k)
        """
        x1, x2 = x[..., 0::2], x[..., 1::2]  # (..., seq_len, d_k/2)
        # gather cos/sin for this sequence
        # token_positions include the maximum seq length, input x could be smaller than it
        cos = self.cos[token_positions][:x.shape[-2],:]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions][:x.shape[-2],:]
        
        # apply rotation
        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)  # (..., seq_len, d_k/2, 2)

        return rearrange(x_rot, "... seq_len d_k l -> ... seq_len (d_k l)")