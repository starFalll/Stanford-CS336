import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from .linear import Linear
from .rope import RoPE
from .head import scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, 
    d_model: int,
    num_heads: int,
    theta: float | None = None,
    max_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_h = num_heads
        self.max_seq_len = max_seq_len
        self.d_v = d_model // num_heads
        self.rope = RoPE(theta=theta, d_k=self.d_v, max_seq_len=max_seq_len) if theta else None
        self.q_proj = Linear(d_model, self.d_v * num_heads) 
        self.k_proj = Linear(d_model, self.d_v * num_heads) 
        self.v_proj = Linear(d_model, self.d_v * num_heads)  
        self.output_proj = Linear(self.d_v * num_heads, d_model) 
        
    
    def forward(self, 
                x: Float[Tensor, " ... sequence_length d_in"], 
                token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> torch.tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(self.max_seq_len)
            heads = [self.rope(rearrange(item, "... seq_len (h k) -> ... h seq_len k", h=self.d_h), token_positions) for item in [q, k]]
            heads.append(rearrange(v, "... seq_len (h k) -> ... h seq_len k", h=self.d_h))
        else:
            heads = [rearrange(item, "... seq_len (h k) -> ... h seq_len k", h=self.d_h) for item in [q, k, v]]
        multi_heads = scaled_dot_product_attention(heads[0], heads[1], heads[2])
        multi_heads = rearrange(multi_heads, "... h seq_len k -> ... seq_len (h k)")
        return self.output_proj(multi_heads)
