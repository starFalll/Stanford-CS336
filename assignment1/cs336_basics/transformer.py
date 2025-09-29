from .rmsnorm import RMSNorm
from .multi_head import CausalMultiHeadSelfAttention
from .positionwise import SwiGLU
from .embedding import Embedding
from .linear import Linear
# from .softmax import Softmax
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class TransFormerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> torch.tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x



class Transformer(nn.Module):
    def __init__(
    self, 
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
):
        super().__init__()
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransFormerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                                                    theta=rope_theta, max_seq_len=context_length) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> torch.tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        # x = Softmax(x, -2)
        return x
    
