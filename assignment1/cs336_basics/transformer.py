from .rmsnorm import RMSNorm
from .multi_head import CausalMultiHeadSelfAttention
from .positionwise import SwiGLU
from .embedding import Embedding
from .linear import Linear
from .softmax import Softmax
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class TransFormerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    
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
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransFormerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, 
                                                    theta=rope_theta, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        # x = Softmax(x, -2)
        return x
    
    def generate(self, input: Int[Tensor, "batch_size sequence_length"], top_p: float, max_new_tokens, eos_token_id, temperature: float = 1.0) -> torch.tensor:
        original_length = len(input)
        for _ in range(max_new_tokens):
            logits = self(input)
            probs = Softmax(logits, -1, temperature)
            next_token = probs[:, -1, :]
            sorted_probs, sorted_indices = torch.sort(next_token, dim = -1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
            mask = cumulative_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs /= sorted_probs.sum()
            # The output is not always the one with the highest probability, but is random, making the generated text more natural.
            next_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)]
            if next_token == eos_token_id:
                break
            input = torch.cat((input, next_token), dim=-1)
        return input[:, original_length:]


    
