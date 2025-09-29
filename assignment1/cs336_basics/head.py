import torch
import torch.nn as nn
from einops import einsum
from .softmax import Softmax

def scaled_dot_product_attention(query: torch.tensor, key: torch.tensor, value: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
    """
    key/query: (batch_size, ..., seq_len, d_k)
    value: (batch_size, ..., seq_len, d_v)
    mask: (seq_len, seq_len)
    """
    node = einsum(query, key, "... n d_k, ... m d_k -> ... n m")
    node = node * key.shape[-1] ** -0.5
    if mask is None:
        mask = torch.tril(torch.ones((key.shape[-2], query.shape[-2])))
    node.masked_fill_(mask==0, float('-inf'))
    node = Softmax(node, -1)
    head = einsum(node, value, "... n m, ... m d -> ... n d")
    return head




