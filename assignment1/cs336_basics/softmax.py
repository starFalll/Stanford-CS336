import torch 

def Softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    exp = torch.exp(x-max_vals)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp
