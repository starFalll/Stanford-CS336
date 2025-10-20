import torch 

def Softmax(x: torch.Tensor, dim: int, temperature: float = 1.0) -> torch.Tensor:
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    exp = torch.exp((x-max_vals)/temperature)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp
