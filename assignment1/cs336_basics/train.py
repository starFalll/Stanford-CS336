from typing import Callable, Optional
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from .softmax import Softmax
import math

def cross_entropy(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # use log-sum-exp trick to avoid overflow
    max_vals, _ = torch.max(logits, dim=-1, keepdim=True)
    shifted = logits - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim = -1, keepdim=True))
    true_logits = shifted[torch.arange(len(targets)), targets]
    l = -true_logits + log_sum_exp
    return torch.mean(l, dtype=torch.float32)



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps):
        """
        lr: learning rate
        betas: hyperparameters
        weight_decay: decay rate
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m = state.get("m", torch.empty(p.shape))
                v = state.get("v", torch.empty(p.shape))
                t = state.get("t", 1)
                m = beta1 * m + (1-beta1)*grad
                v = beta2 * v + (1-beta2)*(grad**2)
                lrt = lr * math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data -= lrt * m / (torch.sqrt(v) + eps)
                p.data = p.data-lr*weight_decay*p.data 
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss