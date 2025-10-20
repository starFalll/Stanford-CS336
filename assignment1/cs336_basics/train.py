import json
import os
import pathlib
from typing import Callable, Iterable, Optional
import typing
import numpy as np
import torch
import torch.nn as nn
import tqdm
from jaxtyping import Float, Int
from torch import Generator, Tensor
from .softmax import Softmax
import math
import numpy.typing as npt
from .transformer import Transformer

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    max_start = len(dataset) - context_length
    start_indices = torch.randint(0, max_start, (batch_size,), device=device)
    dataset_loader = torch.tensor(dataset, dtype=torch.long, device=device)
    sampled_sequence = torch.stack([dataset_loader[i : i + context_length] for i in start_indices])
    target_sequence = torch.stack([dataset_loader[i+1 : i + context_length + 1] for i in start_indices])
    return [sampled_sequence, target_sequence]

def get_val(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str = 'cpu',
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    length = len(dataset)
    batch = (length - context_length - 1) // batch_size
    for i in range(batch):
        start_indices = i * batch_size
        dataset_loader = torch.tensor(dataset, dtype=torch.long, device=device)
        sampled_sequence = torch.stack([dataset_loader[i : i + context_length] for i in range(start_indices, start_indices+batch_size)])
        target_sequence = torch.stack([dataset_loader[i+1 : i + context_length + 1] for i in range(start_indices, start_indices+batch_size)])
        yield [sampled_sequence, target_sequence]

def cross_entropy(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # use log-sum-exp trick to avoid overflow
    max_vals, _ = torch.max(logits, dim=-1, keepdim=True)
    shifted = logits - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim = -1, keepdim=True))
    true_logits = shifted[torch.arange(len(targets)), targets]
    l = -true_logits + log_sum_exp
    return torch.mean(l, dtype=torch.float32)

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    param_list = [p for p in parameters if p.grad is not None]
    if len(param_list) == 0:
        return
    # gradient (for all parameters)
    l2_norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in param_list))
        
    factor = max_l2_norm / (l2_norm + eps)
    if factor < 1.0: 
        for p in param_list:
            p.grad.mul_(factor)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    state = dict()
    state['model'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['iteration'] = iteration
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(state, f)
    else:
        torch.save(state, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            state = torch.load(f)
    else:
        state = torch.load(src)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return state['iteration']


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

CONFIG_PATH = "hyperparameters.json"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.dat")
VAL_DATA_PATH = os.path.join(DATA_DIR, "valid.dat")

def train(device = 'cpu'):
    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = Transformer(**config["model"]).to(device)
    # optimize this computation using compile
    model = torch.compile(model)

    params = {}
    for kv in config.values():
        params.update(kv)
    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    # from params['lr'] to args.lr
    args = DotDict(params)

    # load training and validation data
    train_data = np.memmap(TRAIN_DATA_PATH, np.int32, mode = 'r')
    valid_data = np.memmap(VAL_DATA_PATH, np.int32, mode = 'r')
    
    os.makedirs(args.checkpoint_path, exist_ok=True)    

    # Create optimizer
    optimizer = AdamW(model.parameters(), args.lr, args.betas, args.weight_decay, args.eps)

    # resume checkpoint
    start_iter = 0
    if args.checkpoint:
        print(f"Resume from checkpoint {args.checkpoint}")
        save_path = pathlib.Path(__file__).resolve().parent / f"checkpoints/ckp_iter_{args.checkpoint}.pt"
        start_iter = load_checkpoint(save_path, model, optimizer)
        print(f"Resume iteration:{start_iter}")
    
    for it in tqdm(range(start_iter, args.train_steps), desc = "Training"):
        # -------------training---------------
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        # calculate loss function
        loss = cross_entropy(logits=logits, targets=y)
        # set grad to zero to avoid affecting later calculation
        optimizer.zero_grad()
        loss.backward()
        # to avoid very large grad
        gradient_clipping(model.parameters(), args.max_l2_norm)
        # update learning rate
        lr = learning_rate_schedule(it, args.lr, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr
        
        optimizer.step()

        # -------------validation--------------
        if (it+1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_lossed = []
                count = 0
                for x_val, y_val in get_val(valid_data, args.batch_size, args.context_length):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    loss = cross_entropy(val_logits, y_val)
                    val_lossed.append(loss)
                    count += 1
                    if count >= args.val_batches:
                        break
                val_loss_mean = np.mean(val_lossed)
                print(f"iteration {it}: VALIDATION loss = {val_loss_mean}")

        # -------------save-----------------
        if (it+1) % args.save_interval == 0:
            checkpoint_name = os.path.join(args.checkpoint_path, f"ckp_iter_{it+1}.pt")
            save_checkpoint(model, optimizer, it, checkpoint_name)
            params['train']['checkpoint'] = it+1
            with open(CONFIG_PATH, 'w') as f:
                 json.dump(params, f)
        



    
    