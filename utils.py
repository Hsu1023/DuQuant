import torch
from math import inf
import logging
from termcolor import colored
import sys
import os
import time
import pickle
from tqdm import tqdm
import math


@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def exchange_row_col(_tensor, i, j):
    tensor = _tensor.detach().clone()
    assert isinstance(tensor, torch.Tensor)
    indices_row = torch.arange(tensor.size(0))
    indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
    tensor = tensor[indices_row]

    indices_col = torch.arange(tensor.size(1))
    indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
    tensor = tensor[:, indices_col]
    return tensor

Rot = {}
def get_rot(n, device='cpu'):
    try:
        if Rot.get(n) is None:
            Rot[n] = pickle.load(open("Rot.pkl", "rb"))[n]
        R = Rot[n].to(device)
        random_matrix = torch.randn(n-1, n-1).to(device)
        q, r = torch.linalg.qr(random_matrix)
        q = torch.cat([torch.zeros(n-1, 1).to(device), q], dim=1)
        q = torch.cat([torch.zeros(1, n).to(device), q], dim=0)
        q[0, 0] = 1
        R = torch.matmul(R,q)
        return R
    except Exception as e:
        print(e)
        assert False, 'No such rotate matrix'


def get_hadamard(n): 
    if n == 1:
        return torch.tensor([[1.]], dtype=torch.float32)
    else:
        assert n % 1 == 0, "The size should be divided by 2."
        H_n_minus_1 = get_hadamard(n//2)
        return torch.cat([torch.cat([H_n_minus_1, H_n_minus_1], dim=1),
                          torch.cat([H_n_minus_1, -H_n_minus_1], dim=1)], dim=0) / math.sqrt(2)
