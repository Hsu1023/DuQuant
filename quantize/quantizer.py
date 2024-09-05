import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import math
from utils import get_rot, exchange_row_col, get_hadamard
from quantize.const import CLIPMAX, CLIPMIN
import random



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        swc=None,
        lac=None,
        act_group_size=None,
        quant_method=None,
        block_size=128,
        rotate=True,
        max_rotation_step=1024,
        permutation_times=1,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rotate = rotate
        self.max_rotation_step = max_rotation_step
        self.quant_method = quant_method
        

        init_value = 4.             # init value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        self.is_weight = shape != None
        self.permutation_times = permutation_times
        self.recorded_x_max = None
        self.let_s = None
        self.act_group_size = act_group_size
        self.lac = lac
        self.swc = swc

        self.init_duquant_params = torch.tensor(1)

        if block_size == -1:
            self.block_size = 4096
        else:
            self.block_size = block_size

        if self.rotate is None:
            self.H = get_hadamard(self.block_size)
        elif self.quant_method == 'duquant':
            self.R, self.permutation_list = [], []
            if self.rotate is not False:
                self.init_duquant_params = torch.tensor(0)

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
            
        x_int = round_ste(x.float() / scale).half()    # avoid overflow
        
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax) 

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant  
    
    def permutation_random(self, weight, other=None):
        hidden_dim = weight.shape[-1]
        _mean = {}
        _weight = weight.detach().clone().abs()
        for _ in range(hidden_dim):
            _mean[_] = torch.max(_weight[:, _]).item()
        _mean = sorted(_mean.items(), key=lambda x: x[1], reverse=True)
        top_k = weight.shape[1] // self.block_size
        top_k_channel = []
        paired_list = []


        l = list(set(range(weight.shape[1])))
        random.shuffle(l)
        top_k_channel = top_k_channel + l[len(l)//2 :]
        paired_list = paired_list + l[:len(l)//2]

        top_k_channel = torch.tensor(top_k_channel)
        paired_list = torch.tensor(paired_list)

        ans = []
        top_k_channel, paired_list = top_k_channel.tolist(), paired_list.tolist()
        for i in range(hidden_dim):
            if i in top_k_channel:
                ans.append(paired_list[top_k_channel.index(i)])
            else:
                ans.append(top_k_channel[paired_list.index(i)])
        weight = weight[:, ans]
        return weight, torch.tensor(ans)
 
    def permutation_zigzag(self, weight):
        weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        weight_max = weight.abs().max(dim=0).values
        weight_mean = weight_max.mean().item()
        pairs = [(i, weight_max[i].item()) for i in range(hidden_dim)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        def zigzag(numbers):
            cur = 0
            up = True
            l = [[] for i in range(hidden_dim // self.block_size)]
            for i in range(len(numbers)):
                l[cur].append(numbers[i])
                if up:
                    cur += 1
                    if cur == len(l):
                        cur -= 1
                        up = False
                else:
                    cur -= 1
                    if cur == -1:
                        cur += 1
                        up = True
            return l
        pairs = zigzag(pairs)

        for i in range(len(pairs)):
            pairs[i].sort(key=lambda x: x[1], reverse=True)

        perm = torch.zeros(hidden_dim, dtype=torch.long)
        for i in range(len(pairs)):
            perm[i * self.block_size:(i+1) * self.block_size] = torch.tensor([_[0] for _ in pairs[i]])
        weight = weight[:, perm]
        return weight, perm

    def rotation(self, weight, max_rotation_step=None, other=None, score_func=None):
        if max_rotation_step is None:
            max_rotation_step = self.max_rotation_step
        weight = weight.detach().clone()
        _weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        exchange_ids = []
        peak_values = []
        weight = weight.reshape(-1, self.block_size)
        if other is not None:
            _other = other.detach().clone()
            other = other.reshape(-1, self.block_size)


        Rot = get_rot(self.block_size, weight.device)
        for j in range(max_rotation_step):
            if score_func is not None:
                weight_max = weight.abs().max(dim=0).values
                other_max = other.abs().max(dim=0).values
                r = score_func(weight_max, other_max).argmax().item()
                peak_values.append(score_func(weight_max[r], other_max[r]).item())
                
            else:
                r, c = divmod(weight.argmax().item(), weight.shape[-1])
                r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
                peak_values.append((weight[r, c] - weight[r2, c2]).item())
            exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
            exchange_ids.append(exchange_id)
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(weight)
            weight = torch.matmul(weight, R)
            if other is not None:
                other = torch.matmul(other, R)

        if score_func is not None:
            weight_max = weight.abs().max(dim=0).values
            other_max = other.abs().max(dim=0).values
            r = score_func(weight_max, other_max).argmax().item()
            peak_values.append(score_func(weight_max[r], other_max[r]).item())
        else:
            r, c = divmod(weight.argmax().item(), weight.shape[-1])
            r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
            peak_values.append((weight[r, c] - weight[r2, c2]).item())
        exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
        exchange_ids.append(exchange_id)

        weight = _weight.detach().clone()
        if other is not None:
            other = _other.detach().clone()
        select_length = torch.argmin(torch.tensor(peak_values)).item()
        exchange_ids = exchange_ids[:select_length]
        peak_values = peak_values[:select_length+1]

        R_ = torch.eye(self.block_size).to(weight)
        for exchange_id in exchange_ids:
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(R_)
            R_ = torch.matmul(R_, R)
        weight = torch.matmul(weight.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        if other is not None:
            other = torch.matmul(other.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        return (weight, exchange_ids, R_) if other is None else (weight, other, exchange_ids, peak_values[select_length], R_)
    
    def calculate_std(self, weight):
        weight = weight.abs().max(dim=0).values
        groups = [weight[j * self.block_size: (j+1) * self.block_size] for j in range(weight.shape[0] // self.block_size)]
        group_means = [sum(group) / len(group) for group in groups]
        mean = sum(group_means) / len(group_means)
        variance = sum((x - mean) ** 2 for x in group_means) / len(group_means)
        return math.sqrt(variance)

    def online_duquant_cali(self, weight):
        weight = weight.detach().clone()
        T = {}
        
        self.permutation_list = None
        self.R = None
        for i in range(self.permutation_times):
            weight, _, R = self.rotation(weight)
            if self.R is None:
                self.R = R.unsqueeze(0)
            else:
                self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
            
            weight, perm = self.permutation_zigzag(weight)
            if self.permutation_list is None:
                self.permutation_list = perm.unsqueeze(0)
            else:
                self.permutation_list = torch.cat((self.permutation_list, perm.unsqueeze(0)), dim=0)

        weight, _, R = self.rotation(weight)
        if self.R is None:
            self.R = R.unsqueeze(0)
        else:
            self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
        return weight

    def init_duquant(self, x: torch.Tensor):
        if self.quant_method is None:
            return x
        if self.rotate is None:
                x_shape = x.shape   # (n_tokens, hidden_dim) / (out_features, in_features)
                hadamard = self.H.to(x)
                x = x.reshape(-1, self.block_size)
                x = x.matmul(hadamard).view(x_shape)
        elif self.quant_method == 'duquant':
            if self.rotate:
                if not self.init_duquant_params:
                    x = self.online_duquant_cali(x)
                    self.init_duquant_params = torch.tensor(1)
                else:
                    x_size = x.shape
                    x_type = x.dtype
                    if self.permutation_list is not None:
                        for i in range(len(self.permutation_list)):
                            x = x.reshape(-1, self.block_size)
                            R = self.R[i].to(x)
                            x = x.matmul(R).reshape(x_size).squeeze(0)
                            # if False:
                            if True:
                                if len(self.permutation_list.shape) == 3:
                                    perm = (self.permutation_list[i, 0].to(x.device), self.permutation_list[i, 1].to(x.device))
                                    x[:, perm[0]], x[:, perm[1]] = x[:, perm[1]], x[:, perm[0]]
                                else:
                                    perm = self.permutation_list[i].to(x.device)
                                    x = x[:, perm]
                    if len(self.R) > 0:
                        x = x.reshape(-1, self.block_size)
                        R = self.R[-1].to(x)
                        x = x.matmul(R).reshape(x_size) 
        else:
            raise NotImplementedError
        return x
            

    def forward(self, x: torch.Tensor, return_no_quant=False):
        if hasattr(self, 'smooth_scales'):
            x /= self.smooth_scales.to(x.device)

        # if self.n_bits >= 16 or not self.enable:
        #     return x
        # if self.metric == "fix0to1":
        #     return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
            
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            x = self.init_duquant(x)

        if return_no_quant:
            reduce_shape = [-1]
            xmin = x.amin(reduce_shape, keepdim=True)
            xmax =  x.amax(reduce_shape, keepdim=True)
            if self.swc:
                xmax = self.swc*xmax
                xmin = self.swc*xmin
            elif self.lwc:
                xmax = self.sigmoid(self.upbound_factor)*xmax
                xmin = self.sigmoid(self.lowbound_factor)*xmin
            if self.lac:
                xmax = self.lac*xmax
                xmin = self.lac*xmin
            return x
        
        if self.recorded_x_max is None:
            self.recorded_x_max = x.abs().reshape(-1, x.shape[-1]).max(axis=0).values
        if self.let_s is not None:
            x /= self.let_s

        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True).to(x.device)
        xmax =  x.amax(reduce_shape, keepdim=True).to(x.device)
        if self.swc:
            xmax = self.swc*xmax
            xmin = self.swc*xmin
        elif self.lwc:
            xmax = self.sigmoid(self.upbound_factor.to(x.device))*xmax
            xmin = self.sigmoid(self.lowbound_factor.to(x.device))*xmin
        elif self.lac:
            xmax = self.lac*xmax
            xmin = self.lac*xmin

        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def register_duquant_params(self):
        if self.rotate is not True:
            return
        permutation_list, R = self.permutation_list, self.R
        delattr(self, 'R')
        delattr(self, 'permutation_list')
        delattr(self, 'init_duquant_params')
        self.register_buffer('permutation_list', permutation_list)
        self.register_buffer('R', R)
        self.register_buffer('init_duquant_params', torch.tensor(1))

    def copy_duquant_params(self, quantizer_ref):
        if quantizer_ref.rotate is True:
            assert quantizer_ref.init_duquant_params == True
            self.R = quantizer_ref.R.clone().detach()
            try:
                self.permutation_list = quantizer_ref.permutation_list.clone().detach()
            except:
                self.permutation_list = quantizer_ref.permutation_list
            self.init_duquant_params = torch.tensor(1)
