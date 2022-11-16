import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from module.Linear_base import LinearBase


class LinearSuper(LinearBase):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim, in_index=None, out_index=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters(in_index=in_index, out_index=out_index)

    def _sample_parameters(self, in_index=None, out_index=None):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim, in_index=in_index, out_index=out_index)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        if self.bias is not None:
            return F.linear(x, self.samples['weight'].to(x.device), self.samples['bias'].to(x.device))
        else:
            return F.linear(x, self.samples['weight'].to(x.device))

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops

def sample_weight(weight, sample_in_dim, sample_out_dim, in_index=None, out_index=None):
    
    if in_index is None:
        sample_weight = weight[:, :sample_in_dim]
    else:
        sample_weight = weight.index_select(1, in_index.to(weight.device))

    if out_index is None:
        sample_weight = sample_weight[:sample_out_dim, :]
    else:
        sample_weight = sample_weight.index_select(0, out_index.to(sample_weight.device))

    return sample_weight


def sample_bias(bias, sample_out_dim, out_index=None):
    
    if out_index is None:
        sample_bias = bias[:sample_out_dim]
    else:
        sample_bias = bias.index_select(0, out_index.to(bias.device))

    return sample_bias
