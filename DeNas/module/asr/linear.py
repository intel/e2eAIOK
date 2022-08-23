import torch
import torch.nn as nn
import torch.nn.functional as F

from module.Linear_base import LinearBase


class Linear(LinearBase):
    """
    Computes a linear transformation y = wx + b.
    """

    def __init__(
        self,
        n_neurons,
        input_size=None,
        bias=True,
    ):
        super().__init__(input_size, n_neurons, bias=bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
