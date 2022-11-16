import torch.nn as nn


class LinearBase(nn.Linear):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__(in_dim, out_dim, bias=bias)

    def forward(self, x):
        raise NotImplementedError
