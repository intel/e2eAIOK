#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.autograd import Function
import torch
from typing import Optional, Any, Tuple
import numpy as np

class GradientReverseOp(Function):
    ''' Gradient Reverse Operation

    '''
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ''' forward function

        :param ctx: context
        :param input: input tensor
        :param coeff: coeff to warm start Gradient Reverse Operation
        :return: output
        '''
        ctx.coeff = coeff # save coeff for backward using
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    ''' Gradient Reverse Layer

    '''
    def __init__(self, coeff_alpha: float,coeff_high: float,max_iter: int, enable_step: bool):
        ''' Init method.

        :param coeff_alpha: control warm curve shape. The larger, the steeper
        :param coeff_high: control high of the curve shape
        :param max_iter: max iter for one epoch
        :param enable_step: frozen iter_num or not
        '''
        super(GradientReverseLayer, self).__init__()
        self.coeff_alpha = coeff_alpha
        self.coeff_high = coeff_high
        self.iter_num = 0
        self.max_iter = max_iter
        self.enable_step = enable_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        self.coeff = 2.0*self.coeff_high /(1.0 + np.exp(-self.coeff_alpha * self.iter_num / self.max_iter))- self.coeff_high

        if self.enable_step: # increase iter_num
            self.iter_num += 1
        return GradientReverseOp.apply(input, self.coeff)

