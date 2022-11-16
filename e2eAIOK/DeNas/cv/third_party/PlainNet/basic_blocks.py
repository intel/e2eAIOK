'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import uuid

import PlainNet
from PlainNet.net_struct_utils import _get_right_parentheses_index_, _create_netblock_list_from_str_

class PlainNetBasicBlockClass(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, stride=1, no_create=False, block_name=None, **kwargs):
        super(PlainNetBasicBlockClass, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_create = no_create
        self.block_name = block_name
        if self.block_name is None:
            self.block_name = 'uuid{}'.format(uuid.uuid4().hex)

    def forward(self, x):
        raise RuntimeError('Not implemented')

    def __str__(self):
        return type(self).__name__ + '({},{},{})'.format(self.in_channels, self.out_channels, self.stride)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{},{})'.format(self.block_name, self.in_channels, self.out_channels, self.stride)

    def get_output_resolution(self, input_resolution):
        raise RuntimeError('Not implemented')

    def get_FLOPs(self, input_resolution):
        raise RuntimeError('Not implemented')

    def get_model_size(self):
        raise RuntimeError('Not implemented')

    def set_in_channels(self, c):
        raise RuntimeError('Not implemented')

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert PlainNetBasicBlockClass.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               block_name=tmp_block_name, no_create=no_create), s[idx + 1:]

    @classmethod
    def is_instance_from_str(cls, s):
        if s.startswith(cls.__name__ + '(') and s[-1] == ')':
            return True
        else:
            return False


class AdaptiveAvgPool(PlainNetBasicBlockClass):
    def __init__(self, out_channels, output_size, no_create=False, **kwargs):
        super(AdaptiveAvgPool, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.AdaptiveAvgPool2d(output_size=(self.output_size, self.output_size))

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return type(self).__name__ + '({},{})'.format(self.out_channels // self.output_size**2, self.output_size)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{})'.format(self.block_name,
                                                         self.out_channels // self.output_size ** 2, self.output_size)

    def get_output_resolution(self, input_resolution):
        return self.output_size

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels = c

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert AdaptiveAvgPool.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('AdaptiveAvgPool('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        output_size = int(param_str_split[1])
        return AdaptiveAvgPool(out_channels=out_channels, output_size=output_size,
                               block_name=tmp_block_name, no_create=no_create), s[idx + 1:]


class BN(PlainNetBasicBlockClass):
    def __init__(self, out_channels=None, copy_from=None, no_create=False, **kwargs):
        super(BN, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.BatchNorm2d)
            self.in_channels = copy_from.weight.shape[0]
            self.out_channels = copy_from.weight.shape[0]
            assert out_channels is None or out_channels == self.out_channels
            self.netblock = copy_from

        else:
            self.in_channels = out_channels
            self.out_channels = out_channels
            if no_create:
                return
            else:
                self.netblock = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'BN({})'.format(self.out_channels)

    def __repr__(self):
        return 'BN({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return input_resolution ** 2 * self.out_channels

    def get_model_size(self):
        return self.out_channels

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels = c
        if not self.no_create:
            self.netblock = nn.BatchNorm2d(num_features=self.out_channels)
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert BN.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('BN('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        out_channels = int(param_str)
        return BN(out_channels=out_channels, block_name=tmp_block_name, no_create=no_create), s[idx + 1:]


class ConvKX(PlainNetBasicBlockClass):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, groups=1, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvKX, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            self.groups = copy_from.groups
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride
            self.netblock = copy_from
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            self.groups = groups
            self.kernel_size = kernel_size
            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            else:
                self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                          kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding, bias=False, groups=self.groups)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return type(self).__name__ + '({},{},{},{})'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{},{},{})'.format(self.block_name, self.in_channels, self.out_channels, self.kernel_size, self.stride)

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.out_channels * self.kernel_size ** 2 * input_resolution ** 2 // self.stride ** 2 // self.groups

    def get_model_size(self):
        return self.in_channels * self.out_channels * self.kernel_size ** 2 // self.groups

    def set_in_channels(self, c):
        self.in_channels = c
        if not self.no_create:
            self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding, bias=False)
            self.netblock.train()
            self.netblock.requires_grad_(True)


    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert cls.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        kernel_size = int(split_str[2])
        stride = int(split_str[3])
        return cls(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, no_create=no_create, block_name=tmp_block_name), s[idx + 1:]


class ConvDW(PlainNetBasicBlockClass):
    def __init__(self, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvDW, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            assert self.in_channels == self.out_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride

            self.netblock = copy_from
        else:

            self.in_channels = out_channels
            self.out_channels = out_channels
            self.stride = stride
            self.kernel_size = kernel_size

            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            else:
                self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                          kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding, bias=False, groups=self.in_channels)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'ConvDW({},{},{})'.format(self.out_channels, self.kernel_size, self.stride)

    def __repr__(self):
        return 'ConvDW({}|{},{},{})'.format(self.block_name, self.out_channels, self.kernel_size, self.stride)

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return self.out_channels * self.kernel_size ** 2 * input_resolution ** 2 // self.stride ** 2

    def get_model_size(self):
        return self.out_channels * self.kernel_size ** 2

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels=self.in_channels
        if not self.no_create:
            self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding, bias=False, groups=self.in_channels)
            self.netblock.train()
            self.netblock.requires_grad_(True)



    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert ConvDW.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('ConvDW('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        out_channels = int(split_str[0])
        kernel_size = int(split_str[1])
        stride = int(split_str[2])
        return ConvDW(out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, no_create=no_create, block_name=tmp_block_name), s[idx + 1:]

class ConvKXG2(ConvKX):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvKXG2, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, copy_from=copy_from, no_create=no_create,
                                       groups=2, **kwargs)

class ConvKXG4(ConvKX):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvKXG4, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, copy_from=copy_from, no_create=no_create,
                                       groups=4, **kwargs)


class ConvKXG8(ConvKX):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvKXG8, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, copy_from=copy_from, no_create=no_create,
                                       groups=8, **kwargs)

class ConvKXG16(ConvKX):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvKXG16, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, copy_from=copy_from, no_create=no_create,
                                       groups=16, **kwargs)

class ConvKXG32(ConvKX):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super(ConvKXG32, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, copy_from=copy_from, no_create=no_create,
                                       groups=32, **kwargs)


class Flatten(PlainNetBasicBlockClass):
    def __init__(self, out_channels, no_create=False, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, x):
        return torch.flatten(x, 1)

    def __str__(self):
        return 'Flatten({})'.format(self.out_channels)

    def __repr__(self):
        return 'Flatten({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return 1

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels = c

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Flatten.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('Flatten('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Flatten(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), s[idx + 1:]



class Linear(PlainNetBasicBlockClass):
    def __init__(self, in_channels=None, out_channels=None, bias=True, copy_from=None,
                 no_create=False,  **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Linear)
            self.in_channels = copy_from.weight.shape[1]
            self.out_channels = copy_from.weight.shape[0]
            self.use_bias = copy_from.bias is not None
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels

            self.netblock = copy_from
        else:

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.use_bias = bias
            if not no_create:
                self.netblock = nn.Linear(self.in_channels, self.out_channels,
                                          bias=self.use_bias)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'Linear({},{},{})'.format(self.in_channels, self.out_channels, int(self.use_bias))

    def __repr__(self):
        return 'Linear({}|{},{},{})'.format(self.block_name, self.in_channels, self.out_channels, int(self.use_bias))

    def get_output_resolution(self, input_resolution):
        assert input_resolution == 1
        return 1

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.out_channels

    def get_model_size(self):
        return self.in_channels * self.out_channels + int(self.use_bias)

    def set_in_channels(self, c):
        self.in_channels = c
        if not self.no_create:
            self.netblock = nn.Linear(self.in_channels, self.out_channels,
                                      bias=self.use_bias)
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Linear.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('Linear('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        use_bias = int(split_str[2])

        return Linear(in_channels=in_channels, out_channels=out_channels, bias=use_bias == 1,
            block_name=tmp_block_name, no_create=no_create), s[idx+1 :]



class MaxPool(PlainNetBasicBlockClass):
    def __init__(self, out_channels, kernel_size, stride, no_create=False,  **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'MaxPool({},{},{})'.format(self.out_channels, self.kernel_size, self.stride)

    def __repr__(self):
        return 'MaxPool({}|{},{},{})'.format(self.block_name, self.out_channels, self.kernel_size, self.stride)

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels = c
        if not self.no_create:
            self.netblock = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert MaxPool.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('MaxPool('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        kernel_size = int(param_str_split[1])
        stride = int(param_str_split[2])
        return MaxPool(out_channels=out_channels, kernel_size=kernel_size, stride=stride, no_create=no_create,
                       block_name=tmp_block_name), s[idx + 1:]


class Sequential(PlainNetBasicBlockClass):
    def __init__(self, block_list, no_create=False, **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = block_list[0].in_channels
        self.out_channels = block_list[-1].out_channels
        self.no_create = no_create
        res = 1024
        for block in self.block_list:
            res = block.get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output = x
        for inner_block in self.block_list:
            output = inner_block(output)
        return output

    def __str__(self):
        s = 'Sequential('
        for inner_block in self.block_list:
            s += str(inner_block)
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)
        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)
        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def set_in_channels(self, c):
        self.in_channels = c
        if len(self.block_list) == 0:
            self.out_channels = c
            return

        self.block_list[0].set_in_channels(c)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and isinstance(self.block_list[1], BN):
            self.block_list[1].set_in_channels(last_channels)

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Sequential.is_instance_from_str(s)
        the_right_paraen_idx = _get_right_parentheses_index_(s)
        param_str = s[len('Sequential(')+1:the_right_paraen_idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, ''
        return Sequential(block_list=the_block_list, no_create=no_create, block_name=tmp_block_name), ''


class MultiSumBlock(PlainNetBasicBlockClass):
    def __init__(self, block_list, no_create=False, **kwargs):
        super(MultiSumBlock, self).__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = np.max([x.in_channels for x in block_list])
        self.out_channels = np.max([x.out_channels for x in block_list])
        self.no_create = no_create

        res = 1024
        res = self.block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output = self.block_list[0](x)
        for inner_block in self.block_list[1:]:
            output2 = inner_block(x)
            output = output + output2
        return output

    def __str__(self):
        s = 'MultiSumBlock({}|'.format(self.block_name)
        for inner_block in self.block_list:
            s += str(inner_block) + ';'
        s = s[:-1]
        s += ')'
        return s

    def __repr__(self):
        return str(self)


    def get_output_resolution(self, input_resolution):
        the_res = self.block_list[0].get_output_resolution(input_resolution)
        for the_block in self.block_list:
            assert the_res == the_block.get_output_resolution(input_resolution)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(input_resolution)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def set_in_channels(self, c):
        self.in_channels = c
        for the_block in self.block_list:
            the_block.set_in_channels(c)

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert MultiSumBlock.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('MultiSumBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = _create_netblock_list_from_str_(the_s, PlainNet._all_netblocks_dict_, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_block_list.append(tmp_block_list[0])
            else:
                the_block_list.append(Sequential(block_list=tmp_block_list, no_create=no_create))
        pass  # end while

        if len(the_block_list) == 0:
            return None, s[idx+1:]

        return MultiSumBlock(block_list=the_block_list, block_name=tmp_block_name, no_create=no_create), s[idx+1:]


class MultiCatBlock(PlainNetBasicBlockClass):
    def __init__(self, block_list, no_create=False, **kwargs):
        super(MultiCatBlock, self).__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = np.max([x.in_channels for x in block_list])
        self.out_channels = np.sum([x.out_channels for x in block_list])
        self.no_create = no_create

        res = 1024
        res = self.block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output_list = []
        for inner_block in self.block_list:
            output = inner_block(x)
            output_list.append(output)

        return torch.cat(output_list, dim=1)

    def __str__(self):
        s = 'MultiCatBlock({}|'.format(self.block_name)
        for inner_block in self.block_list:
            s += str(inner_block) + ';'

        s = s[:-1]
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = self.block_list[0].get_output_resolution(input_resolution)
        for the_block in self.block_list:
            assert the_res == the_block.get_output_resolution(input_resolution)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(input_resolution)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def set_in_channels(self, c):
        self.in_channels = c
        for the_block in self.block_list:
            the_block.set_in_channels(c)
        self.out_channels = np.sum([x.out_channels for x in self.block_list])


    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert MultiCatBlock.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('MultiCatBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = _create_netblock_list_from_str_(the_s, PlainNet._all_netblocks_dict_, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_block_list.append(tmp_block_list[0])
            else:
                the_block_list.append(Sequential(block_list=tmp_block_list, no_create=no_create))
            pass  # end if
        pass  # end while

        if len(the_block_list) == 0:
            return None, s[idx+1:]

        return MultiCatBlock(block_list=the_block_list, block_name=tmp_block_name,
                             no_create=no_create), s[idx + 1:]


class RELU(PlainNetBasicBlockClass):
    def __init__(self, out_channels, no_create=False, **kwargs):
        super(RELU, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, x):
        return F.relu(x)

    def __str__(self):
        return 'RELU({})'.format(self.out_channels)

    def __repr__(self):
        return 'RELU({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels = c

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert RELU.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('RELU('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return RELU(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), s[idx+1:]



class ResBlock(PlainNetBasicBlockClass):
    '''
    ResBlock(in_channles, inner_blocks_str). If in_channels is missing, use block_list[0].in_channels as in_channels
    '''
    def __init__(self, block_list, in_channels=None, stride=None, no_create=False, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = block_list[-1].out_channels

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res

        self.proj = None
        if self.stride > 1 or self.in_channels != self.out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x):
        if len(self.block_list) == 0:
            return x

        output = x
        for inner_block in self.block_list:
            output = inner_block(output)

        if self.proj is not None:
            output = output + self.proj(x)
        else:
            output = output + x

        return output

    def __str__(self):
        s = 'ResBlock({},{},'.format(self.in_channels, self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def __repr__(self):
        s = 'ResBlock({}|{},{},'.format(self.block_name, self.in_channels, self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        if self.proj is not None:
            the_flops += self.in_channels * self.out_channels * (the_res / self.stride) ** 2 + \
                         (the_res / self.stride) ** 2 * self.out_channels

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        if self.proj is not None:
            the_size += self.in_channels * self.out_channels + self.out_channels

        return the_size

    def set_in_channels(self, c):
        self.in_channels = c
        if len(self.block_list) == 0:
            self.out_channels = c
            return

        self.block_list[0].set_in_channels(c)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and \
                ( isinstance(self.block_list[0], ConvKX) or isinstance(self.block_list[0], ConvDW)) and \
                isinstance(self.block_list[1], BN):
            self.block_list[1].set_in_channels(last_channels)

        self.proj = None
        if not self.no_create:
            if self.stride > 1 or self.in_channels != self.out_channels:
                self.proj = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                    nn.BatchNorm2d(self.out_channels),
                )
                self.proj.train()
                self.proj.requires_grad_(True)




    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert ResBlock.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        the_stride = None
        param_str = s[len('ResBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit():  # cannot parse in_channels, missing, use default
            in_channels = None
            the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index+1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[0:second_comma_index].isdigit():
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
            pass
        pass

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, s[idx+1:]
        return ResBlock(block_list=the_block_list, in_channels=in_channels,
                        stride=the_stride, no_create=no_create, block_name=tmp_block_name), s[idx+1:]


class ResBlockProj(PlainNetBasicBlockClass):
    '''
    ResBlockProj(in_channles, inner_blocks_str). If in_channels is missing, use block_list[0].in_channels as in_channels
    '''
    def __init__(self, block_list, in_channels=None, stride=None, no_create=False, **kwargs):
        super(ResBlockProj, self).__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = block_list[-1].out_channels

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res


        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
            nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x):
        if len(self.block_list) == 0:
            return x

        output = x
        for inner_block in self.block_list:
            output = inner_block(output)
        output = output + self.proj(x)
        return output

    def __str__(self):
        s = 'ResBlockProj({},{},'.format(self.in_channels, self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def __repr__(self):
        s = 'ResBlockProj({}|{},{},'.format(self.block_name, self.in_channels, self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        if self.proj is not None:
            the_flops += self.in_channels * self.out_channels * (the_res / self.stride) ** 2 + \
                         (the_res / self.stride) ** 2 * self.out_channels

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        if self.proj is not None:
            the_size += self.in_channels * self.out_channels + self.out_channels

        return the_size

    def set_in_channels(self, c):
        self.in_channels = c
        if len(self.block_list) == 0:
            self.out_channels = c
            return

        self.block_list[0].set_in_channels(c)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and \
                ( isinstance(self.block_list[0], ConvKX) or isinstance(self.block_list[0], ConvDW)) and \
                isinstance(self.block_list[1], BN):
            self.block_list[1].set_in_channels(last_channels)

        self.proj = None
        if not self.no_create:
            if self.stride > 1 or self.in_channels != self.out_channels:
                self.proj = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                    nn.BatchNorm2d(self.out_channels),
                )
                self.proj.train()
                self.proj.requires_grad_(True)




    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert ResBlockProj.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        the_stride = None
        param_str = s[len('ResBlockProj('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit():  # cannot parse in_channels, missing, use default
            in_channels = None
            the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index+1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[0:second_comma_index].isdigit():
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, PlainNet._all_netblocks_dict_, no_create=no_create)
            pass
        pass

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, s[idx+1:]
        return ResBlockProj(block_list=the_block_list, in_channels=in_channels,
                        stride=the_stride, no_create=no_create, block_name=tmp_block_name), s[idx+1:]

class SE(PlainNetBasicBlockClass):
    def __init__(self, out_channels=None, copy_from=None,
                 no_create=False, **kwargs):
        super(SE, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            raise RuntimeError('Not implemented')
        else:
            self.in_channels = out_channels
            self.out_channels = out_channels
            self.se_ratio = 0.25
            self.se_channels = max(1, int(round(self.out_channels * self.se_ratio)))
            if no_create or self.out_channels == 0:
                return
            else:
                self.netblock = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Conv2d(in_channels=self.out_channels, out_channels=self.se_channels, kernel_size=1, stride=1,
                              padding=0, bias=False),
                    nn.BatchNorm2d(self.se_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.se_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                              padding=0, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.Sigmoid()
                )

    def forward(self, x):
        se_x = self.netblock(x)
        return se_x * x

    def __str__(self):
        return 'SE({})'.format(self.out_channels)

    def __repr__(self):
        return 'SE({}|{})'.format(self.block_name,self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.se_channels + self.se_channels * self.out_channels + self.out_channels + \
            self.out_channels * input_resolution ** 2

    def get_model_size(self):
        return self.in_channels * self.se_channels + 2 * self.se_channels + self.se_channels * self.out_channels + \
            2 * self.out_channels

    def set_in_channels(self, c):
        self.in_channels = c
        if not self.no_create:
            self.netblock = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=self.out_channels, out_channels=self.se_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(self.se_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.se_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.Sigmoid()
            )
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert SE.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('SE('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return SE(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), s[idx + 1:]



class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(PlainNetBasicBlockClass):
    def __init__(self, out_channels=None, copy_from=None,
                 no_create=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            raise RuntimeError('Not implemented')
        else:
            self.in_channels = out_channels
            self.out_channels = out_channels

    def forward(self, x):
        return SwishImplementation.apply(x)

    def __str__(self):
        return 'Swish({})'.format(self.out_channels)

    def __repr__(self):
        return 'Swish({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return self.out_channels * input_resolution ** 2

    def get_model_size(self):
        return 0

    def set_in_channels(self, c):
        self.in_channels = c
        self.out_channels = c


    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Swish.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('Swish('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Swish(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), s[idx + 1:]



def _add_bn_layer_(block_list):
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, ConvKX) or isinstance(the_block, ConvDW):
            out_channels = the_block.out_channels
            new_bn_block = BN(out_channels=out_channels, no_create=True)
            new_seq_with_bn = Sequential(block_list=[the_block, new_bn_block], no_create=True)
            new_block_list.append(new_seq_with_bn)
        elif hasattr(the_block, 'block_list'):
            new_block_list = _add_bn_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)
        pass
    pass

    return new_block_list


def _remove_bn_layer_(block_list):
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, BN):
            continue
        elif hasattr(the_block, 'block_list'):
            new_block_list = _remove_bn_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)
        pass
    pass

    return new_block_list


def _add_se_layer_(block_list):
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, RELU):
            out_channels = the_block.out_channels
            new_se_block = SE(out_channels=out_channels, no_create=True)
            new_seq_with_bn = Sequential(block_list=[the_block, new_se_block], no_create=True)
            new_block_list.append(new_seq_with_bn)
        elif hasattr(the_block, 'block_list'):
            new_block_list = _add_se_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)
        pass
    pass

    return new_block_list

def _replace_relu_with_swish_layer_(block_list):
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, RELU):
            out_channels = the_block.out_channels
            new_swish_block = Swish(out_channels=out_channels, no_create=True)
            new_block_list.append(new_swish_block)
        elif hasattr(the_block, 'block_list'):
            new_block_list = _replace_relu_with_swish_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)
        pass
    pass

    return new_block_list

def _fuse_convkx_and_bn_(convkx, bn):
    the_weight_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    convkx.weight[:] = convkx.weight * the_weight_scale.view((-1, 1, 1, 1))
    the_bias_shift = (bn.weight * bn.running_mean) / \
                     torch.sqrt(bn.running_var + bn.eps)
    bn.weight[:] = 1
    bn.bias[:] = bn.bias - the_bias_shift
    bn.running_var[:] = 1.0 - bn.eps
    bn.running_mean[:] = 0.0


def _fuse_bn_layer_for_blocks_list_(block_list):
    last_block = None  # type: ConvKX
    with torch.no_grad():
        for the_block in block_list:
            if isinstance(the_block, BN):
                # assert isinstance(last_block, ConvKX) or isinstance(last_block, ConvDW)
                if isinstance(last_block, ConvKX) or isinstance(last_block, ConvDW):
                    _fuse_convkx_and_bn_(last_block.netblock, the_block.netblock)
                else:
                    print('--- warning! Cannot fuse BN={} because last_block={}'.format(the_block, last_block))

                last_block = None
            elif isinstance(the_block, ConvKX) or isinstance(the_block, ConvDW):
                last_block = the_block
            elif hasattr(the_block, 'block_list') and the_block.block_list is not None and \
                    len(the_block.block_list) > 0:
                _fuse_bn_layer_for_blocks_list_(the_block.block_list)
            else:
                pass
            pass
        pass
    pass  # end with




def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'AdaptiveAvgPool': AdaptiveAvgPool,
        'BN': BN,
        'ConvDW': ConvDW,
        'ConvKX': ConvKX,
        'ConvKXG2': ConvKXG2,
        'ConvKXG4': ConvKXG4,
        'ConvKXG8': ConvKXG8,
        'ConvKXG16': ConvKXG16,
        'ConvKXG32': ConvKXG32,
        'Flatten': Flatten,
        'Linear': Linear,
        'MaxPool': MaxPool,
        'MultiSumBlock': MultiSumBlock,
        'MultiCatBlock': MultiCatBlock,
        'PlainNetBasicBlockClass': PlainNetBasicBlockClass,
        'RELU': RELU,
        'ResBlock': ResBlock,
        'ResBlockProj': ResBlockProj,
        'Sequential': Sequential,
        'SE': SE,
        'Swish': Swish,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict
