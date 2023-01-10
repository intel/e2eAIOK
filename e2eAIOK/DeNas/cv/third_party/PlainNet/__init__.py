'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, argparse
from torch import nn
from PlainNet.net_struct_utils import _create_netblock_list_from_str_

_all_netblocks_dict_ = {}

class PlainNet(nn.Module):
    def __init__(self, num_classes=None, plainnet_struct=None, no_create=False, **kwargs):
        super(PlainNet, self).__init__()
        self.num_classes = num_classes
        self.plainnet_struct = plainnet_struct

        if self.plainnet_struct is None:
            raise ValueError("plainnet_struct is None")

        if self.plainnet_struct is None:
            return

        the_s = self.plainnet_struct  # type: str

        block_list, remaining_s = _create_netblock_list_from_str_(the_s, _all_netblocks_dict_,  no_create=no_create, **kwargs)
        assert len(remaining_s) == 0

        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)  # register

    def forward(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def __str__(self):
        s = ''
        for the_block in self.block_list:
            s += str(the_block)
        return s

    def __repr__(self):
        return str(self)

    @classmethod
    def create_netblock_list_from_str(cls, s, no_create=False, **kwargs):
        the_list, remaining_s = _create_netblock_list_from_str_(s, _all_netblocks_dict_, no_create=no_create, **kwargs)
        assert len(remaining_s) == 0
        return the_list

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

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block
        if block_id < len(self.block_list):
            self.block_list[block_id + 1].set_in_channels(new_block.out_channels)

        self.module_list = nn.Module(self.block_list)



from PlainNet import basic_blocks
_all_netblocks_dict_ = basic_blocks.register_netblocks_dict(_all_netblocks_dict_)

from PlainNet import super_blocks
_all_netblocks_dict_ = super_blocks.register_netblocks_dict(_all_netblocks_dict_)

from PlainNet import SuperResKXKX
_all_netblocks_dict_ = SuperResKXKX.register_netblocks_dict(_all_netblocks_dict_)

from PlainNet import SuperResK1KXK1
_all_netblocks_dict_ = SuperResK1KXK1.register_netblocks_dict(_all_netblocks_dict_)

from PlainNet import SuperResIDWEXKX
_all_netblocks_dict_ = SuperResIDWEXKX.register_netblocks_dict(_all_netblocks_dict_)