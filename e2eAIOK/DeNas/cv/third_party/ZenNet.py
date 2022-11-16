'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np
import torch, argparse
from torch import nn
import torch.nn.functional as F
import PlainNet
from PlainNet import global_utils
from PlainNet import basic_blocks, super_blocks, SuperResKXKX, SuperResK1KXK1, SuperResIDWEXKX
import itertools

class DeSearchSpaceXXBL(object):
    seach_space_block_type_list_list = [
        [SuperResK1KXK1.SuperResK1K3K1, SuperResK1KXK1.SuperResK1K5K1, SuperResK1KXK1.SuperResK1K7K1],
        [SuperResKXKX.SuperResK3K3, SuperResKXKX.SuperResK5K5, SuperResKXKX.SuperResK7K7],
    ]

    __block_type_round_channels_base_dict__ = {
            SuperResKXKX.SuperResK3K3: 8,
            SuperResKXKX.SuperResK5K5: 8,
            SuperResKXKX.SuperResK7K7: 8,
            SuperResK1KXK1.SuperResK1K3K1: 8, SuperResK1KXK1.SuperResK1K5K1: 8, SuperResK1KXK1.SuperResK1K7K1: 8,
        }

    __block_type_min_channels_base_dict__ = {
            SuperResKXKX.SuperResK3K3: 8,
            SuperResKXKX.SuperResK5K5: 8,
            SuperResKXKX.SuperResK7K7: 8,
            SuperResK1KXK1.SuperResK1K3K1: 8,
            SuperResK1KXK1.SuperResK1K5K1: 8,
            SuperResK1KXK1.SuperResK1K7K1: 8,
        }

    @classmethod
    def _get_select_student_channels_list(cls, out_channels):
        the_list = [out_channels * 2.5, out_channels * 2, out_channels * 1.5, out_channels * 1.25,
                    out_channels,
                    out_channels / 1.25, out_channels / 1.5, out_channels / 2, out_channels / 2.5]
        the_list = [min(2048, max(8, x)) for x in the_list]
        the_list = [global_utils.smart_round(x, base=8) for x in the_list]
        the_list = list(set(the_list))
        the_list.sort(reverse=True)
        return the_list

    @classmethod
    def _get_select_student_sublayers_list(cls, sub_layers):
        the_list = [sub_layers,
                    sub_layers + 1, sub_layers + 2,
                    sub_layers - 1, sub_layers - 2, ]
        the_list = [max(0, round(x)) for x in the_list]
        the_list = list(set(the_list))
        the_list.sort(reverse=True)
        return the_list

    @classmethod
    def gen_search_space(cls, block_list, block_id):
        the_block = block_list[block_id]
        student_blocks_list_list = []

        if isinstance(the_block, super_blocks.SuperConvKXBNRELU):
            student_blocks_list = []
            student_out_channels_list = cls._get_select_student_channels_list(the_block.out_channels)
            for student_out_channels in student_out_channels_list:
                tmp_block_str = type(the_block).__name__ + '({},{},{},1)'.format(
                    the_block.in_channels, student_out_channels, the_block.stride)
                student_blocks_list.append(tmp_block_str)
            pass
            student_blocks_list = list(set(student_blocks_list))
            assert len(student_blocks_list) >= 1
            student_blocks_list_list.append(student_blocks_list)
        else:
            for student_block_type_list in cls.seach_space_block_type_list_list:
                student_blocks_list = []
                student_out_channels_list = cls._get_select_student_channels_list(the_block.out_channels)
                student_sublayers_list = cls._get_select_student_sublayers_list(sub_layers=the_block.sub_layers)
                student_bottleneck_channels_list = cls._get_select_student_channels_list(the_block.bottleneck_channels)
                for student_block_type in student_block_type_list:
                    for student_out_channels, student_sublayers, student_bottleneck_channels in itertools.product(
                            student_out_channels_list, student_sublayers_list, student_bottleneck_channels_list):

                        # filter smallest possible channel for this block type
                        min_possible_channels = cls.__block_type_round_channels_base_dict__[student_block_type]
                        channel_round_base = cls.__block_type_round_channels_base_dict__[student_block_type]
                        student_out_channels = global_utils.smart_round(student_out_channels, channel_round_base)
                        student_bottleneck_channels = global_utils.smart_round(student_bottleneck_channels,
                                                                            channel_round_base)

                        if student_out_channels < min_possible_channels or student_bottleneck_channels < min_possible_channels:
                            continue
                        if student_sublayers <= 0:  # no empty layer
                            continue
                        tmp_block_str = student_block_type.__name__ + '({},{},{},{},{})'.format(
                            the_block.in_channels, student_out_channels, the_block.stride, student_bottleneck_channels,
                            student_sublayers)
                        student_blocks_list.append(tmp_block_str)
                    pass
                    student_blocks_list = list(set(student_blocks_list))
                    assert len(student_blocks_list) >= 1
                    student_blocks_list_list.append(student_blocks_list)
                pass
            pass  # end for student_block_type_list in seach_space_block_type_list_list:
        pass
        return student_blocks_list_list

class DeSearchSpaceIDWEXKX:
    def __init__(self):
        self.seach_space_block_type_list_list = [
            [SuperResIDWEXKX.SuperResIDWE1K3, SuperResIDWEXKX.SuperResIDWE2K3, SuperResIDWEXKX.SuperResIDWE4K3,
            SuperResIDWEXKX.SuperResIDWE6K3,
            SuperResIDWEXKX.SuperResIDWE1K5, SuperResIDWEXKX.SuperResIDWE2K5, SuperResIDWEXKX.SuperResIDWE4K5,
            SuperResIDWEXKX.SuperResIDWE6K5,
            SuperResIDWEXKX.SuperResIDWE1K7, SuperResIDWEXKX.SuperResIDWE2K7, SuperResIDWEXKX.SuperResIDWE4K7,
            SuperResIDWEXKX.SuperResIDWE6K7],
        ]

        self.__block_type_round_channels_base_dict__ = {
            SuperResIDWEXKX.SuperResIDWE1K3: 8,
            SuperResIDWEXKX.SuperResIDWE2K3: 8,
            SuperResIDWEXKX.SuperResIDWE4K3: 8,
            SuperResIDWEXKX.SuperResIDWE6K3: 8,
            SuperResIDWEXKX.SuperResIDWE1K5: 8,
            SuperResIDWEXKX.SuperResIDWE2K5: 8,
            SuperResIDWEXKX.SuperResIDWE4K5: 8,
            SuperResIDWEXKX.SuperResIDWE6K5: 8,
            SuperResIDWEXKX.SuperResIDWE1K7: 8,
            SuperResIDWEXKX.SuperResIDWE2K7: 8,
            SuperResIDWEXKX.SuperResIDWE4K7: 8,
            SuperResIDWEXKX.SuperResIDWE6K7: 8,
        }

        self.__block_type_min_channels_base_dict__ = {
            SuperResIDWEXKX.SuperResIDWE1K3: 8,
            SuperResIDWEXKX.SuperResIDWE2K3: 8,
            SuperResIDWEXKX.SuperResIDWE4K3: 8,
            SuperResIDWEXKX.SuperResIDWE6K3: 8,
            SuperResIDWEXKX.SuperResIDWE1K5: 8,
            SuperResIDWEXKX.SuperResIDWE2K5: 8,
            SuperResIDWEXKX.SuperResIDWE4K5: 8,
            SuperResIDWEXKX.SuperResIDWE6K5: 8,
            SuperResIDWEXKX.SuperResIDWE1K7: 8,
            SuperResIDWEXKX.SuperResIDWE2K7: 8,
            SuperResIDWEXKX.SuperResIDWE4K7: 8,
            SuperResIDWEXKX.SuperResIDWE6K7: 8,
        }

    def _get_select_student_channels_list(self, out_channels):
        the_list = [out_channels * 2.5, out_channels * 2, out_channels * 1.5, out_channels * 1.25,
                    out_channels,
                    out_channels / 1.25, out_channels / 1.5, out_channels / 2, out_channels / 2.5]
        the_list = [max(8, x) for x in the_list]
        the_list = [global_utils.smart_round(x, base=8) for x in the_list]
        the_list = list(set(the_list))
        the_list.sort(reverse=True)
        return the_list

    def _get_select_student_sublayers_list(self, sub_layers):
        the_list = [sub_layers,
                    sub_layers + 1, sub_layers + 2,
                    sub_layers - 1, sub_layers - 2, ]
        the_list = [max(0, round(x)) for x in the_list]
        the_list = list(set(the_list))
        the_list.sort(reverse=True)
        return the_list

    def gen_search_space(self, block_list, block_id):
        the_block = block_list[block_id]
        student_blocks_list_list = []

        if isinstance(the_block, super_blocks.SuperConvKXBNRELU):
            student_blocks_list = []

            if the_block.kernel_size == 1:  # last fc layer, never change fc
                student_out_channels_list = [the_block.out_channels]
            else:
                student_out_channels_list = self.get_select_student_channels_list(the_block.out_channels)

            for student_out_channels in student_out_channels_list:
                tmp_block_str = type(the_block).__name__ + '({},{},{},1)'.format(
                    the_block.in_channels, student_out_channels, the_block.stride)
                student_blocks_list.append(tmp_block_str)
            pass
            student_blocks_list = list(set(student_blocks_list))
            assert len(student_blocks_list) >= 1
            student_blocks_list_list.append(student_blocks_list)
        else:
            for student_block_type_list in self.seach_space_block_type_list_list:
                student_blocks_list = []
                student_out_channels_list = self._get_select_student_channels_list(the_block.out_channels)
                student_sublayers_list = self._get_select_student_sublayers_list(sub_layers=the_block.sub_layers)
                student_bottleneck_channels_list = self._get_select_student_channels_list(the_block.bottleneck_channels)
                for student_block_type in student_block_type_list:
                    for student_out_channels, student_sublayers, student_bottleneck_channels in itertools.product(
                            student_out_channels_list, student_sublayers_list, student_bottleneck_channels_list):

                        # filter smallest possible channel for this block type
                        min_possible_channels = self.__block_type_round_channels_base_dict__[student_block_type]
                        channel_round_base = self.__block_type_round_channels_base_dict__[student_block_type]
                        student_out_channels = global_utils.smart_round(student_out_channels, channel_round_base)
                        student_bottleneck_channels = global_utils.smart_round(student_bottleneck_channels,
                                                                            channel_round_base)

                        if student_out_channels < min_possible_channels or student_bottleneck_channels < min_possible_channels:
                            continue
                        if student_sublayers <= 0:  # no empty layer
                            continue
                        tmp_block_str = student_block_type.__name__ + '({},{},{},{},{})'.format(
                            the_block.in_channels, student_out_channels, the_block.stride, student_bottleneck_channels,
                            student_sublayers)
                        student_blocks_list.append(tmp_block_str)
                    pass
                    student_blocks_list = list(set(student_blocks_list))
                    assert len(student_blocks_list) >= 1
                    student_blocks_list_list.append(student_blocks_list)
                pass
            pass  # end for student_block_type_list in seach_space_block_type_list_list:
        pass
        return student_blocks_list_list

class DeMainNet(PlainNet.PlainNet):
    def __init__(self, num_classes=None, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None):

        super().__init__(num_classes=num_classes, plainnet_struct=plainnet_struct,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)
        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = basic_blocks.Linear(in_channels=self.last_channels, out_channels=self.num_classes, no_create=no_create)

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        stage_features_list = []
        image_size = x.shape[2]
        output = x

        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        output = F.adaptive_avg_pool2d(output, output_size=1)
        output = torch.flatten(output, 1)
        logit = self.fc_linear(output)
        return stage_features_list, logit

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)

        output = F.adaptive_avg_pool2d(output, output_size=1)

        output = torch.flatten(output, 1)
        output = self.fc_linear(output)
        return output

    def forward_pre_GAP(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        the_flops += self.fc_linear.get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.fc_linear.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, super_blocks.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.fc_linear.in_channels != self.last_channels:
                self.fc_linear.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass

        for superblock in self.block_list:
            if not isinstance(superblock, super_blocks.PlainNetSuperBlockClass):
                continue
            for block in superblock.block_list:
                if not (isinstance(block, basic_blocks.ResBlock) or isinstance(block, basic_blocks.ResBlockProj)):
                    continue
                # print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, basic_blocks.BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                # print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)