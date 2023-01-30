#    Copyright 2022, Intel Corporation.
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch
import torch.nn.functional
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import (ConvDropoutNormNonlin, StackedConvLayers)



class SegDiscriminator(nn.Module):
    BASE_NUM_FILTERS = 50

    def __init__(self, input_channels, threeD=True, 
                 pool_op_kernel_sizes=None,
                 ):

        super(SegDiscriminator, self).__init__()

        if threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
         
        nonlin=nn.LeakyReLU
        weightInitializer=InitWeights_He(1e-2)
        basic_block=ConvDropoutNormNonlin
        num_conv_per_stage=2

        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        # norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.input_channels = input_channels
        self.conv_kwargs = conv_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op

        if conv_op == nn.Conv2d:
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * len(input_channels)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            conv_kernel_sizes = (3, 3)
            conv_pad_sizes = [1, 1]
        elif conv_op == nn.Conv3d:
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * len(input_channels)
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            conv_kernel_sizes = (3, 3, 3)
            conv_pad_sizes = [1, 1, 1]
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.conv_kwargs['kernel_size'] = conv_kernel_sizes
        self.conv_kwargs['padding'] = conv_pad_sizes

        self.conv_blocks_context = []
        for d in range(len(input_channels)):

            if d == 0:
                input_features = input_channels[d]
                output_features = self.BASE_NUM_FILTERS
            else:
                input_features = d * self.BASE_NUM_FILTERS + input_channels[d]
                output_features = (d+1) * self.BASE_NUM_FILTERS
            
            first_stride = pool_op_kernel_sizes[d]

            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))

        # register all modules properly
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
        
    def forward(self, features):
        if not isinstance(features, list) and not isinstance(features, tuple):
            features = [features]

        input = features[0]
        x = self.conv_blocks_context[0](input)

        for i in range(1, len(self.input_channels)):
            x = torch.cat((x, features[i]), dim=1)
            x = self.conv_blocks_context[i](x)
   
        x = self.avgpool(x)
        x = x.view(x.size(0) * x.size(1), -1)

        return x
        

class FCDiscriminator(nn.Module):
    BASE_NUM_FILTERS = 50
    CONV_NUM = 4

    def __init__(self):
        super().__init__()

        input_channel=1
        num_conv_per_stage=2
        pool_op_kernel_size = (2, 2)
        first_stride = pool_op_kernel_size
        conv_kernel_sizes = (3, 3)
        conv_pad_sizes = [1, 1]
        weightInitializer=InitWeights_He(1e-2)
        basic_block=ConvDropoutNormNonlin
        avgpool = nn.AdaptiveAvgPool2d(1)

        conv_op = nn.Conv2d
        norm_op = nn.BatchNorm2d
        dropout_op = nn.Dropout2d
        nonlin=nn.LeakyReLU

        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        # norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.conv_kwargs = conv_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.avgpool = avgpool
        self.conv_kwargs['kernel_size'] = conv_kernel_sizes
        self.conv_kwargs['padding'] = conv_pad_sizes

        self.conv_blocks_context = []
        for d in range(self.CONV_NUM):

            if d == 0:
                input_features = input_channel
            else:
                input_features = d * self.BASE_NUM_FILTERS
            output_features = (d+1) * self.BASE_NUM_FILTERS

            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))

        # register all modules properly
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

        self.final_layer = nn.Sequential(
            nn.Linear(output_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=0, end_dim=2)
        x = torch.unsqueeze(x, 1)

        for i in range(self.CONV_NUM):
            x = self.conv_blocks_context[i](x)
   
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.final_layer(x)

        return x
