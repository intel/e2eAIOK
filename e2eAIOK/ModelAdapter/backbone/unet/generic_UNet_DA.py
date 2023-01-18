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
import numpy as np
import torch.nn.functional
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from .generic_UNet import Generic_UNet


class Generic_UNet_DA(Generic_UNet):

    def __init__(self, threeD, input_channels, base_num_features, num_classes, 
                 num_conv_per_stage=2,  
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                ):
        
        if threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
        
        self.threeD = threeD
        num_pool = len(pool_op_kernel_sizes)
        feat_map_mul_on_downscale=2
        nonlin = nn.LeakyReLU
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        deep_supervision=True
        dropout_in_localization=False
        upscale_logits=False
        convolutional_pooling=True
        convolutional_upsampling=True
        weightInitializer=InitWeights_He(1e-2)
        final_nonlin = lambda x: x
        max_num_features=None
        basic_block=ConvDropoutNormNonlin
        seg_output_use_bias=False

        super().__init__(input_channels, base_num_features, num_classes, 
                num_pool, num_conv_per_stage=num_conv_per_stage,
                feat_map_mul_on_downscale=feat_map_mul_on_downscale, conv_op=conv_op,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, deep_supervision=deep_supervision, 
                dropout_in_localization=dropout_in_localization,
                final_nonlin=final_nonlin, weightInitializer=weightInitializer, 
                pool_op_kernel_sizes=pool_op_kernel_sizes,
                conv_kernel_sizes=conv_kernel_sizes,
                upscale_logits=upscale_logits, convolutional_pooling=convolutional_pooling, 
                convolutional_upsampling=convolutional_upsampling,
                max_num_features=max_num_features, basic_block=basic_block,
                seg_output_use_bias=seg_output_use_bias)
        self.set_loss()

    def set_loss(self, batch_dice=True):
        self.loss = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        ################# Here we wrap the loss for deep supervision ############
        # we need to know the number of outputs of the network
        net_numpool = len(self.pool_op_kernel_sizes)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights
        # now wrap the loss
        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
        ################# END ###################

    def forward(self, x):
        skips = []
        seg_outputs = []
        decoder_feats = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            decoder_feats.append(x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision:
            logits = tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            logits = tuple([seg_outputs[-1]])

        if self.training:
            return logits, skips, decoder_feats[::-1]
        return logits[0]

