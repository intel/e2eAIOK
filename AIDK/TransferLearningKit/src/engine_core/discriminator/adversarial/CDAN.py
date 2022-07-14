#!/usr/bin/python
# -*- coding: utf-8 -*-
from .adversarial_base import AdversarialNetwork
import torch.nn as nn
import torch
import math
from .grl import GradientReverseLayer
import logging

class RandomLayer(nn.Module):
    ''' A non-trainable random layer, see paper: Conditional Adversarial Domain Adaptation

    '''
    def __init__(self, input_dim_list, output_dim):
        ''' init method

        :param input_dim_list: input dim list
        :param output_dim: output dim
        '''
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim)  # random matrix (no trainable)
                              for i in range(self.input_num)]

    def forward(self, input_list):
        ''' forward

        :param input_list: input tensor list, must match 'input_dim_list'
        :return:
        '''
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        ''' move random matrix to cuda

        :return:
        '''
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class CDAN(AdversarialNetwork):
    ''' CDAN , see paper: Conditional Adversarial Domain Adaptation

    '''
    def __init__(self,in_feature, hidden_size,dropout_rate,grl_coeff_alpha,grl_coeff_high,max_iter,
                 backbone_output_size,enable_random_layer,enable_entropy_weight):
        ''' Init method

        :param in_feature: adversarial network input
        :param hidden_size: hidden size
        :param dropout_rate: dropout rate
        :param grl_coeff_alpha: GradientReverseLayer alpha argument to calculate coef
        :param grl_coeff_high: GradientReverseLayer coef high
        :param max_iter: max iter for one epoch
        :param backbone_output_size: backbone output size
        :param enable_random_layer: whether using random layer
        :param enable_entropy_weight: whether using entropy as weight
        '''
        super(CDAN, self).__init__(in_feature, hidden_size,dropout_rate,grl_coeff_alpha,grl_coeff_high,max_iter)
        if enable_random_layer:
            self._random_layer = RandomLayer([in_feature, backbone_output_size], in_feature)
            logging.info("CDAN enable random layer")
        else:
            self._random_layer = None
        self.entropy_grl = GradientReverseLayer(coeff_alpha=grl_coeff_alpha, coeff_high=grl_coeff_high,
                                        max_iter=max_iter, enable_step=self.training)

        self._enable_entropy_weight = enable_entropy_weight
        if self._enable_entropy_weight:
            logging.info("CDAN enable entropy weight")

    def loss(self, discriminator_feature_source, backbone_output_source, backbone_label_source,
             discriminator_feature_target, backbone_output_target, backbone_label_target,
             tensorboard_writer = None):
        ''' discriminator loss function

        :param discriminator_feature_source: discriminator input feature of source
        :param backbone_output_source: backbone logit output of source
        :param backbone_label_source: backbone ground truth of source
        :param discriminator_feature_target: discriminator input feature of target
        :param backbone_output_target: backbone logit output of target
        :param backbone_label_target: backbone ground truth of target
        :param tensorboard_writer: tensorboard writer
        :return: loss
        '''

        backbone_logit_output = torch.concat([backbone_output_source,backbone_output_target],dim=0)
        backbone_prediction = nn.Softmax(dim=1)(backbone_logit_output)

        in_feature = torch.concat([discriminator_feature_source,discriminator_feature_target],dim=0)
        if self._random_layer is None:
            op_out = torch.bmm(backbone_prediction.unsqueeze(2), in_feature.unsqueeze(1))  # outer product of feature and output
            prediction = self.forward(op_out.view(-1, backbone_prediction.size(1) * in_feature.size(1)))
        else:  # random layer for computation efficiency
            random_out = self._random_layer.forward([in_feature,backbone_prediction])  # fixed random layer
            prediction = self.forward(random_out.view(-1, random_out.size(1)))

        if tensorboard_writer is not None:
            tensorboard_writer.add_histogram('CDAN/prediction', prediction, self.iter_num)

        label = torch.concat([self.make_label(discriminator_feature_source.size(0), True),
                                            self.make_label(discriminator_feature_target.size(0), False)
                                            ], dim=0).view(-1, 1)
        if self._enable_entropy_weight:  # quantify the uncertainty of classifier predictions by the entropy, to emphasize those easy-to-transfer exmples
            entropy = torch.mean(torch.special.entr(backbone_prediction), dim=1)
            entropy = self.entropy_grl(entropy)
            entropy.register_hook(lambda grad: -grad.clone()) # gradient reversal layer(GRL)
            entropy = 1.0 + torch.exp(-entropy)               # entropy as weight
            if tensorboard_writer is not None:
                tensorboard_writer.add_histogram('CDAN/entropy', entropy, self.iter_num)
            ########### weight normalization ###########
            source_mask = torch.ones_like(entropy)
            source_mask[discriminator_feature_source.size(0):] = 0
            source_weight = entropy * source_mask
            source_weight_sum =  torch.sum(source_weight).detach().item() # simplify gradient

            target_weight = entropy - source_weight
            target_weight_sum = torch.sum(target_weight).detach().item()  # simplify gradient

            weight = (source_weight /source_weight_sum + target_weight / target_weight_sum).view(-1, 1)
            sum_weight = torch.sum(weight).detach().item()                # simplify gradient
            if tensorboard_writer is not None:
                # tensorboard_writer.add_histogram('CDAN/source_weight', source_weight, self.iter_num)
                tensorboard_writer.add_scalar('CDAN/sum_source_weight', source_weight_sum, self.iter_num)
                # tensorboard_writer.add_histogram('CDAN/target_weight', target_weight, self.iter_num)
                tensorboard_writer.add_scalar('CDAN/target_weight_sum', target_weight_sum, self.iter_num)
                # tensorboard_writer.add_histogram('CDAN/weight', weight, self.iter_num)
                tensorboard_writer.add_scalar('CDAN/sum_weight', sum_weight, self.iter_num)
            return torch.sum(weight * nn.BCELoss(reduction='none')(prediction, label))/sum_weight
        else:
            return nn.BCELoss()(prediction, label)