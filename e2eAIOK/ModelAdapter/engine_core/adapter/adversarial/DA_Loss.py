#    Copyright 2022, Intel Corporation.
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


from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import SegDiscriminator, FCDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.utils.metric import binary_accuracy, accuracy


class SegDomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional[nn.Module] = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True), 
                 sigmoid=True):
        super(SegDomainAdversarialLoss, self).__init__()

        # define gradient reverselayer type
        # grl = GradientReverseLayer()
        # grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.grl = grl
        
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s_list: List[torch.Tensor], f_t_list: List[torch.Tensor],
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = [
            self.grl(torch.cat((f_s, f_t), dim=0))
            for f_s, f_t in zip(f_s_list, f_t_list)
        ]
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((d_s.size(0), 1)).to(d_s.device)
        d_label_t = torch.zeros((d_t.size(0), 1)).to(d_t.device)
        self.domain_discriminator_accuracy = 0.5 * (
                    binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (
            F.binary_cross_entropy_with_logits(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
            F.binary_cross_entropy_with_logits(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
        )


class SegOutDomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional[nn.Module] = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True), 
                 sigmoid=True):
        super().__init__()
        self.grl = grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((d_s.size(0), 1)).to(d_s.device)
            d_label_t = torch.zeros((d_t.size(0), 1)).to(d_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)


class CACDomainAdversarialLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_weights = kwargs.pop("loss_weight")

        self.encoder_domain_adv = self.decoder_domain_adv = self.seg_domain_adv = None

        if self.loss_weights[0] > 0:
            encoder_domain_discri = SegDiscriminator(**kwargs)
            self.encoder_domain_adv = SegDomainAdversarialLoss(encoder_domain_discri)
            
        if self.loss_weights[1] > 0:
            decoder_domain_discri = SegDiscriminator(**kwargs)
            self.decoder_domain_adv = SegDomainAdversarialLoss(decoder_domain_discri)
            
        if self.loss_weights[2] > 0:
            seg_domain_discri = FCDiscriminator()
            self.seg_domain_adv = SegOutDomainAdversarialLoss(seg_domain_discri)

    def get_parameters(self):
        parameters = []
        if self.encoder_domain_adv:
            parameters += [{
                'params': self.encoder_domain_adv.parameters(),
            }]
        if self.decoder_domain_adv:
            parameters += [{
                'params': self.decoder_domain_adv.parameters(),
            }]
        if self.seg_domain_adv:
            parameters += [{
                'params': self.seg_domain_adv.parameters(),
            }]
        return parameters
    
    def get_metrics(self):
        metric = {}
        if self.encoder_domain_adv:
            metric['train/encoder_adv_acc'] = self.encoder_domain_adv.domain_discriminator_accuracy.item() / 100.

        if self.decoder_domain_adv:
            metric['train/decoder_adv_acc'] = self.decoder_domain_adv.domain_discriminator_accuracy.item() / 100.

        if self.seg_domain_adv:
            metric['train/seg_adv_acc'] = self.seg_domain_adv.domain_discriminator_accuracy.item() / 100.
        return metric

    def forward(self, *data):
        source_data, target_data, source_label = data
        source_output, encoder_f_s, decoder_f_s = source_data
        output, encoder_f_t, decoder_f_t = target_data

        loss_list = []
        if self.encoder_domain_adv:
            gan_encoder = self.loss_weights[0] * self.encoder_domain_adv(encoder_f_s, encoder_f_t)
            loss_list.append(gan_encoder)
        if self.decoder_domain_adv:
            gan_decoder = self.loss_weights[1] * self.decoder_domain_adv(decoder_f_s, decoder_f_t)
            loss_list.append(gan_decoder)
            # from thop import profile
            # macs, params = profile(self.decoder_domain_adv, inputs=(decoder_f_s, decoder_f_t, ))
            # print(f'flops: {macs}, params: {params}')
            # sys.exit(0)
        if self.seg_domain_adv:
            gan_seg = self.loss_weights[2] * self.seg_domain_adv(
                source_label[0],
                torch.argmax(source_output[0], dim=1, keepdim=True)
            )
            loss_list.append(gan_seg)

        if not loss_list:
            return None
        l = loss_list[0]
        for item in loss_list[1:]:
            l += item
        return l

