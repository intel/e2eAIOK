# Copyright (c) 2022, Intel Corporation

## refer to https://github.com/megvii-research/mdistiller

# MIT License

# Copyright (c) 2022 MEGVII Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# detectron2

# Copyright 2020 - present, Facebook, Inc

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# RepDistiller

# Copyright (c) 2020, Yonglong Tian

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_distiller import BasicDistiller

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(BasicDistiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""
    def __init__(self, pretrained_model, alpha, beta, temperature, warmup, is_frozen=True, use_saved_logits=True, topk=0, num_classes=10, teacher_type=None):
        ''' Init method.

        :param pretrained_model: the pretrained model as teacher
        :param alpha: the alpha for DKD 
        :param beta: the beta for DKD 
        :param temperature: the temperature for DKD 
        :param warmup: warmup epoches for DKD
        :param is_frozen: whether frozen teacher when training
        :param use_saved_logits: whether train with pre-saved logits
        :param topk: if use logits, save top k logits, 0 means save all logits
        :param num_classes: num of classification classes
        :param teacher_type: teacher model type
        '''
        super(DKD, self).__init__(pretrained_model, is_frozen, use_saved_logits, topk, num_classes, teacher_type)
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup = warmup

    def loss(self,teacher_logits, student_logits,**kwargs):
        ''' Loss function.

        :param teacher_logits: the teacher logits
        :param student_logits: the student logits
        '''
        distiller_loss = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
                            student_logits, teacher_logits,kwargs["target"],
                            self.alpha, self.beta, self.temperature,)
        return distiller_loss
