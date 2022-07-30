#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/27/2022 9:39 AM

import torch.nn as nn
import torch
import logging
from .adapter.adversarial.adversarial_adapter import AdversarialAdapter
from collections import namedtuple

from enum import Enum
class TransferStrategy(Enum):
    '''Transfer Strategy

    '''
    OnlyFinetuneStrategy = 1         # pretraining-finetuning, and the pretrained model is the same as the target model
    OnlyDistillationStrategy = 10    # distillation
    OnlyDomainAdaptionStrategy = 20  # domain adaption
    FinetuneAndDomainAdaptionStrategy = 30 # pretraining-finetuning and domain adaption
    DistillationAndAdaptionStrategy = 40   # distillation and domain adaption


TransferrableModelOutput = namedtuple('TransferrableModelOutput',
                                      ['backbone_output','distiller_output','adapter_output'])

class TransferrableModelLoss:
    ''' TransferrableModel Loss, which is composed by total_loss,backbone_loss,distiller_loss,adapter_loss

    '''
    def __init__(self,total_loss,backbone_loss,distiller_loss,adapter_loss):
        ''' Init method

        :param total_loss: total loss
        :param backbone_loss: backbone loss, could be none
        :param distiller_loss: distiller loss, could be none
        :param adapter_loss: adapter loss, could be none
        '''
        self.total_loss = total_loss
        self.backbone_loss = backbone_loss
        self.distiller_loss = distiller_loss
        self.adapter_loss = adapter_loss

    def backward(self):
        self.total_loss.backward()

class TransferrableModel(nn.Module):
    ''' A wrapper to make model transferrable

    '''
    def __init__(self,backbone,adapter,distiller,transfer_strategy,enable_target_training_label):
        ''' Init method

        :param backbone: the backbone model to be wrapped
        :param adapter: an adapter to perform domain adaption
        :param distiller: a distiller to perform knowledge distillation
        :param transfer_strategy: transfer strategy
        :param enable_target_training_label: During training, whether use target training label or not.
        '''
        super(TransferrableModel, self).__init__()
        self.backbone = backbone
        super(TransferrableModel, self).__dict__['backbone'] = backbone
        self.adapter = adapter
        self.distiller = distiller
        self.transfer_strategy = transfer_strategy
        self.enable_target_training_label = enable_target_training_label

        if self.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
            if not self.enable_target_training_label:
                raise RuntimeError("Must enable target training label when only finetune.")

    def __getattribute__(self, item):
        ''' Change the action for training and evaluation. When training, use self. When evaluation, use self._backbone.
        It is called when visit any property.

        :param item:
        :return:
        '''
        # for special build-in method
        # or train/eval mode change, otherwise it is always in backbone's eval mode
        # or train mode
        if item.startswith("__") or \
                item in ('train','eval','training') or \
                object.__getattribute__(self,"training"):
            return object.__getattribute__(self,item)
        else:   # eval mode
            backbone = object.__getattribute__(self,'backbone')
            return object.__getattribute__(backbone,item)

    def __str__(self):
        _str = 'TransferrableModel:transfer_strategy = %s, enable_target_training_label = %s\n'%(
            self.transfer_strategy, self.enable_target_training_label)
        _str += "\tbackbone:%s\n"%self.backbone
        _str += "\tbackbone:%s\n" % self.adapter
        _str += "\tbackbone:%s\n" % self.distiller
        return _str

    # def init_weight(self):
    #     if self._transfer_strategy == TransferStrategy.OnlyFinetuneStrategy or\
    #         self._transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy:
    #         pass
    #     else:
    #         pass

    def _finetune_forward(self,x):
        ''' forward for OnlyFinetuneStrategy

        :param x: backbone input
        :return: TransferrableModelOutput
        '''
        backbone_output = self.backbone(x)
        return TransferrableModelOutput(backbone_output,None,None)

    def _finetune_loss(self,output,label):
        ''' loss for OnlyFinetuneStrategy

        :param output: backbone predict
        :param label: backbone ground truth
        :return: TransferrableModelLoss
        '''
        backbone_loss = self.backbone.loss(output, label)
        return TransferrableModelLoss(backbone_loss,backbone_loss,None,None)

    def _distillation_forward(self,x):
        ''' forward for OnlyDistillationStrategy

        :param x: backbone input
        :return: TransferrableModelOutput
        '''
        backbone_output = self.backbone(x)
        distiller_input = self.backbone.adapter_input
        distiller_output = self.distiller(distiller_input)
        return TransferrableModelOutput(backbone_output, distiller_output,None)

    def _distillation_loss(self,student_output,teacher_output,student_label):
        ''' loss for OnlyDistillationStrategy

        :param student_output: student output
        :param teacher_output: teacher output
        :param student_label: student label
        :return: TransferrableModelLoss
        '''
        if self.enable_target_training_label:
            student_loss = self.backbone.loss(student_output, student_label)
        else:
            student_loss = 0.0
        distiller_loss = self.distiller.loss(teacher_output, student_output)
        total_loss = student_loss + distiller_loss
        return TransferrableModelLoss(total_loss,student_loss,distiller_loss,None)

    def _adaption_forward(self,x_tgt,x_src):
        ''' forward for OnlyDomainAdaptionStrategy or FinetuneAndDomainAdaptionStrategy

        :param x_tgt: target domain data
        :param x_src: source domain data
        :return: (TransferrableModelOutput for x_tgt, TransferrableModelOutput for x_src)
        '''
        backbone_output_src = self.backbone(x_src)
        adapter_output_src = self.adapter(self.backbone.adapter_input, backbone_output=backbone_output_src) if self.adapter is not None else None
        backbone_output_tgt = self.backbone(x_tgt)
        adapter_output_tgt = self.adapter(self.backbone.adapter_input, backbone_output=backbone_output_tgt) if self.adapter is not None else None
        return (
            TransferrableModelOutput(backbone_output_tgt,None,adapter_output_tgt),
            TransferrableModelOutput(backbone_output_src, None, adapter_output_src),
        )

    def _adaption_loss(self,backbone_output_tgt, backbone_output_src,
                       adapter_output_tgt, adapter_output_src,
                       backbone_label_tgt,backbone_label_src):
        ''' loss for OnlyDomainAdaptionStrategy or FinetuneAndDomainAdaptionStrategy

        :param backbone_output_tgt: backbone main output of target domain
        :param backbone_output_src: backbone main output of source domain
        :param adapter_output_tgt: adapter output of target domain
        :param adapter_output_src: adapter output of source domain
        :param backbone_label_tgt: backbone label of target domain
        :param backbone_label_src: backbone label of source domain
        :return:  TransferrableModelLoss
        '''
        if self.enable_target_training_label:
            target_loss = self.backbone.loss(backbone_output_tgt, backbone_label_tgt)
        else:
            target_loss = 0.0
        src_loss = self.backbone.loss(backbone_output_src, backbone_label_src)
        backbone_loss = src_loss + target_loss

        if self.adapter is None:
            adapter_loss = 0.0
        else:
            adapter_label_tgt = AdversarialAdapter.make_label(adapter_output_tgt.size(0), is_source=False).view(-1, 1)
            adapter_label_src = AdversarialAdapter.make_label(adapter_output_src.size(0), is_source=True).view(-1, 1)
            adapter_loss = self.adapter.loss(adapter_output_tgt, adapter_label_tgt, backbone_output=backbone_output_tgt) + \
                           self.adapter.loss(adapter_output_src, adapter_label_src, backbone_output=backbone_output_src)

        total_loss = backbone_loss + adapter_loss

        return TransferrableModelLoss(total_loss,backbone_loss,None,adapter_loss)

    def _distillation_and_adaption_forward(self,x_tgt,x_src):
        ''' forward for DistillationAndAdaptionStrategy

        :param x_tgt: target domain data
        :param x_src: source domain data
        :return:  (TransferrableModelOutput for x_tgt, TransferrableModelOutput for x_src)
        '''
        backbone_output_src= self.backbone(x_src)
        adapter_output_src = self.adapter(self.backbone.adapter_input)
        distiller_output_src = self.distiller(self.backbone.distiller_input)
        backbone_output_tgt = self.backbone(x_tgt)
        adapter_output_tgt = self.adapter(self.backbone.adapter_input)
        distiller_output_tgt = self.distiller(self.backbone.distiller_input)

        return TransferrableModelOutput(backbone_output_tgt,distiller_output_tgt,adapter_output_tgt), \
               TransferrableModelOutput(backbone_output_src,distiller_output_src,adapter_output_src)

    def _distillation_and_adaption_loss(self,backbone_output_tgt, backbone_output_src,
                       teacher_output_tgt, teacher_output_src,
                       adapter_output_tgt, adapter_output_src,
                       backbone_label_tgt,backbone_label_src):
        ''' loss for DistillationAndAdaptionStrategy

        :param backbone_output_tgt: student backbone main output of target domain
        :param backbone_output_src: student backbone main output of source domain
        :param teacher_output_tgt: teacher output of target domain
        :param teacher_output_src: teacher output of source domain
        :param adapter_output_tgt: adapter output of target domain
        :param adapter_output_src: adapter output of source domain
        :param backbone_label_tgt: student backbone label of target domain
        :param backbone_label_src: student backbone label of source domain
        :return: TransferrableModelLoss
        '''
        if self.enable_target_training_label:
            target_loss = self.backbone.loss(backbone_output_tgt, backbone_label_tgt)
        else:
            target_loss = 0.0
        src_loss = self.backbone.loss(backbone_output_src, backbone_label_src)

        backbone_loss = target_loss + src_loss

        adapter_label_tgt = AdversarialAdapter.make_label(adapter_output_tgt.size(0), is_source=False).view(-1, 1)
        adapter_label_src = AdversarialAdapter.make_label(adapter_output_src.size(0), is_source=True).view(-1, 1)

        distiller_loss = self.distiller.loss(teacher_output_tgt, backbone_output_tgt) + \
                         self.distiller.loss(teacher_output_src, backbone_output_src)
        adapter_loss = self.adapter.loss(adapter_output_tgt, adapter_label_tgt) + \
                       self.adapter.loss(adapter_output_src, adapter_label_src)
        total_loss = backbone_loss + distiller_loss +adapter_loss

        return TransferrableModelLoss(total_loss,backbone_loss,distiller_loss,adapter_loss)

    def forward(self,x):
        ''' forward

        :param x: input, may be tensor or tuple
        :return: TransferrableModelOutput
        '''
        if self.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
            x = x if isinstance(x,torch.Tensor) else x[0] # x may be a tuple
            return self._finetune_forward(x)
        elif self.transfer_strategy == TransferStrategy.OnlyDistillationStrategy:
            x = x if isinstance(x, torch.Tensor) else x[0] # x may be a tuple
            return self._distillation_forward(x)
        elif self.transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy or \
            self.transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy:
            if isinstance(x,torch.Tensor):
                raise RuntimeError("TransferrableModel forward for OnlyDomainAdaptionStrategy should be tuple or list, not be %s"%type(x))
            return self._adaption_forward(x[0],x[1])
        elif self.transfer_strategy == TransferStrategy.DistillationAndAdaptionStrategy:
            if isinstance(x, torch.Tensor):
                raise RuntimeError("TransferrableModel forward for DistillationAndAdaptionStrategy should be tuple or list, not be %s" % type(x))
            return self._distillation_and_adaption_forward(x[0], x[1])
        else:
            raise RuntimeError("Unknown transfer_strategy [%s] " % self.transfer_strategy)

    def loss(self,output,label):
        ''' loss function

        :param output: output of forward()
        :param label: label (may be a tuple)
        :return: TransferrableModelLoss
        '''
        if self.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
            return self._finetune_loss(output.backbone_output,label if isinstance(label, torch.Tensor) else label[0])
        elif self.transfer_strategy == TransferStrategy.OnlyDistillationStrategy:
            return self._distillation_loss(output.backbone_output,output.distiller_output,label if isinstance(label, torch.Tensor) else label[0])
        elif self.transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy or \
            self.transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy:
            return self._adaption_loss(output[0].backbone_output,output[1].backbone_output,
                                       output[0].adapter_output,output[1].adapter_output,
                                       label[0],label[1])
        elif self.transfer_strategy == TransferStrategy.DistillationAndAdaptionStrategy:
            return self._distillation_and_adaption_loss(
                                        output[0].backbone_output, output[1].backbone_output,
                                        output[1].distiller_output, output[1].distiller_output,
                                        output[0].adapter_output,output[1].adapter_output,
                                        label[0], label[1])
        else:
            raise RuntimeError("Unknown transfer_strategy [%s] " % self.transfer_strategy)

    def get_training_metrics(self,output,label,loss_value,metric_fn_map):
        ''' get the training metrics

        :param output: output of forward()
        :param label: label (may be a tuple)
        :param loss_value: output of loss()
        :param metric_fn_map:  metric function map, which map metric name to metric function
        :return: metrics
        '''
        metric_values = {"total_loss": loss_value.total_loss}
        if loss_value.backbone_loss is not None:
            metric_values["backbone_loss"] = loss_value.backbone_loss
        if loss_value.distiller_loss is not None:
            metric_values["distiller_loss"] = loss_value.distiller_loss
        if loss_value.adapter_loss is not None:
            metric_values["adapter_loss"] = loss_value.adapter_loss

        for (metric_name, metric_fn) in sorted(metric_fn_map.items()):
            if self.transfer_strategy in [
                TransferStrategy.OnlyDomainAdaptionStrategy,
                TransferStrategy.FinetuneAndDomainAdaptionStrategy,
                TransferStrategy.DistillationAndAdaptionStrategy]:
                if (isinstance(label, torch.Tensor)):
                    raise RuntimeError("label in adaption must be a collection, not %s" % (type(label)))
                index = 0 if self.enable_target_training_label else 1  # 0 is target domain output, 1 is source domain output
                metric_value = metric_fn(output[index].backbone_output, label[index])
                metric_values[metric_name] = metric_value
            else:
                metric_value = metric_fn(output.backbone_output, label if isinstance(label, torch.Tensor) else label[0])
                metric_values[metric_name] = metric_value

        return metric_values

def make_transferrable(model,adapter,distiller,transfer_strategy,enable_target_training_label):
    ''' make a model transferrable

    :param model: the backbone model
    :param adapter: an adapter
    :param distiller: a distiller
    :param transfer_strategy: transfer strategy
    :param enable_target_training_label: During training, whether use target training label or not.
    :return: a TransferrableModel
    '''
    return TransferrableModel(model,adapter,distiller,transfer_strategy,enable_target_training_label)

def transferrable(adapter,distiller,transfer_strategy,enable_target_training_label):
    ''' a decorator to make instances of class transferrable

    :param adapter: an adapter
    :param distiller: a distiller
    :param transfer_strategy: transfer strategy
    :param enable_target_training_label: During training, whether use target training label or not.
    :return:
    '''
    def wrapper(ModelClass):
        def _wrapper(*args, **kargs):
            return TransferrableModel(ModelClass(*args, **kargs),adapter,distiller,transfer_strategy,enable_target_training_label)
        return _wrapper
    return wrapper