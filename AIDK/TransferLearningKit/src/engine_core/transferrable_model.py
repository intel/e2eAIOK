#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/27/2022 9:39 AM

import torch.nn as nn
import torch
import logging
from .adapter.adversarial.adversarial_adapter import AdversarialAdapter
from collections import namedtuple
import torch.fx

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
        self.__dict__['backbone'] = backbone
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
        if item == '__dict__': # this is special
            return object.__getattribute__(self, item)

        __dict__ = object.__getattribute__(self, "__dict__")
        is_eval = "training" in __dict__ and (not __dict__["training"])
        if is_eval: # eval mode
            if item in ('train',): # from eval to train
                instance = self
            else:
                instance = __dict__['backbone'].__dict__['original'] # the original model
        else: # train mode
            instance = self

        return object.__getattribute__(instance, item)

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
        backbone_output = self.backbone(x)[0]
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
        backbone_output,others = self.backbone(x)
        distiller_input = others[0]
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
        backbone_output_src,others_src = self.backbone(x_src)
        adapter_output_src = self.adapter(others_src[1], backbone_output=backbone_output_src) if self.adapter is not None else None
        backbone_output_tgt,others_tgt = self.backbone(x_tgt)
        adapter_output_tgt = self.adapter(others_tgt[1], backbone_output=backbone_output_tgt) if self.adapter is not None else None
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
        backbone_output_src,others_src= self.backbone(x_src)
        distiller_output_src = self.distiller(others_src[0])
        adapter_output_src = self.adapter(others_src[1])

        backbone_output_tgt,others_tgt = self.backbone(x_tgt)
        distiller_output_tgt = self.distiller(others_tgt[0])
        adapter_output_tgt = self.adapter(others_tgt[1])

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

def extract_distiller_adapter_features(model,intermediate_layer_name_for_distiller,
                                     intermediate_layer_name_for_adapter):

    ''' extract input feature for distiller and adapter

    :param model: model
    :param intermediate_layer_name_for_distiller: the intermediate_layer_name of model for distiller
    :param intermediate_layer_name_for_adapter: the intermediate_layer_name of model for adapter
    :return: modified model
    '''
    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    print("GraphModule")
    gm.graph.print_tabular()

    def find_node(graph,node_name):
        ''' find node from GraphModule by node_name

        :param graph: a GraphModule
        :param node_name: node name
        :return: the target node
        '''
        candidate_nodes =  [node for node in graph.nodes if node.name == node_name]
        if len(candidate_nodes) != 1:
            raise RuntimeError("Can not find layer name [%s] from [%s] : find [%s] result" % (
                node_name, ";".join(node.name for node in graph.nodes), len(candidate_nodes)
            ))
        return candidate_nodes[0]
    ##############         retrieve the node           ##################
    distiller_node = find_node(gm.graph,intermediate_layer_name_for_distiller)
    adapter_node = find_node(gm.graph,intermediate_layer_name_for_adapter)
    output_node = find_node(gm.graph,"output")
    #############          replace the output          ##################
    with gm.graph.inserting_after(output_node):
        original_args = output_node.args[0] # output_node.args is always a tuple, and the first element is the real output
        distiller_adapter_inputs = (distiller_node,adapter_node)
        new_args = (original_args,distiller_adapter_inputs)
        new_node = gm.graph.output(new_args)
        output_node.replace_all_uses_with(new_node)

    gm.graph.erase_node(output_node) # Remove the old node from the graph

    gm.recompile()   # Recompile the forward() method of `gm` from its Graph
    print("After recompile")
    gm.graph.lint()  # Does some checks to make sure the Graph is well-formed.

    return gm

def set_attribute(obj_name,obj,attr_name,attr):
    ''' set attribute for obj

    :param obj_name: obj name
    :param obj: obj
    :param attr_name: attribute name
    :param attr: attribute
    :return:
    '''
    if not hasattr(obj, attr_name):
        obj.__dict__[attr_name] = attr
        logging.info("Set %s for %s"%(attr_name,obj_name))
    else:
        logging.info("Use %s.%s"%(obj_name,attr_name))

def make_transferrable(model,loss,
                       distiller_feature_size,distiller_feature_layer_name,
                       adapter_feature_size,adapter_feature_layer_name,
                       distiller,adapter,transfer_strategy,enable_target_training_label):
    ''' make a model transferrable

    :param model: the backbone model. If model does not have loss method, then use loss argument.
    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param distiller_feature_size: input feature size of distiller
    :param distiller_feature_layer_name: specify the layer output, which is from model, as input feature of distiller
    :param adapter_feature_size: input feature size of adapter
    :param adapter_feature_layer_name: specify the layer output, which is from model, as input feature of adapter
    :param distiller: a distiller
    :param adapter: an adapter
    :param transfer_strategy: transfer strategy
    :param enable_target_training_label: During training, whether use target training label or not.
    :return: a TransferrableModel
    '''
    #################### modify output #######################
    new_model = extract_distiller_adapter_features(model,distiller_feature_layer_name,adapter_feature_layer_name)
    #################### set attribute  #################
    set_attribute("model",model,"loss",loss)
    set_attribute("model", model, "distiller_feature_size", distiller_feature_size)
    set_attribute("model", model, "adapter_feature_size", adapter_feature_size)

    set_attribute("new_model", new_model, "loss", loss)
    set_attribute("new_model", new_model, "distiller_feature_size", distiller_feature_size)
    set_attribute("new_model", new_model, "adapter_feature_size", adapter_feature_size)
    set_attribute("new_model", new_model, "original", model) # remember the orignal one

    return TransferrableModel(new_model,adapter,distiller,transfer_strategy,enable_target_training_label)

def transferrable(loss,distiller_feature_size,distiller_feature_layer_name,
                       adapter_feature_size,adapter_feature_layer_name,
                       distiller,adapter,transfer_strategy,enable_target_training_label):
    ''' a decorator to make instances of class transferrable

    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param distiller_feature_size: input feature size of distiller
    :param distiller_feature_layer_name: specify the layer output, which is from model, as input feature of distiller
    :param adapter_feature_size: input feature size of adapter
    :param adapter_feature_layer_name: specify the layer output, which is from model, as input feature of adapter
    :param distiller: a distiller
    :param adapter: an adapter
    :param transfer_strategy: transfer strategy
    :param enable_target_training_label: During training, whether use target training label or not.
    :return: a wrapper
    '''
    def wrapper(ModelClass):
        def _wrapper(*args, **kargs):
            model = ModelClass(*args, **kargs)
            if not hasattr(model, "loss"):
                setattr(model, "loss", loss)
                logging.info("Set loss function for model")
            else:
                logging.info("Use model.loss()")

            return make_transferrable(model,loss,
                       distiller_feature_size,distiller_feature_layer_name,
                       adapter_feature_size,adapter_feature_layer_name,
                       distiller,adapter,transfer_strategy,enable_target_training_label)
        return _wrapper
    return wrapper