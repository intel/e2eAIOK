import torch.nn as nn
import torch
import torch.fx

from enum import Enum
class TransferStrategy(Enum):
    '''Transfer Strategy

    '''
    OnlyFinetuneStrategy = 1         # pretraining-finetuning, and the pretrained model is the same as the target model
    OnlyDistillationStrategy = 10    # distillation
    OnlyDomainAdaptionStrategy = 20  # domain adaption
    FinetuneAndDomainAdaptionStrategy = 30 # pretraining-finetuning and domain adaption
    DistillationAndDomainAdaptionStrategy = 40   # distillation and domain adaption

ALL_STRATEGIES = [
    TransferStrategy.OnlyFinetuneStrategy,
    TransferStrategy.OnlyDistillationStrategy,
    TransferStrategy.OnlyDomainAdaptionStrategy,
    TransferStrategy.FinetuneAndDomainAdaptionStrategy,
    TransferStrategy.DistillationAndDomainAdaptionStrategy
]

class TransferrableModelOutput:
    ''' TransferrableModel Output, which is composed by backbone_output,distiller_output,adapter_output

    '''
    def __init__(self,backbone_output,distiller_output,adapter_output):
        ''' Init method

        :param backbone_output: backbone output, could be none
        :param distiller_output: distiller output, could be none
        :param adapter_output: adapter output, could be none
        '''
        self.backbone_output = backbone_output
        self.distiller_output = distiller_output
        self.adapter_output = adapter_output

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
    def __init__(self,backbone,adapter,distiller,transfer_strategy,enable_target_training_label,
                backbone_loss_weight=1.0, distiller_loss_weight=0.0, adapter_loss_weight=0.0):
        ''' Init method

        :param backbone: the backbone model to be wrapped
        :param adapter: an adapter to perform domain adaption
        :param distiller: a distiller to perform knowledge distillation
        :param transfer_strategy: transfer strategy
        :param enable_target_training_label: During training, whether use target training label or not.
        '''
        super(TransferrableModel, self).__init__()
        self.backbone = backbone

        self.adapter = adapter
        self.distiller = distiller
        self.transfer_strategy = transfer_strategy
        self.enable_target_training_label = enable_target_training_label
        self.backbone_loss_weight = backbone_loss_weight
        self.distiller_loss_weight = distiller_loss_weight
        self.adapter_loss_weight = adapter_loss_weight

        if self.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
            if not self.enable_target_training_label:
                raise RuntimeError("Must enable target training label when only finetune.")
    
    def get_backbone(self):
        ''' Get backbone

        :return: backbone
        '''
        return self.backbone

    def __str__(self):
        _str = 'TransferrableModel:transfer_strategy = %s, enable_target_training_label = %s\n'%(
            self.transfer_strategy, self.enable_target_training_label)
        _str += "\tbackbone:%s\n"%self.backbone
        _str += "\tadapter:%s\n" % self.adapter
        _str += "\tdistiller:%s\n" % self.distiller
        return _str

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

        :param x: input if not distiller.use_saved_logits; (input, save_values) if distiller.use_saved_logits
        :return: TransferrableModelOutput
        '''
        x_input = x[0] if self.distiller.use_saved_logits else x
        backbone_output = self.backbone(x_input)
        distiller_output = self.distiller(x)
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
        
        distiller_loss = self.distiller.loss(teacher_output, student_output, target=student_label)

        total_loss = self.backbone_loss_weight * student_loss + \
                     self.distiller_loss_weight * distiller_loss

        return TransferrableModelLoss(total_loss,student_loss,distiller_loss,None)

    def _adaption_forward(self,x_tgt,x_src):
        ''' forward for OnlyDomainAdaptionStrategy or FinetuneAndDomainAdaptionStrategy

        :param x_tgt: target domain data
        :param x_src: source domain data
        :return: (TransferrableModelOutput for x_tgt, TransferrableModelOutput for x_src)
        '''
        backbone_output_tgt = self.backbone(x_tgt)
        backbone_output_src = self.backbone(x_src)
        return (
            TransferrableModelOutput(backbone_output_tgt,None, None),
            TransferrableModelOutput(backbone_output_src, None, None),
        )

    def _adaption_loss(self, input_sample, label, source_pred_to_target_pred=lambda x:x):
        ''' loss for OnlyDomainAdaptionStrategy or FinetuneAndDomainAdaptionStrategy

        :param input_sample: tuple of (source_data, target_data)
        :param label: tuple of (source_label, target_label)
        :return:  TransferrableModelLoss
        '''
        source_data, data = input_sample
        source_label, target = label

        source_output, *source_feat = self.backbone(source_data)
        source_loss = self.backbone.loss(source_output, source_label)
        
        target_loss = None
        output, *target_feat = self.backbone(data)
        output = [source_pred_to_target_pred(item) for item in output]
        if self.enable_target_training_label:
            target_loss = self.backbone.loss(output, target)
        
        adv_loss = self.adapter(*(
            (source_output, *source_feat),
            (output, *target_feat),
            source_label
        ))

        # calc total loss
        total_loss = source_loss * self.backbone_loss_weight
        if self.enable_target_training_label:
            total_loss += target_loss * self.backbone_loss_weight
        if adv_loss:
            total_loss += adv_loss

        # return (total_loss, adv_loss, source_loss, target_loss)
        return TransferrableModelLoss(total_loss,source_loss * self.backbone_loss_weight,None,adv_loss)

    def forward(self,x):
        ''' forward

        :param x: input, may be tensor or tuple
        :return: TransferrableModelOutput
        '''
        if self.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
            x = x if isinstance(x,torch.Tensor) else x[0] # x may be a tuple
            return self._finetune_forward(x)
        elif self.transfer_strategy == TransferStrategy.OnlyDistillationStrategy:
            return self._distillation_forward(x)
        elif self.transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy:
            if isinstance(x,torch.Tensor):
                raise RuntimeError("TransferrableModel forward for OnlyDomainAdaptionStrategy should be tuple or list, not be %s"%type(x))
            return self._adaption_forward(x[0],x[1])
        elif self.transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy:
            if isinstance(x,torch.Tensor):
                raise RuntimeError("TransferrableModel forward for FinetuneAndDomainAdaptionStrategy should be tuple or list, not be %s"%type(x))
            return self._adaption_forward(x[0],x[1])
        elif self.transfer_strategy == TransferStrategy.DistillationAndDomainAdaptionStrategy:
            if isinstance(x, torch.Tensor):
                raise RuntimeError("TransferrableModel forward for DistillationAndDomainAdaptionStrategy should be tuple or list, not be %s" % type(x))
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
            label = label if isinstance(label, torch.Tensor) else label[0]
            return self._distillation_loss(output.backbone_output,output.distiller_output,label)
        elif self.transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy or \
            self.transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy:
            return self._adaption_loss(output, label)
        elif self.transfer_strategy == TransferStrategy.DistillationAndDomainAdaptionStrategy:
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
                TransferStrategy.DistillationAndDomainAdaptionStrategy]:
                if (isinstance(label, torch.Tensor)):
                    raise RuntimeError("label in adaption must be a collection, not %s" % (type(label)))

                # 0 is target domain output, 1 is source domain output
                metric_value = metric_fn(output[1].backbone_output, label[1])
                metric_value = metric_value[0] if isinstance(metric_value, list) else metric_value
                metric_values["%s_src_domain"%metric_name] = metric_value
                if self.enable_target_training_label:
                    metric_value = metric_fn(output[0].backbone_output, label[0])
                    metric_value = metric_value[0] if isinstance(metric_value, list) else metric_value
                    metric_values["%s_target_domain" % metric_name] = metric_value
            else:
                metric_value = metric_fn(output.backbone_output, label if isinstance(label, torch.Tensor) else label[0])
                metric_value = metric_value[0] if isinstance(metric_value, list) else metric_value
                metric_values[metric_name] = metric_value

        return metric_values


def _make_transferrable(model, loss,
                        finetunner, distiller, adapter,
                        transfer_strategy, enable_target_training_label,
                        backbone_loss_weight=1.0, distiller_loss_weight=0.0, adapter_loss_weight=0.0):
    ''' make a model transferrable

    :param model: the backbone model. If model does not have loss method, then use loss argument.
    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param finetunner: a finetunner. If no finetune, then could be none.
    :param distiller: a distiller. If no distillation, then could be none.
    :param adapter: an adapter. If no adaption, then could be none.
    :param transfer_strategy: transfer strategy.
    :param enable_target_training_label: During training, whether use target training label or not.
    :param backbone_loss_weight: the weight of backbone model loss, default=1
    :param distiller_loss_weight: the weight of distiller_loss_weight, default=0
    :param adapter_loss_weight: the weight of adapter_loss_weight, default=0
    :return: a TransferrableModel
    '''
    ######## check input #####
    if not hasattr(model,"loss"):
        if loss is None:
            raise RuntimeError("Need loss for model")
        else:
            setattr(model,'loss',loss)
    if transfer_strategy in [TransferStrategy.OnlyDistillationStrategy,
                             TransferStrategy.DistillationAndDomainAdaptionStrategy]:
        if distiller is None:
            raise RuntimeError("Need distiller for Distillation")
    if transfer_strategy in [TransferStrategy.OnlyDomainAdaptionStrategy,
                             TransferStrategy.FinetuneAndDomainAdaptionStrategy,
                             TransferStrategy.DistillationAndDomainAdaptionStrategy]:
        if adapter is None:
            raise RuntimeError("Need adapter for Adaption")
    if transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
        if not enable_target_training_label:
            raise RuntimeError("Need enable_target_training_label for OnlyFinetune")
    ###################### initialize ####################
    if transfer_strategy in [TransferStrategy.OnlyFinetuneStrategy,
                                      TransferStrategy.FinetuneAndDomainAdaptionStrategy]:
        finetunner.finetune_network(model)
    
    return TransferrableModel(model,adapter,distiller,transfer_strategy,enable_target_training_label,backbone_loss_weight,distiller_loss_weight,adapter_loss_weight)

def _transferrable(loss, finetunner, distiller, adapter,
                   transfer_strategy, enable_target_training_label,
                   backbone_loss_weight=1.0, distiller_loss_weight=0.0, adapter_loss_weight=0.0):
    ''' a decorator to make instances of class transferrable

    :param loss: loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param finetunner: a finetunner. If no finetune, then could be none.
    :param distiller: a distiller. If no distillation, then could be none.
    :param adapter: an adapter. If no adaption, then could be none.
    :param transfer_strategy: transfer strategy
    :param enable_target_training_label: During training, whether use target training label or not.
    :return: a wrapper
    '''
    def wrapper(ModelClass):
        def _wrapper(*args, **kargs):
            model = ModelClass(*args, **kargs)
            return _make_transferrable(model, loss,
                                       finetunner, distiller, adapter,
                                       transfer_strategy, enable_target_training_label,
                                       backbone_loss_weight,distiller_loss_weight,adapter_loss_weight)
        return _wrapper
    return wrapper

############ simple API #############
def make_transferrable_with_finetune(model,loss,finetunner):
    ''' make transferrable with finetune strategy

    :param model: the backbone model. If model does not have loss method, then use loss argument.
    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param finetunner: a finetunner.
    :return: a TransferrableModel
    '''
    return _make_transferrable(model=model, loss=loss,
                               finetunner=finetunner, distiller=None, adapter=None,
                               transfer_strategy=TransferStrategy.OnlyFinetuneStrategy,
                               enable_target_training_label=True)

def transferrable_with_finetune(loss,finetunner):
    ''' a decorator to make instances of class transferrable with finetune strategy

    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param finetunner: a finetunner.
    :return: a wrapper
    '''
    return _transferrable(loss=loss,
                          finetunner=finetunner, distiller=None, adapter=None,
                          transfer_strategy=TransferStrategy.OnlyFinetuneStrategy,
                          enable_target_training_label=True)

def make_transferrable_with_knowledge_distillation(model,loss,distiller,
                                                   enable_target_training_label=True,backbone_loss_weight=0.1,distiller_loss_weight=0.9):
    '''  make transferrable with knowledge distillation strategy

    :param model: the backbone model. If model does not have loss method, then use loss argument.
    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param distiller: a distiller.
    :param enable_target_training_label: During training, whether use target training label or not.
    :return: a TransferrableModel
    '''
    return _make_transferrable(model=model, loss=loss,
                               finetunner=None, distiller=distiller, adapter=None,
                               transfer_strategy=TransferStrategy.OnlyDistillationStrategy,
                               enable_target_training_label=enable_target_training_label,
                               backbone_loss_weight=backbone_loss_weight,distiller_loss_weight=distiller_loss_weight)

def transferrable_with_knowledge_distillation(loss,distiller,
                                                   enable_target_training_label,backbone_loss_weight,distiller_loss_weight):
    ''' a decorator to make instances of class transferrable with distillation strategy

    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param distiller: a distiller.
    :param enable_target_training_label: During training, whether use target training label or not.
    :return: a wrapper
    '''
    return _transferrable(loss=loss,
                          finetunner=None, distiller=distiller, adapter=None,
                          transfer_strategy=TransferStrategy.OnlyDistillationStrategy,
                          enable_target_training_label=enable_target_training_label,
                          backbone_loss_weight=backbone_loss_weight,distiller_loss_weight=distiller_loss_weight)

def make_transferrable_with_domain_adaption(model,loss,adapter,
                                            enable_target_training_label,backbone_loss_weight,adapter_loss_weight):
    ''' make transferrable with domain adaption strategy

    :param model: the backbone model. If model does not have loss method, then use loss argument.
    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param adapter: an adapter.
    :param enable_target_training_label:  During training, whether use target training label or not.
    :return: a TransferrableModel
    '''
    return _make_transferrable(model=model, loss=loss,
                               finetunner=None, distiller=None, adapter=adapter,
                               transfer_strategy=TransferStrategy.OnlyDomainAdaptionStrategy,
                               enable_target_training_label=enable_target_training_label,
                               backbone_loss_weight=backbone_loss_weight,adapter_loss_weight=adapter_loss_weight)

def transferrable_with_domain_adaption(loss,adapter,
                                        enable_target_training_label,backbone_loss_weight,adapter_loss_weight):
    ''' a decorator to make instances of class transferrable with adaption strategy

    :param loss : loss function for model,signature: loss(output_logit, label). If model has loss attribute, then loss could be none.
    :param adapter: an adapter.
    :param enable_target_training_label:  During training, whether use target training label or not.
    :return: a wrapper
    '''
    return _transferrable(loss=loss,
                          finetunner=None, distiller=None, adapter=adapter,
                          transfer_strategy=TransferStrategy.OnlyDomainAdaptionStrategy,
                          enable_target_training_label=enable_target_training_label,
                          backbone_loss_weight=backbone_loss_weight,adapter_loss_weight=adapter_loss_weight)
