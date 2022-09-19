#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/4/2022 2:02 PM

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"test"))
import torch
import torchvision
from engine_core.transferrable_model import TransferrableModel,\
    TransferStrategy,TransferrableModelOutput,TransferrableModelLoss,\
    extract_distiller_adapter_features,set_attribute,_make_transferrable,_transferrable,ALL_STRATEGIES,\
    make_transferrable_with_finetune, make_transferrable_with_knowledge_distillation,\
    make_transferrable_with_domain_adaption,make_transferrable_with_finetune_and_domain_adaption,\
    make_transferrable_with_knowledge_distillation_and_domain_adaption
from engine_core.distiller.basic_distiller import BasicDistiller
from engine_core.finetunner.basic_finetunner import BasicFinetunner
from engine_core.adapter.adversarial.dann_adapter import DANNAdapter
from engine_core.adapter.adversarial.adversarial_adapter import AdversarialAdapter
from dataset.office31 import Office31
from dataset.composed_dataset import ComposedDataset
from training.metrics import accuracy
from training.utils import initWeights
from utils import tensor_near_equal
import torch.fx

class TestExtractDistillerAdapterFeatures:
    ''' test extract_distiller_adapter_features

    '''
    def setup(self):
        self.model = torchvision.models.resnet18(pretrained=False)
        self.input = torch.randn([16, 3, 224, 224])
        node_names = [node.name for node in torch.fx.symbolic_trace(self.model).graph.nodes]
        self.adapter_layer_name = node_names[-3] # [... ,avgpool, flatten, fc, output]

    def test_all_component_is_none(self):
        ''' test all component is none

        :return:
        '''
        modified_model = extract_distiller_adapter_features(self.model, "", "")
        assert type(modified_model(self.input)) == tuple
        assert modified_model(self.input)[1] == (None, None)
        assert tensor_near_equal(self.model(self.input) ,modified_model(self.input)[0])
    def test_adapter_is_none(self):
        ''' test adapter is none

        :return:
        '''
        modified_model = extract_distiller_adapter_features(self.model, "x", "")
        assert type(modified_model(self.input)) == tuple
        assert tensor_near_equal(self.input, modified_model(self.input)[1][0])  # distiller input = x
        assert modified_model(self.input)[1][1] is None
        assert tensor_near_equal(self.model(self.input),  modified_model(self.input)[0])
    def test_distiller_is_none(self):
        ''' test distiller is none

        :return:
        '''
        modified_model = extract_distiller_adapter_features(self.model, "", self.adapter_layer_name)
        assert type(modified_model(self.input)) == tuple
        assert modified_model(self.input)[1][0] is None
        assert type(modified_model(self.input)[1][1]) is torch.Tensor
        assert tensor_near_equal(self.model(self.input), modified_model(self.input)[0])
    def test_all_is_valid(self):
        ''' test all component is valid

        :return:
        '''
        modified_model = extract_distiller_adapter_features(self.model, "x", self.adapter_layer_name)
        assert type(modified_model(self.input)) == tuple
        assert tensor_near_equal(self.input, modified_model(self.input)[1][0])  # distiller input = x
        assert type(modified_model(self.input)[1][1]) is torch.Tensor
        assert tensor_near_equal(self.model(self.input), modified_model(self.input)[0])
    def test_invalid_layer_name(self):
        ''' test invalid layer name

        :return:
        '''
        for layer_name in ['invalid_name','ABC',self.adapter_layer_name.upper()]:
            with pytest.raises(RuntimeError) as e:
                extract_distiller_adapter_features(self.model, "x", layer_name)
            assert e.value.args[0].startswith("Can not find layer name")

def test_set_attribute():
    '''

    :return:
    '''
    model = torchvision.models.resnet18(pretrained=False)
    with pytest.raises(AttributeError) as e:
        _tmp = model.original
    assert e.value.args[0] == "'ResNet' object has no attribute 'original'"
    set_attribute("model",model,"original","this is for test")
    assert model.original == "this is for test"

class TestMakeTransferrable:
    ''' test _make_transferrable

    '''
    def setup(self):
        self.label_map = {item.split('\t')[0]: int(item.split('\t')[1]) for item in
                     open("../datasets/office31/label_map.txt").readlines()}
        self.batch_size = 16

    def _create_kwargs(self):
        ''' create kwargs

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        adapter_layer_name = [node.name for node in torch.fx.symbolic_trace(model).graph.nodes][-3]  # [... ,avgpool, flatten, fc, output]
        print("adapter_layer_name for resnet18:%s" % adapter_layer_name)
        teacher_model = torchvision.models.resnet18(pretrained=True)
        new_teacher_model = extract_distiller_adapter_features(teacher_model,'x',adapter_layer_name)

        return {
            'model': model,
            'loss': torch.nn.CrossEntropyLoss(),
            'distiller_feature_size': (3, 224, 224),
            'distiller_feature_layer_name': 'x',
            'adapter_feature_size': 1000,
            'adapter_feature_layer_name': adapter_layer_name,
            'finetunner':BasicFinetunner(torchvision.models.resnet18(pretrained=True),-1,True),
            'distiller': BasicDistiller(new_teacher_model, True),
            'adapter': DANNAdapter(512, 8, 0.0, 5.0, 1.0, 100),
            'training_dataloader': torch.utils.data.DataLoader(
                Office31("../datasets/office31/amazon", self.label_map, None, 'RGB'),
                batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True),
            'adaption_source_domain_training_dataset': Office31("../datasets/office31/webcam", self.label_map, None, 'RGB'),
            'transfer_strategy': TransferStrategy.DistillationAndDomainAdaptionStrategy,
            'enable_target_training_label': True,
            'backbone_loss_weight': 1.0,
            'distiller_loss_weight': 1.0,
            'adapter_loss_weight': 1.0,
        }
    def test_valid_call(self):
        ''' test valid call

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = _make_transferrable(**kwargs)
        tensor1 = torch.ones((self.batch_size, 2))
        tensor0 = torch.zeros((self.batch_size, 2))

        assert tensor_near_equal(kwargs['model'].loss(tensor1,tensor0) ,kwargs['loss'](tensor1, tensor0))
        assert kwargs['model'].distiller_feature_size is kwargs['distiller_feature_size']
        assert kwargs['model'].adapter_feature_size is kwargs['adapter_feature_size']

        assert type(new_model) == TransferrableModel
        assert tensor_near_equal(new_model.backbone.loss(tensor1, tensor0), kwargs['loss'](tensor1, tensor0))
        assert new_model.backbone.distiller_feature_size is kwargs['distiller_feature_size']
        assert new_model.backbone.adapter_feature_size is kwargs['adapter_feature_size']
        assert new_model.backbone.original is kwargs['model']
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is kwargs['distiller']
        assert type(kwargs['training_dataloader'].dataset) is ComposedDataset

    def test_valid_decorator(self):
        ''' test valid _transferrable

        :return:
        '''
        kwargs = self._create_kwargs()
        new_kwargs = {item[0]:item[1] for item in kwargs.items() if item[0] != 'model'}

        @_transferrable(**new_kwargs)
        class TmpModel(torch.nn.Module):
            def __init__(self):
                # TypeError: super() argument 1 must be type, not function
                # return super(A, cls).__new__(cls)
                super().__init__()
                self.m = torchvision.models.resnet18(pretrained=True)
            def forward(self,x):
                return self.m(x)

        new_model = TmpModel()
        tensor1 = torch.ones((self.batch_size, 2))
        tensor0 = torch.zeros((self.batch_size, 2))


        assert type(new_model) == TransferrableModel
        assert tensor_near_equal(new_model.backbone.loss(tensor1, tensor0), kwargs['loss'](tensor1, tensor0))
        assert new_model.backbone.distiller_feature_size is kwargs['distiller_feature_size']
        assert new_model.backbone.adapter_feature_size is kwargs['adapter_feature_size']
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is kwargs['distiller']
        assert type(kwargs['training_dataloader'].dataset) is ComposedDataset

    def test_return_ComposedDataset(self):
        ''' test return ComposedDataset or not

        :return:
        '''
        for transfer_strategy in [TransferStrategy.OnlyFinetuneStrategy,
                                  TransferStrategy.OnlyDistillationStrategy]:
            kwargs = self._create_kwargs()
            kwargs['transfer_strategy'] = transfer_strategy
            _make_transferrable(**kwargs)
            assert type(kwargs['training_dataloader'].dataset) is not ComposedDataset

        for transfer_strategy in [TransferStrategy.OnlyDomainAdaptionStrategy,
                                  TransferStrategy.FinetuneAndDomainAdaptionStrategy,
                                  TransferStrategy.DistillationAndDomainAdaptionStrategy]:
            kwargs = self._create_kwargs()
            kwargs['transfer_strategy'] = transfer_strategy
            _make_transferrable(**kwargs)
            assert type(kwargs['training_dataloader'].dataset) is ComposedDataset
    def test_invalid_loss_call(self):
        ''' test call of invalid loss

        :return:
        '''
        with pytest.raises(RuntimeError) as e:
            kwargs = self._create_kwargs()
            kwargs["loss"] = None
            _make_transferrable(**kwargs)
        assert e.value.args[0] == "Need loss for model"
    def test_invalid_distiller_call(self):
        ''' test call of invalid distiller

        :return:
        '''
        for transfer_strategy in [TransferStrategy.OnlyDistillationStrategy,
                                  TransferStrategy.DistillationAndDomainAdaptionStrategy]:
            # for name in ["distiller_feature_size", "distiller_feature_layer_name", "distiller"]:
            for name in ["distiller_feature_layer_name", "distiller"]:
                with pytest.raises(RuntimeError) as e:
                    kwargs = self._create_kwargs()
                    kwargs['transfer_strategy'] = transfer_strategy
                    kwargs[name] = None
                    _make_transferrable(**kwargs)
                assert e.value.args[0] == "Need %s for Distillation" % name
    def test_invalid_adapter_call(self):
        ''' test call of invalid adapter

        :return:
        '''
        for transfer_strategy in [TransferStrategy.OnlyDomainAdaptionStrategy,
                                  TransferStrategy.FinetuneAndDomainAdaptionStrategy,
                                  TransferStrategy.DistillationAndDomainAdaptionStrategy]:
            for name in ["adapter_feature_size", "adapter_feature_layer_name", "adapter"]:
                with pytest.raises(RuntimeError) as e:
                    kwargs = self._create_kwargs()
                    kwargs['transfer_strategy'] = transfer_strategy
                    kwargs[name] = None
                    _make_transferrable(**kwargs)
                assert e.value.args[0] == "Need %s for Adaption" % name

    def test_invalid_enable_target_training_label_call(self):
        ''' test call of invalid enable_target_training_label

        :return:
        '''
        with pytest.raises(RuntimeError) as e:
            kwargs = self._create_kwargs()
            kwargs['transfer_strategy'] = TransferStrategy.OnlyFinetuneStrategy
            kwargs['enable_target_training_label'] = False
            _make_transferrable(**kwargs)
        assert e.value.args[0] == "Need enable_target_training_label for OnlyFinetune"

    def test_invalid_dataset_loader_call(self):
        ''' test call of invalid dataset_loader

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyDomainAdaptionStrategy
        _make_transferrable(**kwargs)  # dataset_loader is ComposedDataset
        with pytest.raises(RuntimeError) as e:
            _make_transferrable(**kwargs)
        assert e.value.args[0] == "training_dataloader should not be ComposedDataset"
    def test_make_transferrable_with_finetune(self):
        ''' test make_transferrable_with_finetune

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_finetune(model=kwargs['model'],loss=kwargs['loss'],
                                                     finetunner=kwargs['finetunner'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is None
        assert new_model.distiller is None

        assert new_model.backbone.distiller_feature_size is None
        assert new_model.backbone.adapter_feature_size is None
        assert new_model.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy
        assert new_model.enable_target_training_label == True
        assert type(kwargs['training_dataloader'].dataset) is not ComposedDataset
    def test_make_transferrable_with_knowledge_distillation(self):
        ''' test make_transferrable_with_knowledge_distillation

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_knowledge_distillation(
            model=kwargs['model'], loss=kwargs['loss'],
            distiller=kwargs['distiller'],distiller_feature_size=kwargs['distiller_feature_size'],
            distiller_feature_layer_name=kwargs['distiller_feature_layer_name'],
            enable_target_training_label=kwargs['enable_target_training_label'],
            backbone_loss_weight=kwargs['backbone_loss_weight'],
            distiller_loss_weight=kwargs['distiller_loss_weight'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is None
        assert new_model.distiller is kwargs['distiller']

        assert new_model.backbone.distiller_feature_size is kwargs['distiller_feature_size']
        assert new_model.backbone.adapter_feature_size is None
        assert new_model.transfer_strategy == TransferStrategy.OnlyDistillationStrategy
        assert new_model.enable_target_training_label == kwargs['enable_target_training_label']
        assert type(kwargs['training_dataloader'].dataset) is not ComposedDataset
    def test_make_transferrable_with_domain_adaption(self):
        ''' test make_transferrable_with_domain_adaption

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_domain_adaption(
            model=kwargs['model'], loss=kwargs['loss'],
            adapter=kwargs['adapter'],adapter_feature_size=kwargs['adapter_feature_size'],
            adapter_feature_layer_name=kwargs['adapter_feature_layer_name'],
            training_dataloader=kwargs['training_dataloader'],
            adaption_source_domain_training_dataset=kwargs['adaption_source_domain_training_dataset'],
            enable_target_training_label=kwargs['enable_target_training_label'],
            backbone_loss_weight=kwargs['backbone_loss_weight'],
            adapter_loss_weight=kwargs['adapter_loss_weight'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is None

        assert new_model.backbone.distiller_feature_size is None
        assert new_model.backbone.adapter_feature_size is kwargs['adapter_feature_size']
        assert new_model.transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy
        assert new_model.enable_target_training_label == kwargs['enable_target_training_label']
        assert type(kwargs['training_dataloader'].dataset) is ComposedDataset
    def test_make_transferrable_with_finetune_and_domain_adaption(self):
        ''' test make_transferrable_with_finetune_and_domain_adaption

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_finetune_and_domain_adaption(
            model=kwargs['model'], loss=kwargs['loss'],
            finetunner=kwargs['finetunner'],adapter=kwargs['adapter'],
            adapter_feature_size=kwargs['adapter_feature_size'],
            adapter_feature_layer_name=kwargs['adapter_feature_layer_name'],
            training_dataloader=kwargs['training_dataloader'],
            adaption_source_domain_training_dataset=kwargs['adaption_source_domain_training_dataset'],
            enable_target_training_label=kwargs['enable_target_training_label'],
            backbone_loss_weight=kwargs['backbone_loss_weight'],
            adapter_loss_weight=kwargs['adapter_loss_weight'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is None

        assert new_model.backbone.distiller_feature_size is None
        assert new_model.backbone.adapter_feature_size is kwargs['adapter_feature_size']
        assert new_model.transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy
        assert new_model.enable_target_training_label == kwargs['enable_target_training_label']
        assert type(kwargs['training_dataloader'].dataset) is ComposedDataset
    def test_make_transferrable_with_knowledge_distillation_and_domain_adaption(self):
        ''' test make_transferrable_with_knowledge_distillation_and_domain_adaption

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_knowledge_distillation_and_domain_adaption(
            model=kwargs['model'], loss=kwargs['loss'],
            distiller=kwargs['distiller'],adapter=kwargs['adapter'],
            distiller_feature_size=kwargs['distiller_feature_size'],
            distiller_feature_layer_name=kwargs['distiller_feature_layer_name'],
            adapter_feature_size=kwargs['adapter_feature_size'],
            adapter_feature_layer_name=kwargs['adapter_feature_layer_name'],
            training_dataloader=kwargs['training_dataloader'],
            adaption_source_domain_training_dataset=kwargs['adaption_source_domain_training_dataset'],
            enable_target_training_label=kwargs['enable_target_training_label'],
            backbone_loss_weight=kwargs['backbone_loss_weight'],
            distiller_loss_weight=kwargs['distiller_loss_weight'],
            adapter_loss_weight=kwargs['adapter_loss_weight'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is kwargs['distiller']

        assert new_model.backbone.distiller_feature_size is kwargs['distiller_feature_size']
        assert new_model.backbone.adapter_feature_size is kwargs['adapter_feature_size']
        assert new_model.transfer_strategy == TransferStrategy.DistillationAndDomainAdaptionStrategy
        assert new_model.enable_target_training_label == kwargs['enable_target_training_label']
        assert type(kwargs['training_dataloader'].dataset) is ComposedDataset
class TestTransferrableModel:
    ''' Test TransferrableModel

    '''

    def _create_kwargs(self):
        ''' create kwargs

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        adapter_layer_name = [node.name for node in torch.fx.symbolic_trace(model).graph.nodes][-3]  # [... ,avgpool, flatten, fc, output]
        print("adapter_layer_name for resnet18:%s" % adapter_layer_name)
        teacher_model = torchvision.models.resnet18(pretrained=True)
        new_teacher_model = extract_distiller_adapter_features(teacher_model,'x',adapter_layer_name)

        return {
            'model': model,
            'loss': torch.nn.CrossEntropyLoss(),
            'distiller_feature_size': (3, 224, 224),
            'distiller_feature_layer_name': 'x',
            'adapter_feature_size': 1000,
            'adapter_feature_layer_name': adapter_layer_name,
            'finetunner': BasicFinetunner(torchvision.models.resnet18(pretrained=True), -1, True),
            'distiller': BasicDistiller(new_teacher_model, True),
            'adapter': DANNAdapter(512, 8, 0.0, 5.0, 1.0, 100),
            'training_dataloader': torch.utils.data.DataLoader(
                Office31("../datasets/office31/amazon", self.label_map, None, 'RGB'),
                batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True),
            'adaption_source_domain_training_dataset': Office31("../datasets/office31/webcam", self.label_map, None,
                                                                'RGB'),
            'transfer_strategy': TransferStrategy.OnlyDistillationStrategy,
            'enable_target_training_label': True,
            'backbone_loss_weight': 1.0,
            'distiller_loss_weight': 1.0,
            'adapter_loss_weight': 1.0,
        }

    def setup(self):
        self.label_map = {item.split('\t')[0]: int(item.split('\t')[1]) for item in
                     open("../datasets/office31/label_map.txt").readlines()}
        self.batch_size = 16
        self.input = torch.randn([self.batch_size, 3, 224, 224])
        self.num_class = 1000

    def test_create_invalid_strategy(self):
        ''' test create  TransferrableModel with invalid strategy

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        model.__dict__["original"] = torchvision.models.resnet18(pretrained=False)
        finetunner = BasicFinetunner( torchvision.models.resnet18(pretrained=True),-1,True)
        adapter = DANNAdapter(512, 8, 0.0, 5.0, 1.0, 100)
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), True)

        for enable_target_training_label in [True,False]:
            for transfer_strategy in ALL_STRATEGIES:
                if enable_target_training_label == False and transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
                    with pytest.raises(RuntimeError) as e:
                        TransferrableModel(model, adapter, distiller,transfer_strategy, enable_target_training_label)
                    assert e.value.args[0] == "Must enable target training label when only finetune."
                else:
                    TransferrableModel(model, adapter, distiller, transfer_strategy,enable_target_training_label) # valid

    def test_create_invalid_backbone(self):
        ''' create TransferrableModel with invalid backbone

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        finetunner = BasicFinetunner(torchvision.models.resnet18(pretrained=True),-1,True)
        adapter = DANNAdapter(512, 8, 0.0, 5.0, 1.0, 100)
        distiller = BasicDistiller(torchvision.models.resnet18(pretrained=True), True)
        with pytest.raises(RuntimeError) as e:
            TransferrableModel(model, adapter, distiller, TransferStrategy.OnlyFinetuneStrategy,True)
        assert e.value.args[0] == "Backbone must have original attribute."

    def test_getattribute_in_train_mode(self):
        ''' test __getattribute__ in training mode

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.__dict__  # no exception
        ############### training mode ##########
        model.train()
        assert model.backbone.original is kwargs['model']
        assert type(model(self.input)) is TransferrableModelOutput
        assert tensor_near_equal(model(self.input).backbone_output,kwargs['model'](self.input))
        assert type(model.loss(model(self.input),model(self.input))) is TransferrableModelLoss

    def test_getattribute_in_eval_mode(self):
        ''' test __getattribute__ in eval mode

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.__dict__  # no exception
        ############## eval mode ############
        model.eval()
        assert type(model(self.input)) is torch.Tensor
        assert tensor_near_equal(model(self.input),kwargs['model'](self.input))
        assert type(model.loss(model(self.input),model(self.input))) is torch.Tensor
        # ############### train mode again ##########
        model.train()
        assert model.backbone.original is kwargs['model']
        assert type(model(self.input)) is TransferrableModelOutput
        assert tensor_near_equal(model(self.input).backbone_output, kwargs['model'](self.input))
        assert type(model.loss(model(self.input),model(self.input))) is TransferrableModelLoss

    def test_finetune_forward(self):
        ''' test _finetune_forward

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()
        for i in range(0,10):
            x = torch.randn_like(self.input)
            new_output = model._finetune_forward(x)
            assert type(new_output) == TransferrableModelOutput
            assert tensor_near_equal(new_output.backbone_output,kwargs['model'](x))
            assert new_output.distiller_output is None
            assert new_output.adapter_output is None
    def test_finetune_loss(self):
        ''' test _finetune_loss

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()

        output = torch.zeros([self.batch_size,self.num_class])
        label = torch.ones([self.batch_size,self.num_class])

        new_loss = model._finetune_loss(output,label)
        assert type(new_loss) == TransferrableModelLoss
        assert tensor_near_equal(new_loss.total_loss,new_loss.backbone_loss)
        assert tensor_near_equal(new_loss.backbone_loss, kwargs['model'].loss(output,label))
        assert new_loss.distiller_loss is None
        assert new_loss.adapter_loss is None
    def test_distillation_forward(self):
        ''' test _distillation_forward

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()
        for i in range(0, 10):
            x = torch.randn_like(self.input)
            distiller_input = model.backbone(x)[1][0]
            new_output = model._distillation_forward(x)
            assert type(new_output) == TransferrableModelOutput
            assert tensor_near_equal(new_output.backbone_output, kwargs['model'](x))
            assert tensor_near_equal(new_output.distiller_output, kwargs['distiller'](distiller_input)[0])
            assert new_output.adapter_output is None
    def test_distillation_loss(self):
        ''' test _distillation_loss

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()

        student_output = torch.zeros([self.batch_size,self.num_class])
        teacher_output = torch.ones([self.batch_size,self.num_class]) * 0.5
        student_label = torch.zeros([self.batch_size,self.num_class])

        new_loss = model._distillation_loss(student_output,teacher_output,student_label)
        assert type(new_loss) == TransferrableModelLoss
        assert tensor_near_equal(new_loss.total_loss,new_loss.backbone_loss + new_loss.distiller_loss)
        assert tensor_near_equal(new_loss.backbone_loss, kwargs['model'].loss(student_output,student_label))
        assert tensor_near_equal(new_loss.distiller_loss, kwargs['distiller'].loss(teacher_output,student_output))
        assert new_loss.adapter_loss is None
    def test_adaption_forward(self):
        ''' test _adaption_forward

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()
        for i in range(0, 10):
            x1 = torch.randn_like(self.input)
            x2 = torch.randn_like(self.input)
            adapter_input1 = model.backbone(x1)[1][1] # target
            adapter_input2 = model.backbone(x2)[1][1] # source
            new_output = model._adaption_forward(x1,x2)
            assert type(new_output) == tuple
            assert type(new_output[0]) == TransferrableModelOutput
            assert type(new_output[1]) == TransferrableModelOutput
            assert tensor_near_equal(new_output[0].backbone_output, kwargs['model'](x1))
            assert tensor_near_equal(new_output[1].backbone_output, kwargs['model'](x2))
            assert new_output[0].distiller_output is None
            assert new_output[1].distiller_output is None
            assert tensor_near_equal(new_output[0].adapter_output, kwargs['adapter'](adapter_input1))
            assert tensor_near_equal(new_output[1].adapter_output, kwargs['adapter'](adapter_input2))
    def test_adaption_loss(self):
        ''' test _adaption_loss

        :return:
        '''
        for enable_target_training_label in [True,False]:
            kwargs = self._create_kwargs()
            kwargs['enable_target_training_label'] = enable_target_training_label
            model = _make_transferrable(**kwargs)
            model.train()

            backbone_output_tgt = torch.ones([self.batch_size,self.num_class]) * 0.1
            backbone_output_src = torch.ones([self.batch_size,self.num_class]) * 0.2
            adapter_output_tgt  = torch.ones([self.batch_size,1]) * 0.7
            adapter_output_src  = torch.ones([self.batch_size,1]) * 0.5
            if enable_target_training_label:
                backbone_label_tgt  = torch.ones([self.batch_size,self.num_class])
            else:
                backbone_label_tgt = None
            backbone_label_src  = torch.zeros([self.batch_size,self.num_class])

            new_loss = model._adaption_loss(backbone_output_tgt, backbone_output_src,
                           adapter_output_tgt, adapter_output_src,
                           backbone_label_tgt,backbone_label_src)

            assert type(new_loss) == TransferrableModelLoss
            assert tensor_near_equal(new_loss.total_loss,new_loss.backbone_loss + new_loss.adapter_loss)
            if enable_target_training_label:
                assert tensor_near_equal(new_loss.backbone_loss,
                                     kwargs['model'].loss(backbone_output_tgt,backbone_label_tgt) +
                                     kwargs['model'].loss(backbone_output_src, backbone_label_src)
                                     )
            else:
                assert tensor_near_equal(new_loss.backbone_loss,
                                         kwargs['model'].loss(backbone_output_src, backbone_label_src))
            assert new_loss.distiller_loss is None
            adapter_label_tgt = AdversarialAdapter.make_label(adapter_output_tgt.size(0), is_source=False).view(-1, 1)
            adapter_label_src = AdversarialAdapter.make_label(adapter_output_src.size(0), is_source=True).view(-1, 1)
            assert tensor_near_equal(new_loss.adapter_loss,
                                     kwargs['adapter'].loss(adapter_output_tgt, adapter_label_tgt) +
                                     kwargs['adapter'].loss(adapter_output_src, adapter_label_src)
                                     )
    def test_distillation_and_adaption_forward(self):
        ''' test _distillation_and_adaption_forward

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()
        for i in range(0, 10):
            x1 = torch.randn_like(self.input)
            x2 = torch.randn_like(self.input)
            distiller_input1 = model.backbone(x1)[1][0]  # target
            distiller_input2 = model.backbone(x2)[1][0]  # source
            adapter_input1 = model.backbone(x1)[1][1]  # target
            adapter_input2 = model.backbone(x2)[1][1]  # source
            new_output = model._distillation_and_adaption_forward(x1, x2)
            assert tensor_near_equal(new_output[0].backbone_output, kwargs['model'](x1))
            assert tensor_near_equal(new_output[1].backbone_output, kwargs['model'](x2))
            assert tensor_near_equal(new_output[0].distiller_output, kwargs['distiller'](distiller_input1)[0])
            assert tensor_near_equal(new_output[1].distiller_output, kwargs['distiller'](distiller_input2)[0])
            assert tensor_near_equal(new_output[0].adapter_output, kwargs['adapter'](adapter_input1))
            assert tensor_near_equal(new_output[1].adapter_output, kwargs['adapter'](adapter_input2))
    def test_distillation_and_adaption_loss(self):
        ''' test _distillation_and_adaption_loss

        :return:
        '''
        for enable_target_training_label in [True, False]:
            kwargs = self._create_kwargs()
            kwargs['enable_target_training_label'] = enable_target_training_label
            model = _make_transferrable(**kwargs)
            model.train()

            backbone_output_tgt = torch.ones([self.batch_size, self.num_class]) * 0.1
            backbone_output_src = torch.ones([self.batch_size, self.num_class]) * 0.2
            teacher_output_tgt = torch.ones([self.batch_size, self.num_class]) * 0.3
            teacher_output_src = torch.ones([self.batch_size, self.num_class]) * 0.4
            adapter_output_tgt = torch.ones([self.batch_size, 1]) * 0.7
            adapter_output_src = torch.ones([self.batch_size, 1]) * 0.5
            if enable_target_training_label:
                backbone_label_tgt = torch.ones([self.batch_size, self.num_class])
            else:
                backbone_label_tgt = None
            backbone_label_src = torch.zeros([self.batch_size, self.num_class])

            if enable_target_training_label:
                target_loss = kwargs['model'].loss(backbone_output_tgt, backbone_label_tgt)
            else:
                target_loss = 0.0
            src_loss = kwargs['model'].loss(backbone_output_src, backbone_label_src)
            new_loss = model._distillation_and_adaption_loss(backbone_output_tgt, backbone_output_src,
                                            teacher_output_tgt, teacher_output_src,
                                            adapter_output_tgt, adapter_output_src,
                                            backbone_label_tgt, backbone_label_src)

            assert type(new_loss) == TransferrableModelLoss
            assert tensor_near_equal(new_loss.total_loss, new_loss.backbone_loss + new_loss.distiller_loss + new_loss.adapter_loss)
            if enable_target_training_label:
                assert tensor_near_equal(new_loss.backbone_loss,
                                     kwargs['model'].loss(backbone_output_tgt, backbone_label_tgt) +
                                     kwargs['model'].loss(backbone_output_src, backbone_label_src)
                                     )
            else:
                assert tensor_near_equal(new_loss.backbone_loss,
                                         kwargs['model'].loss(backbone_output_src, backbone_label_src)
                                         )
            assert tensor_near_equal(new_loss.distiller_loss,
                                     kwargs['distiller'].loss(teacher_output_tgt, backbone_output_tgt) +
                                     kwargs['distiller'].loss(teacher_output_src, backbone_output_src)
                                     )
            adapter_label_tgt = AdversarialAdapter.make_label(adapter_output_tgt.size(0), is_source=False).view(-1, 1)
            adapter_label_src = AdversarialAdapter.make_label(adapter_output_src.size(0), is_source=True).view(-1, 1)
            assert tensor_near_equal(new_loss.adapter_loss,
                                     kwargs['adapter'].loss(adapter_output_tgt, adapter_label_tgt) +
                                     kwargs['adapter'].loss(adapter_output_src, adapter_label_src)
                                     )
    def test_forward_OnlyFinetuneStrategy(self):
        ''' test forward with OnlyFinetuneStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyFinetuneStrategy
        model = _make_transferrable(**kwargs)
        model.train()
        ############ single input #########
        _input = self.input
        output = model(_input)
        assert type(output) == TransferrableModelOutput
        assert tensor_near_equal(output.backbone_output,kwargs['model'](_input))
        assert output.distiller_output is None
        assert output.adapter_output is None
        ########### double input ############
        _input = (self.input,torch.ones_like(self.input))
        output = model(_input)
        assert type(output) == TransferrableModelOutput
        assert tensor_near_equal(output.backbone_output, kwargs['model'](_input[0]))
        assert output.distiller_output is None
        assert output.adapter_output is None
    def test_forward_OnlyDistillationStrategy(self):
        ''' test forward with OnlyDistillationStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyDistillationStrategy
        model = _make_transferrable(**kwargs)
        model.train()
        ############ single input #########
        _input = self.input
        output = model(_input)
        assert type(output) == TransferrableModelOutput
        assert tensor_near_equal(output.backbone_output, kwargs['model'](_input))
        assert output.distiller_output is not None
        assert output.adapter_output is None
        ########### double input ############
        _input = (self.input, torch.ones_like(self.input))
        output = model(_input)
        assert type(output) == TransferrableModelOutput
        assert tensor_near_equal(output.backbone_output, kwargs['model'](_input[0]))
        assert output.distiller_output is not None
        assert output.adapter_output is None
    def test_forward_OnlyDomainAdaptionStrategy(self):
        ''' test forward with OnlyDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyDomainAdaptionStrategy
        model = _make_transferrable(**kwargs)
        model.train()
        ############ single input #########
        _input = self.input
        with pytest.raises(RuntimeError) as e:
            model(_input)
        assert e.value.args[0].startswith("TransferrableModel forward for OnlyDomainAdaptionStrategy should be tuple or list")
        ########### double input ############
        _input = (self.input, torch.ones_like(self.input))
        output = model(_input)
        assert type(output) == tuple
        assert type(output[0]) == TransferrableModelOutput
        assert type(output[1]) == TransferrableModelOutput
        assert tensor_near_equal(output[0].backbone_output, kwargs['model'](_input[0]))
        assert tensor_near_equal(output[1].backbone_output, kwargs['model'](_input[1]))
        assert output[0].distiller_output is None
        assert output[1].distiller_output is None
        assert output[0].adapter_output is not None
        assert output[1].adapter_output is not None
    def test_forward_FinetuneAndDomainAdaptionStrategy(self):
        ''' test forward with FinetuneAndDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.FinetuneAndDomainAdaptionStrategy
        model = _make_transferrable(**kwargs)
        model.train()
        ############ single input #########
        _input = self.input
        with pytest.raises(RuntimeError) as e:
            model(_input)
        assert e.value.args[0].startswith(
            "TransferrableModel forward for FinetuneAndDomainAdaptionStrategy should be tuple or list")
        ########### double input ############
        _input = (self.input, torch.ones_like(self.input))
        output = model(_input)
        assert type(output) == tuple
        assert type(output[0]) == TransferrableModelOutput
        assert type(output[1]) == TransferrableModelOutput
        assert tensor_near_equal(output[0].backbone_output, kwargs['model'](_input[0]))
        assert tensor_near_equal(output[1].backbone_output, kwargs['model'](_input[1]))
        assert output[0].distiller_output is None
        assert output[1].distiller_output is None
        assert output[0].adapter_output is not None
        assert output[1].adapter_output is not None
    def test_forward_DistillationAndDomainAdaptionStrategy(self):
        ''' test forward DistillationAndDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.DistillationAndDomainAdaptionStrategy
        model = _make_transferrable(**kwargs)
        model.train()
        ############ single input #########
        _input = self.input
        with pytest.raises(RuntimeError) as e:
            model(_input)
        assert e.value.args[0].startswith(
            "TransferrableModel forward for DistillationAndDomainAdaptionStrategy should be tuple or list")
        ########### double input ############
        _input = (self.input, torch.ones_like(self.input))
        output = model(_input)
        assert type(output) == tuple
        assert type(output[0]) == TransferrableModelOutput
        assert type(output[1]) == TransferrableModelOutput
        assert tensor_near_equal(output[0].backbone_output, kwargs['model'](_input[0]))
        assert tensor_near_equal(output[1].backbone_output, kwargs['model'](_input[1]))
        assert output[0].distiller_output is not None
        assert output[1].distiller_output is not None
        assert output[0].adapter_output is not None
        assert output[1].adapter_output is not None
    def test_loss_OnlyFinetuneStrategy(self):
        ''' test loss with OnlyFinetuneStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyFinetuneStrategy
        model = _make_transferrable(**kwargs)
        model.train()

        output = model(self.input)
        label = torch.ones([self.batch_size,self.num_class])
        loss = model.loss(output,label)
        assert type(loss) == TransferrableModelLoss
        assert tensor_near_equal(loss.total_loss, loss.backbone_loss)
        assert loss.distiller_loss is None
        assert loss.adapter_loss is None
    def test_loss_OnlyDistillationStrategy(self):
        ''' test loss with OnlyDistillationStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyDistillationStrategy
        model = _make_transferrable(**kwargs)
        model.train()

        output = model(self.input)
        label = torch.ones([self.batch_size,self.num_class])
        loss = model.loss(output,label)
        assert type(loss) == TransferrableModelLoss
        assert tensor_near_equal(loss.total_loss, loss.backbone_loss + loss.distiller_loss)
        assert loss.distiller_loss is not None
        assert loss.adapter_loss is None
    def test_loss_OnlyDomainAdaptionStrategy(self):
        ''' test loss with OnlyDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.OnlyDomainAdaptionStrategy
        model = _make_transferrable(**kwargs)
        model.train()

        output = model((self.input,self.input+1))
        label = (torch.ones([self.batch_size,self.num_class]),torch.zeros([self.batch_size,self.num_class]))
        loss = model.loss(output,label)
        assert type(loss) == TransferrableModelLoss
        assert tensor_near_equal(loss.total_loss, loss.backbone_loss + loss.adapter_loss)
        assert loss.distiller_loss is None
        assert loss.adapter_loss is not None
    def test_loss_FinetuneAndDomainAdaptionStrategy(self):
        ''' test loss with FinetuneAndDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.FinetuneAndDomainAdaptionStrategy
        model = _make_transferrable(**kwargs)
        model.train()

        output = model((self.input,self.input+1))
        label = (torch.ones([self.batch_size,self.num_class]),torch.zeros([self.batch_size,self.num_class]))
        loss = model.loss(output,label)
        assert type(loss) == TransferrableModelLoss
        assert tensor_near_equal(loss.total_loss, loss.backbone_loss + loss.adapter_loss)
        assert loss.distiller_loss is None
        assert loss.adapter_loss is not None
    def test_loss_DistillationAndDomainAdaptionStrategy(self):
        ''' test loss with DistillationAndDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        kwargs['transfer_strategy'] = TransferStrategy.DistillationAndDomainAdaptionStrategy
        model = _make_transferrable(**kwargs)
        model.train()

        output = model((self.input,self.input+1))
        label = (torch.ones([self.batch_size,self.num_class]),torch.zeros([self.batch_size,self.num_class]))
        loss = model.loss(output,label)
        assert type(loss) == TransferrableModelLoss
        assert tensor_near_equal(loss.total_loss, loss.backbone_loss + loss.distiller_loss + loss.adapter_loss )
        assert loss.total_loss is not None
        assert loss.distiller_loss is not None
        assert loss.adapter_loss is not None
    def test_get_training_metrics(self):
        ''' test get_training_metrics

        :return:
        '''
        for transfer_strategy in ALL_STRATEGIES:
            if transfer_strategy in [TransferStrategy.OnlyDomainAdaptionStrategy,
                                  TransferStrategy.FinetuneAndDomainAdaptionStrategy,
                                  TransferStrategy.DistillationAndDomainAdaptionStrategy]:
                _input = (self.input, self.input + 1)
                label = (torch.ones([self.batch_size, self.num_class]), torch.zeros([self.batch_size, self.num_class]))
            else:
                _input = self.input
                label = torch.ones([self.batch_size, self.num_class])

            for enable_target_training_label in [True,False]:
                if transfer_strategy == TransferStrategy.OnlyFinetuneStrategy \
                        and enable_target_training_label == False:
                    continue # invalid
                kwargs = self._create_kwargs()
                kwargs['transfer_strategy'] = transfer_strategy
                kwargs['enable_target_training_label'] = enable_target_training_label
                model = _make_transferrable(**kwargs)
                model.train()
                output = model(_input)
                loss = model.loss(output, label)
                metric_fn_map = {'acc':accuracy}
                metric_values = model.get_training_metrics(output,label,loss,metric_fn_map)
                metric_keys = [item for item in sorted(metric_values.keys())]
                if transfer_strategy == TransferStrategy.OnlyFinetuneStrategy:
                    assert metric_keys == ["acc","backbone_loss","total_loss"]
                elif transfer_strategy == TransferStrategy.OnlyDistillationStrategy:
                    assert metric_keys == ["acc", "backbone_loss", "distiller_loss", "total_loss"]
                elif transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy:
                    if enable_target_training_label:
                        assert metric_keys == ["acc_src_domain", "acc_target_domain", "adapter_loss","backbone_loss", "total_loss"]
                    else:
                        assert metric_keys == ["acc_src_domain", "adapter_loss", "backbone_loss", "total_loss"]
                elif transfer_strategy == TransferStrategy.FinetuneAndDomainAdaptionStrategy:
                    if enable_target_training_label:
                        assert metric_keys == ["acc_src_domain", "acc_target_domain", "adapter_loss","backbone_loss", "total_loss"]
                    else:
                        assert metric_keys == ["acc_src_domain", "adapter_loss", "backbone_loss", "total_loss"]
                elif transfer_strategy == TransferStrategy.DistillationAndDomainAdaptionStrategy:
                    if enable_target_training_label:
                        assert metric_keys == ["acc_src_domain", "acc_target_domain", "adapter_loss","backbone_loss", "distiller_loss", "total_loss"]
                    else:
                        assert metric_keys == ["acc_src_domain", "adapter_loss", "backbone_loss",  "distiller_loss", "total_loss"]
                else:
                    assert NotImplemented("unknown strategy %s"%transfer_strategy)
    def test_init_weight(self):
        ''' test init_weight
        :return:
        '''

        for strategy in ALL_STRATEGIES:
            kwargs = self._create_kwargs()
            basic_weights = {item[0]: item[1] for item in
                             kwargs['finetunner']._pretrained_network.named_parameters()}
            kwargs['transfer_strategy'] = strategy
            model = _make_transferrable(**kwargs)
            for (name,weight) in model.backbone.named_parameters():
                if strategy in [TransferStrategy.OnlyFinetuneStrategy,
                                TransferStrategy.FinetuneAndDomainAdaptionStrategy]:
                    assert tensor_near_equal(weight,basic_weights[name])
                    assert weight.requires_grad == False
                else:
                    assert not tensor_near_equal(weight, basic_weights[name],1e-3)
                    assert weight.requires_grad == True
            if strategy in [TransferStrategy.OnlyDistillationStrategy,
                            TransferStrategy.DistillationAndDomainAdaptionStrategy]:
                for (name,weight) in model.distiller.pretrained_model.named_parameters():
                    assert tensor_near_equal(weight,basic_weights[name])
