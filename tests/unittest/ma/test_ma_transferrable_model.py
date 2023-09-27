import pytest
import sys
import os
import torch
import torchvision
from e2eAIOK.ModelAdapter.engine_core.transferrable_model import TransferrableModel,\
    TransferStrategy,TransferrableModelOutput,TransferrableModelLoss,\
    _make_transferrable,_transferrable,ALL_STRATEGIES,\
    make_transferrable_with_finetune, make_transferrable_with_knowledge_distillation,\
    make_transferrable_with_domain_adaption
from e2eAIOK.ModelAdapter.engine_core.distiller.basic_distiller import BasicDistiller
from e2eAIOK.ModelAdapter.engine_core.finetunner.basic_finetunner import BasicFinetunner
from e2eAIOK.ModelAdapter.engine_core.adapter.adversarial.dann_adapter import DANNAdapter
from e2eAIOK.ModelAdapter.engine_core.adapter.adversarial.adversarial_adapter import AdversarialAdapter
from e2eAIOK.ModelAdapter.dataset.office31 import Office31
from e2eAIOK.ModelAdapter.backbone.unet.generic_UNet_DA import Generic_UNet_DA
from e2eAIOK.ModelAdapter.engine_core.adapter.adversarial.DA_Loss import CACDomainAdversarialLoss
from e2eAIOK.common.trainer.utils.utils import acc
from e2eAIOK.common.trainer.utils.utils import tensor_near_equal
import torch.fx

class TestMakeTransferrable:
    ''' test _make_transferrable

    '''
    def setup(self):
        # self.label_map = {item.split('\t')[0]: int(item.split('\t')[1]) for item in
        #              open("/home/vmagent/app/data/dataset/office31/label_map.txt").readlines()}
        self.batch_size = 16

    def _create_kwargs(self):
        ''' create kwargs

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        teacher_model = torchvision.models.resnet18(pretrained=True)

        return {
            'model': model,
            'loss': torch.nn.CrossEntropyLoss(),
            'finetunner':BasicFinetunner(torchvision.models.resnet18(pretrained=True),True),
            'distiller': BasicDistiller(teacher_model, True),
            'adapter': DANNAdapter(512, 8, 0.0, 5.0, 1.0, 100),
            # 'training_dataloader': torch.utils.data.DataLoader(
            #     Office31("/home/vmagent/app/data/dataset/office31/amazon", self.label_map, None, 'RGB'),
            #     batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True),
            # 'adaption_source_domain_training_dataset': Office31("/home/vmagent/app/data/dataset/office31/webcam", self.label_map, None, 'RGB'),
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

        assert type(new_model) == TransferrableModel
        assert tensor_near_equal(new_model.backbone.loss(tensor1, tensor0), kwargs['loss'](tensor1, tensor0))
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is kwargs['distiller']

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
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is kwargs['distiller']

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
        for transfer_strategy in [TransferStrategy.OnlyDistillationStrategy]:
            for name in ["distiller"]:
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
        transfer_strategy = TransferStrategy.OnlyDomainAdaptionStrategy
        for name in ["adapter"]:
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
        _make_transferrable(**kwargs)
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

        assert new_model.transfer_strategy == TransferStrategy.OnlyFinetuneStrategy
        assert new_model.enable_target_training_label == True
    def test_make_transferrable_with_knowledge_distillation(self):
        ''' test make_transferrable_with_knowledge_distillation

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_knowledge_distillation(
            model=kwargs['model'], loss=kwargs['loss'],
            distiller=kwargs['distiller'],
            enable_target_training_label=kwargs['enable_target_training_label'],
            backbone_loss_weight=kwargs['backbone_loss_weight'],
            distiller_loss_weight=kwargs['distiller_loss_weight'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is None
        assert new_model.distiller is kwargs['distiller']

        assert new_model.transfer_strategy == TransferStrategy.OnlyDistillationStrategy
        assert new_model.enable_target_training_label == kwargs['enable_target_training_label']

    def test_make_transferrable_with_domain_adaption(self):
        ''' test make_transferrable_with_domain_adaption

        :return:
        '''
        kwargs = self._create_kwargs()
        new_model = make_transferrable_with_domain_adaption(
            model=kwargs['model'], loss=kwargs['loss'],
            adapter=kwargs['adapter'],
            enable_target_training_label=kwargs['enable_target_training_label'],
            backbone_loss_weight=kwargs['backbone_loss_weight'],
            adapter_loss_weight=kwargs['adapter_loss_weight'])

        assert type(new_model) == TransferrableModel
        assert new_model.adapter is kwargs['adapter']
        assert new_model.distiller is None

        assert new_model.transfer_strategy == TransferStrategy.OnlyDomainAdaptionStrategy
        assert new_model.enable_target_training_label == kwargs['enable_target_training_label']

class TestTransferrableModel:
    ''' Test TransferrableModel

    '''

    def _create_kwargs(self):
        ''' create kwargs

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        teacher_model = torchvision.models.resnet18(pretrained=True)

        return {
            'model': model,
            'loss': torch.nn.CrossEntropyLoss(),
            'finetunner': BasicFinetunner(torchvision.models.resnet18(pretrained=True), True),
            'distiller': BasicDistiller(teacher_model, True),
            'adapter': DANNAdapter(512, 8, 0.0, 5.0, 1.0, 100),
            # 'training_dataloader': torch.utils.data.DataLoader(
            #     Office31("/home/vmagent/app/data/dataset/office31/amazon", self.label_map, None, 'RGB'),
            #     batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True),
            # 'adaption_source_domain_training_dataset': Office31("/home/vmagent/app/data/dataset/office31/webcam", self.label_map, None,
            #                                                     'RGB'),
            'transfer_strategy': TransferStrategy.OnlyDistillationStrategy,
            'enable_target_training_label': True,
            'backbone_loss_weight': 1.0,
            'distiller_loss_weight': 1.0,
            'adapter_loss_weight': 1.0,
        }

    def setup(self):
        # self.label_map = {item.split('\t')[0]: int(item.split('\t')[1]) for item in
        #              open("/home/vmagent/app/data/dataset/office31/label_map.txt").readlines()}
        self.batch_size = 16
        self.input = torch.randn([self.batch_size, 3, 224, 224])
        self.num_class = 1000

    def test_create_invalid_strategy(self):
        ''' test create  TransferrableModel with invalid strategy

        :return:
        '''
        model = torchvision.models.resnet18(pretrained=False)
        finetunner = BasicFinetunner( torchvision.models.resnet18(pretrained=True),True)
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

    def test_getattribute_in_train_mode(self):
        ''' test __getattribute__ in training mode

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.__dict__  # no exception
        ############### training mode ##########
        student_label = torch.zeros([self.batch_size,self.num_class])
        model.train()
        assert type(model(self.input)) is TransferrableModelOutput
        assert tensor_near_equal(model(self.input).backbone_output,kwargs['model'](self.input))
        assert type(model.loss(model(self.input),student_label)) is TransferrableModelLoss

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
        assert tensor_near_equal(new_loss.backbone_loss, kwargs['loss'](output,label))
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
            new_output = model._distillation_forward(x)
            assert type(new_output) == TransferrableModelOutput
            assert tensor_near_equal(new_output.backbone_output, kwargs['model'](x))
            assert tensor_near_equal(new_output.distiller_output, kwargs['distiller'](x))
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
        assert tensor_near_equal(new_loss.backbone_loss, kwargs['loss'](student_output,student_label))
        assert tensor_near_equal(new_loss.distiller_loss, kwargs['distiller'].loss(teacher_output,student_output))
        assert new_loss.adapter_loss is None
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
        # _input = (self.input, torch.ones_like(self.input))
        # output = model(_input)
        # assert type(output) == TransferrableModelOutput
        # assert tensor_near_equal(output.backbone_output, kwargs['model'](_input[0]))
        # assert output.distiller_output is not None
        # assert output.adapter_output is None
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
    def test_get_training_metrics(self):
        ''' test get_training_metrics

        :return:
        '''
        for transfer_strategy in [TransferStrategy.OnlyFinetuneStrategy,
                                  TransferStrategy.OnlyDistillationStrategy]:
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
                metric_fn_map = {'acc':acc}
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
            basic_weights = kwargs['finetunner']._pretrained_params
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

class TestTransferrableModelForDomainAdaption:
    ''' Test TransferrableModel For DomainAdaption

    '''
    def _create_kwargs(self):
        ''' create kwargs

        :return:
        '''
        backbone_kwargs = {
            'threeD': True, 
            'input_channels': 1, 
            'base_num_features': 30, 
            'num_classes': self.num_class, 
            'num_conv_per_stage': 2, 
            'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        }
        model = Generic_UNet_DA(**backbone_kwargs)
        model._deep_supervision = False

        adv_kwargs = {
            'input_channels': [30, 60, 120, 240, 320], 
            'threeD': True, 
            'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], 
            'loss_weight': [1.0, 0.0, 0.0]
        }

        return {
            'model': model,
            'loss': None,
            'finetunner': None,
            'distiller': None,
            'adapter': CACDomainAdversarialLoss(**adv_kwargs),
            'transfer_strategy': TransferStrategy.OnlyDomainAdaptionStrategy,
            'enable_target_training_label': False,
            'backbone_loss_weight': 1.0,
            'distiller_loss_weight': 0.0,
            'adapter_loss_weight': 1.0,
        }

    def setup(self):
        self.num_class = 2
        self.batch_size = 2
        self.patch_size = [80, 160, 160]
        self.input_size = [self.batch_size, 1] + self.patch_size
        self.input = torch.randn(self.input_size)

    def test_adaption_forward(self):
        ''' test _adaption_forward

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()
        for i in range(0, 2):
            x1 = torch.randn_like(self.input)
            x2 = torch.randn_like(self.input)
            new_output = model._adaption_forward(x1,x2)
            assert type(new_output) == tuple
            assert type(new_output[0]) == TransferrableModelOutput
            assert type(new_output[1]) == TransferrableModelOutput
            print(new_output[0].backbone_output[0][0].shape)
            print(kwargs['model'](x1)[0][0].shape)
            assert tensor_near_equal(new_output[0].backbone_output[0][0], kwargs['model'](x1)[0][0])
            assert tensor_near_equal(new_output[1].backbone_output[0][0], kwargs['model'](x2)[0][0])
            assert new_output[0].distiller_output is None
            assert new_output[1].distiller_output is None
    
    def test_adaption_loss(self):
        ''' test _adaption_loss

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()

        source_data, target_data = torch.randn_like(self.input), torch.randn_like(self.input)
        source_label, target_label = [torch.zeros(self.input_size)], [torch.zeros(self.input_size)]
        input_sample = (source_data, target_data)
        label = (source_label, target_label)

        source_output, *source_feat = model.backbone(source_data)
        target_output, *target_feat = model.backbone(target_data)
        print(source_output[0].shape, target_output[0].shape)
        new_loss = model._adaption_loss(input_sample, label)
        
        assert type(new_loss) == TransferrableModelLoss
        assert new_loss.distiller_loss is None
        assert tensor_near_equal(new_loss.total_loss,new_loss.backbone_loss + new_loss.adapter_loss)
        assert tensor_near_equal(new_loss.backbone_loss,
                                    model.backbone.loss(source_output, source_label)*model.backbone_loss_weight)

        adpter_loss = model.adapter(*(
            (source_output, *source_feat),
            (target_output, *target_feat),
            source_label
        ))
        assert tensor_near_equal(new_loss.adapter_loss, adpter_loss)

    def test_forward_OnlyDomainAdaptionStrategy(self):
        ''' test forward with OnlyDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
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
        assert tensor_near_equal(output[0].backbone_output[0][0], kwargs['model'](_input[0])[0][0])
        assert tensor_near_equal(output[1].backbone_output[0][0], kwargs['model'](_input[1])[0][0])
        assert output[0].distiller_output is None
        assert output[1].distiller_output is None

    def test_loss_OnlyDomainAdaptionStrategy(self):
        ''' test loss with OnlyDomainAdaptionStrategy

        :return:
        '''
        kwargs = self._create_kwargs()
        model = _make_transferrable(**kwargs)
        model.train()

        input_sample = (self.input,self.input+1)
        label = ([torch.ones(self.input_size)], [torch.zeros(self.input_size)])
        loss = model.loss(input_sample, label)
        assert type(loss) == TransferrableModelLoss
        assert tensor_near_equal(loss.total_loss, loss.backbone_loss + loss.adapter_loss)
        assert loss.distiller_loss is None
        assert loss.adapter_loss is not None


if __name__ == "__main__":
    # test = TestTransferrableModel()
    # test.setup()

    test = TestTransferrableModelForDomainAdaption()
    test.setup()
    # test.test_adaption_forward()
    # test.test_adaption_loss()
    # test.test_forward_OnlyDomainAdaptionStrategy()
    test.test_loss_OnlyDomainAdaptionStrategy()
