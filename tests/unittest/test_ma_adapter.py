import pytest
import sys
import os
from e2eAIOK.ModelAdapter.src.engine_core.adapter.factory import createAdapter
from e2eAIOK.ModelAdapter.src.engine_core.adapter.adversarial.adversarial_adapter import AdversarialAdapter
from e2eAIOK.ModelAdapter.src.engine_core.adapter.adversarial.dann_adapter import DANNAdapter
from e2eAIOK.ModelAdapter.src.engine_core.adapter.adversarial.cdan_adapter import CDANAdapter,RandomLayer
from e2eAIOK.ModelAdapter.src.engine_core.adapter.adversarial.grl import GradientReverseLayer
import torch
import numpy as np

from e2eAIOK.common.trainer.utils.utils import tensor_near_equal

class TestAdapterFactory:
    ''' Test Adapter

    '''
    def test_createDANN(self):
        ''' test create DANN

        :return:
        '''
        for input_size in [4,8,16,32]:
            hidden_size = input_size//2
            for dropout in [0.0,0.1,0.2,0.3]:
                for grl_coeff_alpha in [1.0,2.0,3.0,4.0]:
                    for grl_coeff_high in [0.5,1.0,2.0]:
                        for max_iter in [10,100,200]:
                            kwargs = {
                                'input_size': input_size,
                                'hidden_size': hidden_size,
                                'dropout': dropout,
                                'grl_coeff_alpha': grl_coeff_alpha,
                                'grl_coeff_high': grl_coeff_high,
                                'max_iter': max_iter
                            }
                            adpter = createAdapter("DANN", **kwargs)
                            assert kwargs['input_size'] == adpter.ad_layer1.in_features
                            assert kwargs['hidden_size'] == adpter.ad_layer1.out_features
                            assert abs(kwargs['dropout'] - adpter.dropout1.p) <= 1e-9
                            assert abs(kwargs['grl_coeff_alpha'] - adpter.grl.coeff_alpha) <= 1e-9
                            assert abs(kwargs['grl_coeff_high'] - adpter.grl.coeff_high) <= 1e-9
                            assert kwargs['max_iter'] == adpter.grl.max_iter

    def test_createCDAN(self):
        ''' test create CDAN

        :return:
        '''
        for input_size in [4,8,16,32]:
            hidden_size = input_size//2
            for dropout in [0.0,0.1,0.2,0.3]:
                for grl_coeff_alpha in [1.0,2.0,3.0,4.0]:
                    for grl_coeff_high in [0.5,1.0,2.0]:
                        for max_iter in [10,100,200]:
                            for backbone_output_size in [10,20]:
                                for enable_random_layer in [-1,0,1,2]:
                                    for enable_entropy_weight in [-1,0,1,2]:
                                        kwargs = {
                                            'input_size': input_size,
                                            'hidden_size': hidden_size,
                                            'dropout': dropout,
                                            'grl_coeff_alpha': grl_coeff_alpha,
                                            'grl_coeff_high': grl_coeff_high,
                                            'max_iter': max_iter,
                                            'backbone_output_size': backbone_output_size,
                                            'enable_random_layer': enable_random_layer,
                                            'enable_entropy_weight': enable_entropy_weight,
                                        }
                                        adpter = createAdapter("CDAN", **kwargs)
                                        assert kwargs['input_size'] == adpter.ad_layer1.in_features
                                        assert kwargs['hidden_size'] == adpter.ad_layer1.out_features
                                        assert abs(kwargs['dropout'] - adpter.dropout1.p) <= 1e-9
                                        assert abs(kwargs['grl_coeff_alpha'] - adpter.grl.coeff_alpha) <= 1e-9
                                        assert abs(kwargs['grl_coeff_high'] - adpter.grl.coeff_high)  <= 1e-9
                                        assert kwargs['max_iter'] == adpter.grl.max_iter

                                        if kwargs['enable_random_layer'] > 0:  # enable random layer
                                            assert adpter._random_layer is not None
                                            assert kwargs['backbone_output_size'] == \
                                                   adpter._random_layer.input_dim_list[1]
                                        else:
                                            assert adpter._random_layer is None

                                        assert (kwargs['enable_entropy_weight'] > 0) == adpter._enable_entropy_weight

    def test_createInvalid_Kwargs(self):
        ''' test create invalid adapter by invalid kwargs

        :return:
        '''
        for name in ['DANN', 'CDAN']:
            with pytest.raises(KeyError) as e:
                adapter = createAdapter(name)
            assert e.value.args[0] == 'input_size'

    def test_createInvalid_NotImplemented(self):
        ''' test create invalid adapter by NotImplemented class

        :return:
        '''
        for name in ['ABC','dann','cdan','Dann','cDAN']:
            with pytest.raises(NotImplementedError) as e:
                adapter = createAdapter(name)
            assert e.value.args[0] == "[%s] is not supported"%name

class TestAdversarialAdapter:
    ''' Test AdversarialAdapter

    '''
    def setup(self):
        self.in_feature = 16
        hidden_size = 8
        dropout_rate = 0.0
        grl_coeff_alpha = 5.0
        grl_coeff_high = 1.0
        max_iter = 100
        self.adapter = AdversarialAdapter(self.in_feature, hidden_size,dropout_rate,
                                          grl_coeff_alpha,grl_coeff_high,max_iter)

    # def teardown(self):
    #     print("teardown: run after every test case")
    def test_forward(self):
        ''' test forward

        :return:
        '''
        batch_size = 16
        y = self.adapter(torch.rand([batch_size,self.in_feature]))
        assert y.size(0) == batch_size
        assert y.size(1) == 1
        assert torch.all(y > 0.0).item()
        assert torch.all(y < 1.0).item()

    def test_forward_bad_shape(self):
        ''' test forward with bad shape

        :return:
        '''
        batch_size = 16
        with pytest.raises(RuntimeError) as e:
            self.adapter(torch.rand([batch_size,self.in_feature + 1]))
        assert e.value.args[0].startswith('mat1 and mat2 shapes cannot be multiplied')

    def test_make_label(self):
        ''' test make_label

        :return:
        '''
        for shape in [(1, 2), (2, 1), (2, 4), (4, 2)]:
            assert torch.equal(self.adapter.make_label(shape, is_source=True),torch.ones(shape, dtype=torch.float))
            assert torch.equal(self.adapter.make_label(shape, is_source=False),torch.zeros(shape, dtype=torch.float))

    def test_loss(self):
        ''' test loss

        :return:
        '''
        with pytest.raises(NotImplementedError) as e:
            self.adapter.loss(None,None)
        assert e.value.args[0].startswith("must implement loss function")

class TestDANNAdapter:
    ''' Test DANNAdapter

    '''

    def setup(self):
        self.in_feature = 16
        hidden_size = 8
        dropout_rate = 0.0
        grl_coeff_alpha = 5.0
        grl_coeff_high = 1.0
        max_iter = 100
        self.adapter = DANNAdapter(self.in_feature, hidden_size, dropout_rate,
                                          grl_coeff_alpha, grl_coeff_high, max_iter)
    def test_loss(self):
        ''' test loss

        :return:
        '''
        batch_size = 16
        label =  torch.range(1,batch_size) % 2
        prob = torch.range(1,batch_size)/16.0
        assert torch.abs(self.adapter.loss(label, label)).item() <= 1e-9 # zero loss
        assert  torch.equal(torch.nn.BCELoss()(prob,label),self.adapter.loss(prob,label))

    def test_loss_bad_prob(self):
        ''' test loss with bad prob

        :return:
        '''
        batch_size = 16
        label = torch.range(1, batch_size) % 2
        prob = torch.range(1, batch_size) / 16.0 + 1.0 # greater than 1.0
        with pytest.raises(RuntimeError) as e:
            self.adapter.loss(prob,label)
        assert e.value.args[0] == "all elements of input should be between 0 and 1"

class TestCDANAdapter:
    ''' Test CDANAdapter

    '''
    def setup(self):
        self.in_feature = 16
        hidden_size = 8
        dropout_rate = 0.0
        grl_coeff_alpha = 5.0
        grl_coeff_high = 1.0
        max_iter = 100
        self.backbone_output_size = 10
        enable_random_layer = True
        enable_entropy_weight = True
        self.adapter = CDANAdapter(self.in_feature, hidden_size, dropout_rate,
                                   grl_coeff_alpha, grl_coeff_high, max_iter,
                                   self.backbone_output_size,enable_random_layer,enable_entropy_weight)

    def test_forward_input_if_use_random_layer(self):
        ''' test _forward_input if use random_layer

        :return:
        '''
        batch_size = 16
        adapter_input = torch.zeros([batch_size, self.in_feature])
        backbone_output = torch.zeros([batch_size, self.backbone_output_size])
        new_input = self.adapter._forward_input(adapter_input, backbone_output)
        assert batch_size == new_input.size(0)
        assert self.in_feature == new_input.size(1)

    def test_forward_input_if_no_random_layer(self):
        ''' test _forward_input if no random_layer

        :return:
        '''
        batch_size = 16
        adapter_input = torch.zeros([batch_size, self.in_feature])
        backbone_output = torch.zeros([batch_size, self.backbone_output_size])
        layer = self.adapter._random_layer
        self.adapter._random_layer = None
        new_input = self.adapter._forward_input(adapter_input, backbone_output)
        assert batch_size == new_input.size(0)
        assert self.in_feature * self.backbone_output_size == new_input.size(1)
        self.adapter._random_layer = layer

    def test_forward(self):
        ''' test forward

        :return:
        '''
        batch_size = 16
        x = torch.zeros([batch_size, self.in_feature])
        with pytest.raises(KeyError) as e:
            self.adapter.forward(x)
        assert e.value.args[0].startswith("backbone_output")

        backbone_output = torch.zeros([batch_size, self.backbone_output_size])
        self.adapter.forward(x,backbone_output=backbone_output) # no exception

    def test_normalized_entropy_weight(self):
        ''' test _normalized_entropy_weight

        :return:
        '''
        #################### same weight #########################
        backbone_output = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        predict_weight = self.adapter._normalized_entropy_weight(backbone_output)
        # each sample is the same weight, because they have the same entropy
        target_weight = torch.tensor([ 0.5, 0.5])
        assert tensor_near_equal(target_weight , predict_weight)
        ################### different weight ########################
        backbone_output = torch.tensor([[1.0, 100.0], [1.0, 1.0]])
        predict_weight = self.adapter._normalized_entropy_weight(backbone_output)
        # the first sample is more focused, because it has less entropy
        # 0.5*(-log_e(0.5)) = 0.34657359, 1 + exp(-0.34657359) = 1.70710678
        target_weight = torch.tensor([2.0/(2.0 + 1.70710678), 1.70710678/(2.0+1.70710678)])
        assert tensor_near_equal(target_weight , predict_weight)

    def test_loss_if_no_entropy_weight(self):
        ''' test loss if no entropy_weight

        :return:
        '''
        batch_size = 16
        label = torch.range(1, batch_size) % 2
        prob = torch.range(1, batch_size) / 16.0
        self.adapter._enable_entropy_weight = False

        assert torch.abs(self.adapter.loss(label, label)).item() <= 1e-9  # zero loss
        self.adapter.loss(prob, label) # no exception
        with pytest.raises(RuntimeError) as e:
            self.adapter.loss(prob + 1.0, label)
        assert e.value.args[0] == "all elements of input should be between 0 and 1"
        self.adapter._enable_entropy_weight = True

    def test_loss_if_use_entropy_weight(self):
        '''test loss if use entropy_weight

        :return:
        '''
        self.adapter._enable_entropy_weight = False
        batch_size = 16
        label = torch.range(1, batch_size) % 2
        prob = torch.range(1, batch_size) / 16.0
        loss1 = self.adapter.loss(prob, label) # no exception
        ############### witout backbone_output kwargs ########
        self.adapter._enable_entropy_weight = True
        with pytest.raises(KeyError) as e:
            self.adapter.loss(prob,label)
        assert e.value.args[0].startswith("backbone_output")
        ############## different weight ###################
        backbone_output = torch.randn([batch_size,self.backbone_output_size])
        assert torch.abs(self.adapter.loss(label, label,backbone_output = backbone_output)).item() <= 1e-9  # zero loss
        loss2 = self.adapter.loss(prob,label,backbone_output = backbone_output)
        assert not tensor_near_equal(loss2 , loss1, 1e-3) # loss2 is entropy weighted with different weight
        ############## same weight ###################
        backbone_output = torch.zeros([batch_size, self.backbone_output_size])
        assert torch.abs(self.adapter.loss(label, label, backbone_output=backbone_output)).item() <= 1e-9  # zero loss
        loss3 = self.adapter.loss(prob,label,backbone_output = backbone_output)
        assert tensor_near_equal(loss3, loss1, 1e-5) # loss3 is entropy weighted with same weight

class TestRandomLayer:
    ''' Test RandomLayer

    '''
    def setup(self):
        self.input_dim_list = [16,24]
        self.output_dim = 20
        self.random_layer = RandomLayer(self.input_dim_list,self.output_dim)
        assert self.random_layer.input_num == len(self.input_dim_list)
        assert len(self.random_layer.random_matrix) == len(self.input_dim_list)
        assert self.random_layer.random_matrix[0].shape == torch.Size([self.input_dim_list[0],
                                                                      self.output_dim])
        assert self.random_layer.random_matrix[1].shape == torch.Size([self.input_dim_list[1],
                                                                      self.output_dim])
    def test_forward(self):
        ''' test forward

        :return:
        '''
        batch_size = 16
        input_list = [torch.randn([batch_size,i]) for i in self.input_dim_list]
        result = self.random_layer(input_list)
        assert result.shape ==  torch.Size([batch_size,self.output_dim])

class TestGradientReverseLayer:
    ''' Test GradientReverseLayer

    '''

    def setup(self):
        self.coeff_alpha = 5.0
        self.coeff_high = 1.0
        self.max_iter = 100
        self.enable_step = False
        self.grl = GradientReverseLayer(self.coeff_alpha,self.coeff_high,self.max_iter,self.enable_step)

    def test_forward_if_eable_step(self):
        ''' test forward if enable_step = True

        :return:
        '''
        batch_size = 16
        input_size = 8
        input = torch.randn([batch_size, input_size], requires_grad=True)
        self.grl.enable_step = True
        for step in range(1, self.max_iter + 1):
            input.grad = None
            output = self.grl(input)
            assert self.grl.iter_num == step
            assert torch.equal(input, output)
            target_coeff = 2.0 * self.coeff_high / (
                        1.0 + np.exp(-self.coeff_alpha * (step - 1) / self.max_iter)) - self.coeff_high
            assert abs(self.grl.coeff - target_coeff) <= 1e-9
            torch.sum(output).backward()
            assert tensor_near_equal(input.grad, -target_coeff*torch.ones_like(input.grad))  # grad is -1 * target_coeff

    def test_forward_if_no_enable_step(self):
        ''' test forward if enable_step = False

        :return:
        '''
        batch_size = 16
        input_size = 8
        input = torch.randn([batch_size, input_size], requires_grad=True)
        self.grl.enable_step = False
        for step in range(1, self.max_iter + 1):
            input.grad = None
            output = self.grl(input)
            assert self.grl.iter_num == 0
            assert torch.equal(input, output)
            assert abs(self.grl.coeff) <= 1e-9
            torch.sum(output).backward()
            assert tensor_near_equal(input.grad,torch.zeros_like(input.grad))   # grad is always 0.0