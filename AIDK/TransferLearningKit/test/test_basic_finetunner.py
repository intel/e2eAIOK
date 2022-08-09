#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/8/2022 11:14 AM

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"test"))
import torch
import torchvision
from engine_core.finetunner.basic_finetunner import BasicFinetunner
from utils import tensor_near_equal
from copy import deepcopy

class TestBasicFinetunner:
    ''' Test BasicFinetunner

    '''
    def setup(self):
        self.pretrained_model = torchvision.models.resnet18(pretrained=True)
        self.finetunner = BasicFinetunner(self.pretrained_model,-1)
    def test_set_node_hierarchy(self):
        ''' test _set_node_hierarchy

        :return:
        '''
        node = BasicFinetunner.GraphNode('a','b','c',-1)
        for hierarchy in range(0,100):
            new_node = self.finetunner._set_node_hierarchy(node,hierarchy)
            assert node.name == new_node.name
            assert node.args == new_node.args
            assert node.target == new_node.target
            assert new_node.hierarchy ==  hierarchy
    def test_get_successors(self):
        ''' test _get_successors

        :return:
        '''
        successor_map = self.finetunner.node_successors
        assert [item for item in sorted(successor_map['x'])] == ['conv1']
        assert [item for item in sorted(successor_map['conv1'])] == ['bn1']
        assert [item for item in sorted(successor_map['bn1'])] == ['relu']
        assert [item for item in sorted(successor_map['fc'])] == ['output']
        assert [item for item in sorted(successor_map['output'])] == []
    def test_get_precursors(self):
        ''' test _get_precursors

        :return:
        '''
        precursor_map = self.finetunner.node_precursors
        assert [item for item in sorted(precursor_map['x'])] == []
        assert [item for item in sorted(precursor_map['conv1'])] == ['x']
        assert [item for item in sorted(precursor_map['bn1'])] == ['conv1']
        assert [item for item in sorted(precursor_map['output'])] == ['fc']
    def test_assign_hierarchy_with_output_based(self):
        ''' test _assign_hierarchy with output based

        :return:
        '''
        finetunner = BasicFinetunner(self.pretrained_model,-1,True,True)
        node_map = finetunner._node_map
        for (k,v) in node_map.items():
            print(k,v.hierarchy)
        assert node_map['x'].hierarchy == 0
        assert node_map['conv1'].hierarchy == 1
        assert node_map['bn1'].hierarchy == 2
        assert node_map['relu'].hierarchy == 3
        assert node_map['maxpool'].hierarchy == 4
        assert node_map['layer1_0_conv1'].hierarchy == 0
        assert node_map['layer1_0_bn1'].hierarchy == 1
        assert node_map['layer1_0_relu'].hierarchy == 2
        assert node_map['layer1_0_conv2'].hierarchy == 3
        assert node_map['layer1_0_bn2'].hierarchy == 4
        assert node_map['avgpool'].hierarchy == 27

    def test_assign_hierarchy_with_input_based(self):
        ''' test _assign_hierarchy with input based

        :return:
        '''
        finetunner = BasicFinetunner(self.pretrained_model, -1,True,False)
        node_map = finetunner._node_map
        assert node_map['x'].hierarchy == 0
        assert node_map['conv1'].hierarchy == 1
        assert node_map['bn1'].hierarchy == 2
        assert node_map['relu'].hierarchy == 3
        assert node_map['maxpool'].hierarchy == 4
        assert node_map['layer1_0_conv1'].hierarchy == 5
        assert node_map['layer1_0_bn1'].hierarchy == 6
        assert node_map['layer1_0_relu'].hierarchy == 7
        assert node_map['layer1_0_conv2'].hierarchy == 8
        assert node_map['avgpool'].hierarchy == 27

    def test_buid_node_graph(self):
        ''' test _buid_node_graph

        :return:
        '''
        self.test_get_successors()
        self.test_get_precursors()
        self.test_assign_hierarchy_with_output_based()
        self.test_assign_hierarchy_with_input_based()
    def test_is_same_structure(self):
        ''' test _is_same_structure

        :return:
        '''
        assert self.finetunner._is_same_structure(torchvision.models.resnet18(pretrained=False))
        assert not self.finetunner._is_same_structure(torchvision.models.resnet34(pretrained=False))
        ######## frozen #########
        target_network = torchvision.models.resnet18(pretrained=False)
        for p in target_network.parameters():
            p.requires_grad = False
        assert self.finetunner._is_same_structure(target_network)
    def test_get_node_by_name(self):
        ''' test _get_node_by_name

        :return:
        '''
        for name in ['abc','X','Layer1_0*','Output']:
            assert self.finetunner._get_node_by_name(name) is None

        node = self.finetunner._get_node_by_name('x')
        assert node is not None
        assert node.name == 'x'
        assert node.hierarchy == 0

        node = self.finetunner._get_node_by_name('layer1_0.*')
        assert node is not None
        assert node.name == 'layer1_0_bn2'

        node = self.finetunner._get_node_by_name('layer1.*')
        assert node is not None
        assert node.name == 'layer1_1_bn2'

        node = self.finetunner._get_node_by_name('fc.*')
        assert node is not None
        assert node.name == 'fc'

    def test_get_precursor_of_node(self):
        ''' test _get_precursor_of_node

        :return:
        '''
        assert sorted([item for item in self.finetunner._get_precursor_of_node("x")]) == []
        assert sorted([item for item in self.finetunner._get_precursor_of_node("conv1")]) == ['x']
        assert sorted([item for item in self.finetunner._get_precursor_of_node("bn1")]) ==['conv1', 'x']
        assert sorted([item for item in self.finetunner._get_precursor_of_node("layer1_0_bn2")]) == ['bn1', 'conv1', 'layer1_0_bn1', 'layer1_0_conv1', 'layer1_0_conv2', 'layer1_0_relu', 'maxpool', 'relu', 'x']
        assert sorted([item for item in self.finetunner._get_precursor_of_node("layer1_1_bn2")]) == ['add', 'bn1', 'conv1', 'layer1_0_bn1', 'layer1_0_bn2', 'layer1_0_conv1', 'layer1_0_conv2', 'layer1_0_relu', 'layer1_0_relu_1', 'layer1_1_bn1', 'layer1_1_conv1', 'layer1_1_conv2', 'layer1_1_relu', 'maxpool', 'relu', 'x']
        assert sorted([item for item in self.finetunner._get_precursor_of_node("add_1")]) == ['add', 'bn1', 'conv1', 'layer1_0_bn1', 'layer1_0_bn2', 'layer1_0_conv1', 'layer1_0_conv2', 'layer1_0_relu', 'layer1_0_relu_1', 'layer1_1_bn1', 'layer1_1_bn2', 'layer1_1_conv1', 'layer1_1_conv2', 'layer1_1_relu', 'maxpool', 'relu', 'x']

    def test_get_finetuned_state_keys_by_name(self):
        ''' test _get_finetuned_state_keys by layer name

        :return:
        '''
        for name in ["abc",'X','Conv1','FC']:
            with pytest.raises(RuntimeError) as e:
                self.finetunner._get_finetuned_state_keys(name)
            assert e.value.args[0] == 'Can not find node by name [%s]'%name
        assert self.finetunner._get_finetuned_state_keys("") == ['bn1.bias', 'bn1.num_batches_tracked', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'conv1.weight', 'fc.bias', 'fc.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.num_batches_tracked', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']
        assert self.finetunner._get_finetuned_state_keys(None) == self.finetunner._get_finetuned_state_keys("")
        assert self.finetunner._get_finetuned_state_keys("x") == []
        assert self.finetunner._get_finetuned_state_keys("conv1") == ['conv1.weight']
        assert self.finetunner._get_finetuned_state_keys("bn1") == ['bn1.bias', 'bn1.num_batches_tracked', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'conv1.weight']
        assert self.finetunner._get_finetuned_state_keys("avgpool") == ['bn1.bias', 'bn1.num_batches_tracked', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'conv1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.num_batches_tracked', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.bn1.bias', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.num_batches_tracked', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.bn1.bias', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.bias', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.weight', 'layer2.1.bn2.bias', 'layer2.1.bn2.num_batches_tracked', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.weight', 'layer3.0.bn2.bias', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.bias', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.weight', 'layer3.1.bn1.bias', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.num_batches_tracked', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.bn1.bias', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.weight', 'layer4.0.bn2.bias', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.bias', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.weight', 'layer4.1.bn2.bias', 'layer4.1.bn2.num_batches_tracked', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']
        assert self.finetunner._get_finetuned_state_keys("fc") == self.finetunner._get_finetuned_state_keys("")

    def test_get_finetuned_state_keys_by_hierarchy(self):
        ''' test _get_finetuned_state_keys by layer hierarchy

        :return:
        '''
        assert self.finetunner._get_finetuned_state_keys("conv1") == self.finetunner._get_finetuned_state_keys(1)
        assert self.finetunner._get_finetuned_state_keys("bn1") == self.finetunner._get_finetuned_state_keys(2)
        assert self.finetunner._get_finetuned_state_keys("avgpool") != self.finetunner._get_finetuned_state_keys(27) # different
        assert self.finetunner._get_finetuned_state_keys(0) == []

    def test_finetune_network(self):
        ''' test finetune_network

        :return:
        '''
        for is_frozen in [False, True]:
            for top_finetuned_layer in ["conv1","bn1","avgpool","fc",0,1,2,-1,-2]:
                target_network = torchvision.models.resnet18(pretrained=False)
                finetunner = BasicFinetunner(torchvision.models.resnet18(pretrained=True),top_finetuned_layer,is_frozen,False)
                ############# save old dict ##########
                old_state_dict = deepcopy(target_network.state_dict())
                ############ finetune ################
                finetunned_dict_keys = self.finetunner._get_finetuned_state_keys(top_finetuned_layer)
                finetunner.finetune_network(target_network)
                new_state_dict = target_network.state_dict()
                named_parameters = {name: parameter for (name, parameter) in target_network.named_parameters()}
                ################ check ###############
                for key in finetunned_dict_keys:
                    if key.endswith("num_batches_tracked"): # statistic for bn layer, could be 0 for finetunned model
                        continue
                    if type(old_state_dict[key]) is torch.Tensor:
                        assert not tensor_near_equal(old_state_dict[key], new_state_dict[key],1e-3)
                    else:
                        assert old_state_dict[key] != new_state_dict[key]
                    if key in named_parameters:
                        assert named_parameters[key].requires_grad == (not is_frozen)