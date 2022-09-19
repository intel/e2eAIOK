#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/8/2022 9:19 AM
import logging
import torch.fx
import re
from collections import namedtuple

class BasicFinetunner:
    ''' Basic finetunner

    '''
    GraphNode = namedtuple('GraphNode',['name','args','target','hierarchy'])
    # name: node name, i.e. layer1_0_conv1
    # args: node args, must be a tuple, i.e.  (maxpool,)
    # target: node target, i.e. layer1.0.conv1
    # hierarchy: an integer to indicate layer hierarchy, input hierarchy is 0, and output hierarchy is the biggest

    def __init__(self,pretrained_network,top_finetuned_layer,is_frozen=True):
        ''' init method

        :param pretrained_network: the pretrained_network
        :param top_finetuned_layer: the top finedtuned layer.
                                    Layers under top_finetuned_layer (include) will be finetuned.
                                    Could be: layer name(regular expression supported)
                                              or hierarchy number  (-1 means output.hierarchy)
                                    If it is None or "" or 0, then all layers will be finetuned.
        :param is_frozen: whether frozen the finetunned bottom layers
        '''
        self._pretrained_network = pretrained_network
        self.is_frozen = is_frozen
        self._top_finetuned_layer = top_finetuned_layer
        self._buid_node_graph()
        self.finetuned_state_keys = self._get_finetuned_state_keys(top_finetuned_layer)
        self.pretrained_state_dict = {k: v for (k, v)
                                      in self._pretrained_network.state_dict().items()
                                      if k in self.finetuned_state_keys}
    def __str__(self):
        _str = "BasicFinetunner:\n\t"
        _str += "pretrained_network:%s\n\t"%self._pretrained_network
        _str += "top_finetuned_layer:%s\n\t" % self._top_finetuned_layer
        _str += "is_frozen:%s\n\t" % self.is_frozen
        return  _str

    def _set_node_hierarchy(self,node,hierarchy):
        ''' set node hierarchy

        :param node: original node
        :param hierarchy: hierarchy num
        :return: new node
        '''
        return self.GraphNode(node.name, node.args, node.target,hierarchy)

    def _get_successors(self):
        ''' get node successors

        :return:
        '''
        self.node_successors = {}
        for (name, node) in self._node_map.items():
            for arg in node.args:
                if type(arg) == int:
                    continue
                elif type(arg) == torch.fx.node.Node:
                    if arg.name in self.node_successors:
                        self.node_successors[arg.name].add(name)
                    else:
                        self.node_successors[arg.name] = set([name,])
                else:
                    raise RuntimeError("Unknown arg type: [%s]" % type(arg))
        if 'output' not in self.node_successors:
            self.node_successors['output'] = [] # successor of output is []

    def _get_precursor_of_node(self,node_name):
        ''' get precursor of node

        :param node_name: node name
        :return: precursors of node_name
        '''


        queue = [node_name]
        result = set()
        if node_name not in self.node_precursors:
            return result

        while queue:
            name = queue.pop()
            precursors = self.node_precursors[name]
            result.update(precursors)
            queue.extend(precursors)
        return result

    def _get_precursors(self):
        ''' get node precursors

        :return:
        '''
        self.node_precursors = {}
        for (name, node) in self._node_map.items():
            self.node_precursors[name] = set(arg.name for arg in node.args if type(arg) == torch.fx.node.Node)

    def _assign_hierarchy(self):
        ''' assign hierarchy

        :return:
        '''
        queue = [(self._input_node.name, 0)]  # from input
        already_names = set()
        while queue:
            (name, hierarchy) = queue.pop()  # remove last
            ##################### current node #################
            new_node = self._set_node_hierarchy(self._node_map[name], hierarchy)
            self._node_map[name] = new_node
            already_names.add(name)
            #################### next one #########################
            next_names = self.node_successors[name]     # successor for next item
            for next_name in sorted(next_names): # sorted to make deterministic
                if next_name in already_names:  # every layer has one hierarchy
                    logging.info("[%s] has already exist, use the exist one"%next_name)
                else:
                    queue.insert(0, (next_name, hierarchy + 1))

    def _buid_node_graph(self):
        ''' construct node graph

        :return:
        '''
        gm: torch.fx.GraphModule = torch.fx.symbolic_trace(self._pretrained_network)
        print("Pretrained Model GraphModule")
        gm.graph.print_tabular()
        self._node_map = {}
        self._input_node = None
        self._output_node = None
        ####################### iterate nodes #####################
        for node in gm.graph.nodes:
            self._node_map[node.name] = self.GraphNode(node.name, node.args, node.target,-1)
            if node.op == 'placeholder':
                self._input_node= node
            elif node.op == 'output':
                self._output_node = node
        if self._input_node is None:
            raise RuntimeError("Invalid input node")
        if self._output_node is None:
            raise RuntimeError("Invalid output node")
        ########################## setup hierarchy ###################
        self._get_successors()
        self._get_precursors()
        self._assign_hierarchy()

    def _is_same_structure(self,target_network):
        ''' whether target_network is same structure with pretrained_network

        :param target_network: target network
        :return: whether these 2 network have the same structure(according to state_dict)
        '''
        state1 = self._pretrained_network.state_dict() # include all trainable parameters and untrainable buffers
        state2 = target_network.state_dict() # include all trainable parameters and untrainable buffers
        if len(state1) != len(state2): # not the same length
            return False
        if len(set([item[0] for item in state1]) - set([item [0] for item in state2])) != 0: # have difference
            return False
        return True

    def _get_node_by_name(self,name_pattern):
        ''' get node by name_pattern

        :param name_pattern: name pattern
        :return: the matched node with the higher hierarchy
        '''
        result = None
        for node in sorted(self._node_map.values(),key=lambda x:(x.hierarchy,x.name)): # from bottom hierarchy to top hierarchy
            if re.match(name_pattern, node.name) or (type(node.target) == str and re.match(name_pattern, node.target)):
                result = node
        return result

    def _get_finetuned_state_keys(self, top_finetuned_layer):
        ''' get finetuned state_dict keys

        :param top_finetuned_layer: the top finedtuned layer. See `finetune_network`.
        :return: the finetuned state_dict keys
        '''
        if  top_finetuned_layer is None or top_finetuned_layer == "": # all layers are finetunned
            logging.info("All layers are finetuned")
            return [item for item in sorted(self._pretrained_network.state_dict())]
        else:
            if type(top_finetuned_layer) is str:
                target_node = self._get_node_by_name(top_finetuned_layer)
                if target_node is None:
                    raise RuntimeError("Can not find node by name [%s]"%top_finetuned_layer)
                else:
                    finetunned_module_names= [self._node_map[item].target for item in self._get_precursor_of_node(target_node.name)] + [target_node.target]
            elif type(top_finetuned_layer) is int:
                if top_finetuned_layer < 0:
                    top_finetuned_layer = top_finetuned_layer + (1 + self._node_map[self._output_node.name].hierarchy)
                finetunned_module_names = [node.target for (name,node) in self._node_map.items() if node.hierarchy <= top_finetuned_layer]
            else:
                raise RuntimeError("top_finetuned_layer type must be string or int, but got: %s"%type(top_finetuned_layer))

            finetuned_modules = [(name,module) for (name,module) in self._pretrained_network.named_modules()
                                 if name in finetunned_module_names]
            logging.info("Finetuned modules is:%s"%(", ".join([item[0] for item in finetuned_modules])))

            finetuned_state_keys = []
            for (module_name, module) in finetuned_modules:
                finetuned_state_keys.extend(["%s.%s"%(module_name,item) for item in module.state_dict()])
            return [item for item in sorted(finetuned_state_keys)]

    def finetune_network(self,target_network):
        ''' finetune target network

        :param target_network: the target network
        :return:
        '''
        #################### whether the same network ###################
        if not self._is_same_structure(target_network):
            raise RuntimeError("Target network is not the same structure with pretrained model."
                          "\n\tTarget network weights:%s."
                          "\n\tPretrained network weights:%s"%(
                ", ".join([item[0] for item in target_network.named_parameters()]),
                ", ".join([item[0] for item in self._pretrained_network.named_parameters()])))
            return
        ############################## finetune #######################
        target_network.load_state_dict(self.pretrained_state_dict, strict=False)
        ########################## frozen #####################################
        if self.is_frozen:
            for (param_name, param) in target_network.named_parameters():
                if param_name in self.finetuned_state_keys:
                    logging.info("set param_name[%s] untrainable" % param_name)
                    param.requires_grad = False