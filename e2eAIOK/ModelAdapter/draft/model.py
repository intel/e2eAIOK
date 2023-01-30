import sys
import os
import torch.nn as nn
import torch
import logging
import torch.fx


def extract_distiller_adapter_features(model,intermediate_layer_name_for_distiller,
                                     intermediate_layer_name_for_adapter):

    ''' extract input feature for distiller and adapter
    :param model: model
    :param intermediate_layer_name_for_distiller: the intermediate_layer_name of model for distiller. Maybe None or empty
    :param intermediate_layer_name_for_adapter: the intermediate_layer_name of model for adapter. Maybe None or empty
    :return: modified model
    '''
    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    # print("GraphModule")
    # gm.graph.print_tabular()

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
    distiller_node = find_node(gm.graph,intermediate_layer_name_for_distiller) if intermediate_layer_name_for_distiller else None
    adapter_node = find_node(gm.graph,intermediate_layer_name_for_adapter) if intermediate_layer_name_for_adapter else None
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
    # print("After recompile")
    gm.graph.lint()  # Does some checks to make sure the Graph is well-formed.

    return gm