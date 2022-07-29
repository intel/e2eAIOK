#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 7/18/2022 8:37 AM
import logging
import re

def layerNamePatternMatch(source_name,source_name_pattern_map):
    ''' Check whether source_name match the pattern

    :param source_name: source layer name
    :param source_name_pattern_map: target pattern and whether trainable
    :return: (match or not, trainable or not)
    '''
    for (pattern, trainable) in source_name_pattern_map.items():
        if re.match(pattern,source_name):
            return (True,trainable)
    return (False,None)

def copyParameterFromPretrained(source_model,pretrained_state_dict,source_name_pattern_map,name_map = None):
    ''' Copy Parameter From pretrained model

    :param source_model: source model, whose parameter will be replaced
    :param pretrained_state_dict: pretrained model's state dict
    :param source_name_pattern_map: source layer name pattern to decide which layer parameter should be replaced and whether trainable
    :param name_map: a map from source model layer name to target model layer name. If not given, only target model layer name is used.
    :return:
    '''
    for (name, tensor) in pretrained_state_dict.items():
        logging.info("layers in pretrained_state_dict: layer name[%s], size[%s]"%(name, tensor.size()))
    if not name_map:
        name_map = {x[0]:x[0] for x in pretrained_state_dict.items()}

    state_dict = {}
    untrainable_set = set()
    for (source_name, target_name) in name_map.items():
        (match,trainable) = layerNamePatternMatch(source_name,source_name_pattern_map)
        if match:
            logging.info("source_name[%s] match pattern, and use target_name[%s]"%(source_name,target_name))
            state_dict[source_name] = pretrained_state_dict[target_name]
            if int(trainable) == 0 :
                untrainable_set.add(source_name)
                logging.info("source_name[%s] untrainable" % source_name)

    source_model.load_state_dict(state_dict,False)
    if untrainable_set:
        for (param_name, param) in source_model.named_parameters():
            if param_name in untrainable_set:
                logging.info("set param_name[%s] untrainable" % param_name)
                param.requires_grad = False
    else:
        logging.info("no untrainable parameters")
