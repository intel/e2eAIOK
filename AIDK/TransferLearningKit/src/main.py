#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from engine_core.transfer_learning_engine import TLEngine
from engine_core.task_manager import TaskManager
from engine_core.model_manager import ModelManager
from sklearn.model_selection import train_test_split
import numpy as np
import logging

def stradifiedDataset(dataset,ratio_list):
    ####################### check #######################
    if np.abs(np.sum(ratio_list) - 1.0) >= 1e-8:
        logging.error("Sum of ratio_list not equal 1.0: [%s]"%(",".join(str(i) for i in ratio_list)))
        raise RuntimeError("Sum of ratio_list not equal 1.0: [%s]"%(",".join(str(i) for i in ratio_list)))
    if len(ratio_list) == 1:
        logging.info("ratio_list only one element")
        return dataset
    ############################## split ##################
    idx = 0
    labels = [] # all label
    for (no,(_,label)) in enumerate(dataset):
        idx += 1
        labels.append(label)
    logging.info('dataset size:%s'%idx)

    head,*tail_list = ratio_list # deep copy
    X = np.arange(0,idx)         # init X
    Y = labels
    datasets = []
    while tail_list:
        head_X, tail_X, head_Y, tail_Y = train_test_split(X,Y,train_size=head, stratify=Y)
        datasets.append(torch.utils.data.Subset(dataset, head_X))
        ##### reset #####
        tail_list = [i/(1-head) for i in tail_list] # normalize
        head, *tail_list = tail_list
        X = tail_X
        Y = tail_Y

    datasets.append(torch.utils.data.Subset(dataset, X)) # the remain

    return datasets

def createDatasets(task_manager):
    ''' create all datasets

    :param task_manager: task manager
    :return: (train_source_dataset,train_target_dataset, validate_target_dataset, test_target_dataset)
    '''
    datasets = task_manager.createDatasets()
    train_source_dataset = datasets['train_source']
    train_target_dataset = datasets['train_target']

    if 'test_target' in datasets:
        (validate_target_dataset, test_target_dataset) = stradifiedDataset(datasets['test_target'], [0.5,0.5])
    else: # no test dataset
        (train_target_dataset, validate_target_dataset, test_target_dataset) = stradifiedDataset(train_target_dataset, [0.7,0.2,0.1])
    return (train_source_dataset,train_target_dataset,
            validate_target_dataset, test_target_dataset)

if __name__ == '__main__':
    task_manager = TaskManager("./engine_core/task_config.xml")
    torch.manual_seed(task_manager.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    ######################## create dataset and dataloader #####################
    (train_source_dataset, train_target_dataset, validate_target_dataset, test_target_dataset) = createDatasets(task_manager)
    train_source_loader = torch.utils.data.DataLoader(train_source_dataset,
        batch_size=task_manager.batch_size, shuffle=True,
        num_workers=task_manager.num_worker, drop_last=True)
    train_target_loader = torch.utils.data.DataLoader(train_target_dataset,
        batch_size=task_manager.batch_size, shuffle=True,
        num_workers=task_manager.num_worker, drop_last=True)
    validate_target_loader = torch.utils.data.DataLoader(validate_target_dataset,
        batch_size=task_manager.batch_size, shuffle=True,
        num_workers=task_manager.num_worker, drop_last=True)
    test_target_loader = torch.utils.data.DataLoader(test_target_dataset,
        batch_size=task_manager.batch_size, shuffle=True,
        num_workers=task_manager.num_worker, drop_last=True)
    ####################### align iter num ####################
    train_source_iter_num = len(train_source_loader)
    train_target_iter_num = len(train_target_loader)

    if train_source_iter_num > train_target_iter_num:
        num_iter_per_epoch = train_source_iter_num
    else:
        num_iter_per_epoch = train_target_iter_num
    ########################## create other components ############
    tensorboard_writer = task_manager.createTensorboardWriter()
    discriminator = task_manager.createDiscriminator(num_iter_per_epoch)
    backbone = task_manager.createBackbone()
    model_manager = ModelManager(task_manager.model_dir)
    logging.info("task_manager:%s"%task_manager)
    print(task_manager)
    ################################### train and evaluate ###################
    engine = TLEngine(task_manager,model_manager,backbone,discriminator,tensorboard_writer)
    # torch.autograd.set_detect_anomaly(False)
    # with torch.autograd.detect_anomaly():
    model_name = engine.train(train_source_loader,train_target_loader,validate_target_loader,
                    train_source_iter_num ,train_target_iter_num)
    engine.evaluate(model_name,test_target_loader)
