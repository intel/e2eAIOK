#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from engine_core.transfer_learning_engine import TLEngine
from engine_core.task_manager import TaskManager
from engine_core.model_manager import ModelManager
from torch.utils.data import random_split
import logging

def createDatasets(task_manager):
    ''' create all datasets

    :param task_manager: task manager
    :return: (train_source_dataset,train_target_dataset, validate_target_dataset, test_target_dataset)
    '''
    datasets = task_manager.createDatasets()
    train_source_dataset = datasets['train_source']
    train_target_dataset = datasets['train_target']

    if 'test_target' in datasets:
        all_test_dataset = datasets['test_target']
        all_test_size = len(all_test_dataset)
        validate_size = all_test_size // 2
        test_size = all_test_size - validate_size
        logging.info("validate size [%s], test size [%s]" % (validate_size, test_size))
        (validate_target_dataset, test_target_dataset) = random_split(all_test_dataset, [validate_size, test_size])
    else: # no test dataset
        all_size = len(train_target_dataset)
        train_size = int(all_size * 0.7)
        validate_size = int(all_size * 0.2)
        test_size = all_size - validate_size - train_size
        (train_target_dataset, validate_target_dataset, test_target_dataset) = random_split(
            train_target_dataset, [train_size, validate_size, test_size])
        logging.info("validate size [%s], test size [%s]" % (validate_size, test_size))
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
