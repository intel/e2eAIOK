#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author : Hua XiaoZhuan          
# @Time   : 8/16/2022 1:10 PM
from training.task import Task
import os
import datetime
import torch.distributed as dist
import torch
import torch.nn as nn
import time
import logging
from training.metrics import accuracy
import torchvision
from torch.utils.data import random_split

def main(rank, world_size, enable_transfer_learning):
    ''' main function

    :param rank: rank of the process
    :param world_size: worker num
    :param enable_transfer_learning: enable transfer learning
    :return:
    '''
    if world_size <= 1:
        world_size = 1
        rank = -1
    is_distributed = (world_size > 1)  # distributed flag
    #################### configuration ################
    torch.manual_seed(0)
    if is_distributed:
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300))
        log_filename = "../log/%s_rank_%s.txt" % (int(time.time()), rank)
        tensorboard_filename_suffix = "_rank%s" % (rank)
        profile_trace_file_training = "./training_profile_rank%s_"%(rank)
        profile_trace_file_inference = './test_profile_rank%s_'%(rank)
        model_save_path = "../model/pretrain_resnet50_cifar10_rank%s.pth"%(rank)
    else:
        log_filename = "../log/%s.txt" % (int(time.time()))
        tensorboard_filename_suffix = ""
        profile_trace_file_training = "./training_profile"
        profile_trace_file_inference = './test_profile'
        model_save_path = "../model/pretrain_resnet50_cifar10.pth"

    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')
    ##################### parameters ##############
    data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_dataset = torchvision.datasets.CIFAR10(root="../datasets", train=True,
                                                     download=False, transform=data_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="../datasets", train=False,
                                                     download=False, transform=data_transform)
    train_len = len(train_dataset)
    valid_len = train_len // 5
    train_dataset, validation_dataset = random_split(train_dataset, [train_len-valid_len, valid_len])

    kwargs = {
        ############### global ##############
        'is_distributed' : is_distributed,
        'enable_transfer_learning' : enable_transfer_learning,
        'training_epochs' : 240,
        'logging_interval_step' : 10,
        'validate_metric_fn_map' : {'acc': accuracy},
        'earlystop_metric' : 'acc',
        'rank' : rank,
        'model_saved_path' : model_save_path,
        ############### dataset #############
        'data_train_dataset' : train_dataset,
        'data_validate_dataset': validation_dataset,
        'data_test_dataset':  test_dataset,
        'data_num_workers' : 1, # 0 means that the data will be loaded in the main process
        'data_batch_size' : 128, # larger is better
        'data_drop_last' : False,
        ############## model ############
        'model_num_classes' : 10,
        'model_backbone_name' : 'resnet50_v2',
        'model_loss' : nn.CrossEntropyLoss(reduction='mean'),
        'finetune_pretrained_path' : '../model/resnet50_cifar100.pth',
        'finetune_top_finetuned_layer' : 'avgpool',
        'finetune_frozen' : False,
        'finetune_pretrained_num_classes' : 100,
        ############## optimizer ############
        'optimizer_learning_rate' : 0.01,
        'optimizer_weight_decay' : 0.0, # L2 penalty
        ############### lr_scheduler ##########
        'scheduler_gamma' : 0.99,
        ######## tensorboard_writer ########
        'tensorboard_dir' : "../tensorboard_log",
        'tensorboard_filename_suffix' : tensorboard_filename_suffix,
        ######### early_stopping ###########
        'earlystop_tolerance_epoch' : 10,
        'earlystop_delta' : 0.001,
        'earlystop_is_max' : True,
        ######### profiler ##############
        'profile_skip_first' : 1,
        'profile_wait':1,
        'profile_warmup':1,
        'profile_active':2,
        'profile_repeat':1,
        'profile_activities':'cpu',
        'profile_trace_file_training':profile_trace_file_training,
        'profile_trace_file_inference':profile_trace_file_inference,
    }
    ###################### task ###############
    task = Task(**kwargs)
    task.run()
    ########### destroy dist #########
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.description = 'Must set world_size.'
    parser.add_argument('-s',"--world_size",default=1, help="The worker num. World_size <= 0 means no parallel.", type=int)
    parser.add_argument('-r',"--rank", help="The current rank. Begins from 0.", type=int)
    parser.add_argument('-t', "--transfer", help="Enable Transfer Learning. Transfer > 0 means true, else false", type=int)
    args = parser.parse_args()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "8089"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    world_size = args.world_size
    rank = args.rank
    enable_transfer_learning = (args.transfer > 0)
    main(rank,world_size,enable_transfer_learning)