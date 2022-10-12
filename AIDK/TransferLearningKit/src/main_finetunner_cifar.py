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
import optuna
from optuna.trial import TrialState
import argparse
from functools import partial

def objective(args,trial):
    ''' Optuna optimize objective

    Args:
        args: parsed args
        trial: a trial object

    Returns: a metric to measure the performance of model

    '''
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "8089"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    world_size = args.world_size
    rank = args.rank
    enable_transfer_learning = (args.transfer > 0)
    return main(rank, world_size, enable_transfer_learning, trial)

def main(rank, world_size, enable_transfer_learning, trial):
    ''' main function
    :param rank: rank of the process
    :param world_size: worker num
    :param enable_transfer_learning: enable transfer learning
    :param trial: Optuna Trial. If None, then training without automl.
    :return: validation metric. If has Earlystopping, using the best metric; Else, using the last metric.
    '''
    if world_size <= 1:
        world_size = 1
        rank = -1
    is_distributed = (world_size > 1)  # distributed flag
    #################### configuration ################
    torch.manual_seed(0)
    LOG_DIR = "../log"  # to save training log
    PROFILE_DIR = '.'  # to save profiling result
    MODEL_DIR = "../model"  # to save trained model
    DATASET_DIR = "../datasets"  # where dataset located
    TENSORBOARD_DIR = "../tensorboard_log"  # to save tensorboard log
    enable_ipex = True  # intel-extension-for-pytorch

    if is_distributed:
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300))
    log_filename = "%s/%s%s%s.txt" % (LOG_DIR, int(time.time()),
                                      "" if trial is None else "_trial%s" % trial.number,
                                      "" if not is_distributed else "_rank%s" % rank)
    tensorboard_filename_suffix = "" if not is_distributed else "_rank%s" % (rank)
    profile_trace_file_training = "%s/training_profile%s%s_" % (PROFILE_DIR,
                                                                "" if trial is None else "_trial%s" % trial.number,
                                                                "" if not is_distributed else "_rank%s" % rank)
    profile_trace_file_inference = '%s/test_profile%s%s_' % (PROFILE_DIR,
                                                             "" if trial is None else "_trial%s" % trial.number,
                                                             "" if not is_distributed else "_rank%s" % rank)
    model_save_path = "%s/pretrain_resnet50_cifar10%s%s.pth" % (MODEL_DIR,
                                                                "" if trial is None else "_trial%s" % trial.number,
                                                                "" if not is_distributed else "_rank%s" % rank)
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')
    ##################### parameters ##############
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.Resize(112),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
        torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),  # CIFAR100
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(112),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #CIFAR10
        torchvision.transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),  # CIFAR100
    ])

    train_dataset = torchvision.datasets.CIFAR100(root=DATASET_DIR, train=True,
                                                 download=False, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root=DATASET_DIR, train=False,
                                                download=False, transform=transform_test)
    kwargs = {
        ############### global ##############
        'is_distributed': is_distributed,
        'enable_transfer_learning': enable_transfer_learning,
        'enable_ipex': enable_ipex,  # intel-extension-for-pytorch
        'training_epochs': 3,
        'logging_interval_step': 10,
        'validate_metric_fn_map': {'acc': accuracy},
        'earlystop_metric': 'acc',
        'rank': rank,
        'model_saved_path': model_save_path,
        ############### dataset #############
        'data_train_dataset': train_dataset,
        'data_validate_dataset': test_dataset,
        'data_test_dataset': test_dataset,
        'data_num_workers': 1,  # 0 means that the data will be loaded in the main process
        'data_batch_size': 128,  # larger is better
        'data_drop_last': False,
        ############## model ############
        'model_num_classes': 100,
        'model_backbone_name': 'resnet50_timm',
        'model_loss': nn.CrossEntropyLoss(reduction='mean'),
        'finetune_pretrained_state_dict': torch.load('../pretrained/resnet50_miil_21k.pth')['state_dict'],
        'finetune_top_finetuned_layer': 'global_pool_flatten',
        'finetune_frozen': False,
        'finetune_pretrained_num_classes': 11221,
        ############## optimizer ############
        'optimizer_learning_rate': trial.suggest_float("lr", 0.001, 0.1, log=True) if trial is not None else 0.02,
        # classifier
        'optimizer_finetune_learning_rate': trial.suggest_float("lr_finetuned", 0.001, 0.1,
                                                                log=True) if trial is not None else 0.02,  # finetunning
        'optimizer_weight_decay': trial.suggest_float("weight_decay", 0.0001, 0.1,
                                                      log=True) if trial is not None else 0.0005,  # L2 penalty
        'optimizer_momentum': 0.9,
        ############### lr_scheduler ##########
        'scheduler_T_max': 200,  # max epoch
        ######## tensorboard_writer ########
        'tensorboard_dir': TENSORBOARD_DIR,
        'tensorboard_filename_suffix': tensorboard_filename_suffix,
        ######### early_stopping ###########
        'earlystop_tolerance_epoch': 200,
        'earlystop_delta': 0.001,
        'earlystop_is_max': True,
        'earlystop_limitation': 1.0,
        ######### profiler ##############
        'profile_skip_first': 1,
        'profile_wait': 1,
        'profile_warmup': 1,
        'profile_active': 2,
        'profile_repeat': 1,
        'profile_activities': 'cpu',
        'profile_trace_file_training': profile_trace_file_training,
        'profile_trace_file_inference': profile_trace_file_inference,
    }
    ###################### task ###############
    task = Task(**kwargs)
    if trial is not None:
        logging.info("Begin trial %s" % trial.number)
        print("[%s]: Begin trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
    metric = task.run()
    ########### destroy dist #########
    if is_distributed:
        dist.destroy_process_group()
    if trial is not None:
        logging.info("End trial %s" % trial.number)
        print("[%s]: End trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Must set world_size.'
    parser.add_argument('-s', "--world_size", default=1, help="The worker num. World_size <= 0 means no parallel.",type=int)
    parser.add_argument('-r', "--rank", default=0, help="The current rank. Begins from 0.", type=int)
    parser.add_argument('-t', "--transfer", default=1,help="Enable Transfer Learning. Transfer > 0 means true, else false", type=int)
    parser.add_argument('-R', "--trial_round", default=0,help="The hyper-param tunning round. trial_round <= 0 means no tunning.", type=int)
    args = parser.parse_args()

    trial_round = parser.parse_args().trial_round
    if trial_round > 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(objective,args), n_trials=trial_round, timeout=None)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        print("No Trial")
        objective(args,None)