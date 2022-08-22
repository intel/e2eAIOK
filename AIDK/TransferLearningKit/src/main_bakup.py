#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import time
from dataset.image_list import ImageList
from torch.utils.data.distributed import DistributedSampler
from engine_core.backbone.factory import createBackbone
from engine_core.adapter.factory import createAdapter
from engine_core.transferrable_model import make_transferrable_with_domain_adaption,set_attribute
from training.train import Trainer,Evaluator
import torch.optim as optim
from training.utils import EarlyStopping,initWeights
from training.metrics import accuracy
import logging
from torchvision import transforms
from functools import partial
import torch.nn as nn
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer as ZeRO
from torch.distributed.algorithms.join import Join
import datetime

def create_dataset_loader(**kwargs):
    ''' create dataset loader

    :param kwargs: keyword args
    :return: (train_loader,validate_loader,test_loader)
    '''
    num_workers = kwargs['num_workers']
    batch_size = kwargs['batch_size']
    is_distributed = kwargs['is_distributed']
    transform = kwargs['transform']

    test_dataset = ImageList("../datasets/USPS_vs_MNIST/MNIST",
                             open("../datasets/USPS_vs_MNIST/MNIST/mnist_test.txt").readlines(),
                             transform, 'L')
    train_dataset = ImageList("../datasets/USPS_vs_MNIST/MNIST",
                              open("../datasets/USPS_vs_MNIST/MNIST/mnist_train.txt").readlines(),
                              transform, 'L')
    test_len = len(test_dataset)
    test_dataset, validation_dataset = random_split(test_dataset, [test_len // 2, test_len - test_len // 2])

    logging.info("train_dataset:" + str(train_dataset))
    logging.info("validation_dataset:" + str(validation_dataset))
    logging.info("test_dataset:" + str(test_dataset))

    if is_distributed:
        train_loader = torch.utils.data.DataLoader(train_dataset,  # only split train dataset
                                                   batch_size=batch_size, shuffle=False,
                                                   # shuffle is conflict with sampler
                                                   num_workers=num_workers, drop_last=True,
                                                   sampler=DistributedSampler(train_dataset))
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,  # only split train dataset
                                                   batch_size=batch_size, shuffle=True,
                                                   # shuffle is conflict with sampler
                                                   num_workers=num_workers, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(validation_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=True)
    return (train_loader,validate_loader,test_loader)

def create_model_optimizer(**kwargs):
    ''' create model and optimizer

    :param kwargs: keyword args
    :return: (model, optimizer)
    '''
    num_classes = kwargs['num_classes']
    train_loader = kwargs['train_loader']
    epoch_steps = kwargs['epoch_steps']
    transform = kwargs['transform']
    learning_rate = kwargs['learning_rate']
    weight_decay = kwargs['weight_decay']
    momentum = kwargs['momentum']
    is_distributed = kwargs['is_distributed']
    loss = kwargs['loss']
    enable_transfer_learning = kwargs['enable_transfer_learning']

    model = createBackbone('LeNet', num_classes=num_classes)
    set_attribute("model", model, "loss", loss)
    set_attribute("model", model, "init_weight", partial(initWeights, model))
    logging.info('backbone:%s' % model)

    if enable_transfer_learning:

        adapter_feature_size = 500
        adapter_feature_layer_name = 'fc_layers_2'
        adapter = createAdapter('CDAN', input_size=adapter_feature_size * num_classes, hidden_size=adapter_feature_size,
                                dropout=0.0, grl_coeff_alpha=5.0, grl_coeff_high=1.0, max_iter=epoch_steps,
                                backbone_output_size=num_classes, enable_random_layer=0, enable_entropy_weight=0)
        # adapter = None # createAdapter('DANN',input_size=model.adapter_size,hidden_size=500,dropout=0.0,
        #                         grl_coeff_alpha=5.0,grl_coeff_high=1.0,
        #                         max_iter=epoch_steps)
        source_domain_dataset = ImageList("../datasets/USPS_vs_MNIST/USPS",
                                          open("../datasets/USPS_vs_MNIST/USPS/usps_train.txt").readlines(),
                                          transform, 'L')

        model = make_transferrable_with_domain_adaption(model, loss, initWeights,
                                                        adapter, adapter_feature_size, adapter_feature_layer_name,
                                                        train_loader,source_domain_dataset, enable_target_training_label=False)
        logging.info('adapter:%s' % adapter)
        logging.info('transferrable model:%s' % model)

    if is_distributed:
        logging.info("training with DistributedDataParallel")
        model = DDP(model)
        optimizer = ZeRO(filter(lambda p: p.requires_grad, model.parameters()),optim.SGD,
                                         lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    return (model,optimizer)

def seconds_stats(record_map,process_name,is_begin,):
    ''' stats duration (seconds)

    :param record_map: already records
    :param process_name: process name
    :param is_begin: is begin?
    :return:
    '''
    if is_begin:
        record_map['begin'] = datetime.datetime.now()
        _str = "Begine %s" % process_name
    else:
        total_seconds = (datetime.datetime.now() - record_map['begin']).total_seconds()
        _str = "%s total seconds:%s" % (process_name,total_seconds)
        del record_map['begin']

    print(_str)
    logging.info(_str)


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
    is_distributed = (world_size > 1) # distributed flag
    #################### configuration ################
    tensorboard_dir = "../tensorboard_log"
    model_saved_path = "../model/model.pth"
    torch.manual_seed(0)
    if is_distributed:
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300))
        log_filename = "../log/%s_rank_%s.txt" % (int(time.time()),rank)
        tensorboard_filename_suffix = "_rank%s" % (rank)
    else:
        log_filename ="../log/%s.txt" % (int(time.time()))
        tensorboard_filename_suffix = ""

    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')
    ##################### parameters ##############
    kwargs = {
        "batch_size" : 128,
        "num_workers" : 1,  # means that the data will be loaded in the main process
        "learning_rate" : 0.002,
        "weight_decay" : 0.0005,
        "momentum" : 0.9,
        "num_classes" : 10,
        "transform" : transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            # transforms.Normalize(
            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225])
        ]),
        "is_distributed" : is_distributed,
        "loss":nn.CrossEntropyLoss(),
        "enable_transfer_learning" : enable_transfer_learning
    }
    ########################## dataset loader ##################
    (train_loader, validate_loader, test_loader) = create_dataset_loader(**kwargs)
    kwargs['epoch_steps'] = len(train_loader)
    logging.info("epoch_steps:%s" % kwargs['epoch_steps'])

    ######################### model and optimizer ##############
    kwargs['train_loader'] = train_loader
    (model,optimizer) = create_model_optimizer(**kwargs)
    ######################### trainer and evaluator ##############
    tensorboard_writer = SummaryWriter(tensorboard_dir, filename_suffix=tensorboard_filename_suffix)
    early_stopping = EarlyStopping(tolerance_epoch=3, delta=0.0001, is_max=True)
    logging.info('tensorboard_writer :%s' % tensorboard_writer)
    logging.info('early_stopping :%s' % early_stopping)

    training_epochs = 2
    logging_interval_step = 10

    validate_metric_fn_map = {'acc': accuracy}
    trainer = Trainer(model, optimizer, early_stopping, validate_metric_fn_map, 'acc', training_epochs,
                      tensorboard_writer, logging_interval_step, rank=rank)
    logging.info("trainer:%s" % trainer)

    evaluator = Evaluator(validate_metric_fn_map, tensorboard_writer)
    logging.info("evaluator:%s" % evaluator)
    #################################### train and evaluate ###################
    stats_map = dict() # for time stats
    seconds_stats(stats_map,"Training",True)
    if is_distributed:
        with Join([model, optimizer]):
            trainer.train(train_loader, kwargs['epoch_steps'], validate_loader, model_saved_path)
    else:
        trainer.train(train_loader, kwargs['epoch_steps'], validate_loader, model_saved_path)
    if (not is_distributed) or (is_distributed and rank == 0): # print rank 0
        seconds_stats(stats_map,"Training",False)
    ################################### test ###################################
    if (not is_distributed) or (is_distributed and rank == 0): # only test once
        trained_model = createBackbone('LeNet', num_classes=kwargs['num_classes'])
        trained_model.load_state_dict(torch.load(model_saved_path))
        set_attribute("trained_model",trained_model,"loss",kwargs['loss'])

        seconds_stats(stats_map,"Test",True)
        evaluator.evaluate(trained_model,test_loader)
        seconds_stats(stats_map,"Test",False)
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
        os.environ["MASTER_PORT"] = "8087"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    world_size = args.world_size
    rank = args.rank
    enable_transfer_learning = (args.transfer > 0)
    main(rank,world_size,enable_transfer_learning)
    # if world_size > 1:
    #     print("Parallel Training")
    #     mp.spawn(main,
    #          args=(world_size,),
    #          nprocs=world_size,
    #          join=True)
    # else:
    #     print("UnParallel Training")
    #     main(-1,0)

