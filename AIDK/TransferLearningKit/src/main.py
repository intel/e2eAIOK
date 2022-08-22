#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import time
from dataset.image_list import ImageList
from torch.utils.data.distributed import DistributedSampler
from engine_core.backbone.factory import createBackbone
from engine_core.adapter.factory import createAdapter
from engine_core.transferrable_model import _make_transferrable,make_transferrable_with_domain_adaption
from engine_core.transferrable_model import TransferStrategy,extract_distiller_adapter_features,set_attribute
from training.train import Trainer, Evaluator
import torch.optim as optim
from training.utils import EarlyStopping,initWeights
from training.metrics import accuracy
import logging
from torchvision import transforms
from functools import partial
import torch.nn as nn
import argparse
from cfg import CFG as cfg
from cfg import show_cfg
from dataset import get_dataset
from dataset.imagenet import get_imagenet_val_loader
from engine_core.distiller import KD, DKD
from engine_core.finetunner import BasicFinetunner
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer as ZeRO
from torch.distributed.algorithms.join import Join
import datetime

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

def get_transfer_setting(cfg,num_classes,epoch_steps):
    # get adapter
    if cfg.ADAPTER.TYPE == "NONE":
        adapter_feature_size = None
        adapter_feature_layer_name = None
        adapter = None
    elif cfg.ADAPTER.TYPE == "CDAN":
        adapter_feature_size = cfg.ADAPTER.FEATURE_SIZE
        adapter_feature_layer_name = cfg.ADAPTER.FEATURE_LAYER_NAME
        adapter = createAdapter('CDAN', input_size=adapter_feature_size * num_classes, hidden_size=adapter_feature_size,
                                dropout=0.0, grl_coeff_alpha=5.0, grl_coeff_high=1.0, max_iter=epoch_steps,
                                backbone_output_size=num_classes, enable_random_layer=0, enable_entropy_weight=0)
    else:
        logging.error("[%s] is not supported"%cfg.ADAPTER.TYPE)
        raise NotImplementedError("[%s] is not supported"%cfg.ADAPTER.TYPE)

    # get source dataset
    if cfg.SOURCE_DATASET.TYPE == "NONE":
        adaption_source_domain_training_dataset = None
    elif cfg.SOURCE_DATASET.TYPE == "USPS_vs_MNIST":
        transform = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor(),])
        adaption_source_domain_training_dataset = ImageList(os.path.join(cfg.SOURCE_DATASET.PATH, "USPS"),
                                    open(os.path.join(cfg.SOURCE_DATASET.PATH, "USPS/usps_train.txt")).readlines(),
                                    transform,'L')
    else:
        logging.error("[%s] is not supported"%cfg.SOURCE_DATASET.TYPE)
        raise NotImplementedError("[%s] is not supported"%cfg.SOURCE_DATASET.TYPE)

    # get distiller
    distiller_feature_size = None
    if cfg.DISTILLER.TYPE == "NONE":
        distiller_feature_layer_name = None
        distiller = None
    else:
        teacher_model = createBackbone(cfg.DISTILLER.TEACHER.TYPE, pretrain = cfg.DISTILLER.TEACHER.PRETRAIN, num_classes = num_classes)
        if cfg.DISTILLER.TYPE == "KD":
            distiller_feature_layer_name = "x"
            new_teacher_model = extract_distiller_adapter_features(teacher_model,distiller_feature_layer_name,adapter_feature_layer_name)
            distiller = KD(new_teacher_model, cfg.DISTILLER.TEACHER.IS_FROZEN, cfg.KD.TEMPERATURE, cfg.KD.CE_LOSS_WEIGHT, cfg.KD.KD_LOSS_WEIGHT)
        elif cfg.DISTILLER.TYPE == "DKD":
            distiller_feature_layer_name = "x"
            new_teacher_model = extract_distiller_adapter_features(teacher_model,distiller_feature_layer_name,adapter_feature_layer_name)
            distiller = DKD(new_teacher_model, cfg.DISTILLER.TEACHER.IS_FROZEN, cfg)
        else:
            logging.error("[%s] is not supported"%cfg.DISTILLER.TYPE)
            raise NotImplementedError("[%s] is not supported"%cfg.DISTILLER.TYPE)

    # get finetuner
    if cfg.FINETUNE.TYPE == "NONE":
        finetuner = None
    elif cfg.FINETUNE.TYPE == "Basic":
        top_finetuned_layer = None
        is_frozen = False
        finetuner = BasicFinetunner(model, top_finetuned_layer, is_frozen)
    else:
        logging.error("[%s] is not supported"%cfg.FINETUNE.TYPE)
        raise NotImplementedError("[%s] is not supported"%cfg.FINETUNE.TYPE)
    
    # get strategy
    if cfg.EXPERIMENT.STRATEGY == "OnlyFinetuneStrategy":
        strategy = TransferStrategy.OnlyFinetuneStrategy
    elif cfg.EXPERIMENT.STRATEGY == "OnlyDistillationStrategy":
        strategy = TransferStrategy.OnlyDistillationStrategy
    elif cfg.EXPERIMENT.STRATEGY == "OnlyDomainAdaptionStrategy":
        strategy = TransferStrategy.OnlyDomainAdaptionStrategy
    elif cfg.EXPERIMENT.STRATEGY == "FinetuneAndDomainAdaptionStrategy":
        strategy = TransferStrategy.FinetuneAndDomainAdaptionStrategy
    elif cfg.EXPERIMENT.STRATEGY == "DistillationAndDomainAdaptionStrategy":
        strategy = TransferStrategy.DistillationAndDomainAdaptionStrategy
    else:
        raise NotImplementedError("[%s] is not supported"%CFG.EXPERIMENT.STRATEGY)

    backbone_loss_weight = cfg.EXPERIMENT.LOSS.BACKBONE
    distiller_loss_weight = cfg.EXPERIMENT.LOSS.DISTILLER
    adapter_loss_weight = cfg.EXPERIMENT.LOSS.ADAPTER

    setting = (adapter_feature_size, adapter_feature_layer_name, adapter, adaption_source_domain_training_dataset, \
                distiller_feature_size, distiller_feature_layer_name, distiller, finetuner, strategy,
                backbone_loss_weight, distiller_loss_weight, adapter_loss_weight)

    return setting


def main(args):
    #################### configuration ################
    world_size = args.world_size
    rank = args.rank
    if world_size <= 1:
        world_size = 1
        rank = -1
    is_distributed = (world_size > 1) # distributed flag

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    torch.manual_seed(0)

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    show_cfg(cfg)

    os.makedirs(os.path.join(cfg.EXPERIMENT.MODEL_SAVE),exist_ok=True)
    os.makedirs(os.path.join(cfg.EXPERIMENT.MODEL_SAVE, cfg.EXPERIMENT.PROJECT),exist_ok=True)
    os.makedirs(os.path.join(cfg.EXPERIMENT.MODEL_SAVE, cfg.EXPERIMENT.PROJECT,cfg.EXPERIMENT.TAG),exist_ok=True)
    model_dir = os.path.join(cfg.EXPERIMENT.MODEL_SAVE, cfg.EXPERIMENT.PROJECT,cfg.EXPERIMENT.TAG)
    os.makedirs(os.path.join(model_dir,"tensorboard"),exist_ok=True)
   
    if is_distributed:
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300))
        log_filename = os.path.join(model_dir,"%s_rank_%s.txt" % (int(time.time()),rank)) 
        tensorboard_filename_suffix = "_rank%s" % (rank)
    else:
        log_filename = os.path.join(model_dir,"%s.txt"%int(time.time()))
        tensorboard_filename_suffix = ""  

    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')

    ######################## create dataset and dataloader #####################
    train_loader, val_loader, test_loader, num_data, num_classes = get_dataset(cfg)
    epoch_steps = len(train_loader)
    logging.info("epoch_steps:%s"%epoch_steps)
    
    ############################# Load Model #############################
    loss = nn.CrossEntropyLoss()
    model = createBackbone(cfg.MODEL.TYPE, pretrain = cfg.MODEL.PRETRAIN, num_classes = num_classes)
    set_attribute("model", model, "loss", loss)
    set_attribute("model", model, "init_weight", partial(initWeights,model))
    logging.info('backbone:%s' % model)

    ########################### Make model transferrable###################
    setting = get_transfer_setting(cfg,num_classes,epoch_steps)
    adapter_feature_size, adapter_feature_layer_name, adapter, adaption_source_domain_training_dataset, \
    distiller_feature_size, distiller_feature_layer_name, distiller, finetuner, strategy, \
    backbone_loss_weight, distiller_loss_weight, adapter_loss_weight = setting

    if cfg.EXPERIMENT.STRATEGY == "OnlyDomainAdaptionStrategy":
        model = make_transferrable_with_domain_adaption(model,loss,initWeights,
                            adapter, adapter_feature_size, adapter_feature_layer_name,
                            train_loader,adaption_source_domain_training_dataset,
                            enable_target_training_label=False,
                            backbone_loss_weight=backbone_loss_weight, 
                            adapter_loss_weight=adapter_loss_weight)
        logging.info('adapter:%s' % adapter)
        logging.info('transferrable model:%s' % model)
    else: #to do, split into several APIs
        model = _make_transferrable(model, loss, initWeights,
                                    finetuner, distiller, adapter,
                                    distiller_feature_size, distiller_feature_layer_name,
                                    adapter_feature_size, adapter_feature_layer_name,
                                    train_loader, adaption_source_domain_training_dataset,
                                    strategy, enable_target_training_label=True, 
                                    backbone_loss_weight=backbone_loss_weight, 
                                    distiller_loss_weight=distiller_loss_weight, 
                                    adapter_loss_weight=adapter_loss_weight)
        logging.info('adapter:%s' % adapter)
        logging.info('distiller:%s' % distiller)
        logging.info('finetuner:%s' % finetuner)
        logging.info('transferrable model:%s' % model)
    
    model = model.to(device)
    if not args.eval:
        if is_distributed:
            logging.info("training with DistributedDataParallel")
            model = DDP(model)
            optimizer = ZeRO(filter(lambda p: p.requires_grad, model.parameters()),optim.SGD,
                                                lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
    ############################ create other components ############
    tensorboard_writer = SummaryWriter(os.path.join(model_dir,"tensorboard"), filename_suffix=tensorboard_filename_suffix)
    logging.info('tensorboard_writer :%s' % tensorboard_writer)
    
    if cfg.SOLVER.EARLY_STOP.FLAG and not args.eval:
        early_stopping = EarlyStopping(tolerance_epoch=3, delta=0.0001, is_max=True)
        logging.info('early_stopping :%s' % early_stopping)
    else:
        early_stopping = None
    
    init_weight = False
    if cfg.MODEL.PRETRAIN == "NONE":
        init_weight = True
    logging.info('init_weight :%s' % init_weight)

    #################################### train and evaluate ###################
    stats_map = dict() # for time stats
    validate_metric_fn_map = {'acc':accuracy}
    evaluator = Evaluator(validate_metric_fn_map, tensorboard_writer,device)
    logging.info("evaluator:%s"%evaluator)
    # only for evaluation
    if args.eval:
        seconds_stats(stats_map,"Test",True)
        metric_values = evaluator.evaluate(model, test_loader)
        seconds_stats(stats_map,"Test",False)
        print(metric_values)
    # for train
    else:
        trainer = Trainer(model,optimizer,early_stopping,validate_metric_fn_map,'acc', cfg.SOLVER.EPOCHS,
                    tensorboard_writer,cfg.EXPERIMENT.LOG_INTERVAL_STEP,finetuner,cfg.MODEL.PRETRAIN,\
                    cfg.DISTILLER.TEACHER.PRETRAIN, device,init_weight,rank)
        logging.info("trainer:%s"%trainer)
        ############ train and evaluate ###############
        seconds_stats(stats_map,"Training",True)
        if is_distributed:
            with Join([model, optimizer]):
                trainer.train(train_loader, epoch_steps, val_loader,cfg,model_dir, args.resume)
        else:
            trainer.train(train_loader, epoch_steps, val_loader,cfg,model_dir, args.resume)
        if (not is_distributed) or (is_distributed and rank == 0): # print rank 0
            seconds_stats(stats_map,"Training",False)
        ############ test ###############
        if (not is_distributed) or (is_distributed and rank == 0): # only test once
            trained_model = createBackbone(cfg.MODEL.TYPE, num_classes = num_classes)
            trained_model.load_state_dict(torch.load(os.path.join(model_dir, "backbone_best.pth"),map_location="cpu"))
            set_attribute("trained_model",trained_model,"loss",loss)
            seconds_stats(stats_map,"Test",True)
            evaluator.evaluate(trained_model,test_loader)
            seconds_stats(stats_map,"Test",False)
        ########### destroy dist #########
        if is_distributed:
            dist.destroy_process_group()

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.description = 'Must set world_size.'
    parser.add_argument("--cfg", type=str, default="../config/adapter/usps_vs_minist_CDAN.yaml")
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('-s',"--world_size",default=1, help="The worker num. World_size <= 1 means no parallel.", type=int)
    parser.add_argument('-r',"--rank", default=0, help="The current rank. Begins from 0.", type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "8087"

    main(args)
    print(f"Totally take {(time.time()-start_time)} seconds")

    