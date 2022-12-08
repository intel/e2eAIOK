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
from cfg import cfg, show_cfg
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
    return main(args, trial)

def main(args, trial):
    ''' main function
    :param args: args parameters.
    :param trial: Optuna Trial. If None, then training without automl.
    :return: validation metric. If has Earlystopping, using the best metric; Else, using the last metric.
    '''
    #################### merge cfg ################
    world_size = args.world_size
    rank = args.rank
    if world_size <= 1:
        world_size = 1
        rank = -1
    is_distributed = (world_size > 1) # distributed flag
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)
    torch.manual_seed(cfg.experiment.seed)
    # cfg.profiler.activities = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    print("Model Name:%s\nData Name:%s\nTransfer Learning Strategy:%s\nEnable DDP:%s%s\nTraining epochs:%s"%(
            cfg.model.type, 
            cfg.dataset.type,
            cfg.experiment.strategy, 
            (world_size > 1),
            "" if (world_size <= 1) else "(rank = %s)"%rank ,
           cfg.solver.epochs)
    )
    #################### dir conguration ################
    prefix = "%s%s%s%s%s"%(cfg.model.type,
                     "_%s"%cfg.experiment.strategy if cfg.experiment.strategy else "",
                     "_%s"%cfg.dataset.type,
                     "_trial%s" % trial.number if trial is not None else "",
                     "_rank%s" % rank if is_distributed else "")
    prefix_time = "%s_%s"%(prefix,int(time.time()))
    cfg.experiment.tag = cfg.experiment.tag + "%s" % ("_dist%s" % world_size if is_distributed else "")
    root_dir = os.path.join(cfg.experiment.model_save, cfg.experiment.project,cfg.experiment.tag)
    LOG_DIR = os.path.join(root_dir,"log")                      # to save training log
    PROFILE_DIR = os.path.join(root_dir,"profile")              # to save profiling result
    model_save_path = os.path.join(root_dir, prefix)
    cfg.experiment.tensorboard_dir = os.path.join(cfg.experiment.tensorboard_dir,"%s_%s"%(cfg.experiment.tag,prefix))  # to save tensorboard log
    os.makedirs(LOG_DIR,exist_ok=True)
    os.makedirs(PROFILE_DIR,exist_ok=True) 
    os.makedirs(cfg.experiment.tensorboard_dir,exist_ok=True)
    os.makedirs(model_save_path,exist_ok=True)
 
    cfg.profiler.trace_file_training = os.path.join(PROFILE_DIR,"training_profile_%s"%prefix_time)
    cfg.profiler.trace_file_inference = os.path.join(PROFILE_DIR,"test_profile_%s"%prefix_time)
    
    #################### logging conguration ################
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    
    log_filename = os.path.join(LOG_DIR, "%s.txt"%prefix_time)
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s %(levelname)s [%(filename)s %(funcName)s %(lineno)d]: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filemode='w')
    ################ init dist ################
    if is_distributed:
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300))
    ##################### Optuna hyper params ################
    if trial is not None:
        cfg.solver.optimizer.lr = trial.suggest_float("lr", 0.001, 0.1, log=True)
        cfg.finetuner.learning_rate = trial.suggest_float("lr_finetuned", 0.001, 0.1, log=True) 
        cfg.solver.optimizer.weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.1,log=True)
    if int(cfg.distiller.save_logits) + int(cfg.distiller.use_saved_logits) + int(cfg.distiller.check_logits) >=2:
        raise RuntimeError("Can not save teacher logits, train students with logits or check logits together!")
    if cfg.distiller.save_logits:
        os.makedirs(cfg.distiller.logits_path, exist_ok=True)
    if cfg.distiller.use_saved_logits or cfg.distiller.check_logits:
        if not os.path.exists(cfg.distiller.logits_path):
            raise RuntimeError("Need teacher saved logits!")
    ##################### show conguration ################
    # cfg.freeze()
    show_cfg(cfg)
    ##################### functions ##############
    validate_metric_fn_map = {'acc': accuracy}
    model_loss = nn.CrossEntropyLoss(reduction='mean')
    ###################### create task ###############
    task = Task(cfg, model_save_path, model_loss, is_distributed)
    if trial is not None:
        logging.info("[%s]: Begin trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
        print("[%s]: Begin trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
    metric = task.run(validate_metric_fn_map, rank, eval=args.eval, resume=args.resume)
    ############### destroy dist ###############
    if is_distributed:
        dist.destroy_process_group()
    if trial is not None:
        logging.info("End trial %s" % trial.number)
        print("[%s]: End trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
    return metric

if __name__ == '__main__':
    # usage: python main.py --cfg ../config/demo/cifar100_kd_vit_res18.yaml --opts solver.epochs 1 dataset.path /xxx/yyy
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.description = 'Must set world_size.'
    parser.add_argument("--cfg", type=str, default="../config/finetuner/cifar100_res50PretrainI21k.yaml")
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('-s',"--world_size",default=1, help="The worker num. World_size <= 0 means no parallel.", type=int)
    parser.add_argument('-r', "--rank", default=0, help="The current rank. Begins from 0.", type=int)
    parser.add_argument('-R', "--trial_round", default=0,help="The hyper-param tunning round. trial_round <= 0 means no tunning.", type=int)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
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

    print(f"Totally take {(time.time()-start_time)} seconds")