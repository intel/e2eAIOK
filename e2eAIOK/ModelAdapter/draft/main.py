from easydict import EasyDict as edict
import os, time, datetime
import yaml
import torch
import torch.nn as nn
import logging
import torchvision
import optuna
from optuna.trial import TrialState
import argparse
from functools import partial
import sys
import e2eAIOK
from e2eAIOK.ModelAdapter.training import ModelAdapterTask
from e2eAIOK.common.utils import update_dict
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
e2eaiok_dir = e2eAIOK.__path__[0]

def is_safe_path(basedir, path):
    return os.path.abspath(path).startswith(basedir)

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
    return main(args, trial)

def main(args, trial):
    ''' main function
    :param args: args parameters.
    :param trial: Optuna Trial. If None, then training without automl.
    :return: validation metric. If has Earlystopping, using the best metric; Else, using the last metric.
    '''
    #################### merge cfg ################
    with open(os.path.join(e2eaiok_dir, "common/default.conf")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(e2eaiok_dir, "ModelAdapter/default_ma.conf")) as f:
        cfg = update_dict(cfg, yaml.safe_load(f))
    if not is_safe_path("/home/vmagent/app/e2eaiok", args.cfg):
        print(f"{args.cfg} is not safe.")
        sys.exit()      
    if os.access(args.cfg, os.R_OK):
        with open(args.cfg) as f:
            cfg = edict(update_dict(cfg, yaml.safe_load(f)))
    with open(args.cfg) as f:
        cfg = edict(update_dict(cfg, yaml.safe_load(f)))
    torch.manual_seed(cfg.seed)

    #################### dir conguration ################
    is_distributed = ext_dist.my_size > 1
    prefix = "%s%s%s%s%s"%(cfg.model_type,
                     "_%s"%cfg.experiment.strategy if cfg.experiment.strategy else "",
                     "_%s"%cfg.data_set,
                     "_trial%s" % trial.number if trial is not None else "",
                     "_rank%s" % ext_dist.my_rank if is_distributed else "")
    prefix_time = "%s_%s"%(prefix,int(time.time()))
    cfg.experiment.tag = cfg.experiment.tag + "%s" % ("_dist%s" % ext_dist.my_size if is_distributed else "")
    root_dir = os.path.join(cfg.output_dir, cfg.experiment.project,cfg.experiment.tag)
    LOG_DIR = os.path.join(root_dir,"log")                      # to save training log
    PROFILE_DIR = os.path.join(root_dir,"profile")              # to save profiling result
    model_save_path = os.path.join(root_dir, prefix)
    cfg.tensorboard_dir = os.path.join(cfg.tensorboard_dir,"%s_%s"%(cfg.experiment.tag,prefix))  # to save tensorboard log
    cfg.profiler_config.trace_file = os.path.join(PROFILE_DIR,"profile_%s"%prefix_time)
    if not is_safe_path("/home/vmagent/app/data", LOG_DIR):
        print(f"{LOG_DIR} is not safe.")
        sys.exit()
    if not is_safe_path("/home/vmagent/app/data", PROFILE_DIR):
        print(f"{PROFILE_DIR} is not safe.")
        sys.exit()
    if not is_safe_path("/home/vmagent/app/data", cfg.tensorboard_dir):
        print(f"{cfg.tensorboard_dir} is not safe.")
        sys.exit()
    if not is_safe_path("/home/vmagent/app/data", model_save_path):
        print(f"{model_save_path} is not safe.")
        sys.exit()    
    os.makedirs(LOG_DIR,exist_ok=True)
    os.makedirs(PROFILE_DIR,exist_ok=True) 
    os.makedirs(cfg.tensorboard_dir,exist_ok=True)
    os.makedirs(model_save_path,exist_ok=True)

    if int(cfg.distiller.save_logits) + int(cfg.distiller.use_saved_logits) + int(cfg.distiller.check_logits) >=2:
        raise RuntimeError("Can not save teacher logits, train students with logits or check logits together!")
    if cfg.distiller.save_logits:
        os.makedirs(cfg.distiller.logits_path, exist_ok=True)
    if cfg.distiller.use_saved_logits or cfg.distiller.check_logits:
        if not os.path.exists(cfg.distiller.logits_path):
            raise RuntimeError("Need teacher saved logits!")
 
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
    ext_dist.init_distributed(backend=cfg.dist_backend)
    ##################### Optuna hyper params ################
    if trial is not None:
        cfg.learning_rate = trial.suggest_float("lr", 0.001, 0.1, log=True)
        cfg.finetuner.learning_rate = trial.suggest_float("lr_finetuned", 0.001, 0.1, log=True) 
        cfg.weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.1,log=True)
    ##################### show conguration ################
    if torch.__version__.startswith('1.12') and "enable_ipex" in cfg and cfg.enable_ipex:
        logging.warning("See abnormal behavior in dataloader when enable IPEX in PyTorch 1.12, set enable_ipex to False!")
        print("See abnormal behavior in dataloader when enable IPEX in PyTorch 1.12, set enable_ipex to False!")
        cfg.enable_ipex = False
    print("configurations:")
    print(cfg)
    ###################### create task ###############
    task = ModelAdapterTask(cfg, model_save_path, is_distributed)
    if trial is not None:
        logging.info("[%s]: Begin trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
        print("[%s]: Begin trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
    metric = task.run(eval=args.eval, resume=args.resume)
    ############### destroy dist ###############
    if is_distributed:
        dist.destroy_process_group()
    if trial is not None:
        logging.info("End trial %s" % trial.number)
        print("[%s]: End trial %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), trial.number))
    return metric

if __name__ == '__main__':
    # usage: python main.py --cfg ../config/demo/cifar100_kd_vit_res18.yaml --opts train_epochs 1 dataset.path /xxx/yyy
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="../config/baseline/cifar100_resnet18_LRdecay10.conf")
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('-R', "--trial_round", default=0,help="The hyper-param tunning round. trial_round <= 0 means no tunning.", type=int)
    # parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
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