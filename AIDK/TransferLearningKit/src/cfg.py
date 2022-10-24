from yacs.config import CfgNode as CN

def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.experiment = cfg.experiment
    dump_cfg.optimize = cfg.optimize
    dump_cfg.profiler = cfg.profiler
    dump_cfg.dataset = cfg.dataset
    dump_cfg.source_dataset = cfg.source_dataset
    dump_cfg.finetuner = cfg.finetuner
    dump_cfg.distiller = cfg.distiller
    dump_cfg.adapter = cfg.adapter
    dump_cfg.solver = cfg.solver
    print(dump_cfg.dump())


cfg = CN()
############################## TLK Global Setting ##########################
# Experiment
cfg.experiment = CN()
cfg.experiment.project = "TLK"
cfg.experiment.tag = "default"
cfg.experiment.strategy = ""
cfg.experiment.seed = 0
cfg.experiment.model_save = "/home/vmagent/app/data/model/"
cfg.experiment.model_save_interval = 40
cfg.experiment.log_interval_step = 10
cfg.experiment.tensorboard_dir = ""
cfg.experiment.tensorboard_filename_suffix = ""
cfg.experiment.loss = CN()
cfg.experiment.loss.backbone = 1.0
cfg.experiment.loss.distiller = 0.0
cfg.experiment.loss.adapter = 0.0

# optimize
cfg.optimize = CN()
cfg.optimize.enable_ipex = False

#profiler
cfg.profiler = CN()
cfg.profiler.skip_first = 1
cfg.profiler.wait = 1
cfg.profiler.warmup = 1
cfg.profiler.active = 2
cfg.profiler.repeat = 1
cfg.profiler.activities = 'cpu'
cfg.profiler.trace_file_training = ""
cfg.profiler.trace_file_inference = ""

# Dataset
cfg.dataset = CN()
cfg.dataset.type = ""
cfg.dataset.path = ""
cfg.dataset.num_workers = 2
cfg.dataset.data_drop_last = False
cfg.dataset.train_transform = "default"
cfg.dataset.test_transform = "default"
cfg.dataset.val = CN()
cfg.dataset.val.batch_size = 128
cfg.dataset.test = CN()
cfg.dataset.test.batch_size = 128

# Dataset - source
cfg.source_dataset = CN()
cfg.source_dataset.type = ""
cfg.source_dataset.path = ""
cfg.source_dataset.num_workers = 2
cfg.source_dataset.val = CN()
cfg.source_dataset.val.batch_size = 128
cfg.source_dataset.test = CN()
cfg.source_dataset.test.batch_size = 128

# Model
cfg.model = CN()
cfg.model.type = "resnet18_v2"
cfg.model.pretrain = ""   # "", True - default pretrain model, path
 
# Finetune
cfg.finetuner = CN()
cfg.finetuner.type = ""
cfg.finetuner.pretrain = ""
cfg.finetuner.pretrained_num_classes = 10
cfg.finetuner.finetuned_lr= 0.01
cfg.finetuner.frozen = False

# Distiller
cfg.distiller = CN()
cfg.distiller.type = ""  # Vanilla as default
cfg.distiller.feature_size = ""
cfg.distiller.feature_layer_name = "x"
cfg.distiller.teacher = CN()
cfg.distiller.teacher.type = "resnet50_v2"
cfg.distiller.teacher.pretrain = ""
cfg.distiller.teacher.is_frozen = True
cfg.distiller.save_logits = False
cfg.distiller.use_saved_logits = False
cfg.distiller.check_logits = False
cfg.distiller.logits_path = ""
cfg.distiller.logits_topk = 0
cfg.distiller.save_logits_start_epoch = 1

# Transfer
cfg.adapter = CN()
cfg.adapter.type = ""
cfg.adapter.feature_size = 500
cfg.adapter.feature_layer_name = "x"

# Solver
cfg.solver = CN()
cfg.solver.batch_size = 64
cfg.solver.start_epoch = 1
cfg.solver.epochs = 240
cfg.solver.warmup = 0

cfg.solver.optimizer = CN()
cfg.solver.optimizer.type = "SGD"
cfg.solver.optimizer.lr = 0.05
cfg.solver.optimizer.weight_decay = 0.0001
cfg.solver.optimizer.momentum = 0.9

cfg.solver.scheduler = CN()
cfg.solver.scheduler.type = ""
cfg.solver.scheduler.lr_decay_stages = [150, 180, 210]
cfg.solver.scheduler.lr_decay_rate = 0.1
cfg.solver.scheduler.T_max = 200
cfg.solver.scheduler.patience = 10

cfg.solver.early_stop = CN()
cfg.solver.early_stop.metric = "acc"
cfg.solver.early_stop.flag = True
cfg.solver.early_stop.tolerance_epoch = 3
cfg.solver.early_stop.delta = 0.0001
cfg.solver.early_stop.is_max = True
cfg.solver.early_stop.limitation = 1.0

######################## For Distiller #####################
# kd cfg
cfg.kd = CN()
cfg.kd.temperature = 4

# dkd(Decoupled Knowledge Distillation) cfg
cfg.dkd = CN()
cfg.dkd.alpha = 1.0
cfg.dkd.beta = 8.0
cfg.dkd.temperature = 4.0
cfg.dkd.warmup = 20