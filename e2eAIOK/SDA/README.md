# Smart Democratization advisor (SDA)

# INTRODUCTION
A user-guided tool to facilitate automation of built-in model democratization via parameterized models, it generates yaml files based on user choice, provided build-in intelligence through parameterized models and leverage SigOpt for HPO. SDA converts the manual model tuning and optimization to assisted autoML and autoHPO. SDA provides a list of build-in optimized models ranging from RecSys, CV, NLP, ASR and RL. 

# Getting Start
## install with pip (require preinstall spark)
```
pip install e2eAIOK
```

## examples

example 1: builtin model with default best hyper-parameter
``` python
from e2eAIOK import SDA

settings = dict()
settings["data_path"] = "/home/vmagent/app/dataset/criteo/"
settings["ppn"] = 2
settings["ccl_worker_num"] = 4
settings["enable_sigopt"] = True

sda = SDA(model="dlrm", settings=settings) # default settings
sda.launch()

hydro_model = sda.snapshot()
hydro_model.explain()
```
``` console
2022-04-26 05:27:09,901 - HYDRO.AI.SDA - INFO - ### Ready to submit current task  ###
2022-04-26 05:27:09,902 - HYDRO.AI.SDA - INFO - Model Advisor created
2022-04-26 05:27:09,903 - HYDRO.AI.SDA - INFO - model parameter initialized
2022-04-26 05:27:09,903 - HYDRO.AI.SDA - INFO - start to launch training
...
[0] Finished training it 256/256 of epoch 0, 411.28 ms/it, loss 0.129076, accuracy 96.669 %
[1] Finished training it 256/256 of epoch 0, 411.10 ms/it, loss 0.128426, accuracy 96.695 %
[0] :::MLLOG {"namespace": "", "time_ms": 1650952435427, "event_type": "INTERVAL_START", "key": "eval_start", "value": null, "metadata": {"file": "/home/vmagent/app/e2eaiok/modelzoo/dlrm/dlrm/dlrm_s_pytorch.py", "lineno": 1366, "epoch_num": 2.0}}
2022-04-26 05:27:14,748 - sigopt - INFO - Training completed based in sigopt suggestion, took 4.843278169631958 secs
2022-04-26 05:27:14,749 - HYDRO.AI.SDA - INFO - training script completed

===============================================
***    Best Trained Model    ***
===============================================
  Model Type: dlrm
  Model Saved Path: /home/vmagent/app/e2eaiok/result/dlrm/20220426_052421/
  Sigopt Experiment id is None
  === Result Metrics ===
    accuracy: 0.8025
===============================================
```

example 2: create your own model
``` python
from e2eAIOK import SDA

# global settings
settings = dict()
settings["data_path"] = "/home/vmagent/app/dataset/xxx/"
settings["enable_sigopt"] = True

# model settings
model_info = dict()
# config for model
model_info["score_metrics"] = [("accuracy", "maximize"), ("training_time", "minimize")]
model_info["execute_cmd_base"] = "/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python /home/vmagent/app/e2eaiok/modelzoo/dlrm/dlrm/launch.py"
model_info["result_file_name"] = "best_auc.txt"

# config for sigopt
model_info["experiment_name"] = "dlrm"
model_info["sigopt_config"] = [
    {'name':'learning_rate','bounds':{'min':5,'max':50},'type':'int'},
    {'name':'lamb_lr','bounds':{'min':5,'max':50},'type':'int'},
    {'name':'warmup_steps','bounds':{'min':2000,'max':4500},'type':'int'},
    {'name':'decay_start_steps','bounds':{'min':4501,'max':9000},'type':'int'},
    {'name':'num_decay_steps','bounds':{'min':5000,'max':15000},'type':'int'},
    {'name':'sparse_feature_size','grid': [128,64,16],'type':'int'},
    {'name':'mlp_top_size','bounds':{'min':0,'max':7},'type':'int'},
    {'name':'mlp_bot_size','bounds':{'min':0,'max':3},'type':'int'}]
model_info["observation_budget"] = 1

# register model to SDA
sda = SDA(settings=settings) # default settings
sda.register(model_info)
sda.launch()

hydro_model = sda.snapshot()
hydro_model.explain()
```

## use cases
* [DLRM](http://vsr140:8891/notebooks/builtin/dlrm/DLRM_DEMO.ipynb) - [Readme](modelzoo/dlrm/README.md) - recsys, facebook, pytorch_mlperf
* [DIEN](http://vsr140:8892/notebooks/builtin/dien/DIEN_DEMO.ipynb) - [Readme](modelzoo/dien/README.md) - recsys, alibaba, tensorflow
* [WnD](http://vsr140:8892/notebooks/builtin/wnd/WND_DEMO.ipynb) - [Readme](modelzoo/WnD/README.md) - recsys, google, tensorflow
* [RNNT](http://vsr140:8890/notebooks/builtin/rnnt/RNNT_DEMO.ipynb) - [Readme](modelzoo/rnnt/README.md) - speech recognition, pytorch
* [RESNET](http://vsr140:8892/notebooks/builtin/resnet/RESNET_DEMO.ipynb) - [Readme](modelzoo/resnet/README.md) - computer vision, tensorflow
* [BERT](http://vsr140:8892/notebooks/builtin/bert/BERT_DEMO.ipynb) - [Readme](modelzoo/bert/README.md) - Natual Language Processing, tensorflow
* [MiniGO](http://sr141:8888/notebooks/demo/MiniGo_DEMO.ipynb) - [Readme](modelzoo/minigo/README.md) - minimalist engine modeled after AlphaGo Zero, tensorflow

## LICENSE
* Apache 2.0

## Dependency
* python 3.*
