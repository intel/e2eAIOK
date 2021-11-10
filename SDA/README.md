## Introduction

Smart Democratization Advisor (SDA) is a tool to facilitate [Sigopt](https://sigopt.com) recipes generations with human instructed intelligence.

## Quick start

* Quick start

#### Prepare dataset
download from ${path will be provided later}
* Internal COPY
172.16.8.30://mnt/nvme0/dataset

Expected data format
```
└── [  73]  criteo
    ├── [1.8K]  criteo_meta.yaml
    ├── [ 13G]  test_data.bin
    └── [625G]  train_data.bin


└── [  61]  outbrain
    ├── [1.3K]  outbrain_meta.yaml
    └── [  31]  tfrecords
        ├── [4.0K]  eval.parquet
        └── [4.0K]  train.parquet
    └── [  34]  meta
        └── [  26]  transformed_metadata
```


#### Enter Docker ENV
```bash
cd SDA
docker run -it --privileged --network host --device=/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v `pwd`:/home/vmagent/app/bluewhale/SDA -w /home/vmagent/app/ docker.io/xuechendi/oneapi-aikit:sda /bin/bash
```

#### Prepare sigopt TOKEN
https://app.sigopt.com/tokens/info

#### run test(WnD using outbrain dataset)
```bash
# after enter docker
cd bluewhale/SDA
source config_proxy
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force

SIGOPT_API_TOKEN=${TOKEN} python main.py --train_path "/home/vmagent/app/dataset/outbrain/tfrecords/train/part*" --eval_path "/home/vmagent/app/dataset/outbrain/tfrecords/eval/part*" \
--dataset_meta_path /home/vmagent/app/dataset/outbrain/outbrain_meta.yaml --model WnD \
--python_executable /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python --ppn ${parallelism} --ccl_worker_num ${parallelism} --metric MAP --metric_threshold 0.6553 --num_epochs 8 --global_batch_size 524288 \
--training_time_threshold 1800 \
--program /home/vmagent/app/bluewhale/examples/WnD/TensorFlow2/main.py \
--deep_dropout 0.1 \
--observation_budget 3
```

#### run test(WnD using criteo dataset)
```bash
# after enter docker
cd bluewhale/SDA
source config_proxy
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force

SIGOPT_API_TOKEN=${TOKEN} python main.py --train_path "/home/vmagent/app/dataset/criteo/train_data.bin" --eval_path "/home/vmagent/app/dataset/criteo/test_data.bin" \
--dataset_meta_path /home/vmagent/app/dataset/criteo/criteo_meta.yaml --model WnD \
--python_executable /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python --ppn ${parallelism} --ccl_worker_num ${parallelism} --metric AUC --metric_threshold 0.6553 --num_epochs 1 --global_batch_size 524288 \
--training_time_threshold 1800 \
--program /home/vmagent/app/bluewhale/examples/WnD/TensorFlow2/main.py \
--deep_warmup_epochs 1 \
--observation_budget 3
```

#### run test(DLRM using criteo dataset)
```bash
# after enter docker
cd bluewhale/SDA
source config_proxy
export LD_LIBRARY_PATH=/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/
source /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/.local/env/setvars.sh

 SIGOPT_API_TOKEN=${TOKEN} seed_num=$(date +%s) python main.py --train_path /home/vmagent/app/dataset/criteo/train_data.bin --eval_path /home/vmagent/app/dataset/criteo/test_data.bin \
--dataset_meta_path /home/vmagent/app/dataset/criteo/criteo_meta.yaml --model DLRM \
--python_executable /opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python --ppn 2 --metric AUC --metric_threshold 0.8025 \
--hosts localhost
```

## Advanced

* Using sda.yaml file to Config SDA

  ```yaml
  train:
    # train/test dataset format, support TFRecords and Binary
    # dataset metadata path
    dataset_meta_path: /outbrain2/tfrecords
    # evaluation dataset location
    eval_path: /mnt/sdd/outbrain2/tfrecords/eval/part*
    # define model used for optimization
    model: WnD
    # train dataset location
    train_path: /mnt/sdd/outbrain2/tfrecords/train/part*
  ```

* Config file for sigopt

  ```yaml for WnD
  experiment: WnD
  metrics:
  - name: training_time
    objective: minimize
    threshold: 1800
  - name: MAP
    objective: maximize
    threshold: 0.6553
  observation_budget: 40
  parameters:
  - name: deep_learning_rate
    bounds:
      max: 0.1
      min: 0.0001
    transformation: log
    type: double
  project: sda
  ```
  ```yaml for DLRM
  experiment: DLRM
  metrics:
  - name: AUC
    objective: maximize
  observation_budget: 40
  parameters:
  - name: learning_rate
    bounds:
      max: 50
      min: 5
    type: int
  - name: lamb_lr
    bounds:
      max: 50
      min: 5
    type: int
  - name: warmup_steps
    bounds:
      max: 4500
      min: 2000
    type: int
  - name: decay_start_steps
    bounds:
      max: 9000
      min: 4501
    type: int
  - name: num_decay_steps
    bounds:
      max: 15000
      min: 5000
    type: int
  - name: sparse_feature_size
    grid:
      128
      64
      16    
    type: int
  - name: mlp_top_size
    bounds:
      max: 7
      min: 0    
    type: int
  - name: mlp_bot_size
    bounds:
      max: 3
      min: 0
    type: int
  project: sda
  ```

* Command line arguments for cluster arch

  SDA support distributed training automatically, the arguments for cluster arch are:

  | parameter | comment |
  | --------- | ------- |
  | --hosts | Define hosts to launch training, separated by spaces |
  | --ppn | Define worker number per node for distributed training |
  | --cores | Define node CPU cores used for training |

* Common command line arguments for model

  | parameter | comment |
  | --------- | ------- |
  | --global_batch_size | Global batch size for train and evaluation |
  | --num_epochs | Number training epochs |
  | --metric | Model evaluation metric |
  | --observation_budget | Define total number of sigopt optimization loop |
  | --metric_threshold | Model evaluation metric threshold used for early stop |
  | --training_time_threshold | Define training time threshold for sigopt optmization metric |

* Quick demo

  To quick evaluate the SDA, the `run.sh` script provides a good startup demo. This demo launches WnD model with outbrain dataset. And you can also evaluate the demo of DLRM model with Criteo Terabyte using the `run_dlrm.sh` script.

## Programming guide

SDA provides an easy way to extends different models. 

In order to extends SDA for other unsupported models, user should implement model launcher which inherits `BaseModelLauncher` and implement `parse_args`, `generate_sigopt_yaml` and `launch` functions.

* `parse_args`

  Parse model specific argument from command line

* `generate_sigopt_yaml`

  Generate sigopt yaml file for model

* `launch`

  Construct launch command for model optimization with sigopt. It is an easy way for user to extend exist training code with SDA without modifying the training code.
