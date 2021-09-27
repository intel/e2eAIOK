## Introduction

Smart Democratization Advisor (SDA) is a tool to facilitate [Sigopt](https://sigopt.com) recipes generations with human instructed intelligence.

## Quick start

* Quick start

```bash
python main.py --train_path ${trainset_path} --eval_path ${evalset_path} \
--dataset_meta_path ${meta_path} --model ${model} --dataset_format ${dataset_format}
```

SDA config file will be generated based on command line arguments and saved at `sda.yaml`.
Sigopt config file will be generated based on command line arguments and saved at `models/$model/sigopt.yaml`

* Config file for SDA

  ```yaml
  train:
    # train/test dataset format, support TFRecords and Binary
    dataset_format: TFRecords
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

  ```yaml
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

## Advanced

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

  To quick evaluate the SDA, the `run.sh` script provides a good startup demo. This demo launches WnD model with outbrain dataset

## Programming guide

SDA provides an easy way to extends different models. 

In order to extends SDA for other unsupported models, user should implement model launcher which inherits `BaseModelLauncher` and implement `parse_args`, `generate_sigopt_yaml` and `launch` functions.

* `parse_args`

  Parse model specific argument from command line

* `generate_sigopt_yaml`

  Generate sigopt yaml file for model

* `launch`

  Construct launch command for model optimization with sigopt. It is an easy way for user to extend exist training code with SDA without modifying the training code.