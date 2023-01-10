# Intel® End-to-End AI Optimization Kit for WnD
## Original source disclose
Source repo: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep

Google's [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

---

# Quick Start
## Enviroment Setup
``` bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b tensorflow -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```

## Enter Docker
```
sshpass -p docker ssh ${host0} -p 12344
```

## Workflow Prepare
``` bash
# prepare model codes
cd /home/vmagent/app/e2eaiok/modelzoo/WnD/TensorFlow2
bash patch_wnd.patch

# Download Dataset
# download and unzip dataset from https://www.kaggle.com/c/outbrain-click-prediction/data to /home/vmagent/app/dataset/outbrain/orig

# source spark env
source /home/spark-env.sh

# Start services
# only if there is no spark service running, may check ${localhost}:8080 to confirm
/home/start_spark_service.sh
```

## Data Process
```
cd /home/vmagent/app/e2eaiok/modelzoo/WnD/TensorFlow2; sh scripts/spark_preproc.sh
```

## Training
```
# edit config
cat /home/vmagent/app/e2eaiok/conf/e2eaiok_defaults_wnd_example.conf
### GLOBAL SETTINGS ###
observation_budget: 1
save_path: /home/vmagent/app/e2eaiok/result/
ppn: 2
ccl_worker_num: 2
global_batch_size: 524288
num_epochs: 20
cores: 104
iface: lo
hosts:
- localhost
```

```
export OMP_NUM_THREADS=30
cd /home/vmagent/app/e2eaiok; python run_e2eaiok.py --data_path /home/vmagent/app/dataset/outbrain/ --model_name wnd --conf conf/e2eaiok_defaults_wnd_example.conf 
```

## Inference

Edit scripts/inference.sh
```
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate tensorflow

export OMP_NUM_THREADS=30

time horovodrun -np 8 -H ${node1}:2,${node2}:2,${node3}:2,${node4}:2 --network-interface ${interface} \
/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python -u main.py \
  --train_data_pattern '/home/vmagent/app/dataset/outbrain/train/part*' \
  --eval_data_pattern '/home/vmagent/app/dataset/outbrain/valid/part*' \
  --model_dir ./checkpoints \
  --dataset_meta_file data/outbrain/outbrain_meta.yaml \
  --deep_learning_rate 0.00048 \
  --linear_learning_rate 0.8 \
  --eval_batch_size 1048576 \
  --evaluate \
  --use_checkpoint \
  --benchmark \
  --benchmark_warmup_steps 50 \
  --benchmark_steps 100 \
  --metric MAP --deep_hidden_units 128 128 64
```
`bash scripts/inference.sh`
