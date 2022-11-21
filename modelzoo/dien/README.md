# Intel® End-to-End AI Optimization Kit for DIEN
## Original source disclose
Source repo: https://github.com/alibaba/ai-matrix

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
cd /home/vmagent/app/e2eaiok/modelzoo/dien/train
sh patch_dien.sh

# Download Dataset
cd /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/
./download_dataset /home/vmagent/app/dataset/

# source spark env
source /home/spark-env.sh

# Start services
# only if there is no spark service running, may check ${localhost}:8080 to confirm
/home/start_spark_service.sh
```

## Data Process
```
cd /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/;
python preprocessing.py --train --dataset_path /home/vmagent/app/dataset/amazon_reviews/
python preprocessing.py --test --dataset_path /home/vmagent/app/dataset/amazon_reviews/
```

## Training
```
# edit /home/vmagent/app/dataset/amazon_reviews/meta.yaml
uid_voc: /home/vmagent/app/dataset/amazon_reviews/uid_voc.pkl
mid_voc: /home/vmagent/app/dataset/amazon_reviews/mid_voc.pkl
cat_voc: /home/vmagent/app/dataset/amazon_reviews/cat_voc.pkl
```

```
cd /home/vmagent/app/e2eaiok/; python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien 
```

## Distributed Training
```
# edit below config with correct nic and hosts name
cat conf/e2eaiok_defaults_dien_example.conf
ppn: ${num_copy}
iface: ${nic}
hosts:
- ${node1}
- ${node2}
- ${node3}
- ${node4}

# run distributed training
cd /home/vmagent/app/e2eaiok/modelzoo/dien/feature_engineering/; sh split_for_distribute.sh
cd /home/vmagent/app/e2eaiok/; python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews_distributed --model_name dien  --conf conf/e2eaiok_defaults_dien_example.conf
```

## Result
```
# after training, you'll see info as below
<!-- 
We found the best model! Here is the model explaination

===============================================
***    Best Trained Model    ***
===============================================
  Model Type: dien
  Model Saved Path: ${path to your result}
  Sigopt Experiment id is None
  === Result Metrics ===
    AUC: 0.8205973396674585
    training_time: 410.986137151718
=============================================== 
-->
```

## Inference
```
cd /home/vmagent/app/e2eaiok/modelzoo/dien/
rm /home/vmagent/app/e2eaiok/modelzoo/dien/train/ai-matrix/dnn_best_model_trained/ -rf
cp -r ${path to your result}/dnn_best_model/ dnn_best_model_trained
sh infer.sh
<!--
----------------------------------------------------------------
Running inference with num_instance of 64
----------------------------------------------------------------
Inference System time in miliseconds is: 63859
Inference Throughput is
205670
Inference prepare avg is
13.0636
Inference eval avg is
34.5014
-->
```
