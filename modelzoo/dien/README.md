# Intel® End-to-End AI Optimization Kit for DIEN


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intel/e2eAIOK/blob/main/demo/builtin/dien/DIEN_DEMO.ipynb)&emsp;&emsp;  <img width="20" height="20" src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"> [View source on GitHub](https://github.com/intel/e2eAIOK/blob/main/demo/builtin/dien/DIEN_DEMO.ipynb)
---

## Original source disclose
Source repo: https://github.com/alibaba/ai-matrix

---

# Quick Start
## option 1: Use notebook (recommended)
* setup notebook
``` bash
docker run --shm-size=100g -it --privileged --network host -v `pwd`:/home/vmagent/app -w /home/vmagent/app ubuntu /bin/bash
apt-get update -y &&  DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python-is-python3
pip install jupyter
jupyter notebook --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --notebook-dir /home/vmagent/app/
```

* open jupyter notebook
access "http://${hostname}:8888/notebooks/demo/builtin/dien/DIEN_DEMO.ipynb"

* scroll down to 'Getting Started' and follow guide to run code blocks


## option 2: use docker
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
```

## Download data
```
# Download Dataset
cd /home/vmagent/app/e2eaiok/modelzoo/dien/
! wget wget https://zenodo.org/record/3463683/files/data.tar.gz
! wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books.json.gz
! wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
! tar -jxvf data.tar.gz
! gunzip reviews_Books.json.gz
! gunzip meta_Books.json.gz

! mkdir -p data/raw_data
! mkdir -p data/train
! mkdir -p data/valid

! mv *json data/raw_data/; cp data/local_test_splitByUser data/raw_data/
! mv data/local_test_splitByUser data/valid/
```

## Data Process
```
cd /home/vmagent/app/e2eaiok/modelzoo/dien;
python3 feature_engineering/preprocessing.py --train --dataset_path `pwd`/data
python3 feature_engineering/preprocessing.py --test --dataset_path `pwd`/data
```

## Training
```
data_path=/home/vmagent/app/e2eaiok/modelzoo/dien; cd /home/vmagent/app/e2eaiok/; python -u run_e2eaiok.py --data_path ${data_path}/data --model_name dien 
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
conda activate tensorflow

best_train_model_path=`find /home/vmagent/app/e2eaiok/result/ -name dnn_best_model | sort -u | tail -1`
data_path=`pwd`"/data/"

cd /home/vmagent/app/e2eaiok/modelzoo/dien/; rm -rf ./dnn_best_model_trained; cp -r ${best_train_model_path} ./dnn_best_model_trained
cd /home/vmagent/app/e2eaiok/modelzoo/dien/; python train/ai-matrix/script/train.py --mode=test --advanced --slice_id=0 --batch_size=128 --num-inter-threads=1 --num-intra-threads=1 --train_path ${data_path}/train/local_train_splitByUser --test_path ${data_path}/valid/local_test_splitByUser --meta_path ${data_path}/meta.yaml

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