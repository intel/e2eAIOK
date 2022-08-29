# Intel Optimized Solutions for RecSys Challenge 2021 

# RecSys Challenge 2021 
The challenge focuses on a real-world task of tweet engagement prediction in a dynamic environment. Please check the [website](https://recsys.acm.org/recsys21/challenge/) on the details to get the dataset. 

# Quick Start

## Prepare
```
export path_to_e2eaiok_dataset=`pwd`/e2eaiok_dataset # put the downloaded dataset here
export path_to_e2eaiok=`pwd`/e2eAIOK
git clone https://github.com/intel/e2eAIOK.git
git submodule update --init --recursive
```

## Environment setup
You can choose to use AIOK docker or prepare the environment by yourself.

### User AIOK Docker
```
cd ${path_to_e2eaiok}/Dockerfile-ubuntu18.04/
docker build -t e2eaiok-pytorch-spark . -f DockerfilePytorch-spark
cd ${path_to_e2eaiok}
docker run --shm-size=10g -it --privileged --network host -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-pytorch-spark /bin/bash
source /etc/profile.d/spark-env.sh
pip install pyrecdp
Install hadoop and update the setting
```

### Prepare environment by yourself
Install the following libs:
 * Spark 3.0
 * Hadoop 3.2
 * lightgbm 3.2.1
 * XGBoost 1.3.3
 * transformers 4.13.0
 * pyrecdp
   

## E2E Train
1. Data Processing
    * Put original dataset(train, valid, test, valid_split_index) at HDFS: /recsys2021/oridata
    * Create two folds in HDFS to save data for stage1 and stage2: /recsys2021/datapre_stage1/, /recsys2021/datapre_stage2/
    * Preprocess train data:
        ``` bash
        cd data_preprocess
        python datapre.py train
        ```
    * Preprocess valid data:
        ``` bash
        python datapre.py valid_stage1
        python datapre.py valid_stage2
        ```
2. Train to get the model(take lgbm as example, xgboost is the same)
    ```bash
    cd model/lgbm
    python train_stage1.py  # train model for stage1 and save prediction
    python train_merge12.py # merge prediction from stage1 to data
    python train_stage2.py  # train model for stage2
    ```

## E2E Inference
1. Preprocess test dataset
    ``` bash
    cd data_preprocess
    python datapre.py inference_decoder
    python datapre.py inference_join
    ```
2. Predict (take lgbm as example, xgboost is the same)
    * On single node
    ```bash
    cd model/lgbm
    python inference.py 
    ```
    * On multi nodes(if needed)
    ```bash
    cd model/lgbm/inference_distributed
    python split_data.py # split test data into multi parts
    # Copy data to other nodes 
    python inference.py  # Run on multi nodes simultaneously
    ```
