# Intel Democratized Solutions for RecSys Challenge 2021 

# How to Use

## Prepare
1. Environment
    * Spark 3.0
    * Hadoop 3.2
    * lightgbm 3.2.1
    * XGBoost 1.3.3
2. Prepare 
    * install pyrecdp
    ```bash
    pip install pyrecdp
    ```
    * Put original dataset(train, valid, test, valid_split_index) at HDFS: /recsys2021/oridata
    * Create two folds in HDFS to save data for stage1 and stage2: /recsys2021/datapre_stage1/, /recsys2021/datapre_stage2/

## E2E Train
1. Preprocess train and valid dataset
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
