# Intel® End-to-End AI Optimization Kit for DLRM
## Original source disclose
Source repo: https://github.com/facebookresearch/dlrm

---

# Quick Start

## Enviroment Setup
``` bash
# Setup ENV
git clone https://github.com/intel/e2eAIOK.git
cd e2eAIOK
git submodule update --init --recursive
python3 scripts/start_e2eaiok_docker.py -b pytorch -w ${host0} ${host1} ${host2} ${host3} --proxy ""
```

## Enter Docker
```
sshpass -p docker ssh ${host0} -p 12346
```
## Workflow Prepare
``` bash
# prepare model codes
cd /home/vmagent/app/e2eaiok/modelzoo/dlrm
sh patch_dlrm.sh

# Download Dataset
# Download from https://labs.criteo.com/2013/12/download-terabyte-click-logs/ and unzip them
ls /home/vmagent/app/dataset/criteo/raw_data
day_0  day_10  day_12  day_14  day_16  day_18  day_2   day_21  day_23  day_4  day_6  day_8
day_1  day_11  day_13  day_15  day_17  day_19  day_20  day_22  day_3   day_5  day_7  day_9

# source spark env
source /home/spark-env.sh

# Start services
# only if there is no spark service running, may check ${localhost}:8080 to confirm
sh /home/start_spark_service.sh
```

## Data Process

* suggest to build a HDFS cluster with at least 3 datanodes
* required ~1T HDFS capacity for raw data and processed data storage
* required ~1.5T Spark Shuffle capacity total, 500G per node

```
# 1. add your HDFS master node in below files
convert_to_parquet.py
preprocessing.py
convert_to_binary.py

# 2. upload and transform downloaded data as parquet to HDFS, may take 30 mins
# please only start 1 spark worker co-located with raw data
python convert_to_parquet.py

# 3. process data, may take about 1 hour
# start spark workers on all three nodes
python preprocessing.py

# 4. download processed data and convert to binary, taking about 1 hour
python convert_to_binary.py

# 5. check result
ll /home/vmagent/app/dataset/criteo/output/
total 683357412
         4096  ./
         4096  ../
          474  day_fea_count.npz
  14261971040  test_data.bin
 671231630720  train_data.bin
  14261970880  valid_data.bin
```

## Training
```
cd /home/vmagent/app/e2eaiok/; python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/criteo --model_name dlrm 
```

## Inference
```
cd /home/vmagent/app/e2eaiok/modelzoo/dlrm/; bash run_inference.sh
```
