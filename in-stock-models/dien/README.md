## Quick start

#### Prepare docker
```
wget https://github.com/intel-innersource/frameworks.bigdata.bluewhale/archive/refs/heads/main.zip
unzip main.zip
 
wget https://github.com/oap-project/recdp/archive/refs/heads/master.zip
unzip master.zip
 
# For CPU
current_path=`pwd`
docker run --privileged --network host -v ${current_path}/frameworks.bigdata.bluewhale-main/examples/dien/train/ai-matrix/horovod:/home/xxx/dien -v ${current_path}/recdp-master/:/home/vmagent/app/recdp -it xuechendi/ubuntu-tf2.4-horovod-recdp /bin/bash
/home/start_service.sh

# For GPU
current_path=`pwd`
docker run --privileged --network host -v ${current_path}/frameworks.bigdata.bluewhale-main/examples/dien/train/ai-matrix/gpu:/home/xxx/dien -v ${current_path}/recdp-master/:/home/vmagent/app/recdp -it xuechendi/ubuntu-gpu-tf2-py3-recdp /bin/bash
/home/start-service.sh

source /etc/profile.d/spark-env.sh
# if running on multiple nodes, only run below on master
${SPARK_HOME}/sbin/start-history-server.sh
```

#### Prepare dataset
```
cd /home/vmagent/app/recdp/examples/python_tests/dien/
./download_dataset
ls /home/vmagent/app/recdp/examples/python_tests/dien/raw_data
local_test_splitByUser  meta_Books.json  reviews_Books.json
```

#### Train - Preprocessing data
```
cd /home/xxx/dien
# modify preprocessing_for_training.py and preprocessing_for_downloaded_test to change spark://${master}:7077
python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_training.py
python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_downloaded_test.py
```

#### Train
```
# training on scalable CPU
source /etc/profile.d/oneccl.sh
cd /home/xxx/dien
# modify train.sh to specify multiple nodes’ ip and nic; num-intra-threads set to num_cores/4;
./train

# training on GPU
cd /home/xxx/dien
./train
```

#### Inference data preprocess
```
# modify preprocessing_for_inference.py to change spark://${master}:7077 and NUM_INSTS
# For NUM_INSTS, CPU should set same as num_cores/2; For GPU, set as 8
python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_inference.py
```

#### Inference
```
cp –r dnn_best_model dnn_best_model_trained
# modify infer.sh to change NUM_INSTANCES and may uncomment distributed inference 
./infer.sh
```

#### Result processing
```
# For train result
grep -r 'time breakdown’ .
grep –r test_auc .

# For inference result
echo 'Inference Throughput is '; grep performance -r ./ | awk '{sum+=$NF}END{print sum}'
echo 'Inference prepare avg is '; grep "time breakdown" -r ./ | awk '{sum+=$7}END{print sum/NR}'
echo 'Inference eval avg is '; grep "time breakdown" -r ./ | awk '{sum+=$11}END{print sum/NR}'
```
