# Intel Optimized DIEN
## Original source disclose
Source repo: https://github.com/alibaba/ai-matrix

---

# Quick Start

## set path
```
export path_to_e2eaiok_dataset=`pwd`/e2eaiok_dataset
export path_to_e2eaiok=`pwd`/e2eAIOK
mkdir -p ${path_to_e2eaiok_dataset}
```

## Install
```
git clone https://github.com/intel/e2eAIOK.git
git submodule update --init --recursive
cd ${path_to_e2eaiok}/modelzoo/dien/train
sh patch_dien.sh
```

## Environment setup
```
cd ${path_to_e2eaiok}/Dockerfile-ubuntu18.04/
docker build -t e2eaiok-tensorflow-spark . -f DockerfileTensorflow-spark
```

## Download Dataset
```
cd ${path_to_e2eaiok}/modelzoo/dien/feature_engineering/
./download_dataset ${path_to_e2eaiok_dataset}
ls ${path_to_e2eaiok_dataset}/amazon_reviews
j2c_test  output  raw_data
```

### activate docker and conda
```
cd ${path_to_e2eaiok}
docker run --shm-size=10g -it --privileged --network host --device=/dev/dri -v ${path_to_e2eaiok_dataset}:/home/vmagent/app/dataset -v `pwd`/:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ e2eaiok-tensorflow-spark /bin/bash
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
cd /home/vmagent/app/e2eaiok/
python setup.py install
```

### Data Process
```
# Now you are running inside docker, /home/vmagent/app/ 
python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_training.py
python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_downloaded_test.py
python /home/vmagent/app/recdp/examples/python_tests/dien/preprocessing_for_inference.py
```

### Training
```
# test our best parameter
python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien --no_sigopt

# optional: use SDA to search best parameter
SIGOPT_API_TOKEN=$SIGOPT_API_TOKEN python run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien
```

### Inference
```
cp –r dnn_best_model dnn_best_model_trained
# modify infer.sh to change NUM_INSTANCES and may uncomment distributed inference 
./infer.sh
```

### Result Process
```
# For train result
grep -r 'time breakdown’ .
grep –r test_auc .

# For inference result
echo 'Inference Throughput is '; grep performance -r ./ | awk '{sum+=$NF}END{print sum}'
echo 'Inference prepare avg is '; grep "time breakdown" -r ./ | awk '{sum+=$7}END{print sum/NR}'
echo 'Inference eval avg is '; grep "time breakdown" -r ./ | awk '{sum+=$11}END{print sum/NR}'
```