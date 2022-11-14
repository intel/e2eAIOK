# prepare data
# tree dataset/resnet/
# dataset/resnet/
# ├── meta.yaml
# ├── train
# └── valid
# change hosts and eth in conf/e2eaiok_defaults_resnet_example.conf
# iface: ${eth0}
# hosts:
#   - ${host_name}


RANDOM_SEED=`date +%s`

QUALITY=0.759

set -e



export OMP_NUM_THREADS=14


export KMP_BLOCKTIME=1

# Use e2eaiok API
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/resnet/" --model_name resnet --conf conf/e2eaiok_defaults_resnet_example.conf --executable_python /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python --program /home/vmagent/app/e2eaiok/modelzoo/resnet/mlperf_resnet/imagenet_main.py



# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/resnet/" --model_name resnet --conf conf/e2eaiok_defaults_resnet_example.conf --executable_python /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python --program /home/vmagent/app/e2eaiok/modelzoo/resnet/mlperf_resnet/imagenet_main.py