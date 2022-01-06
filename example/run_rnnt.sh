# prepare data
# tree -d 1 ../dataset/LibriSpeech/
# ../dataset/LibriSpeech/
# ├── metadata
# │   └── *.pkl
# ├── meta.yaml
# ├── train
# │   └── train-*-wav
# ├── valid
# │   └── dev-*-wav
# └── sentencepieces
# │   └── *.model
# │   └── *.vocab

# change hosts in conf/hydroai_defaults_rnnt_example.conf
# hosts:
#   - ${host_name}

source /opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/.local/env/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/rnnt/lib/python3.7/site-packages/torch/lib/
export LD_PRELOAD=/opt/intel/oneapi/intelpython/latest/envs/rnnt/lib/libiomp5.so

# Use hydro.ai API
SIGOPT_API_TOKEN=${TOKEN} python run_hydroai.py --data_path "/home/vmagent/app/dataset/LibriSpeech" --model_name rnnt --conf conf/hydroai_defaults_rnnt_example.conf

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/LibriSpeech" --model_name rnnt --conf conf/hydroai_defaults_rnnt_example.conf