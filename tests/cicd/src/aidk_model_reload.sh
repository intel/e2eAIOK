#/bin/bash

cd /home/vmagent/app/hydro.ai
echo "s" | /opt/intel/oneapi/intelpython/latest/bin/python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH