#/bin/bash

cd /home/vmagent/app/hydro.ai
printf "s\n" | /opt/intel/oneapi/intelpython/latest/bin/python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH --no_sigopt --interactive