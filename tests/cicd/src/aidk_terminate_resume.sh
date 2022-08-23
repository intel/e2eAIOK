#/bin/bash

python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH --no_sigopt &
kill -15 $(pgrep python)
python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH --no_sigopt