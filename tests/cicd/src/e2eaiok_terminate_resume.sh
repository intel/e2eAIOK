#/bin/bash

python run_e2eaiok.py --model_name $MODEL_NAME --data_path $DATA_PATH  --custom_result_path $CUSTOM_RESULT_PATH &
kill -15 $(pgrep python)
python run_e2eaiok.py --model_name $MODEL_NAME --data_path $DATA_PATH  --custom_result_path $CUSTOM_RESULT_PATH