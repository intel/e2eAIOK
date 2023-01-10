#/bin/bash

printf "s\n" | python run_e2eaiok.py --model_name $MODEL_NAME --data_path $DATA_PATH  --custom_result_path $CUSTOM_RESULT_PATH --interactive