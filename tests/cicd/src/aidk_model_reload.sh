#/bin/bash

printf "s\n" | python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH --no_sigopt --custom_result_path $CUSTOM_RESULT_PATH --interactive