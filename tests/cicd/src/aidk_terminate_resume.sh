#/bin/bash

cd /home/vmagent/app/hydro.ai

printf "========= Start Config Sigopt =========\n"
printf "%s\ny\ny\n" $SIGOPT_API_TOKEN | sigopt config
printf "\n========= End Config Sigopt ==========\n"

printf "========= Start First Job in background   ========="
nohup printf "y\ny\n" | /opt/intel/oneapi/intelpython/latest/bin/python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH &
sleep 3
printf "========= First Job Started in background ========="

echo "========= Start Pipeline Terminated ========="
kill -15 $(pgrep python)
echo "=========== End Pipeline Terminated ========="

echo "========== Start Resume Context  =========="
printf "y\ny\n" | /opt/intel/oneapi/intelpython/latest/bin/python run_hydroai.py --model_name $MODEL_NAME --data_path $DATA_PATH
echo "========== Finish Resume Context =========="