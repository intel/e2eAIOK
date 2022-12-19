#!/bin/bash

source $APP_DIR/scripts/setup.sh
NODE_IP=$(hostname -I | awk -F' ' '{print $1}')

bash run_prepare_env.sh $1 $NODE_IP
bash run_data_process.sh $1 $NODE_IP
bash run_inference.sh $1 $NODE_IP

