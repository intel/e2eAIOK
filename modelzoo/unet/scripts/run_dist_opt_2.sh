#    Copyright 2022, Intel Corporation.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#!/bin/bash
set -x

echo "############################## setting args ##############################"
master=$1
node_rank=$2
epochs=$3

echo "############################## setting env ##############################"
export nnUNet_raw_data_base="/home/vmagent/app/data/adaptor_large/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/vmagent/app/data/adaptor_large/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/vmagent/app/data/adaptor_large/nnUNet_trained_models"
pretrained_weights_path="/home/vmagent/app/data/adaptor_large/pre-trained-model/model_final_checkpoint-600.model"

echo "############################## 4 node opt model ##############################"
# -exp_name 'cpu-test-epoch-20' \
# --initial_lr 0.02 \
# --epochs 20
# -no_train -val
# --ipex 

torchrun \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$master:29400 \
    --nnodes=4 \
    --nproc_per_node=1 \
    --no_python \
    --node_rank $node_rank \
    --master_addr=$master \
    --master_port=23900 \
    nnUNet_train_da \
        3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
        -p nnUNetPlansv2.1_trgSp_kits19 \
        -sp nnUNetPlansv2.1_trgSp_kits19 \
        --epochs $epochs --loss_weights 1 0 1 0 0 \
        --backend gloo \
        -pretrained_weights $pretrained_weights_path
    