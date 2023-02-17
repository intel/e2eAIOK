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

echo "############################## setting env ##############################"
master=`hostname`
# hosts format: hostname1,hostname2,...
hosts=$1
hosts_num=$2
OMP_NUM_THREADS=1
# epochs: usually 20
epochs=$3

echo "############################## 4 node opt model ##############################"
# -exp_name 'cpu-test-epoch-20' \
# --initial_lr 0.02 \
# --epochs 20
# -no_train -val
# --ipex 

export MASTER_ADDR=$master && \
    export MASTER_PORT=23100 && \
    mpirun \
        -genv nnUNet_raw_data_base="/home/vmagent/app/data/adaptor_large/nnUNet_raw_data_base" \
        -genv nnUNet_preprocessed="/home/vmagent/app/data/adaptor_large/nnUNet_preprocessed" \
        -genv RESULTS_FOLDER="/home/vmagent/app/data/adaptor_large/nnUNet_trained_models" \
        -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
        -n $hosts_num -ppn 1 \
        -hosts $hosts \
        -print-rank-map \
        -verbose \
        nnUNet_train \
        3d_fullres nnUNetTrainerV2 507 1 \
        -p nnUNetPlansv2.1_trgSp_kits19 \
        --epochs $epochs \
        --backend gloo \
        -pretrained_weights "/home/vmagent/app/data/adaptor_large/pre-trained-model/model_final_checkpoint-600.model"



# export nnUNet_raw_data_base="/home/vmagent/app/data/adaptor_large/nnUNet_raw_data_base" && \
# export nnUNet_preprocessed="/home/vmagent/app/data/adaptor_large/nnUNet_preprocessed" && \
# export RESULTS_FOLDER="/home/vmagent/app/data/adaptor_large/nnUNet_trained_models" && \
# python -m intel_extension_for_pytorch.cpu.launch --distributed \
#     --nproc_per_node=1 --nnodes=2 \
#     --master_addr=vsr257 \
#     --master_port=23900 \
#     --hostfile /home/vmagent/app/e2eAIOK/modelzoo/unet/bkp/hosts \
#     /home/vmagent/app/e2eAIOK/modelzoo/unet/nnUNet/nnunet/run/run_training.py \
#         3d_fullres nnUNetTrainerV2 507 1 \
#         -p nnUNetPlansv2.1_trgSp_kits19 \
#         --epochs 1 \
#         --backend gloo \
#         -pretrained_weights "/home/vmagent/app/data/adaptor_large/pre-trained-model/model_final_checkpoint-600.model"
    