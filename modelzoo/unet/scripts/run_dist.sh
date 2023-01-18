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


echo "############################## 4 node ##############################"
# -exp_name 'cpu-test-epoch-20' \
# --initial_lr 0.02 \
# --epochs 20
# -no_train -val

export MASTER_ADDR=vsr214 && \
    export MASTER_PORT=23900 && \
    mpirun \
        -genv nnUNet_raw_data_base="/home/vmagent/app/data/adaptor_large/nnUNet_raw_data_base" \
        -genv nnUNet_preprocessed="/home/vmagent/app/data/adaptor_large/nnUNet_preprocessed" \
        -genv RESULTS_FOLDER="/home/vmagent/app/data/adaptor_large/nnUNet_trained_models" \
        -genv OMP_NUM_THREADS=23 \
        -n 4 -ppn 1 \
        -hosts vsr214,vsr215,vsr216,vsr218 \
        -print-rank-map \
        -verbose \
        python -u nnUnet/nnunet/run/run_training_da.py \
        3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
        -p nnUNetPlansv2.1_trgSp_kits19 \
        -sp nnUNetPlansv2.1_trgSp_kits19 \
        --epochs 20 --loss_weights 1 0 1 0 0 \
        --backend gloo --ipex \
        -pretrained_weights /home/vmagent/app/data/adaptor_large/pre-trained-model/model_final_checkpoint-600.model



