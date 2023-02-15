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

unset MASTER_ADDR

echo "############################## setting env ##############################"
export nnUNet_raw_data_base="/home/vmagent/app/data/adaptor_large/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/vmagent/app/data/adaptor_large/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/vmagent/app/data/adaptor_large/nnUNet_trained_models"
pre_trained_model_path="/home/vmagent/app/data/adaptor_large/pre-trained-model/model_final_checkpoint-600.model"


echo "############################## 1 node stock model ##############################"
# -exp_name 'cpu-test-epoch-20' \
# -no_train -val
epochs=$1

nnUNet_train \
    3d_fullres nnUNetTrainerV2 507 1 \
    -p nnUNetPlansv2.1_trgSp_kits19 \
    --epochs $epochs