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
export nnUNet_raw_data_base="/home/vmagent/app/data/adaptor_large/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/vmagent/app/data/adaptor_large/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/vmagent/app/data/adaptor_large/nnUNet_trained_models"
# nnUNetTrainer_DA_V2, nnUNetTrainerV2
trainer=nnUNetTrainer_DA_V2


############################################# predict #############################################
# -chk model_latest \

# # inference
time nnUNet_predict \
    -i ${nnUNet_raw_data_base}/nnUNet_raw_data/Task507_KiTS_kidney/testTr/ \
    -o ${nnUNet_raw_data_base}/nnUNet_raw_data/Task507_KiTS_kidney/predict/ \
    -f 1 \
    -t 507 -m 3d_fullres -p nnUNetPlansv2.1_trgSp_kits19 \
    --disable_tta \
    -tr $trainer \
    --overwrite_existing \
    --disable_mixed_precision 

# # evaluate with given label    
nnUNet_evaluate_folder \
    -ref ${nnUNet_raw_data_base}/nnUNet_raw_data/Task507_KiTS_kidney/labelsTr \
    -pred ${nnUNet_raw_data_base}/nnUNet_raw_data/Task507_KiTS_kidney/predict \
    -l 1 \
    --common
