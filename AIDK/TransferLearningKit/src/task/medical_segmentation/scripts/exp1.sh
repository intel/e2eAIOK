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

############################################# env #############################################
# pip install SimpleITK==2.0.2
# python -m pip install intel_extension_for_pytorch==1.11.200 -f https://developer.intel.com/ipex-whl-stable
# docker-compose run --rm --service-ports kits19_cpu bash

############################################# 预处理 #############################################
# nnUNet_plan_and_preprocess -t 507 --verify_dataset_integrity
# nnUNet_plan_and_preprocess -t 507 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 -no_pp
# nnUNet_plan_and_preprocess -t 507 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 -no_plan
# nnUNet_plan_and_preprocess -t 508 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19
# python nnunet/experiment_planning/change_patch_size.py


############################################# 跑实验 #############################################

# cac evaluate
# nnUNet_train_da 3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
#     -p nnUNetPlansv2.1_trgSp_kits19 \
#     -sp nnUNetPlansv2.1_trgSp_kits19 \
#     --loss_weights 1 0 1 0 0 -val -chk model_latest

# erm
# nnUNet_train 3d_fullres nnUNetTrainerV2 508 1 --epochs 800 -p nnUNetPlansv2.1_trgSp_kits19 -c


# cac
# nnUNet_train_da 3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
#     -p nnUNetPlansv2.1_trgSp_kits19 \
#     -sp nnUNetPlansv2.1_trgSp_kits19 \
#     --epochs 100 --loss_weights 0 1 0 0 0 -c \
#     -exp_name 'pre-600-lr-0-2-finetune-epoch-100' 


# nnUNet_train_da 3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
#     -p nnUNetPlansv2.1_trgSp_kits19 \
#     -sp nnUNetPlansv2.1_trgSp_kits19 \
#     --epochs 60 --loss_weights 0 1 0 0 0 \
#     -exp_name 'pre-600-lr-0-2-finetune-epoch-60' \
#     -pretrained_weights /data/nnUNet_trained_models/nnUNet/3d_fullres/Task508_AMOS_kidney/nnUNetTrainerV2__nnUNetPlansv2.1_trgSp_kits19/fold_1/model_final_checkpoint-600.model


# erm evaluate
# nnUNet_train_da 3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
#     -p nnUNetPlansv2.1_trgSp_kits19 \
#     -sp nnUNetPlansv2.1_trgSp_kits19 \
#     --loss_weights 1 0 0 0 0 \
#     -val -chk model_final_checkpoint-800


# orancle
nnUNet_train 3d_fullres nnUNetTrainerV2 507 1 \
    -p nnUNetPlansv2.1_trgSp_kits19 \
    --epochs 1000 -c --fp16


# directory
# input_dir=/data/nnUNet_trained_models/nnUNet/3d_fullres/Task502_KiTS_tumor_small/nnUNetTrainer_DA_V2__nnUNetPlansv2.1_trgSp_kits19
# output_dir=/data/nnUNet_trained_models/nnUNet/3d_fullres/Task502_KiTS_tumor_small
# mv $input_dir $output_dir"/cac-enc-dec-seg"

exit 0


############################################# 预测评估 #############################################

# # predict
# # -chk model_latest \
# nnUNet_predict \
#     -i /data/nnUNet_raw_data_base/nnUNet_raw_data/Task040_KiTS/imagesTs/ \
#     -o /data/tmp \
#     -f 1 \
#     -t 507 -m 3d_fullres -p nnUNetPlansv2.1_trgSp_kits19 \
#     --disable_tta \
#     -tr nnUNetTrainer_DA_V2

# # evaluate with given label    
# nnUNet_evaluate_folder \
#     -ref /mnt/DP_disk1/yu/nnUNet_raw_data_base/nnUNet_raw_data/Task507_KiTS_kidney/labelsTr \
#     -pred /data/tmp \
#     -l 1 \
#     --common
