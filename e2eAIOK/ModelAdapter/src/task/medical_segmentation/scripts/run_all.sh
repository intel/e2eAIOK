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
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
conda activate pytorch-1.10.0

cd /home/vmagent/app/TLK/src/task/medical_segmentation
cd third_party/nnUNet/nnunet && patch -p1 < ../../../kits19.patch
cd - && cp -r third_party/nnUNet/nnunet .
cd /home/vmagent/app/TLK/src/task/medical_segmentation/nnunet
pip install -e .

export nnUNet_raw_data_base="/home/vmagent/app/dataset/TLK/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/vmagent/app/dataset/TLK/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/vmagent/app/dataset/TLK/nnUNet_trained_models"
pre_trained_model_path="${RESULTS_FOLDER}/nnUNet/3d_fullres/Task508_AMOS_kidney/nnUNetTrainerV2__nnUNetPlansv2.1_trgSp_kits19/fold_1/model_final_checkpoint.model"


############################################# data preprocess #############################################
# cd /home/vmagent/app/TLK/src/task/medical_segmentation/nnunet
# python dataset_conversion/amos_convert_label.py
# python dataset_conversion/kits_convert_label.py basic
# nnUNet_plan_and_preprocess -t 507 --verify_dataset_integrity
# nnUNet_plan_and_preprocess -t 508 --verify_dataset_integrity
# nnUNet_plan_and_preprocess -t 508 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19
# !nnUNet_plan_and_preprocess -t 507 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 -no_pp
# python dataset_conversion/kits_convert_label.py intensity
# nnUNet_plan_and_preprocess -t 507 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 -no_plan


############################################# training #############################################
nnUNet_train 3d_fullres nnUNetTrainerV2 508 1 --epochs 1 -p \
    nnUNetPlansv2.1_trgSp_kits19 --disable_postprocessing_on_folds --ipex

nnUNet_train_da 3d_fullres nnUNetTrainer_DA_V2 508 507 1 \
    -p nnUNetPlansv2.1_trgSp_kits19 \
    -sp nnUNetPlansv2.1_trgSp_kits19 \
    --epochs 1 --loss_weights 1 0 1 0 0 \
    --ipex \
    -pretrained_weights $pre_trained_model_path


############################################# predict & evaluate #############################################
# nnUNet_predict \
#     -i ${nnUNet_raw_data_base}/nnUNet_raw_data/Task507_KiTS_kidney/imagesTr/ \
#     -o /home/vmagent/app/dataset/prediction \
#     -f 1 \
#     -t 507 -m 3d_fullres -p nnUNetPlansv2.1_trgSp_kits19 \
#     --disable_tta \
#     -tr nnUNetTrainer_DA_V2

# nnUNet_evaluate_folder \
#     -ref ${nnUNet_raw_data_base}/nnUNet_raw_data/Task507_KiTS_kidney/labelsTr \
#     -pred /home/vmagent/app/dataset/prediction \
#     -l 1 \
#     --common