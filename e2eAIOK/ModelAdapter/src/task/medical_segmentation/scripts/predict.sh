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

output_path=/data/predict_pretrain

# prepare env
# docker-compose run --rm --service-ports kits19 bash
# pip install -e .

# prepare data
# python nnunet/dataset_conversion/Task040_KiTS.py

# preprocess
# python nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 40 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19 --verify_dataset_integrity
# nnUNet_plan_and_preprocess -t 500 -pl2d None -pl3d ExperimentPlanner3D_v21_customTargetSpacing_kits19

# for train
# python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 40 1 -c -p nnUNetPlansv2.1_trgSp_kits19
# nnUNet_train 3d_fullres nnUNetTrainerV2 500 1 --epochs 50 -p nnUNetPlansv2.1_trgSp_kits19
# nnUNet_train_da 3d_fullres nnUNetTrainer_DA 29 40 1 --epochs 50 --loss_weights 1 0 1 -p nnUNetPlansv2.1_trgSp_kits19 -sp nnUNetPlansv2.1_trgSp_kits19

# for evaluation
# python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 40 1 -val -chk model_latest -p nnUNetPlansv2.1_trgSp_kits19
# nnUNet_train 3d_fullres nnUNetTrainerV2 40 1 -val -p nnUNetPlansv2.1_trgSp_kits19 -chk model_latest
# nnUNet_train_da 3d_fullres nnUNetTrainer_DA 29 40 1 -val -p nnUNetPlansv2.1_trgSp_kits19 -sp nnUNetPlansv2.1_trgSp_kits19 -chk model_latest

# choose model
# nnUNet_determine_postprocessing -t 48 -tr nnUNetTrainerV2 -m 3d_fullres
# nnUNet_find_best_configuration -m 3d_fullres -t 48

# pretrain predict
# nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task048_KiTS_clean/imagesTs/ -o /data/predict_pretrain -t 48 -m 3d_fullres --num_threads_preprocessing 10

# normal predict
# python nnunet/inference/predict_simple.py \
#     -i /data/nnUNet_raw_data_base/nnUNet_raw_data/Task040_KiTS/imagesTs/ \
#     -o $output_path -t 40 -m 3d_fullres -p nnUNetPlansv2.1_trgSp_kits19 \
#     -chk model_latest

# submission
# cd $output_path
# rename "s/case/prediction/" *
# zip prediction.zip prediction_*.nii.gz

# other util
# nnUNet_find_best_configuration, nnUNet_determine_postprocessing, nnUNet_evaluate_folder