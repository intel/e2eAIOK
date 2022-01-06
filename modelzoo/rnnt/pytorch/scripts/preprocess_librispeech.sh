# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash

DATA_DIR="/mnt/sdd/LibriSpeech/LibriSpeech"

python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/train-clean-100 \
    --dest_dir ${DATA_DIR}/train-clean-100-wav \
    --output_json ${DATA_DIR}/librispeech-train-clean-100-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/train-clean-360 \
    --dest_dir ${DATA_DIR}/train-clean-360-wav \
    --output_json ${DATA_DIR}/librispeech-train-clean-360-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/train-other-500 \
    --dest_dir ${DATA_DIR}/train-other-500-wav \
    --output_json ${DATA_DIR}/librispeech-train-other-500-wav.json


python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/dev-clean \
    --dest_dir ${DATA_DIR}/dev-clean-wav \
    --output_json ${DATA_DIR}/librispeech-dev-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/dev-other \
    --dest_dir ${DATA_DIR}/dev-other-wav \
    --output_json ${DATA_DIR}/librispeech-dev-other-wav.json


python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/test-clean \
    --dest_dir ${DATA_DIR}/test-clean-wav \
    --output_json ${DATA_DIR}/librispeech-test-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_DIR}/test-other \
    --dest_dir ${DATA_DIR}/test-other-wav \
    --output_json ${DATA_DIR}/librispeech-test-other-wav.json

bash scripts/create_sentencepieces.sh /sentencepieces

python scripts/tokenize_transcripts.py --output_dir /metadata/ --model /sentencepieces/librispeech1023.model ${DATA_DIR}/librispeech-*.json
