#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
echo "begin train kaggle dataset"

python /home/xianyang/BlueWhale-poc/dlrm/dlrm_s_pytorch_horovod.py --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=/mnt/DP_disk1/dlrm/kaggle/tmp/train.txt --processed-data-file=/mnt/DP_disk1/dlrm/kaggle/tmp/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1000 --print-time --test-freq=5000 --test-mini-batch-size=128 --test-num-workers=16 --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle $dlrm_extra_option 2>&1 | tee run_kaggle_mlperf_horovod_two_node.log

echo "done"
