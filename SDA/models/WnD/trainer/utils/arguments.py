# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tensorflow2 WideAndDeep Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )

    locations = parser.add_argument_group('datasets parameters')

    locations.add_argument('--train_data_pattern', type=str, default='/outbrain/tfrecords/train/part*', nargs='+',
                           help='Pattern of training file names. For example if training files are train_000.tfrecord, '
                                'train_001.tfrecord then --train_data_pattern is train_*')

    locations.add_argument('--eval_data_pattern', type=str, default='/outbrain/tfrecords/eval/part*', nargs='+',
                           help='Pattern of eval file names. For example if eval files are eval_000.tfrecord, '
                                'eval_001.tfrecord then --eval_data_pattern is eval_*')

    locations.add_argument('--transformed_metadata_path', type=str, default='/outbrain/tfrecords',
                           help='Path to transformed_metadata for feature specification reconstruction, only available for TFRecords')
    
    locations.add_argument('--prebatch_size', type=int, default=4096, help='Dataset prebatch size, only available for TFRecords')

    locations.add_argument('--dataset_format', type=str, default='TFRecords', help='train/test dataset format, support TFRecords and binary')

    locations.add_argument('--model_dir', type=str, default='/outbrain/checkpoints',
                           help='Destination where model checkpoint will be saved')

    training_params = parser.add_argument_group('training parameters')

    training_params.add_argument('--training_set_size', type=int, default=59761827,
                                 help='Number of samples in the training set')

    training_params.add_argument('--global_batch_size', type=int, default=131072,
                                 help='Total size of training batch')

    training_params.add_argument('--eval_batch_size', type=int, default=131072,
                                 help='Total size of evaluation batch')

    training_params.add_argument('--num_epochs', type=int, default=20,
                                 help='Number of training epochs')

    training_params.add_argument('--amp', default=False, action='store_true',
                                 help='Enable automatic mixed precision conversion')

    training_params.add_argument('--xla', default=False, action='store_true',
                                 help='Enable XLA conversion')

    training_params.add_argument('--linear_learning_rate', type=float, default=-1,
                                 help='Learning rate for linear model')

    training_params.add_argument('--deep_learning_rate', type=float, default=-1,
                                 help='Learning rate for deep model')

    training_params.add_argument('--deep_warmup_epochs', type=float, default=-1,
                                 help='Number of learning rate warmup epochs for deep model')
    
    training_params.add_argument('--metric', type=str, default='AUC', help='Evaluation metric')

    training_params.add_argument('--metric_threshold', type=float, default=0, help='Metric threshold used for training early stop')

    model_construction = parser.add_argument_group('model construction')

    model_construction.add_argument('--deep_hidden_units', type=int, default=[], nargs="+",
                                    help='Hidden units per layer for deep model, separated by spaces')

    model_construction.add_argument('--deep_dropout', type=float, default=-1,
                                    help='Dropout regularization for deep model')

    return parser.parse_args()
