# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Modifications copyright Intel
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

import json
import logging
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf

from data.outbrain.dataset import create_dataset
from data.outbrain.features import FeatureMeta


def init(args, logger):
    hvd.init()

    init_logger(
        full=hvd.rank() == 0,
        args=args,
        logger=logger
    )

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    
    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)

def init_logger(args, full, logger):
    if full:
        logger.setLevel(logging.INFO)
        logger.warning('command line arguments: {}'.format(json.dumps(vars(args))))
    else:
        logger.setLevel(logging.ERROR)


def create_config(args):
    logger = logging.getLogger('tensorflow')

    init(args, logger)

    features = FeatureMeta(args.dataset_meta_file)
    train, eval, steps_per_epoch = create_dataset(args, features)
    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train,
        'eval_dataset': eval, 
        'features': features
    }

    return config
