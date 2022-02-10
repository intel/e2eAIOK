import json
import logging

import horovod.tensorflow.keras as hvd
import tensorflow as tf

from tf.data.dataset import create_dataset
from tf.data.features import FeatureMeta


def init(args):
    hvd.init()

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(16)

    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)
    
    init_logger(full=hvd.rank() == 0)

def init_logger(full):
    if full:
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level = logging.ERROR,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_config(args):
    init(args)
    logger = logging.getLogger('upm')
    logger.info('command line arguments: {}'.format(json.dumps(vars(args))))

    features = FeatureMeta(args.dataset_meta_file)
    train, eval, steps_per_epoch = create_dataset(args, features)
    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train,
        'eval_dataset': eval, 
        'features': features
    }

    return config