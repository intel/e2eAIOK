import json
import logging

import pt.extend_distributed as ext_dist
from pt.data.dataset import create_dataset
from pt.data.features import FeatureMeta

def init(args):
    ext_dist.init_distributed(backend='ccl')
    init_logger(full=True)

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
    train, eval, steps_per_epoch = create_dataset(args)
    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train,
        'eval_dataset': eval, 
        'features': features
    }

    return config