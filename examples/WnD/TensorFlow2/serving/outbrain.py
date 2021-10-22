import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from zoo.serving.client import InputQueue, OutputQueue
from data.outbrain.dataloader import TFRecordsDataset
from data.outbrain.features import FeatureMeta
import time
import numpy as np
import argparse

PREBATCH_SIZE = 4096
DISPLAY_ID_COLUMN = 'display_id'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tensorflow2 WideAndDeep Model serving',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )
    parser.add_argument('--dataset_meta_file', type=str, default='data/outbrain/outbrain_meta.yaml',
                        help='dataset meta file')
    parser.add_argument('--dataset_pattern', type=str, default='/mnt/sdd/outbrain2/tfrecords/eval/part*', 
                        help='Pattern of serving dataset. For example if dataset files are file_000.tfrecord, '
                            'file_001.tfrecord then --dataset_pattern is file_*')
    
    return parser.parse_args()

def load_dataset(args):
    features = FeatureMeta(args.dataset_meta_file)
    dataloader = TFRecordsDataset(features.tfrecords_meta_path, features.features_keys, features.label)

    dataset = dataloader.input_fn(
        args.dataset_pattern,
        records_batch_size=1,
        shuffle=False,
        num_gpus=1,
        id=0)
    return dataset

def run():
    args = parse_args()
    dataset = load_dataset(args)

    input_api = InputQueue()
    output_api = OutputQueue()
    output_api.dequeue()

    for (x, y) in dataset:
        for index in range(PREBATCH_SIZE):
            record = {}
            for k in x:
                record[k] = x[k][index].numpy()
            input_api.enqueue("wnd"+str(index), **record)
        break
    time.sleep(2)
    preds = output_api.dequeue()
    print(preds)


if __name__ == "__main__":
    run()

