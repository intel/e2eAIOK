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

from functools import partial
import math
import os
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

from data.outbrain.features import get_features_keys

class BinDataset:
    """Binary version of dataset loader"""

    def __init__(self, data_file, metadata,
                 batch_size=1, bytes_per_feature=4, drop_last_batch=False):
        # dataset
        self.tar_fea = metadata.target   # target
        self.den_fea = metadata.numerical_features  # dense  features
        self.spa_fea = metadata.categorical_features  # sparse features
        self.tar_len = 1
        self.den_len = len(self.den_fea)
        self.spa_len = len(self.spa_fea)
        self.tot_fea = self.tar_len + self.den_len + self.spa_len

        self.batch_size = batch_size
        self.bytes_per_batch = (bytes_per_feature * self.tot_fea * batch_size)

        data_file_size = os.path.getsize(data_file)
        self.num_batches = int(math.ceil(data_file_size / self.bytes_per_batch)) if not drop_last_batch \
                            else int(math.floor(data_file_size / self.bytes_per_batch))
        # self.num_batches = data_file_size // self.bytes_per_batch

        bytes_per_sample = bytes_per_feature * self.tot_fea
        self.num_samples = data_file_size // bytes_per_sample

        if hvd.size() > 1:
            self.bytes_per_rank = self.bytes_per_batch // hvd.size()
        else:
            self.bytes_per_rank = self.bytes_per_batch

        if hvd.size() > 1 and self.num_batches * self.bytes_per_batch > data_file_size:
            last_batch = (data_file_size % self.bytes_per_batch) // bytes_per_sample
            self.bytes_last_batch = last_batch // hvd.size() * bytes_per_sample
        else:
            self.bytes_last_batch = self.bytes_per_rank

        if self.bytes_last_batch == 0:
            self.num_batches = self.num_batches - 1
            self.bytes_last_batch = self.bytes_per_rank

        print('data file:', data_file, 'number of batches:', self.num_batches)
        self.file = open(data_file, 'rb')

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError()
        my_rank = hvd.rank() if hvd.size() > 1 else 0
        rank_size = self.bytes_last_batch if idx == (self.num_batches - 1) else self.bytes_per_rank 
        self.file.seek(idx * self.bytes_per_batch + rank_size * my_rank, 0)
        raw_data = self.file.read(rank_size)

        array = np.frombuffer(raw_data, dtype=np.int32).reshape(-1, self.tot_fea)
        features = {}
        for i, f in enumerate(self.den_fea):
            numerical_feature = array[:, i+self.tar_len].view(dtype=np.float32)
            features[f] = tf.convert_to_tensor(numerical_feature)

        for i, f in enumerate(self.spa_fea):
            features[f] = tf.convert_to_tensor(array[:, i+self.tar_len+self.den_len])
        click = tf.convert_to_tensor(array[:, 0], dtype=tf.float32)
        
        return features, click



def _consolidate_batch(elem):
    label = elem.pop('label')
    reshaped_label = tf.reshape(label, [-1, label.shape[-1]])
    features = get_features_keys()

    reshaped_elem = {
        key: tf.reshape(elem[key], [-1, elem[key].shape[-1]])
        for key in elem
        if key in features
    }

    return reshaped_elem, reshaped_label

def train_input_fn(
        filepath_pattern,
        feature_spec,
        records_batch_size,
        num_gpus=1,
        id=0):

    dataset = tf.data.Dataset.list_files(
        file_pattern=filepath_pattern,
        shuffle=False
    )

    dataset = tf.data.TFRecordDataset(
        filenames=dataset,
        num_parallel_reads=1
    )

    dataset = dataset.shard(num_gpus, id)

    dataset = dataset.shuffle(records_batch_size)

    dataset = dataset.batch(
        batch_size=records_batch_size,
        drop_remainder=False
    )

    dataset = dataset.apply(
        transformation_func=tf.data.experimental.parse_example_dataset(
            features=feature_spec,
            num_parallel_calls=1
        )
    )

    dataset = dataset.map(
        map_func=partial(
            _consolidate_batch
        ),
        num_parallel_calls=None
    )

    dataset = dataset.prefetch(
        buffer_size=1
    )

    return dataset


def eval_input_fn(
        filepath_pattern,
        feature_spec,
        records_batch_size,
        num_gpus=1,
        id=0):
    dataset = tf.data.Dataset.list_files(
        file_pattern=filepath_pattern,
        shuffle=False
    )

    dataset = tf.data.TFRecordDataset(
        filenames=dataset,
        num_parallel_reads=1
    )

    dataset = dataset.shard(num_gpus, id)

    dataset = dataset.batch(
        batch_size=records_batch_size,
        drop_remainder=False
    )

    dataset = dataset.apply(
        transformation_func=tf.data.experimental.parse_example_dataset(
            features=feature_spec,
            num_parallel_calls=1
        )
    )

    dataset = dataset.map(
        map_func=partial(
            _consolidate_batch
        ),
        num_parallel_calls=None
    )
    dataset = dataset.prefetch(
        buffer_size=1
    )

    return dataset
