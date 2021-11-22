from functools import partial
import math
import os
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import tensorflow_transform as tft


class BinDataset:
    """Binary version of dataset loader"""

    def __init__(self, data_file, metadata,
                 batch_size=1, bytes_per_feature=4, drop_last_batch=False):
        # dataset
        self.tar_fea = metadata.label_meta   # target
        self.den_fea = metadata.numerical_meta  # dense  features
        self.spa_fea = metadata.categorical_meta  # sparse features
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
        for k, v in self.den_fea.items():
            numerical_feature = array[:, v['indice']].view(dtype=np.float32)
            features[k] = tf.convert_to_tensor(numerical_feature)

        for k, v in self.spa_fea.items():
            features[k] = tf.convert_to_tensor(array[:, v['indice']])
        click = tf.convert_to_tensor(array[:, list(self.tar_fea.values())[0]['indice']], dtype=tf.float32)
        
        return features, click


class TFRecordsDataset:
    def __init__(self, meta_path, features, label):
        self.meta_path = meta_path
        self.features = features
        self.label = label
        self.feature_spec = tft.TFTransformOutput(meta_path).transformed_feature_spec()

    def consolidate_batch(self, elem, label, features):
        label = elem.pop(label)
        reshaped_label = tf.reshape(label, [-1, label.shape[-1]])

        reshaped_elem = {
            key: tf.reshape(elem[key], [-1, elem[key].shape[-1]])
            for key in elem
            if key in features
        }

        return reshaped_elem, reshaped_label

    def input_fn(self,
            filepath_pattern,
            records_batch_size,
            shuffle=False,
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

        if shuffle:
            dataset = dataset.shuffle(records_batch_size)

        dataset = dataset.batch(
            batch_size=records_batch_size,
            drop_remainder=False
        )

        dataset = dataset.apply(
            transformation_func=tf.data.experimental.parse_example_dataset(
                features=self.feature_spec,
                num_parallel_calls=1
            )
        )

        dataset = dataset.map(
            map_func=partial(
                self.consolidate_batch, label=self.label, features=self.features
            ),
            num_parallel_calls=None
        )

        dataset = dataset.prefetch(
            buffer_size=1
        )

        return dataset