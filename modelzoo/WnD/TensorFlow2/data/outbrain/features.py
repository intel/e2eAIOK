import logging
import tensorflow as tf
import yaml

class FeatureMeta:
    def __init__(self, meta_file):
        with open(meta_file) as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        self.numerical_meta = meta['numerical_features']
        self.categorical_meta = meta['categorical_features']
        self.label_meta = meta['label']
        self.display_id_meta = meta['display_id'] if 'display_id' in meta else None
        self.get_features_keys()

        self.tfrecords_meta_path = meta['tfrecords_meta_path'] if 'tfrecords_meta_path' in meta else None
        self.prebatch_size = meta['prebatch_size'] if 'prebatch_size' in meta else None
        self.dataset_format = meta['dataset_format'] if 'dataset_format' in meta else 'TFRecords'
        self.training_set_size = meta['training_set_size'] if 'training_set_size' in meta else None
    
    def get_features_keys(self):
        self.numerical_keys = list(self.numerical_meta.keys())
        self.categorical_keys = list(self.categorical_meta.keys())
        self.label = list(self.label_meta.keys())[0]
        self.features_keys = self.numerical_keys + self.categorical_keys
        # for f in self.numerical_features:
        #     self.numerical_keys.append(f['name'])
        #     self.features_keys.append(f['name'])
        # for f in self.categorical_features:
        #     self.categorical_keys.append(f['name'])
        #     self.features_keys.append(f['name'])
        if self.display_id_meta is not None:
            self.features_keys.extend((self.display_id_meta.keys()))
        print(f'All feature columns: {self.features_keys}')

    def get_feature_columns(self):
        logger = logging.getLogger('tensorflow')
        wide_columns, deep_columns = [], []

        for column_name in self.categorical_keys:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=self.categorical_meta[column_name]['voc_size'])
            wrapped_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=self.categorical_meta[column_name]['emb_dim'],
                combiner='mean')

            wide_columns.append(categorical_column)
            deep_columns.append(wrapped_column)

        numerics = [tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
                    for column_name in self.numerical_keys]

        wide_columns.extend(numerics)
        deep_columns.extend(numerics)

        logger.warning('deep columns: {}'.format(len(deep_columns)))
        logger.warning('wide columns: {}'.format(len(wide_columns)))
        logger.warning('wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))

        return wide_columns, deep_columns
