import logging
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
        if self.display_id_meta is not None:
            self.features_keys.extend((self.display_id_meta.keys()))
        print(f'All feature columns: {self.features_keys}')