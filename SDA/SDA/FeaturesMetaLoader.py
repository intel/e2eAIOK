import logging
import yaml

class FeaturesMetaLoader:
    def __init__(self, meta_file):
        with open(meta_file) as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        self.numerical_meta = meta['numerical_features'] if 'numerical_features' in meta else None
        self.categorical_meta = meta['categorical_features'] if 'categorical_features' in meta else None
        self.label_meta = meta['label'] if 'label' in meta else None
        self.display_id_meta = meta['display_id'] if 'display_id' in meta else None

        self.tfrecords_meta_path = meta['tfrecords_meta_path'] if 'tfrecords_meta_path' in meta else None
        self.prebatch_size = meta['prebatch_size'] if 'prebatch_size' in meta else None
        self.dataset_format = meta['dataset_format'] if 'dataset_format' in meta else 'TFRecords'
        self.training_set_size = meta['training_set_size'] if 'training_set_size' in meta else None