from e2eAIOK.common.utils import *
from e2eAIOK.utils.hydroconfig import *

class DummyDataProcessor:
    def __init__(self, model_name):
        # TODO: This DataProcessor will later add recdp data process call
        # if model_name.lower() == 'dien':
        #     data_processor = recdp.DIENDataProcessor()
        # else:
        #     data_processor = recdp.GeneralDataProcessor()
        self.model_name = model_name
        pass

class HydroDataLoader:
    def __init__(self, data_processor):
        self.data_processor = data_processor

class TFRecordDataLoader(HydroDataLoader):
    def __init__(self, meta, train_path, valid_path, data_processor):
        super().__init__(data_processor)
        self.meta = meta
        self.train_path = train_path
        self.valid_path = valid_path

class BinaryDataLoader(HydroDataLoader):
    def __init__(self, meta, train_path, valid_path, data_processor):
        super().__init__(data_processor)
        self.meta = meta
        self.train_path = train_path
        self.valid_path = valid_path

class ParquetDataLoader(HydroDataLoader):
    def __init__(self, meta, train_path, valid_path, data_processor):
        super().__init__(data_processor)
        self.meta = meta
        self.train_path = train_path
        self.valid_path = valid_path

class CSVDataLoader(HydroDataLoader):
    def __init__(self, meta, train_path, valid_path, data_processor):
        super().__init__(data_processor)
        self.meta = meta
        self.train_path = train_path
        self.valid_path = valid_path

class ForwardDataLoader(HydroDataLoader):
    def __init__(self, meta_path, train_path, valid_path, data_processor):
        super().__init__(data_processor)
        self.meta_path = meta_path
        self.train_path = train_path
        self.valid_path = valid_path

    def __str__(self):
      tmp = {'meta_path': self.meta_path, 'train_path': self.train_path, 'valid_path': self.valid_path}
      return f"dataloader.hydrodataloader.ForwardDataLoader({tmp})"

    def get_meta(self):
        return self.meta_path

    def get_train(self):
        return self.train_path

    def get_valid(self):
        return self.valid_path

class HydroDataLoaderAdvisor:
    @staticmethod
    def create_data_loader(data_path, model_name):
        dataset_list = list_dir(data_path)
        meta = init_meta()
        meta.update(parse_config(dataset_list['meta']))
        data_processor = DummyDataProcessor(model_name)
        if meta['dataset_format'].lower().startswith("tfrecord"):
            #return TFRecordDataLoader(meta, dataset_list['train'], dataset_list['valid'], data_processor)
            print("data format is tfrecords")
        if meta['dataset_format'].lower().startswith("binary"):
            #return BinaryDataLoader(meta, dataset_list['train'], dataset_list['valid'], data_processor)
            print("data format is binary")
        if meta['dataset_format'].lower().startswith("parquet"):
            #return ParquetDataLoader(meta, dataset_list['train'], dataset_list['valid'], data_processor)
            print("data format is parquet")
        if meta['dataset_format'].lower().startswith("csv"):
            #return CSVDataLoader(meta, dataset_list['train'], dataset_list['valid'], data_processor)
            print("data format is csv")
        
        return ForwardDataLoader(dataset_list['meta'], dataset_list['train'], dataset_list['valid'], data_processor)
