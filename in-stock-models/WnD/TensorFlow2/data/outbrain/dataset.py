import horovod.tensorflow.keras as hvd
import tensorflow_transform as tft

from data.outbrain.dataloader import TFRecordsDataset, BinDataset

def create_dataset(args, features):
    num_gpus = hvd.size()
    gpu_id = hvd.rank()
    if features.dataset_format == "TFRecords":
        if features.tfrecords_meta_path is None:
            raise RuntimeError('Missing TFRecords dataset meta path! Please provide meta path in dataset meta file')
        if features.prebatch_size is None:
            raise RuntimeError('Please define prebatch size for TFRecords dataset in dataset meta file')

        steps_per_epoch = features.training_set_size / args.global_batch_size
        dataloader = TFRecordsDataset(features.tfrecords_meta_path, features.features_keys, features.label)
        train_dataset = dataloader.input_fn(
            args.train_data_pattern, 
            records_batch_size=args.global_batch_size // num_gpus // features.prebatch_size,
            shuffle=True,
            num_gpus=num_gpus,
            id=gpu_id)
    
        eval_dataset = dataloader.input_fn(
            args.eval_data_pattern,
            records_batch_size=args.eval_batch_size // num_gpus // features.prebatch_size,
            shuffle=False,
            num_gpus=num_gpus,
            id=gpu_id)
        return train_dataset, eval_dataset, steps_per_epoch
    elif features.dataset_format == "binary":
        train_dataset = BinDataset(args.train_data_pattern, features, args.global_batch_size)
        eval_dataset = BinDataset(args.eval_data_pattern, features, args.eval_batch_size)
        steps_per_epoch = len(train_dataset)
        return train_dataset, eval_dataset, steps_per_epoch
    else:
        raise RuntimeError(f'Dataset format {features.dataset_format} is not supported!')