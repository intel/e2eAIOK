import horovod.tensorflow.keras as hvd
import tensorflow_transform as tft

from data.outbrain.dataloader import train_input_fn, eval_input_fn, BinDataset

def create_dataset(args):
    num_gpus = hvd.size()
    gpu_id = hvd.rank()
    steps_per_epoch = args.training_set_size / args.global_batch_size
    if args.dataset_format == "TFRecords":
        feature_spec = tft.TFTransformOutput(
            args.transformed_metadata_path
        ).transformed_feature_spec()
    
        train_spec_input_fn = train_input_fn(
            num_gpus=num_gpus,
            id=gpu_id,
            filepath_pattern=args.train_data_pattern,
            feature_spec=feature_spec,
            records_batch_size=args.global_batch_size // num_gpus // args.prebatch_size,
        )
    
        eval_spec_input_fn = eval_input_fn(
            num_gpus=num_gpus,
            id=gpu_id,
            filepath_pattern=args.eval_data_pattern,
            feature_spec=feature_spec,
            records_batch_size=args.eval_batch_size // num_gpus // args.prebatch_size
        )
        return train_spec_input_fn, eval_spec_input_fn, steps_per_epoch
    elif args.dataset_format == "binary":
        print('binary')
        # train_dataset = BinDataset(args.train_dataset_path, metadata, args.global_batch_size)
        # test_dataset = BinDataset(args.eval_dataset_path, metadata, args.eval_batch_size)
        # steps_per_epoch = train_dataset.__len__
    else:
        raise RuntimeError(f'Dataset format {args.dataset_format} is not supported!')