from pt.data.dataloader import BinDataset

def create_dataset(args):
    train_dataset = BinDataset(args.train_data_pattern, args.global_batch_size)
    eval_dataset = BinDataset(args.eval_data_pattern, args.eval_batch_size)
    steps_per_epoch = len(train_dataset)
    return train_dataset, eval_dataset, steps_per_epoch