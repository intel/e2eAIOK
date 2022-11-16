import torch

from trainer.data import cv_build_datasets
from trainer.data import nlp_build_datasets

class DataBuilder():
    """
    The basic data builder class for all dataset

    Note:
        You should implement specfic build_dataset function in the build_dataset.py under data folder
    """
    def __init__(self,args):
        self.args = args
    
    '''
    build the dataloader for train and validation dataset
    '''
    def get_data(self, ext_dist):
        args = self.args
        if args.data_set in ["CIFAR10","CIFAR100"]:
            dataset_train, _ = cv_build_datasets.build_dataset(is_train=True, args=args)
            dataset_val, _ = cv_build_datasets.build_dataset(is_train=False, args=args)
        elif args.data_set in ["SQuADv1.1"]:
            dataset_train, train_examples, train_dataset, labels = nlp_build_datasets.build_dataset(is_train=True, args=args)
            dataset_val, val_examples, val_dataset, val_features, tokenizer = nlp_build_datasets.build_dataset(is_train=False, args=args)
        else:
            pass

        if ext_dist.my_size > 1:
            num_tasks = ext_dist.dist.get_world_size()
            global_rank = ext_dist.dist.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last= True
            )
            
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            if not args.data_set in ["SQuADv1.1"]:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:   
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.eval_batch_size,
            sampler=sampler_val, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )
        if args.data_set in ["CIFAR10","CIFAR100"]:
            return data_loader_train, data_loader_val
        elif args.data_set in ["SQuADv1.1"]:
            return data_loader_train, data_loader_val, train_examples, val_examples, val_dataset, val_features, tokenizer
        
             