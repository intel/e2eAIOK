import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_datasets import build_dataset
import numpy as np
import torch


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
        if args.data_set in ["CIFAR10","CIFAR100","IMGNET"]:
            dataset_train, _ = build_dataset(is_train=True, args=args)
            dataset_val, _ = build_dataset(is_train=False, args=args)
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
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=int(2 * args.batch_size),
            sampler=sampler_val, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )
        return data_loader_train, data_loader_val
        


            
            
            
        
            