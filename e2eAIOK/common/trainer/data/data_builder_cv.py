import os
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist


from e2eAIOK.common.trainer.data_builder import DataBuilder

class DataBuilderCV(DataBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_dataloader(self):
        """
            create training/evaluation dataloader
        """
        dataset_train, dataset_val = self.prepare_dataset()

        if ext_dist.my_size > 1:
            num_tasks = ext_dist.dist.get_world_size()
            global_rank = ext_dist.dist.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last= True
            )
            
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,  shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=sampler_train,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_mem
        )

        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=self.cfg.eval_batch_size,
            sampler=sampler_val, 
            num_workers=self.cfg.num_workers,
            shuffle=False,
            drop_last=False
        )
        
        return dataloader_train, dataloader_val