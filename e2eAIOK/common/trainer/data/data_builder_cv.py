import os
import torch
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
# from e2eAIOK.common.trainer.data.data_utils.data_utils import channels_last_collate
from e2eAIOK.common.trainer.data_builder import DataBuilder

class DataBuilderCV(DataBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_dataloader(self):
        """
            create training/evaluation/test dataloader
        """
        ### generate datasets
        if self.dataset_train is None or self.dataset_val is None:
            self.prepare_dataset()

        ### get configurations
        drop_last =  self.cfg.drop_last if "drop_last" in self.cfg else False
        pin_memory = self.cfg.pin_mem if "pin_mem" in self.cfg else False

        ### check whether distributed or enable_ipex
        if ext_dist.my_size > 1:
            num_tasks = ext_dist.dist.get_world_size()
            global_rank = ext_dist.dist.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                self.dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last= drop_last)
            sampler_val = torch.utils.data.DistributedSampler(
                self.dataset_val, num_replicas=num_tasks, rank=global_rank,  shuffle=False)
            if self.dataset_test is not None:
                sampler_test = torch.utils.data.DistributedSampler(
                self.dataset_test, num_replicas=num_tasks, rank=global_rank,  shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(self.dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(self.dataset_test)
        
        # if "enable_ipex" in self.cfg and self.cfg.enable_ipex:
        #     collate_fn = channels_last_collate
        # else:
        collate_fn = None

        ### generate dataloader
        dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train, 
            sampler=sampler_train,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )

        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size=self.cfg.eval_batch_size,
            sampler=sampler_val, 
            num_workers=self.cfg.num_workers,
            drop_last=False,
            collate_fn=collate_fn
        )

        if self.dataset_test is None:
            return dataloader_train, dataloader_val
        else:
            dataloader_test = torch.utils.data.DataLoader(
                self.dataset_test, 
                batch_size=self.cfg.eval_batch_size,
                sampler=sampler_test, 
                num_workers=self.cfg.num_workers,
                drop_last=False,
                collate_fn=collate_fn
            )
        return dataloader_train, dataloader_val, dataloader_test