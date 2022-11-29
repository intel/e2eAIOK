import sys
import torch
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist

class DataBuilder():
    """
    The basic data builder class for all dataset

    Note:
        You should implement specfic build_dataset function in the build_dataset.py under data folder
    """
    def __init__(self,cfg):
        self.cfg = cfg
    
    def prepare_dataset(self):
        """
            prepare training/evaluation dataset
        """
        raise NotImplementedError("prepare_dataset is abstract.")
    
    def get_dataloader(self):
        """
            create training/evaluation dataloader
        """
        dataset_train, dataset_val = self.prepare_dataset()

        if ext_dist.my_size > 1:
            num_tasks = ext_dist.dist.get_world_size()
            global_rank = ext_dist.dist.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank
            )
            
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank)
        else:
            sampler_val = None
            sampler_train = None

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=sampler_train,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            drop_last=True,
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
