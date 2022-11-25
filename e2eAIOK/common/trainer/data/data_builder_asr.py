import torch
from e2eAIOK.common.trainer.data_builder import DataBuilder
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.DeNas.asr.data.dataio.dataset import dataio_prepare
from e2eAIOK.DeNas.asr.data.dataio.batch import PaddedBatch

class DataBuilderASR(DataBuilder):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.tokenizer = tokenizer

    def get_dataloader(self):
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
            collate_fn=PaddedBatch,
            shuffle=(ext_dist.my_size <= 1),
            drop_last=True,
        )

        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=self.cfg.eval_batch_size,
            sampler=sampler_val, 
            num_workers=self.cfg.num_workers,
            collate_fn=PaddedBatch,
            shuffle=False,
            drop_last=False
        )
        
        return dataloader_train, dataloader_val