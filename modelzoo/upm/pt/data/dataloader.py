from functools import partial
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import pt.extend_distributed as ext_dist

class BinDataset(Dataset):
    """Binary version of criteo dataset."""

    def __init__(self, data_file,
                 batch_size=1, bytes_per_feature=4):
        # dataset
        self.tar_fea = 1   # single target
        self.den_fea = 13  # 13 dense  features
        self.spa_fea = 26  # 26 sparse features
        self.tad_fea = self.tar_fea + self.den_fea
        self.tot_fea = self.tad_fea + self.spa_fea

        self.batch_size = batch_size
        self.bytes_per_batch = (bytes_per_feature * self.tot_fea * batch_size)

        data_file_size = os.path.getsize(data_file)
        self.num_batches = math.ceil(data_file_size / self.bytes_per_batch)

        bytes_per_sample = bytes_per_feature * self.tot_fea
        self.num_samples = data_file_size // bytes_per_sample

        if ext_dist.my_size > 1:
            self.bytes_per_rank = self.bytes_per_batch // ext_dist.my_size
        else:
            self.bytes_per_rank = self.bytes_per_batch

        if ext_dist.my_size > 1 and self.num_batches * self.bytes_per_batch > data_file_size:
            last_batch = (data_file_size % self.bytes_per_batch) // bytes_per_sample
            self.bytes_last_batch = last_batch // ext_dist.my_size * bytes_per_sample
        else:
            self.bytes_last_batch = self.bytes_per_rank

        if self.bytes_last_batch == 0:
            self.num_batches = self.num_batches - 1
            self.bytes_last_batch = self.bytes_per_rank

        print('data file:', data_file, 'number of batches:', self.num_batches)
        self.file = open(data_file, 'rb')

        # hardcoded for now
        self.m_den = 13

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        my_rank = ext_dist.dist.get_rank() if ext_dist.my_size > 1 else 0
        rank_size = self.bytes_last_batch if idx == (self.num_batches - 1) else self.bytes_per_rank 
        self.file.seek(idx * self.bytes_per_batch + rank_size * my_rank, 0)
        raw_data = self.file.read(rank_size)
        array = np.frombuffer(raw_data, dtype=np.int32)
        tensor = torch.from_numpy(array).view((-1, self.tot_fea))

        # x_int_batch=tensor[:, 1:14]
        # x_cat_batch=tensor[:, 14:]
        x_batch = tensor[:, 1:]
        y_batch=tensor[:, 0]
        x_batch = torch.log(torch.tensor(x_batch, dtype=torch.float) + 1)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

        return x_batch, y_batch