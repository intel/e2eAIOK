# coding=utf-8
# Copyright (c) 2022, Intel Corporation

# MIT License
# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# =======================================================================================
# MIT license
# =======================================================================================
# - [Swin Transformer](https://github.com/microsoft/swin-transformer)
# - [CLIP](https://github.com/openai/CLIP)

# =======================================================================================
# Apache license 2.0
# =======================================================================================
# - [LeViT](https://github.com/facebookresearch/LeViT)
# - [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

# =======================================================================================
# BSD-style license
# =======================================================================================
# - [PyTorch](https://github.com/pytorch/pytorch)

import os
import multiprocessing
import torch
import torch.distributed as dist
import numpy as np
from .aug_random import AugRandomContext
from .manager import TxtManager


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, logits_path, num_classes, topk, write):
        super().__init__()
        self.dataset = dataset
        self.logits_path = logits_path
        self.epoch = multiprocessing.Value('i', 0)
        self.topk = topk
        self.num_classes = num_classes
        self.write_mode = write
        self.keys = self._get_keys()
        self._manager = (None, None)

    def __getitem__(self, index: int):
        if self.write_mode:
            return self.__getitem_for_write(index)
        return self.__getitem_for_read(index)

    def __getitem_for_write(self, index: int):
        # get an augmentation seed
        key = self.keys[index]
        seed = np.int32(np.random.randint(0, 1 << 31))
        with AugRandomContext(seed=int(seed)):
            item = self.dataset[index]
        return (item, (key, seed))

    def __getitem_for_read(self, index: int):
        key = self.keys[index]
        values = self._get_saved_logits(key)
        seed = values[0]
        with AugRandomContext(seed=seed):
            item = self.dataset[index]

        if self.topk == 0:
            return ((item[0], (values[1], np.int32(seed))), item[1])
        elif self.topk > 0:
            return ((item[0], (values[1], values[2], np.int32(seed))), item[1])
        
    def _get_saved_logits(self, key: str):
        manager = self.get_manager()
        bstr: bytes = manager.read(key)
        # parse the augmentation seed
        seed = int(np.frombuffer(bstr[:4], dtype=np.int32))
        # parse the logits index and value
        # copy logits_index and logits_value to avoid warning of written flag from PyTorch
        bstr = bstr[4:]
        if self.topk == 0:
            logits_value = np.frombuffer(
                bstr[:self.num_classes * 4], dtype=np.float32).copy()
            return seed, logits_value
        elif self.topk > 0:
            logits_value = np.frombuffer(
                bstr[:self.topk * 2], dtype=np.float16).copy()
            bstr = bstr[self.topk * 2:]
            logits_index = np.frombuffer(
                bstr[:self.topk * 2], dtype=np.int16).copy()
            return seed, logits_value, logits_index

    def _build_manager(self, logits_path: str):
        if self.topk == 0: # save all logits
            # num_classes * 4 bytes for logits + 4 bytes for seed
            item_size = self.num_classes * 4 + 4
        elif self.topk > 0: # save topk logits
            # topk * [idx, value] * 2 bytes  for logits + 4 bytes for seed
            item_size = self.topk * 2 * 2 + 4
        rank = get_rank()
        return TxtManager(logits_path, item_size, rank)

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        self._manager = (None, None)

    def get_manager(self):
        epoch = self.epoch.value
        if epoch != self._manager[0]:
            if self.topk ==0 :
                logits_path = os.path.join(
                    self.logits_path, f'logits_epoch{self.epoch.value}')
            elif self.topk > 0:
                logits_path = os.path.join(
                    self.logits_path, f'logits_top{self.topk}_epoch{self.epoch.value}')
            self._manager = (epoch, self._build_manager(logits_path))
        return self._manager[1]

    def __len__(self):
        return len(self.dataset)

    def _get_keys(self):
        if hasattr(self.dataset, 'get_keys'):
            keys = self.dataset.get_keys()
            if self.write_mode:
                # we only check key unique in the write mode
                assert len(keys) == len(set(keys)), 'keys must be unique'
            return keys
        return [str(i) for i in range(len(self))]
