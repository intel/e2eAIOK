import math
from collections.abc import Iterable
from typing import List, Union

import modin.pandas as mpd
import numpy as np
import pandas
import ray
import torch
from torch.utils.data import Dataset, DistributedSampler

# we use 4 bytes for block size, this means each block can contain
# 4294967296 records
BLOCK_SIZE_BIT = 32


class ModinDataset(Dataset):
    def __init__(self,
                 df: mpd.DataFrame = None,
                 feature_columns: List[str] = None,
                 feature_shapes: List[Union[List, int]] = None,
                 feature_types: List[torch.dtype] = None,
                 label_column: str = None,
                 label_type: torch.dtype = None):
        self._data = self._get_df_partitions(df)
        self._feature_columns = feature_columns
        self._feature_shapes = feature_shapes
        self._feature_types = feature_types
        self._label_column = label_column
        self._label_type = label_type

        self._feature_tensor = None
        self._label_tensor = None
        self._resolved_data = None
        self._previous_partition_index = -1

        self._check_and_convert()

    def _get_df_partitions(self, df: mpd.DataFrame):
        partitions = df._query_compiler._modin_frame._partitions
        result = []
        for partition in partitions:
            result.append(([inner.oid for inner in partition], partition[0].length()))
        return result

    def _check_and_convert(self):
        # convert to list for convenience
        if not isinstance(self._feature_columns, List):
            self._feature_columns = [self._feature_columns]

        if self._feature_shapes:
            if not isinstance(self._feature_shapes, list):
                self._feature_shapes = [self._feature_shapes]

            assert len(self._feature_columns) == len(self._feature_shapes), \
                "The feature_shapes size must match the feature_columns"
            for i in range(len(self._feature_shapes)):
                if not isinstance(self._feature_shapes[i], Iterable):
                    self._feature_shapes[i] = [self._feature_shapes[i]]

        if self._feature_types:
            if not isinstance(self._feature_types, list):
                self._feature_types = [self._feature_types]

            assert len(self._feature_columns) == len(self._feature_types), \
                "The feature_types size must match the feature_columns"
            for i in range(len(self._feature_types)):
                assert all(isinstance(dtype, torch.dtype) for dtype in self._feature_types), \
                    "All value in feature_types should be torch.dtype instance"

        if not self._feature_shapes and self._feature_types:
            assert all(dtype == self._feature_types[0] for dtype in self._feature_types), \
                "All dtypes should be same when feature_shapes doesn't provide"

        if not self._feature_types:
            self._feature_types = [torch.float] * len(self._feature_columns)

        if not self._label_type:
            self._label_type = torch.float

    def _convert_to_tensor(self, df):
        if self._feature_shapes:
            tensors = []
            for col, shape, dtype in zip(self._feature_columns, self._feature_shapes,
                                         self._feature_types):
                column = df[col].values
                if column.dtype == np.object:
                    if isinstance(column[0], np.ndarray):
                        column = np.stack(column)
                    elif isinstance(column[0], (list, tuple)):
                        column = list(column)
                    else:
                        raise Exception(
                            f"Column {col}'s type: {type(column[0])} is not supported. It must "
                            "be numpy built in type or numpy object of (ndarray, list, tuple)")

                t = torch.as_tensor(column, dtype=dtype)
                if shape != [0]:
                    t = t.view(*(-1, *shape))
                tensors.append(t)
            self._feature_tensor = tensors
        else:
            feature_columns = (self._feature_columns if
                               len(self._feature_columns) > 1 else self._feature_columns[0])
            feature_df = df[feature_columns].values
            t = torch.as_tensor(feature_df, dtype=self._feature_types[0])
            self._feature_tensor = [t]

        label_df = df[self._label_column].values
        self._label_tensor = torch.as_tensor(label_df, dtype=self._label_type)

    def _get_next(self, index):
        label = self._label_tensor[index]
        features = [tensor[index] for tensor in self._feature_tensor]
        return (*features, label)

    def _resolve_with_indices(self, indices):
        if self._resolved_data is None:
            resolved = [None] * len(self._data)
            for i in indices:
                dfs = ray.get(self._data[i][0])
                assert all([isinstance(df, pandas.DataFrame) for df in dfs])
                if len(dfs) == 0:
                    resolved[i] = dfs[0]
                else:
                    resolved[i] = pandas.concat(dfs, axis=1)
            self._resolved_data = resolved

    def partition_sizes(self):
        return [d[1] for d in self._data]

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            # TODO: add support
            raise Exception("Multiple processes loading is not supported")

        global BLOCK_SIZE_BIT
        partition_index = index >> BLOCK_SIZE_BIT
        partition_inner_index = (partition_index << BLOCK_SIZE_BIT) ^ index
        if partition_index != self._previous_partition_index:
            self._previous_partition_index = partition_index
            df = self._resolved_data[partition_index]
            self._convert_to_tensor(df)
        return self._get_next(partition_inner_index)

    def __len__(self):
        return sum([d[1] for d in self._data])


class ModinDatasetSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, init_lazy=True):
        assert isinstance(dataset, ModinDataset)
        self._args = (dataset, num_replicas, rank, shuffle)
        self._inited = False

        self._block_indices = None
        self._selected_indices = None

        if not init_lazy:
            self._init_lazy()

    def _init_lazy(self):
        """
        This is a workaround because of ray sgd call initialize the data creator before of
        setup distributed components.
        """
        if not self._inited:
            super(ModinDatasetSampler, self).__init__(*self._args)
            self._split_blocks()
            self._inited = True

    def _split_blocks(self):
        num_blocks = int(math.ceil(len(self.dataset.partition_sizes()) * 1.0 / self.num_replicas))
        total_block_size = num_blocks * self.num_replicas
        g = torch.Generator()
        g.manual_seed(0)
        if self.shuffle:
            total_indices = torch.randperm(len(self.dataset.partition_sizes()), generator=g).tolist()
        else:
            total_indices = list(range(len(self.dataset.partition_sizes())))
        # add extra samples to make it evenly divisible
        while len(total_indices) != total_block_size:
            total_indices += total_indices[:(total_block_size - len(total_indices))]
        assert len(total_indices) == total_block_size, f"{len(total_indices)}, {total_block_size}"

        indices = total_indices[self.rank: total_block_size: self.num_replicas]
        assert len(indices) == num_blocks

        def select(i, current_size, selected) -> int:
            block_size = self.dataset.partition_sizes()[i]
            tmp = current_size + block_size
            if tmp < self.num_samples:
                selected.append((i, block_size))
                current_size = tmp
            elif tmp >= self.num_samples:
                selected.append((i, (self.num_samples - current_size)))
                current_size = self.num_samples
            return current_size

        total_size = 0
        selected_indices = []
        for i in indices:
            total_size = select(i, total_size, selected_indices)
            if total_size == self.num_samples:
                break

        step = 1
        while total_size < self.num_samples:
            index = total_indices[(self.rank + step) % len(total_indices)]
            total_size = select(index, total_size, selected_indices)
            step += self.num_replicas

        assert total_size == self.num_samples

        block_indices = []
        packed_selected_indices = []
        global BLOCK_SIZE_BIT
        for index, size in selected_indices:
            block_indices.append(index)
            # we use 4 Bytes for the block inner index
            packed_selected_indices.append([((index << BLOCK_SIZE_BIT) | i) for i in range(size)])
        self._block_indices = block_indices
        self._selected_indices = packed_selected_indices

    @property
    def block_indices(self):
        return self._block_indices

    def __iter__(self):
        self._init_lazy()
        self.dataset._resolve_with_indices(self._block_indices)
        # deterministically shuffle based on epoch
        np.random.seed(self.epoch)
        block_indices = list(range(len(self._block_indices)))
        if self.shuffle:
            np.random.shuffle(block_indices)

        indices = []
        for index in block_indices:
            tmp = self._selected_indices[index]
            tmp = np.copy(tmp)
            if self.shuffle:
                np.random.shuffle(tmp)
            indices += tmp.tolist()

        return iter(indices)

    def __len__(self):
        self._init_lazy()
        return self.num_samples
