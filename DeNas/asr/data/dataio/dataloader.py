import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import logging
from torch.utils.data import DistributedSampler
import torch.distributed as dist

from asr.data.dataio.batch import PaddedBatch
from asr.data.dataio.dataset import DynamicItemDataset
from asr.data.dataio.sampler import ReproducibleRandomSampler, DistributedSamplerWrapper
from asr.utils.checkpoints import (
    mark_as_saver,
    mark_as_loader,
)


logger = logging.getLogger(__name__)

def get_dataloader(dataset, batch_size, distributed):
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(not distributed), collate_fn=PaddedBatch, drop_last=True, sampler=sampler)
    return dataloader

def make_dataloader(dataset, stage, dist, **loader_kwargs):
    if stage == 'train':
        loader_kwargs = train_loader_specifics(dataset, dist, loader_kwargs)

    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        loader_kwargs["collate_fn"] = PaddedBatch
    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        del loader_kwargs["shuffle"]
    # Create the loader
    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(dataset, **loader_kwargs)
    else:
        dataloader = SaveableDataLoader(dataset, **loader_kwargs)
    return dataloader


def train_loader_specifics(dataset, distributed_launch, loader_kwargs):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    sampler = loader_kwargs.get("sampler", None)
    shuffle = loader_kwargs.get("shuffle", False)
    if shuffle and not distributed_launch:
        if sampler is not None:
            raise ValueError(
                "Cannot specify both shuffle=True"
                "and a sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        train_sampler = sampler
        loader_kwargs["sampler"] = train_sampler
        # Delete the shuffle flag, since you cannot specify both a sampler and
        # shuffling:
        del loader_kwargs["shuffle"]
    # Possibly make a DistributedSampler or a wrapper for some other sampler
    if distributed_launch and not isinstance(dataset, IterableDataset):
        drop_last = loader_kwargs.get("drop_last", False)
        # num_replicas arg is equal to world_size
        # and retrieved automatically within
        # DistributedSampler obj.
        if sampler is not None:
            train_sampler = DistributedSamplerWrapper(
                sampler,
                rank=rank,
                drop_last=drop_last,
                shuffle=shuffle,
            )
            # with DistributedSamplerWrapper, one must disable shuffling for dataloader
            loader_kwargs["shuffle"] = False
            loader_kwargs["sampler"] = train_sampler
        elif loader_kwargs.get("batch_sampler") is None:
            # no sampler and batch-sampler
            train_sampler = DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, drop_last=drop_last
            )
            # with DistributedSamplerWrapper, one must disable shuffling for dataloader
            loader_kwargs["shuffle"] = False
            loader_kwargs["sampler"] = train_sampler
        else:  # batch_sampler was specified
            train_sampler = DistributedSamplerWrapper(
                loader_kwargs.get("batch_sampler", None),
                rank=rank,
                shuffle=False,
            )
            loader_kwargs["batch_sampler"] = train_sampler
    elif distributed_launch and isinstance(dataset, IterableDataset):
        logger.warning(
            "Cannot automatically solve distributed sampling "
            "for IterableDataset."
        )
    return loader_kwargs

class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speechbrain_recovery_skip_to = None
        self._speechbrain_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        self._speechbrain_iterator = iterator
        return iterator

    @mark_as_saver
    def _speechbrain_save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speechbrain_iterator is None:
            to_save = None
        else:
            to_save = self._speechbrain_iterator._num_yielded
        with open(path, "w") as fo:
            fo.write(str(to_save))

    @mark_as_loader
    def _speechbrain_load(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speechbrain_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return
        if end_of_epoch:
            # Don't load at end of epoch, as we actually want to start a fresh
            # epoch iteration next.
            return
        with open(path) as fi:
            saved = fi.read()
            if saved == str(None):
                # Saved at a point where e.g. an iterator did not yet exist.
                return
            else:
                self._speechbrain_recovery_skip_to = int(saved)
