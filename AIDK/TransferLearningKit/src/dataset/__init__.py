from .cifar100 import get_cifar100_dataset
from .cifar10 import get_cifar10_dataset
from .imagenet import get_imagenet_dataset
from .usps_vs_minist import get_usps_vs_minist_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
import torch

def channels_last_collate(batch):
    """Custom collate fn for channels_last.
    Arguments:
        batch: (tuple) A tuple of images and labels
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim and to channels_last
            2) (list of tensors) labels
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    data = torch.stack(data, 0).to(memory_format=torch.channels_last)
    target = torch.LongTensor(target)
    
    return data, target

def get_dataset(cfg):
    if cfg.dataset.type == "USPS_vs_MNIST":
        train_dataset, valid_dataset, test_dataset, num_classes = get_usps_vs_minist_dataset(cfg=cfg)
    elif cfg.dataset.type == "cifar100":
        train_dataset, valid_dataset, test_dataset, num_classes = get_cifar100_dataset(cfg=cfg)
    elif cfg.dataset.type == "cifar10":
        train_dataset, valid_dataset, test_dataset, num_classes = get_cifar10_dataset(cfg=cfg)
    elif cfg.dataset.type == "imagenet":
        train_dataset, valid_dataset, test_dataset, num_classes = get_imagenet_dataset(cfg=cfg)
    else:
        raise NotImplementedError(cfg.dataset.type)

    num_data = {"train": len(train_dataset),
                "valid": len(valid_dataset),
                "test": len(test_dataset)}

    logging.info("train_dataset:" + str(train_dataset))
    logging.info("validate_dataset:" + str(valid_dataset))
    logging.info("test_dataset:" + str(test_dataset))
    logging.info("num_classes:" + str(num_classes))
    logging.info("num_data:" + str(num_data))
    return train_dataset, valid_dataset, test_dataset, num_classes, num_data

def get_dataloader(cfg,train_dataset, valid_dataset, test_dataset, is_distributed=False,enable_ipex=False):
    train_loader_kwargs = {
        'dataset' : train_dataset,
        'batch_size' : cfg.solver.batch_size,
        'shuffle' : True,
        'num_workers' : cfg.dataset.num_workers,
        'drop_last' : cfg.dataset.data_drop_last,
    }

    validate_loader_kwargs = {
        'dataset' : valid_dataset,
        'batch_size' : cfg.dataset.val.batch_size,
        'shuffle' : False,
        'num_workers' : cfg.dataset.num_workers,
        'drop_last' : False,
    }

    test_loader_kwargs = {
        'dataset' : test_dataset,
        'batch_size' : cfg.dataset.test.batch_size,
        'shuffle' : False,
        'num_workers' : cfg.dataset.num_workers,
        'drop_last' : False,
    }
    if is_distributed:
        train_loader_kwargs['shuffle'] = False # shuffle is conflict with sampler
        train_loader_kwargs['sampler'] = DistributedSampler(train_dataset)
    if enable_ipex:
        train_loader_kwargs['collate_fn'] = channels_last_collate
        validate_loader_kwargs['collate_fn'] = channels_last_collate
        test_loader_kwargs['collate_fn'] = channels_last_collate

    train_loader = DataLoader(**train_loader_kwargs)
    validate_loader = DataLoader(**validate_loader_kwargs)
    test_loader = DataLoader(**test_loader_kwargs)

    return train_loader, validate_loader, test_loader


