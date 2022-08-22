from .cifar100 import get_cifar100_dataloaders
from .imagenet import get_imagenet_dataloaders
from .usps_vs_minist import get_usps_vs_minist_dataloaders


def get_dataset(cfg,is_distributed=False):
    if cfg.DATASET.TYPE == "USPS_vs_MNIST":
        train_loader, val_loader, test_loader, num_data = get_usps_vs_minist_dataloaders(
            path=cfg.DATASET.PATH,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            is_distributed=is_distributed
        )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cifar100":
        train_loader, test_loader, num_data = get_cifar100_dataloaders(
            path=cfg.DATASET.PATH,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        val_loader = test_loader
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        train_loader, test_loader, num_data = get_imagenet_dataloaders(
            path=cfg.DATASET.PATH,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        val_loader = test_loader
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, test_loader, num_data, num_classes
