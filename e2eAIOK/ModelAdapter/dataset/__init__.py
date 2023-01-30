from .cifar import DataBuilderCIFARMA
from e2eAIOK.common.trainer.data.cv.data_builder_uspsvsminist import DataBuilderUSPSMinist
import logging
import torch

# def channels_last_collate_distiller_train(batch):
#     # distiller : ((data, (logits, seed)), target)
#     data = [item[0][0] for item in batch]
#     data = torch.stack(data, 0).to(memory_format=torch.channels_last)

#     logits = [torch.from_numpy(item[0][1][0]) for item in batch]
#     logits = torch.stack(logits,0)

#     seed = [item[0][1][1] for item in batch]
#     seed = torch.Tensor(seed).type(torch.int32)

#     target = [item[1] for item in batch]
#     target = torch.LongTensor(target)

#     return [data, [logits, seed]], target

def createDataBuilder(cfg):
    if cfg.data_set == "USPS_vs_MNIST":
        dataBuilder = DataBuilderUSPSMinist(cfg=cfg)
        num_classes = 10
    elif cfg.data_set == "cifar10":
        dataBuilder = DataBuilderCIFARMA(cfg=cfg)
        num_classes = 10
    elif cfg.data_set == "cifar100":
        dataBuilder = DataBuilderCIFARMA(cfg=cfg)
        num_classes = 100
    else:
        raise NotImplementedError(f"Not support dataset {cfg.data_set}")
    return dataBuilder, num_classes


