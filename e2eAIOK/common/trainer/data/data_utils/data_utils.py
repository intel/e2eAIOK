import torch

#optimize for ipex
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