import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import intel_extension_for_pytorch as ipex
import torch_ccl
import torch_optimizer as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(400, 500)

    def forward(self, input):
        return self.linear(input)


if __name__ == "__main__":
    
    os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
  
    # Initialize the process group with ccl backend
    dist.init_process_group(backend='ccl')
    model = Model()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Lamb(model.parameters(), lr=0.001)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    loss_fn = nn.MSELoss()
    if dist.get_world_size() > 1:
        model=DDP(model)

    for i in range(6000):
        input = torch.randn(2, 400)
        labels = torch.randn(2, 500)

        # forward
        res = model(input)
        # print(res)
        L=loss_fn(res, labels)

        optimizer.zero_grad()
        # backward
        L.backward()

        # update
        optimizer.step()