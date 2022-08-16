import os
import builtins
import numpy as np
import torch
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
try:
    import torch_ccl
except ImportError as e:
    print(e)
    torch_ccl = False

my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1

def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default


def init_distributed(rank = -1, size = -1, backend='gloo'):
    global my_rank
    global my_size
    global my_local_rank
    global my_local_size
    #guess Rank and size
    if rank == -1:
        rank = env2int(['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'RANK'], 0)
    if size == -1:
        size = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1)
    if not os.environ.get('RANK', None) and rank != -1: os.environ['RANK'] = str(rank)
    if not os.environ.get('WORLD_SIZE', None) and size != -1: os.environ['WORLD_SIZE'] = str(size)
    if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
    if not os.environ.get('MASTER_ADDR', None): os.environ['MASTER_ADDR'] = '127.0.0.1'
    if size > 1:
        print(F"world_size:{size},rank:{rank}")
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
        my_local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if backend == 'ccl':
            print("Using CCL_ATL_TRANSPORT=%s" % os.environ.get('CCL_ATL_TRANSPORT', '(default)'))
            print("Using CCL_ATL_SHM=%s" % os.environ.get('CCL_ATL_SHM', '(default)'))
