# coding=utf-8
# Copyright (c) 2022, Intel. and its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import builtins
import numpy as np
import torch
from packaging import version
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
try:
    if torch.__version__ < "1.12.0":
        import torch_ccl
    else:
        import oneccl_bindings_for_pytorch as torch_ccl
except ImportError as e:
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
    global my_size
    global my_rank
    global my_local_rank
    global my_local_size

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'])
    if backend == '' and num_mpi_ranks > 1:
        if torch_ccl and env2int(['CCL_WORKER_COUNT']) > 0:
            backend = 'ccl'
        elif dist.is_mpi_available():
            backend = 'mpi'
        else:
            print("WARNING: MPI multi-process launch detected but PyTorch MPI backend not available.")
            backend = 'gloo'
    if backend != '':
        #guess Rank and size
        if rank == -1:
            rank = env2int(['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'RANK'], 0)
        if size == -1:
            size = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1)
        if not os.environ.get('RANK', None) and rank != -1: os.environ['RANK'] = str(rank)
        if not os.environ.get('WORLD_SIZE', None) and size != -1: os.environ['WORLD_SIZE'] = str(size)
        if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
        if not os.environ.get('MASTER_ADDR', None):
            local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
            if local_size != size and backend != 'mpi':
                print("Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default")
                print("If this run hangs, try exporting rank 0's hostname as MASTER_ADDR")
            os.environ['MASTER_ADDR'] = '127.0.0.1'
    if size > 1:
        print(F"world_size:{size},rank:{rank}")
        print(F"extend_distributed backend:{backend}")
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
        my_local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if my_rank == 0: print("Running on %d ranks using %s backend" % (my_size, backend))
        if backend == 'ccl':
            print(f"Using CCL_ATL_TRANSPORT={os.environ.get('CCL_ATL_TRANSPORT', '(default)')}")
            print(f"Using CCL_ATL_SHM={os.environ.get('CCL_ATL_SHM', '(default)')}")
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1
