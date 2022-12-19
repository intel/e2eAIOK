import torch.nn.parallel
import torch.distributed as dist
import torch_ccl
import os

def test_docker_torchccl():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

    backend = 'ccl'
    dist.init_process_group(backend)
    my_rank = dist.get_rank()
    my_size = dist.get_world_size()
    print("my rank = %d  my size = %d" % (my_rank, my_size))

    x = torch.ones([2, 2])
    y = torch.ones([4, 4])
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        for _ in range(10):
            dist.all_reduce(x)
            dist.all_reduce(y)
    dist.barrier()
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))

    assert my_size == 2