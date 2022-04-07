import horovod.tensorflow as hvd
import os

def test_docker_horovod():
    os.environ['HOROVOD_CPU_OPERATIONS'] = "CCL"
    os.environ['CCL_ATL_TRANSPORT'] = "mpi"

    hvd.init()

    print(f"hvd size is {hvd.size()}, hvd rank is {hvd.rank()}")
    assert hvd.size() == 2