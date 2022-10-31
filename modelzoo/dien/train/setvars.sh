import os
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['HOROVOD_CPU_OPERATIONS'] = 'CCL'
os.environ['CCL_WORKER_COUNT'] = '1'
os.environ['CCL_WORKER_AFFINITY'] = '0, 32'
os.environ['HOROVOD_THREAD_AFFINITY'] = '1, 33'
os.environ['I_MPI_PIN_PROCESSOR_EXCLUDE_LIST'] = '0, 1, 32, 33'
