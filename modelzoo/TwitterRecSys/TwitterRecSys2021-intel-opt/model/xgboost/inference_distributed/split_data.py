import os, time, gc, sys, glob
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
very_start = time.time()

path = "/mnt/sdb/xinyao/2optimize/nvidia2021/3mergeall/recsys2021-intel-opt"
data_path = f"{path}/data"

distributed_nodes = 4

if __name__ == "__main__":
    ######## Load data
    t1 = time.time()
    test = pd.read_parquet(f'{data_path}/stage12_test')  
    print(test.shape)
    print(f"load data took {time.time() - t1} s")

    ######## split data
    t1 = time.time()
    indexs = [i for i in range(distributed_nodes)]
    step = int(len(test)/distributed_nodes)
    tests = []
    for i in range(distributed_nodes):
        if i<distributed_nodes-1:
            tests.append(test[i*step:(i+1)*step])
        else:
            tests.append(test[i*step:])
        
    for i in range(len(tests)):
        tests[i].to_parquet(f"{path}/data/stage12_test_{i}.parquet")

    print(f"totally took {time.time() -very_start} s")