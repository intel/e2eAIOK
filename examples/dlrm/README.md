1. Use run_spark.sh to run spark ETL job, the data was saved on hdfs folder of /dlrm/input
   Test and Validation data is under folder of /dlrm/output/test and validation
   Then concat the generated small binary files under the folder of train, test, validation to train_data.bin, test_data.bin and val_data.bin 
   And scp these three bin files under the same folder such as /mnt/DP_disk6/binary_dataset
2. Install the dlrm conda env follow [Intel MLPerf DLRM Benchmark](https://github.com/mlperf/training_results_v0.7/tree/master/Intel/benchmarks/dlrm/1-node-4s-cpx-pytorch)
3.Use dlrm/run_and_time.sh to run DLRM training

