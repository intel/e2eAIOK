# Intel Optimized E2E Solutions for DLRM

# Launch training with SDA
`python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm --conf conf/e2eaiok_defaults_dlrm_example.conf`

# For Spark data processing
   ```bash
   Use run_spark.sh to run spark ETL job, the data was saved on hdfs folder of /dlrm/input
   Test and Validation data is under folder of /dlrm/output/test and validation
   Then concat the generated small binary files under the folder of train, test, validation to train_data.bin, test_data.bin and val_data.bin 
   And scp these three bin files under the same folder such as /mnt/DP_disk6/binary_dataset
   ```

# How to reproduce our optimized DLRM training performance

## Prepare
1. Environment
    * torch 1.5.0a0+b58f89b
    * torch-ccl 1.0
    * torch-ipex 0.1
    * oneCCL 2021.1-beta07-1
2. Prepare 
    * Build the Intel mlperf dlrm environemnt reference the [link](https://github.com/mlperf/training_results_v0.7/tree/master/Intel/benchmarks/dlrm/1-node-4s-cpx-pytorch)
    * Or pull the docker image `xuechendi/oneapi-aikit:legacy_hydro.ai`
    *  Prepare the numpy processed Criteo binary dataset, which includes: day_fea_count.npz, test_data.bin, train_data.bin,  val_data.bin

## DLRM Trainining
1. Add the necessary setting in the run_and_time_launch.sh:DATA_PATH,master_address...
1. Run the run_and_time_launch.sh

## DLRM Inference
1. If you want to run dlrm inference, need add '--save-model' argument when training
2. After training completed, change the 'dlrm_s_pytorch.py' to dlrm_s_pytorch_inference.py in the run_and_time_launch.sh
3. Run the run_and_time_launch.sh


# For Model Compression on DLRM
1. Use run_compress.sh to run model compress on DLRM, the configuration was saved as yaml in ./model_compress/
2. Install the model compression running env:
   ```bash
   (1) Install the dlrm conda env as above;

   (2) Install the distiller
   ```