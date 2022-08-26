# Notes
This ls DLRM for ipex v0.2. You can use docker `xuechendi/oneapi-aikit:legacy_hydro.ai` to evaluate DLRM performance.

# Launch training with SDA
`python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm --conf conf/e2eaiok_defaults_dlrm_example.conf`

# For DLRM
1. Use run_spark.sh to run spark ETL job, the data was saved on hdfs folder of /dlrm/input
   Test and Validation data is under folder of /dlrm/output/test and validation
   Then concat the generated small binary files under the folder of train, test, validation to train_data.bin, test_data.bin and val_data.bin 
   And scp these three bin files under the same folder such as /mnt/DP_disk6/binary_dataset
2. Install the dlrm conda env follow [Intel MLPerf DLRM Benchmark](https://github.com/mlperf/training_results_v0.7/tree/master/Intel/benchmarks/dlrm/1-node-4s-cpx-pytorch)
3.Use dlrm/run_and_time.sh to run DLRM training

------
# For Model Compression on DLRM
1. Use run_compress.sh to run model compress on DLRM, the configuration was saved as yaml in ./model_compress/
2. Install the model compression running env:
   
   (1) Install the dlrm conda env as above;

   (2) Install the distiller follow the [link](https://teams.microsoft.com/l/file/52DEC602-9C2D-44CA-BC06-41D4850204B3?tenantId=46c98d88-e344-4ed4-8496-4ed7712e255d&fileType=docx&objectUrl=https%3A%2F%2Fintel.sharepoint.com%2Fsites%2FIAGS-SSP-SMPS-DPO-AnalyticsStorage%2FShared%20Documents%2FGeneral%2FNew%20Projects%2F2021_07%20Model%20Compression%2FDistiller%20Guide.docx&baseUrl=https%3A%2F%2Fintel.sharepoint.com%2Fsites%2FIAGS-SSP-SMPS-DPO-AnalyticsStorage&serviceName=teams&threadId=19:2fcb2b3c8b824e7ca5216b10d5624574@thread.skype&groupId=69adf55a-c293-4328-b4e3-bf0e344435e4);

