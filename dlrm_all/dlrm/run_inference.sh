set -ex
seed_num=$(date +%s)

export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

# model inference
echo "start model inference"
infer_start=$(date +%s)
cd ./dlrm
/opt/intel/oneapi/intelpython/latest/envs/pytorch_mlperf/bin/python -u ./launch_inference.py --distributed --config-path="../data_processing/config_infer.yaml" --save-path="../data_processing/data_info.txt"  --ncpu_per_proc=1 --nproc_per_node=2 --nnodes=4 --world_size=8 --hostfile ../hosts --master_addr="10.112.228.4" $dlrm_extra_option 2>&1 | tee run_inference_${seed_num}.log
infer_end=$(date +%s)
infer_spend=$(( infer_end - infer_start ))
echo inference time is ${infer_spend} seconds.