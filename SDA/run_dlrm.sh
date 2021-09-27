source ~/.local/env/setvars.sh

seed_num=$(date +%s)
#export CCL_ATL_SHM=0
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

python main.py --ppn=2 --train_path=/mnt/DP_disk1/binary_dataset/train_data.bin --eval_path=/mnt/DP_disk1/binary_dataset/val_data.bin --dataset_meta_path=/mnt/DP_disk1/binary_dataset/day_fea_count.npz --hosts=10.1.0.132 --day-feature-count=/mnt/DP_disk1/binary_dataset/day_fea_count.npz --python_executable=~/sw/miniconda3/envs/dlrm/bin/python  --metric=AUC --metric_threshold=0.8025