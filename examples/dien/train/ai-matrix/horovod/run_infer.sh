NUM_INSTANCES=$1
NUM_INSTANCES_END=$2
OUT_SHIFT=$3
batch=128
CORES_PER_INST=1
mkdir -p results
for (( i=${NUM_INSTANCES} ; i < ${NUM_INSTANCES_END}; i++ ))
do
    numactl -C$((${i} - ${NUM_INSTANCES})) --localalloc python script/train.py --mode=test --advanced --slice_id=${i} --batch_size=$batch --num-inter-threads=1 --num-intra-threads=1 2>results/result_infer_${batch}_$((${i}+${OUT_SHIFT}))_err.txt >results/result_infer_${batch}_$((${i}+${OUT_SHIFT})).txt &
done
