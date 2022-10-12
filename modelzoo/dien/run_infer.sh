NUM_INSTANCES=$1
NUM_INSTANCES_END=$2
OUT_SHIFT=$3
batch=128
CORES_PER_INST=1
NUM_INSTANCES_FIRST=$((${NUM_INSTANCES}+1))
mkdir -p results
for (( i=${NUM_INSTANCES_FIRST} ; i < ${NUM_INSTANCES_END}; i++ ))
do
    numactl -C$((${i} - ${NUM_INSTANCES})) --localalloc /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python train/ai-matrix/script/train.py --mode=test --advanced --slice_id=0 --batch_size=$batch --num-inter-threads=1 --num-intra-threads=1 --train_path /home/vmagent/app/dataset/amazon_reviews/train/local_train_splitByUser --test_path /home/vmagent/app/dataset/amazon_reviews/valid/local_test_splitByUser --meta_path /home/vmagent/app/dataset/amazon_reviews/meta.yaml 2>results/result_infer_${batch}_$((${i}+${OUT_SHIFT}))_err.txt >results/result_infer_${batch}_$((${i}+${OUT_SHIFT})).txt &
done
i=${NUM_INSTANCES}
numactl -C$((${i} - ${NUM_INSTANCES})) --localalloc /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python train/ai-matrix/script/train.py --mode=test --advanced --slice_id=0 --batch_size=$batch --num-inter-threads=1 --num-intra-threads=1 --train_path /home/vmagent/app/dataset/amazon_reviews/train/local_train_splitByUser --test_path /home/vmagent/app/dataset/amazon_reviews/valid/local_test_splitByUser --meta_path /home/vmagent/app/dataset/amazon_reviews/meta.yaml 2>results/result_infer_${batch}_$((${i}+${OUT_SHIFT}))_err.txt >results/result_infer_${batch}_$((${i}+${OUT_SHIFT})).txt
