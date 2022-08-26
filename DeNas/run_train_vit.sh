# training script to load, build and train from the generated best ViT network architecture 
RANDOM_SEED=`date +%s`
export OMP_NUM_THREADS=18

export KMP_BLOCKTIME=1

export KMP_AFFINITY="granularity=fine,compact,1,0"

#Single process training
# python -u ./trainer/train.py --domain vit --conf ../conf/denas/cv/aidk_denas_train_vit.config 2>&1 | tee ViT_train_${RANDOM_SEED}.log
#Distributed training
python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2  ./trainer/train.py --domain vit --conf ../conf/denas/cv/aidk_denas_train_vit.config --dist-train 2>&1 | tee ViT_distributed_training_${RANDOM_SEED}.log


