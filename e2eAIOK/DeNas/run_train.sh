# training script to load, build and train from the generated best ViT and CNN network architecture 
RANDOM_SEED=`date +%s`
export OMP_NUM_THREADS=18

export KMP_BLOCKTIME=1

export KMP_AFFINITY="granularity=fine,compact,1,0"


#CNN single process training
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 /home/vmagent/app/e2eAIOK/e2eAIOK/common/trainer/train.py --domain cnn --conf /home/vmagent/app/e2eAIOK/conf/denas/cv/e2eaiok_denas_train_cnn.conf 2>&1 | tee CNN_training_${RANDOM_SEED}.log

