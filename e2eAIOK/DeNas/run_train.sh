# training script to load, build and train from the generated best CV/NLP/ASR network architecture 
RANDOM_SEED=`date +%s`

#CNN single process training
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/train.py \
    --domain cnn --conf /home/vmagent/app/e2eaiok/conf/denas/cv/e2eaiok_denas_train_cnn.conf 2>&1 | tee CNN_training_${RANDOM_SEED}.log

#ASR single process training
# python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/train.py \
#     --domain asr --conf /home/vmagent/app/e2eaiok/conf/denas/asr/e2eaiok_denas_train_asr.conf --random_seed 74443 2>&1 | tee ASR_training_${RANDOM_SEED}.log
