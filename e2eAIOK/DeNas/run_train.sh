# training script to load, build and train from the generated best CV/NLP/ASR network architecture 
RANDOM_SEED=`date +%s`

#ASR single process training
# python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/train.py \
#     --domain asr --conf /home/vmagent/app/e2eaiok/conf/denas/asr/e2eaiok_denas_train_asr.conf --random_seed 74443 2>&1 | tee ASR_training_${RANDOM_SEED}.log

#CNN Single node two process traing
#python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1  /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/train.py \
#      --domain cnn --conf /home/vmagent/app/e2eaiok/conf/denas/cv/e2eaiok_denas_train_cnn.conf 2>&1 | tee CNN_training_${RANDOM_SEED}.log

#BERT single process training
python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=1 --nnodes=1 /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/train.py \
   --domain bert --conf /home/vmagent/app/e2eaiok/conf/denas/nlp/e2eaiok_denas_train_bert.conf 2>&1 | tee BERT_training_${RANDOM_SEED}.log

#VIT Single node two processes traing
#python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1  /home/vmagent/app/e2eaiok/e2eAIOK/DeNas/train.py \
#      --domain vit --conf /home/vmagent/app/e2eaiok/conf/denas/cv/e2eaiok_denas_train_vit.conf 2>&1 | tee VIT_training_${RANDOM_SEED}.log
