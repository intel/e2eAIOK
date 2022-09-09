# training script to load, build and train from the generated best Bert network architecture 
#RANDOM_SEED=`date +%s`

#/opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/bin/python ./nlp/supernet_train.py --data_dir /home/vmagent/app/dataset/SQuAD \
#--model /home/vmagent/app/dataset/bert-base-uncased/ \
#--task_name squad1 \
#--arches_file best_model_structure.txt \
#--no_cuda 2>&1 |tee Bert_train_4_epochs_${RANDOM_SEED}.log

# training script to load, build and train from the generated best ViT network architecture 
RANDOM_SEED=`date +%s`
export OMP_NUM_THREADS=18
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

python -m intel_extension_for_pytorch.cpu.launch --distributed --nproc_per_node=2 --nnodes=1 ./trainer/train.py --domain bert --conf /home/vmagent/app/hydro.ai/conf/denas/nlp/aidk_denas_train_bert.conf 2>&1 | tee BERT_distributed_training_${RANDOM_SEED}.log