# training script to load, build and train from the generated best Bert network architecture 
RANDOM_SEED=`date +%s`

/opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/bin/python ./nlp/supernet_train.py --data_dir /home/vmagent/app/dataset/SQuAD \
--model /home/vmagent/app/dataset/bert-base-uncased/ \
--task_name squad1 \
--arches_file best_model_structure.txt \
--no_cuda 2>&1 |tee Bert_train_4_epochs_${RANDOM_SEED}.log