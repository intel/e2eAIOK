source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
seed_num=$(date "+%Y%m%d-%H%M%S")
export MODEL_DIR=$(pwd)
export PARENT_DIR=$(dirname $(dirname $(dirname $MODEL_DIR)))
export CHECKPOINT_DIR=$MODEL_DIR/pre-trained-model/bert-large-uncased/wwm_uncased_L-24_H-1024_A-16
export OUTPUT_DIR=$MODEL_DIR/models/out/test/$(seed_num)
export DATASET_DIR=$PARENT_DIR/dataset/SQuAD
export HOROVOD_CPU_OPERATIONS=CCL
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

export MASTER_ADDR="10.1.0.157"
export CCL_WORKER_COUNT=2
export CCL_WORKER_AFFINITY="16,17,34,35"
export HOROVOD_THREAD_AFFINITY="53,71"
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="16,17,34,35,52,53,70,71"
#export OMP_PROC_BIND=true
#export KMP_BLOCKTIME=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
#export MASTER_PORT="12345"
#export MPI_NUM_PROCESSES=2

#One process (Quick Start)
#bash /home/vmagent/app/hydro.ai/modelzoo/bert/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/fp32_squad_training.sh

#Multiprocess
export MPI_NUM_PROCESSES=2
export NOINSTALL="True"
/opt/intel/oneapi/intelpython/latest/envs/tensorflow-2.5.0/bin/python \
/home/vmagent/app/hydro.ai/modelzoo/bert/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision=fp32 \
  --mode=training \
  --framework=tensorflow \
  --batch-size=24 \
  --output-dir $OUTPUT_DIR \
  --host_file=$MODEL_DIR/hosts \
  --mpi_num_processes=$MPI_NUM_PROCESSES \
  --num-intra-threads 36 \
  --num-inter-threads 2 \
  -- train_option=SQuAD \
     vocab_file=$CHECKPOINT_DIR/vocab.txt \
     config_file=$CHECKPOINT_DIR/bert_config.json \
     init_checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
     do_train=True \
     train_file=$DATASET_DIR/train-v1.1.json \
     do_predict=True \
     predict_file=$DATASET_DIR/dev-v1.1.json \
     learning_rate=3e-5 \
     num_train_epochs=2 \
     max_seq_length=384 \
     doc_stride=128 \
     optimized_softmax=True \
     experimental_gelu=False \
     do_lower_case=True \
     num_hidden_layers=24 \
     attention_probs_dropout_prob=0.1 \
     hidden_dropout_prob=0.1 \
     warmup_proportion=0.1 \
| tee ${OUTPUT_DIR}/bert_2nodes_test.log
#
#&&

#Evaluation
#/opt/intel/oneapi/intelpython/latest/envs/tensorflow-2.5.0/bin/python /home/vmagent/app/hydro.ai/modelzoo/bert/quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/evaluate-v1.1.py /home/vmagent/app/hydro.ai/modelzoo/bert/models/out/dev-v1.1.json ${OUTPUT_DIR}/predictions.json --output_file ${OUTPUT_DIR}/best_res.txt