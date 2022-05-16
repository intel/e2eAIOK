# Bert for Intel® Architecture

Bidirectional Encoder Representations from Transformers (BERT) is a method of pre-training language representations, meaning that it trains a general-purpose "language understanding" model on large text corpus, and then use that model for downstream NLP tasks, like question and answering.

## Enabling Bert training on CPU

Refer to [Intel AI FP32 Bert training Guide](https://github.com/IntelAI/models/blob/master/benchmarks/language_modeling/tensorflow/bert_large/training/fp32/README.md) and generally follow steps below.

## Quick start scripts

### SQuAD 1.1 dataset prepare
To run on SQuAD, you will first need to download the dataset. The SQuAD v1.1 can be found here:
- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [evaluate-v1.1.json](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Download these to some directory ````$SQUAD_DIR````.

### Pre-trained models
Download and extract one of BERT large pretrained models from [Google BERT repository](https://github.com/google-research/bert#pre-trained-models) to ````$CHECKPOINT_DIR````. 

### Single node

````
# Setup environment
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
export CHECKPOINT_DIR=<path to the pretrained bert model directory>
export DATASET_DIR=<path to the dataset being used>
export OUTPUT_DIR=<directory where checkpoints and log files will be saved>
mkdir -p $OUTPUT_DIR
# Run scripts
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/<script name>.sh
````

### Multiple nodes multiple process

Edit "hosts" to configure the ip and interfaces:

````
<IP_ADDRESS>:<NETWORK_INTERFACE>
````

Run scipts

````
# Setup environment
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu_icc --force
export CHECKPOINT_DIR=<path to the pretrained bert model directory>
export DATASET_DIR=<path to the dataset being used>
export OUTPUT_DIR=<directory where checkpoints and log files will be saved>

export HOROVOD_CPU_OPERATIONS=CCL
export MASTER_ADDR="*"
export CCL_WORKER_COUNT=*
export CCL_WORKER_AFFINITY="*"
export HOROVOD_THREAD_AFFINITY="*"
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="*"

export MPI_NUM_PROCESSES=<number of processes>
export NOINSTALL="True"

if [ -d "${OUTPUT_DIR}" ]; then
  rm -rf $OUTPUT_DIR
fi
mkdir -p $OUTPUT_DIR

/opt/intel/oneapi/intelpython/latest/envs/tensorflow-2.5.0/bin/python \
/home/vmagent/app/mnt/tianyi/frameworks.bigdata.bluewhale/modelzoo/bert/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision=fp32 \
  --mode=training \
  --framework=tensorflow \
  --batch-size=24 \
  --output-dir $OUTPUT_DIR \
  --mpi_num_processes=$MPI_NUM_PROCESSES \
  --num-intra-threads 36 \
  --num-inter-threads 2 \
  -- train_option=SQuAD \
     vocab_file=$CHECKPOINT_DIR/vocab.txt \
     config_file=$CHECKPOINT_DIR/bert_config_new.json \
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
| tee ${OUTPUT_DIR}/bert_2nodes_test.log
````

### Evaluation

Evaluation scipts

````
/opt/intel/oneapi/intelpython/latest/envs/tensorflow-2.5.0/bin/python ./quickstart/language_modeling/tensorflow/bert_large/training/cpu/fp32/evaluate-v1.1.py /home/vmagent/app/hydro.ai/modelzoo/bert/models/out/dev-v1.1.json ${OUTPUT_DIR}/predictions.json --output_file ${OUTPUT_DIR}/best_res.txt
````
