#################################################### step 1: get input args ####################################################
# cpu, cuda:0
device=$1
pretrained_model_path=$2
peft_model_path=$3
output_path_prefix=$4


#################################################### step 2: formulate model_args ####################################################
# formulate model_args
model_args='use_accelerate=True,trust_remote_code=True'
if [ -z $peft_model_path ]; then
    peft_model_path=""
else
    peft_model_path=",peft="$peft_model_path
fi
model_args=$model_args',pretrained='${pretrained_model_path}${peft_model_path}

dtype=float16
if [ "$device" == "cpu" ]; then
    dtype=bfloat16
fi
model_args=$model_args','$dtype


#################################################### step 3: begin evaluating ####################################################
cd /home/vmagent/app/lm-evaluation-harness

# evaluate on hellaswag
python main.py \
	--device $device \
	--model hf-causal-experimental \
	--model_args $model_args \
	--tasks hellaswag  --num_fewshot 10 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-hellaswag

# evaluate on mmlu
python main.py \
	--device $device \
	--model hf-causal-experimental \
	--model_args $model_args \
	--tasks hendrycksTest*  --num_fewshot 5 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-mmlu

# evaluate on arc_challenge
python main.py \
	--device $device \
	--model hf-causal-experimental \
	--model_args $model_args \
	--tasks arc_challenge  --num_fewshot 25 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-arc_challenge

# evaluate on truthfulqa_mc
python main.py \
	--device $device \
	--model hf-causal-experimental \
	--model_args $model_args \
	--tasks truthfulqa_mc  --num_fewshot 0 \
	--batch_size auto --max_batch_size 32 \
	--output_path $output_path_prefix-truthqa


#################################################### step 4: calculate average acc of MMLU task ####################################################

cd /home/vmagent/app/deltatuner/tests/evaluate_benchmark
bash cal_avg_acc.sh $output_path_prefix-mmlu





