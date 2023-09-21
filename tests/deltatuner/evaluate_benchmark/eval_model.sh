#################################################### option 1: eval llama2 on gpu ####################################################
device=cuda:0
pretrained_model_path=/home/vmagent/app/data/Llama-2-7b-hf
peft_model_path=''
output_path_prefix=/home/vmagent/app/data/llm-eval/llama2-7b-gpu

bash eval_scripts.sh \
	$device \
	$pretrained_model_path \
	$peft_model_path \
	$output_path_prefix


#################################################### option 2: eval mpt on gpu ####################################################
device=cuda:0
pretrained_model_path=/home/vmagent/app/data/mpt-7b
peft_model_path=''
output_path_prefix=/home/vmagent/app/data/llm-eval/mpt-7b-gpu

bash eval_scripts.sh \
	$device \
	$pretrained_model_path \
	$peft_model_path \
	$output_path_prefix

#################################################### option 3: eval llama2 on cpu ####################################################
device=cpu
pretrained_model_path=/home/vmagent/app/data/Llama-2-7b-hf
peft_model_path=''
output_path_prefix=/home/vmagent/app/data/llm-eval/llama2-7b-cpu

bash eval_scripts.sh \
	$device \
	$pretrained_model_path \
	$peft_model_path \
	$output_path_prefix

#################################################### option 4: eval mpt on cpu ####################################################
device=cpu
pretrained_model_path=/home/vmagent/app/data/mpt-7b
peft_model_path=''
output_path_prefix=/home/vmagent/app/data/llm-eval/mpt-7b-cpu

bash eval_scripts.sh \
	$device \
	$pretrained_model_path \
	$peft_model_path \
	$output_path_prefix


