set -x

cd /home/vmagent/app/e2eaiok/e2eAIOK/deltatuner
pip install -e .

cd /home/vmagent/app/e2eaiok

DATA_PATH="/home/vmagent/app/data"
LOG_PATH=$DATA_PATH"/dtuner_test/log"
MODEL_SAVE_PATH=$DATA_PATH"/dtuner_test/models"

mkdir -p $LOG_PATH $MODEL_SAVE_PATH

# fine-tune with ssf
# mosaicml/mpt-7b gpt2 EleutherAI/gpt-j-6b bigscience/bloom-560m facebook/opt-125m EleutherAI/gpt-neo-125m tiiuae/falcon-7b
model_name_list="EleutherAI/gpt-j-6b bigscience/bloom-560m facebook/opt-125m EleutherAI/gpt-neo-125m tiiuae/falcon-7b"
for model_name in $model_name_list
do
    model_name_or_path=${model_name}
    short_model_name=`echo $model_name | cut -d/ -f2`
    model_save_path=${MODEL_SAVE_PATH}"/"${short_model_name}"_ssf"
    log_save_path=$LOG_PATH"/"${short_model_name}"_ssf-1epoch.log"
    python example/instruction_tuning_pipeline/finetune_clm.py \
        --model_name_or_path $model_name_or_path \
        --train_file "$DATA_PATH/alpaca_data.json" \
        --dataset_concatenation \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --validation_split_percentage 30 \
        --do_train \
        --do_eval \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --logging_steps 100 \
        --save_total_limit 1 \
        --log_level info \
        --save_strategy epoch \
        --trust_remote_code True \
        --output_dir $model_save_path \
        --algo ssf \
        --denas False \
        --debugs \
        --no_cuda \
        2>&1 | tee $log_save_path
    # rm -rf ~/.cache/huggingface/hub
done
