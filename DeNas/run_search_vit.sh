#This is the script for the ViT network architecture search, have three level size(base, small and tiny) supernet can be used.
RANDOM_SEED=`date +%s`

python -u evolution.py --gp --change_qk --relative_position --model_type "transformer" --dist-eval --cfg ../conf/denas/cv/supernet_vit/supernet_base.conf --data-set CIFAR 2>&1 |tee ViT_search_10_epochs_${RANDOM_SEED}.log