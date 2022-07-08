# training script to load, build and train from the generated best ViT network architecture 
RANDOM_SEED=`date +%s`

python -u cv/supernet_train.py --data-path ~/data/pytorch_cifar10 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ../conf/denas/cv/supernet_vit/supernet_base.conf --epochs 500 --warmup-epochs 0 \
--output ./ --batch-size 64 2>&1 |tee ViT_train_500_epochs_${RANDOM_SEED}.log

