#This is the script for the ViT network architecture search, have three level size(base, small and tiny) supernet can be used.
RANDOM_SEED=`date +%s`

python -u search.py --domain vit --conf ../conf/denas/cv/aidk_denas_vit.conf 2>&1 | tee ViT_search_10_epochs_${RANDOM_SEED}.log