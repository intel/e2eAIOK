# dataset layout
#  tree ../dataset/amazon_reviews/
# ../dataset/amazon_reviews/
# ├── cat_voc.pkl
# ├── meta.yaml
# ├── mid_voc.pkl
# ├── train
# │   └── local_train_splitByUser
# ├── uid_voc.pkl
# └── valid
#     └── local_test_splitByUser

# Use e2eaiok API
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/amazon_reviews" --model_name dien

# Use SDA API
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/amazon_reviews" --model_name dien

# run from modelzoo
/opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/python /home/vmagent/app/e2eaiok/modelzoo/dien/train/ai-matrix/script/train.py --train_path ../dataset/amazon_reviews/train/local_train_splitByUser --test_path ../dataset/amazon_reviews/valid/local_test_splitByUser --meta_path ../dataset/amazon_reviews/meta.yaml --saved_path /home/vmagent/app/e2eaiok/result/dien/20211208_005646/4e88981a4459b8c1b028bc574f05a532 --num-intra-threads 32 --num-inter-threads 4 --mode train --embedding_device cpu --model DIEN --slice_id 0 --advanced true --data_type FP32 --seed 3 --batch_size 512
