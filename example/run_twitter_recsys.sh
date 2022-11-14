# prepare data
# tree ../dataset/TwitterRecSys2021Dataset/
# ../dataset/TwitterRecSys2021Dataset/
# ├── stage1
# │   └── metadata.yaml
# │   └── train
# │   └── valid
# ├── stage2
# │   └── train
# │   └── valid
# ├── stage2_pred
# │   └── metadata.yaml
# │   └── train
# │   └── valid

##### Use e2eaiok API
#train stage1 model for four targets
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_reply.conf  --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_retweet.conf --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_retweet_with_comment.conf --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_like.conf --no_model_cache
#Merge best prediction from stage1
python modelzoo/TwitterRecSys2021/model_e2eaiok/xgboost/train_merge12.py --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset --reply_pred_path ${reply_best_model_save_path}/xgboost_pred_stage1_reply.csv --retweet_pred_path ${retweet_best_model_save_path}/xgboost_pred_stage1_retweet.csv --retweet_with_comment_pred_path ${retweet_with_comment_best_model_save_path}/xgboost_pred_stage1_retweet_with_comment.csv --like_pred_path ${like_best_model_save_path}/xgboost_pred_stage1_like.csv 
#train stage2 model for four targets
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_reply.conf  --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_retweet.conf --no_model_cache 
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_retweet_with_comment.conf --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_like.conf --no_model_cache

##### Use SDA API
#train stage1 model for four targets
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_reply.conf  --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_retweet.conf --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_retweet_with_comment.conf --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage1 --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage1_like.conf --no_model_cache
#Merge best prediction from stage1
python modelzoo/TwitterRecSys2021/model_e2eaiok/xgboost/train_merge12.py --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset --reply_pred_path ${reply_best_model_save_path}/xgboost_pred_stage1_reply.csv --retweet_pred_path ${retweet_best_model_save_path}/xgboost_pred_stage1_retweet.csv --retweet_with_comment_pred_path ${retweet_with_comment_best_model_save_path}/xgboost_pred_stage1_retweet_with_comment.csv --like_pred_path ${like_best_model_save_path}/xgboost_pred_stage1_like.csv 
#train stage2 model for four targets
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_reply.conf  --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_retweet.conf --no_model_cache 
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_retweet_with_comment.conf --no_model_cache
SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --model_name twitter_recsys --data_path /home/vmagent/app/dataset/TwitterRecSys2021Dataset/stage2_pred --conf conf/TwitterRecSys/e2eaiok_defaults_twitter_recsys_stage2_like.conf --no_model_cache
