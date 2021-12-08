import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from features import *
from multiprocessing import Pool
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
very_start = time.time()

path = "/mnt/sdb/xinyao/2optimize/nvidia2021/3mergeall/recsys2021-intel-opt"
data_path = f"{path}/data"
model_save_path = f"{path}/models"  ## model saving path
pred_save_path = f"{path}/result"   ## prediction result saving path

distributed_node_index = 0 #0, 1, 2, 3
DEBUG = False
stage1_thread = 16
stage2_thread = 24

if __name__ == "__main__":
    ######## Load data
    t1 = time.time()
    test = pd.read_parquet(f'{data_path}/stage12_test_{distributed_node_index}.parquet')
    if DEBUG:
        test = test[:10000]
    print(test.shape)
    print(f"load data took {time.time() - t1} s")

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    
    stage1_feature_list = {}
    stage1_feature_list["reply_timestamp"] = stage1_reply_features
    stage1_feature_list["retweet_timestamp"] = stage1_retweet_features
    stage1_feature_list["retweet_with_comment_timestamp"] = stage1_comment_features
    stage1_feature_list["like_timestamp"] = stage1_like_features

    stage2_feature_list = {}
    stage2_feature_list["reply_timestamp"] = stage2_reply_features
    stage2_feature_list["retweet_timestamp"] = stage2_retweet_features
    stage2_feature_list["retweet_with_comment_timestamp"] = stage2_comment_features
    stage2_feature_list["like_timestamp"] = stage2_like_features

    ######## Stage1 prediction
    t1 = time.time()
    # Split data into multi-instance
    indexs = [i for i in range(stage1_thread)]
    step = int(len(test)/stage1_thread)
    tests = []
    for i in range(stage1_thread):
        if i<stage1_thread-1:
            tests.append(test[i*step:(i+1)*step])
        else:
            tests.append(test[i*step:])

    # Predict with multiprocessing
    def prediction1(index):
        start = time.time()
        test1 = tests[index]
        oof = np.zeros((len(test1),len(label_names)))

        for numlabel in range(4):
            name = label_names[numlabel]
            X_test = test1[stage1_feature_list[name]]

            model = lgb.Booster(model_file=f"{model_save_path}/lgbm_{name}_stage1.txt")
            oof[:, numlabel] += model.predict(X_test)
        print(f"{index} predict took {time.time() - start} s")
        return oof

    pool = Pool(stage1_thread)
    preds1 = pool.map(prediction1,indexs)
    pool.close()
    pool.join()
    gc.collect()
    
    # Merge result
    predss1 = np.concatenate(preds1)
    for i in range(4):
        test[f"pred_{label_names[i]}"] = predss1[:,i]
    print(test.shape)
    print(f"stage1 took {time.time() -t1} s")

    ######## Stage2 prediction
    t1 = time.time()
    # Split data into multi-instance
    indexs = [i for i in range(stage2_thread)]
    step = int(len(test)/stage2_thread)
    tests = []
    for i in range(stage2_thread):
        if i<stage2_thread-1:
            tests.append(test[i*step:(i+1)*step])
        else:
            tests.append(test[i*step:])

    # Predict with multiprocessing
    def prediction2(index):
        start = time.time()
        test1 = tests[index]
        oof = np.zeros((len(test1),len(label_names)))

        for numlabel in range(4):
            name = label_names[numlabel]
            X_test = test1[stage2_feature_list[name]]

            model = lgb.Booster(model_file=f"{model_save_path}/lgbm_{name}_stage2.txt")
            oof[:, numlabel] += model.predict(X_test)
        print(f"{index} predict took {time.time() - start} s")
        return oof

    pool = Pool(stage2_thread)
    preds2 = pool.map(prediction2,indexs)
    pool.close()
    pool.join()
    gc.collect()

    # Merge result
    predss2 = np.concatenate(preds2)
    for i in range(4):
        test[f"pred2_{label_names[i]}"] = predss2[:,i]
    print(test.shape)
    print(f"stage2 took {time.time() -t1} s")

    ######## Save to csv
    t1 = time.time()
    test[['engaging_user_id', 'tweet_id', f"pred2_{label_names[0]}",f"pred2_{label_names[1]}",f"pred2_{label_names[2]}",f"pred2_{label_names[3]}"]].to_csv(f"{pred_save_path}/lgbm_inference_{distributed_node_index}.csv",index=0)
    print(f"save took {time.time() -t1} s")

    print(f"totally took {time.time() -very_start} s")