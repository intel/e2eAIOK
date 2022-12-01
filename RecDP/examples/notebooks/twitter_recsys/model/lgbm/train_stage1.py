#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, average_precision_score
from features import *
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
very_start = time.time()

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

path = "/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/model/"
data_path = "/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/datapre_stage1/"  ## train and valid data path
model_save_path = f"{path}/models"  ## model saving path
pred_save_path = f"{path}/result"   ## prediction result saving path

num_iterations = 1000
DEBUG = False

if __name__ == "__main__":
    ######## Load data
    if DEBUG:
        num_iterations = 10
        model_save_path = f"{model_save_path}/test/"
        pred_save_path = f"{pred_save_path}/test/"
        train = pd.read_parquet(glob.glob(f'{data_path}/stage1_train/*.parquet')[0])[:10000]
        valid = pd.read_parquet(glob.glob(f'{data_path}/stage1_valid/*.parquet')[0])[:10000]
    else:
        train = pd.read_parquet(f'{data_path}/stage1_train')
        valid = pd.read_parquet(f'{data_path}/stage1_valid')
    print(train.shape)
    print(valid.shape)

    for col in valid.columns:
        if valid[col].dtype=='bool':
            train[col] = train[col].astype('int8')
            valid[col] = valid[col].astype('int8')

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list = []
    feature_list.append(stage1_reply_features)
    feature_list.append(stage1_retweet_features)
    feature_list.append(stage1_comment_features)
    feature_list.append(stage1_like_features)
    for i in range(4):
        print(len(feature_list[i]))

    ######## Train and predict
    params_rely = {
            'num_leaves': 191,
            'learning_rate': 0.08206038394079608,
            'max_depth':77,
            'lambda_l1': 2.0976251913389987,
            'lambda_l2': 15,
            'colsample_bynode': 0.8,
            'colsample_bytree':  0.5833237141883345,
            'bagging_fraction': 0.8344415420350142,
            'bagging_freq': 9,
            'metric':'average_precision',
            'max_bin': 194,
            'min_data_in_leaf': 1568,
            'early_stopping_rounds':20,
            }
    params_retweet = {
            'num_leaves': 254, 
            'learning_rate': 0.08998445387572881, 
            'max_depth': 22,
            'lambda_l1': 43.292096932514255, 
            'lambda_l2': 34, 
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.22758004664181256, 
            'bagging_fraction': 0.8438528982539478, 
            'bagging_freq': 5, 
            'metric':'average_precision',
            'max_bin': 151, 
            'min_data_in_leaf': 1148,
            'early_stopping_rounds':20,
            }
    params_comment = {
            'num_leaves': 295, 
            'learning_rate': 0.10046413580022834, 
            'max_depth': 35, 
            'lambda_l1': 10.520357464572335, 
            'lambda_l2': 20, 
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.18117676445729075, 
            'bagging_fraction': 0.6725305772267222, 
            'bagging_freq': 10,
            'metric':'average_precision',
            'max_bin': 275, 
            'min_data_in_leaf': 1880,
            'early_stopping_rounds':20,
            }
    params_like = {
            'num_leaves': 300, 
            'learning_rate': 0.09256233543023894, 
            'max_depth': 100, 
            'lambda_l1': 39.804767697094405, 
            'lambda_l2': 53, 
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.2790877465865267, 
            'bagging_fraction': 0.7721156929402684, 
            'bagging_freq': 10, 
            'metric':'average_precision',
            'max_bin': 294, 
            'min_data_in_leaf': 1098,
            'early_stopping_rounds':20,
            }
    paramss = [params_rely,params_retweet,params_comment,params_like]

    oof = np.zeros((len(valid),len(label_names)))
    for numlabel in range(4):
        start = time.time()
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)

        X_train = train[feature_list[numlabel]]
        X_valid = valid[feature_list[numlabel]]
        trainD = lgb.Dataset(data=X_train,label=train[name],categorical_feature=set([]))
        validationD = lgb.Dataset(data=X_valid, label=valid[name], categorical_feature=set([]))

        print("Training.....")
        params_tmp = paramss[numlabel]
        params_tmp['num_iterations'] = num_iterations
        params_tmp['seed'] = 1
        model = lgb.train(params_tmp,train_set=trainD,valid_sets=validationD,categorical_feature=set([]))
        model.save_model(filename = f"{model_save_path}/lgbm_{name}_stage1.txt")
        
        print('Predicting...')
        oof[:, numlabel] += model.predict(X_valid)

        del trainD,validationD
        print('Took %.1f seconds'%(time.time()-start))

    ######## Merge prediction to data and save
    for i in range(4):
        valid[f"pred_{label_names[i]}"] = oof[:,i]
    
    valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}",f"pred_{label_names[1]}",f"pred_{label_names[2]}",f"pred_{label_names[3]}"]].to_csv(f"{pred_save_path}/lgbm_pred_stage1.csv",index=0)
    
    ######## Evaluate the performance
    txts = ''
    sumap = 0
    sumrce = 0
    for i in range(4):
        ap = compute_AP(oof[:,i],valid[label_names[i]].values)
        rce = compute_rce_fast(oof[:,i],valid[label_names[i]].values)
        txt = f"{label_names[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt)

        txts += "%.4f" % ap + ' '
        txts += "%.4f" % rce + ' '
        sumap += ap
        sumrce += rce
    print(txts)
    print("AVG AP: ", sumap/4.)
    print("AVG RCE: ", sumrce/4.)
    
    print('This notebook took %.1f seconds'%(time.time()-very_start))

