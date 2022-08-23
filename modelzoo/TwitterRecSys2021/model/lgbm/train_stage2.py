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

path = "/mnt/sdb/xinyao/2optimize/nvidia2021/3mergeall/recsys2021-intel-opt"
data_path = f"{path}/data"  ## train and valid data path
model_save_path = f"{path}/models"  ## model saving path

DEBUG = False

if __name__ == "__main__":
    ######## Load data
    train = pd.read_parquet(f'{data_path}/stage2_train_pred.parquet')
    valid = pd.read_parquet(f'{data_path}/stage2_valid_pred.parquet')
    if DEBUG:
        model_save_path = f"{model_save_path}/test/"
        train = train[:10000]
        valid = valid[:10000]
    print(train.shape)
    print(valid.shape)

    for col in valid.columns:
        if valid[col].dtype=='bool':
            train[col] = train[col].astype('int8')
            valid[col] = valid[col].astype('int8')

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list = []
    feature_list.append(stage2_reply_features)
    feature_list.append(stage2_retweet_features)
    feature_list.append(stage2_comment_features)
    feature_list.append(stage2_like_features)
    for i in range(4):
        print(len(feature_list[i]))

    ######## Train and predict
    params_rely = {
            'num_leaves': 320,
            'learning_rate': 0.06185900351327167,
            'max_depth': 148,
            'lambda_l1': 63.55517147274657,
            'lambda_l2': 26,
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.7938198090322777,
            'bagging_fraction': 0.9855437776164211,
            'bagging_freq': 8,
            'metric':'average_precision',
            'max_bin': 275,
            'min_data_in_leaf': 901,
            'early_stopping_rounds':20,
            'num_iterations': 292
            }
    params_retweet = {
            'num_leaves': 319, 
            'learning_rate': 0.060048057001888866, 
            'max_depth': 48,
            'lambda_l1': 43.90089811295544, 
            'lambda_l2': 58, 
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.7208637394038915, 
            'bagging_fraction': 0.9590119474604384, 
            'bagging_freq': 7, 
            'metric':'average_precision',
            'max_bin': 213, 
            'min_data_in_leaf': 587,
            'early_stopping_rounds':20,
            'num_iterations': 500
            }
    params_comment = {
            'num_leaves': 258, 
            'learning_rate': 0.06244145690797791, 
            'max_depth': 34, 
            'lambda_l1': 24.695347381231425, 
            'lambda_l2': 53, 
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.602797161693379, 
            'bagging_fraction': 0.9758513757550573, 
            'bagging_freq': 8,
            'metric':'average_precision',
            'max_bin': 321, 
            'min_data_in_leaf': 1632,
            'early_stopping_rounds':20,
            'num_iterations': 161
            }
    params_like = {
            'num_leaves': 485, 
            'learning_rate': 0.07084796360984123, 
            'max_depth': 123, 
            'lambda_l1': 20.120115298848653, 
            'lambda_l2': 56, 
            'colsample_bynode': 0.8,
            'colsample_bytree': 0.661998023399019, 
            'bagging_fraction': 0.9967692455311306, 
            'bagging_freq': 6, 
            'metric':'average_precision',
            'max_bin': 306, 
            'min_data_in_leaf': 782,
            'early_stopping_rounds':20,
            'num_iterations': 212
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
        model = lgb.train(params_tmp,train_set=trainD,valid_sets=validationD,categorical_feature=set([]))
        model.save_model(filename = f"{model_save_path}/lgbm_{name}_stage2.txt")
        
        print('Predicting...')
        oof[:, numlabel] += model.predict(X_valid)

        del trainD,validationD
        print('Took %.1f seconds'%(time.time()-start))
        
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

