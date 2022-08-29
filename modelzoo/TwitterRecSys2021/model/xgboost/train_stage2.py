#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import xgboost as xgb
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

path = "/path/to/processed/data"
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
    xgb_parms = { 
        'max_depth':8, 
        'learning_rate':0.1, 
        'subsample':0.8,
        'colsample_bytree':0.8, 
        'eval_metric':'logloss',
        'objective':'binary:logistic',
        'tree_method':'hist',
        "random_state":42
    }

    oof = np.zeros((len(valid),len(label_names)))
    for numlabel in range(4):
        start = time.time()
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)
        
        dtrain = xgb.DMatrix(data=train[feature_list[numlabel]], label=train[name])
        dvalid = xgb.DMatrix(data=valid[feature_list[numlabel]], label=valid[name])

        print("Training.....")
        model = xgb.train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=250,
                early_stopping_rounds=25,
                #maximize=True,
                verbose_eval=25) 
        model.save_model(f"{model_save_path}/xgboost_{name}_stage2.model")

        print('Predicting...')
        oof[:,numlabel] = model.predict(dvalid)
        
        print("took %.1f seconds" % ((time.time()-start)))
        
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

