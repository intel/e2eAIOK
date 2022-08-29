#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, average_precision_score
from features import *
import argparse
import yaml
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

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',
                        type=str,
                        default="stage1",
                        help='training stage, can be stage1 or stage2')
    parser.add_argument('--target',
                        type=str,
                        default="reply",
                        help='training label names, can be reply, retweet, retweet_with_comment or like')
    parser.add_argument('--train_data_path',
                        type=str,
                        default="/datapath/stage1/train/train",
                        help='path to training data')
    parser.add_argument('--valid_data_path',
                        type=str,
                        default="/datapath/stage1/valid/valid",
                        help='path to validation data')
    parser.add_argument('--model_save_path',
                        type=str,
                        default="/datapath/models",
                        help='path for model and result saving')
    parser.add_argument('--max_depth',
                        type=int,
                        default=8)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1)
    parser.add_argument('--subsample',
                        type=float,
                        default=0.8)
    parser.add_argument('--colsample_bytree',
                        type=float,
                        default=0.8)
    parser.add_argument('--num_boost_round',
                        type=int,
                        default=250)
    return parser.parse_args(args)

DEBUG = False
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(args)

    ######## Load data
    train = pd.read_parquet(args.train_data_path)
    valid = pd.read_parquet(args.valid_data_path)
    if DEBUG:
        num_iterations = 10
        train = train[:10000]
        valid = valid[:10000]
    print(train.shape)
    print(valid.shape)

    for col in valid.columns:
        if valid[col].dtype=='bool':
            train[col] = train[col].astype('int8')
            valid[col] = valid[col].astype('int8')

    ######## Feature list for each target
    label_name = f'{args.target}_timestamp'
    features = feature_list[args.stage][label_name]
    print(label_name, len(features))

    ######## Train and predict
    xgb_parms = { 
        'max_depth':args.max_depth, 
        'learning_rate':args.learning_rate, 
        'subsample':args.subsample,
        'colsample_bytree':args.colsample_bytree, 
        'eval_metric':'logloss',
        'objective':'binary:logistic',
        'tree_method':'hist',
        "random_state":42
    }

    oof = np.zeros((len(valid),1))
    start = time.time()
    
    dtrain = xgb.DMatrix(data=train[features], label=train[label_name])
    dvalid = xgb.DMatrix(data=valid[features], label=valid[label_name])

    print("Training.....")
    model = xgb.train(xgb_parms, 
            dtrain=dtrain,
            evals=[(dtrain,'train'),(dvalid,'valid')],
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=25,
            verbose_eval=25) 
    model.save_model(f"{args.model_save_path}/xgboost_{args.stage}_{label_name}.model")

    print('Predicting...')
    oof[:,0] = model.predict(dvalid)
    
    print("Training and predicting took %.1f seconds" % ((time.time()-start)))

    ######## Merge prediction to data and save
    valid[f"pred_{label_name}"] = oof[:,0]
    valid[["tweet_id","engaging_user_id",f"pred_{label_name}"]].to_csv(f"{args.model_save_path}/xgboost_pred_{args.stage}_{args.target}.csv",index=0)
    
    ######## Evaluate the performance
    ap = compute_AP(oof[:,0],valid[label_name].values)
    rce = compute_rce_fast(oof[:,0],valid[label_name].values)
    txt = f"{label_name:20} AP:{ap:.5f} RCE:{rce:.5f}"
    print(txt)

    results = {}
    results["AP"] = float(ap)
    results["RCE"] = float(rce)
    saved_path = os.path.join(args.model_save_path, "result.yaml")
    with open(saved_path, 'w') as f:
        yaml.dump(results, f)
    
    print('This notebook took %.1f seconds'%(time.time()-very_start))

