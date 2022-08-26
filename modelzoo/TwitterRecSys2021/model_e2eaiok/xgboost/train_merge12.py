import pandas as pd
import numpy as np
import time
import argparse
import sys
very_start = time.time()

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default="/home/vmagent/app/dataset/xinyao/TwitterRecSys2021-intel-opt/TwitterRecSys2021Dataset")
    parser.add_argument('--reply_pred_path',
                        type=str,
                        default="/home/vmagent/app/hydro.ai/result/twitter_recsys/20211217_055149/2f9dd8a6fe67fe190b0c1e015c6f60d5/xgboost_pred_stage1_reply.csv")
    parser.add_argument('--retweet_pred_path',
                        type=str,
                        default="/home/vmagent/app/hydro.ai/result/twitter_recsys/20211217_055240/d71e647911eadd50fed5693c8e02b436/xgboost_pred_stage1_retweet.csv")
    parser.add_argument('--retweet_with_comment_pred_path',
                        type=str,
                        default="/home/vmagent/app/hydro.ai/result/twitter_recsys/20211217_055303/d5aee6594f7c76ec0dc0b3e5ca1aaaa8/xgboost_pred_stage1_retweet_with_comment.csv")
    parser.add_argument('--like_pred_path',
                        type=str,
                        default="/home/vmagent/app/hydro.ai/result/twitter_recsys/20211217_055325/0eb3fe9e620acc16459bad6c08c7a7e1/xgboost_pred_stage1_like.csv")
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    df1 = pd.read_parquet(f"{args.data_path}/stage2/train")
    df2 = pd.read_parquet(f"{args.data_path}/stage2/valid")

    index_cols = ['tweet_id', 'engaging_user_id']
    ##merge reply
    preds_reply = pd.read_csv(args.reply_pred_path)
    df1 = df1.merge(preds_reply, on=index_cols, how="left")
    df2 = df2.merge(preds_reply, on=index_cols, how="left")
    ##merge retweet
    preds_retweet = pd.read_csv(args.retweet_pred_path)
    df1 = df1.merge(preds_retweet, on=index_cols, how="left")
    df2 = df2.merge(preds_retweet, on=index_cols, how="left")
    ##merge retweet_with_comment
    preds_retweet_with_comment = pd.read_csv(args.retweet_with_comment_pred_path)
    df1 = df1.merge(preds_retweet_with_comment, on=index_cols, how="left")
    df2 = df2.merge(preds_retweet_with_comment, on=index_cols, how="left")
    ##merge like
    preds_like = pd.read_csv(args.like_pred_path)
    df1 = df1.merge(preds_like, on=index_cols, how="left")
    df2 = df2.merge(preds_like, on=index_cols, how="left")

    df1.to_parquet(f"{args.data_path}/stage2_pred/train/stage2_train_xgboost_pred1.parquet")
    df2.to_parquet(f"{args.data_path}/stage2_pred/valid/stage2_valid_xgboost_pred1.parquet")

    print('This notebook took %.1f seconds'%(time.time()-very_start))