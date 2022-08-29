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
                        default="")
    parser.add_argument('--reply_pred_path',
                        type=str,
                        default="")
    parser.add_argument('--retweet_pred_path',
                        type=str,
                        default="")
    parser.add_argument('--retweet_with_comment_pred_path',
                        type=str,
                        default="")
    parser.add_argument('--like_pred_path',
                        type=str,
                        default="")
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