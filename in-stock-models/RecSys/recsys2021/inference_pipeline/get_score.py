import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, log_loss
from utils import Timer
import glob

def load_result(fname):
    names = ['tweet_id', 'engaging_user_id', 'reply_timestamp', 'retweet_timestamp',
             'retweet_with_comment_timestamp', 'like_timestamp']

    dtypes = {'tweet_id': str, 'engaging_user_id': str, 'reply_timestamp': float, 'retweet_timestamp': float, 'retweet_with_comment_timestamp': float,
              'like_timestamp': float}
    return pd.read_csv(fname, sep=',', header=None, names=names, dtype=dtypes)

def load_validate(fname):   
    names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
    'language', 'tweet_timestamp', 'engaged_with_user_id', 'engaged_with_user_follower_count', 'engaged_with_user_following_count',
    'engaged_with_user_is_verified', 'engaged_with_user_account_creation', 'engaging_user_id', 'engaging_user_follower_count',
    'engaging_user_following_count', 'engaging_user_is_verified', 'engaging_user_account_creation', 'engagee_follows_engager',
    'reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

    dtypes = {'text_tokens': str, 'hashtags': str, 'tweet_id': str, 'present_media': str, 'present_links': str,
    'present_domains': str, 'tweet_type': str, 'language': str, 'tweet_timestamp': float, 'engaged_with_user_id': str,
    'engaged_with_user_follower_count': 'int64', 'engaged_with_user_following_count': 'int64',
    'engaged_with_user_is_verified': bool, 'engaged_with_user_account_creation': 'int64', 'engaging_user_id': str,
    'engaging_user_follower_count': 'int64', 'engaging_user_following_count': 'int64', 'engaging_user_is_verified': bool,     
    'engaging_user_account_creation': 'int64', 'engagee_follows_engager': bool, 'reply_timestamp': float,
    'retweet_timestamp': float, 'retweet_with_comment_timestamp': float, 'like_timestamp': float}
    part_files = glob.glob(os.path.join(fname, "part-*"))
    with Timer(F'loading {part_files} w/ lable'):
        part_dfs = [pd.read_csv(f, sep='\x01', header=None, names=names, dtype=dtypes) for f in part_files]
    valid_df = pd.concat(part_dfs, ignore_index=True)
    #valid_df = pd.read_csv(fname, sep='\x01', header=None, names=names, dtype=dtypes)
    tag_names = ['tweet_id', 'engaging_user_id']
    ycols = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    gdt = valid_df[tag_names + ycols]
    for c in ycols:
        gdt[c] = valid_df[c].fillna(0).astype(int)
    gdt.loc[gdt['reply_timestamp'] > 0, 'reply_timestamp'] = 1
    gdt.loc[gdt['retweet_timestamp'] > 0, 'retweet_timestamp'] = 1
    gdt.loc[gdt['retweet_with_comment_timestamp'] > 0, 'retweet_with_comment_timestamp'] = 1
    gdt.loc[gdt['like_timestamp'] > 0, 'like_timestamp'] = 1
    return gdt


def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


def get_score(ytest, pred):
    ycols = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    pred = pred.rename(columns = dict(zip(pred.columns, ycols)))
    ytest = ytest.rename(columns = dict(zip(ytest.columns, ycols)))
    ap_score = [average_precision_score(ytest[ycols[i]], pred[ycols[i]]) for i in range(4)]
    rce_score = [compute_rce_fast(pred[ycols[i]], ytest[ycols[i]]) for i in range(4)]
    for name, (ap, rce) in zip(ycols, zip(ap_score, rce_score)):
        print(f'{name}: {ap} {rce}')


def main(result, validate):
    ycols = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    with Timer(f'loading {result}'):
        res_df = load_result(result)
    with Timer('loading validate'):
        valid_df = load_validate(validate)
    align_df = pd.merge(res_df, valid_df, on=['tweet_id', 'engaging_user_id'], how='left')
    print(F"Nan count after join: {align_df['reply_timestamp_y'].isna().sum()}, {align_df['retweet_timestamp_y'].isna().sum()}, {align_df['retweet_with_comment_timestamp_y'].isna().sum()}, {align_df['like_timestamp_y'].isna().sum()}")
    with Timer('get score'):
        get_score(align_df[[F'{c}_y' for c in ycols]], align_df[[F'{c}_x' for c in ycols]])


if __name__ == '__main__':
    try:
        result, validate = sys.argv[1:3]
    except ValueError:
        print("input sys.args are incorrect, using default values, inputs are ", sys.argv[1:3])
        #sys.exit(f'Usage: {sys.argv[0]} estimators-dir model-dir df-in.parquet|df-in.tsv answer.csv')
        result = "/mnt/nvme2/chendi/BlueWhale/sample_0_3/inference_pipeline/results.csv"
        validate = "/mnt/nvme2/chendi/BlueWhale/sample_0_3/inference_pipeline/validate_dataset" 
    main(result, validate)  
