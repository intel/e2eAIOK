#!/env/bin/python

import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

#import modin.config
#modin.config.Backend.put('omnisci')

#import modin.pandas as pd

import sys
import glob
import json
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
import timeit
import re
from sklearn.metrics import average_precision_score, log_loss

from utils import Timer, Settings, get_estimator, get_is_rt, CountEncoder, FrequencyEncoder, get_join_cols_from_model
import lightgbm as lgb
import gc
from transformers import BertTokenizer


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def load_tsv(fname):
    try:
        names = ['text_tokens', 'hashtags', 'tweet_id', 'present_media', 'present_links', 'present_domains', 'tweet_type',
                'language', 'tweet_timestamp', 'engaged_with_user_id', 'engaged_with_user_follower_count', 'engaged_with_user_following_count',
                'engaged_with_user_is_verified', 'engaged_with_user_account_creation', 'engaging_user_id', 'engaging_user_follower_count',
                'engaging_user_following_count', 'engaging_user_is_verified', 'engaging_user_account_creation', 'engagee_follows_engager']

        dtypes = {'text_tokens': str, 'hashtags': str, 'tweet_id': str, 'present_media': str, 'present_links': str,
                'present_domains': str, 'tweet_type': str, 'language': str, 'tweet_timestamp': float, 'engaged_with_user_id': str,
                'engaged_with_user_follower_count': 'int64', 'engaged_with_user_following_count': 'int64',
                'engaged_with_user_is_verified': bool, 'engaged_with_user_account_creation': 'int64', 'engaging_user_id': str,
                'engaging_user_follower_count': 'int64', 'engaging_user_following_count': 'int64', 'engaging_user_is_verified': bool,
                'engaging_user_account_creation': 'int64', 'engagee_follows_engager': bool}
        part_files = glob.glob(os.path.join(fname, "*part*"))
        with Timer(F'loading {part_files} wo/ label'):
            part_dfs = [pd.read_csv(f, sep='\x01', header=None, names=names, dtype=dtypes) for f in part_files]
        return pd.concat(part_dfs, ignore_index=True)
    except ValueError:
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
        part_files = glob.glob(os.path.join(fname, "*part*"))
        with Timer(F'loading {part_files} w/ lable'):
            part_dfs = [pd.read_csv(f, sep='\x01', header=None, names=names, dtype=dtypes) for f in part_files]
        return pd.concat(part_dfs, ignore_index=True)


def _fix_columns(df_cols, xcols):
    if Settings.using_lgbm:
        return xcols
    '''
    HACK Alert!
    Some columns are made with typo or can contain extra suffixes, hack up xcols.
    FIXME: make sure the typos and double columns during the merge are gone.
           Should just get rid of duplicate estimators during preparation and no typos in ingress data. 
    '''
    result = []
    for col in xcols:
        if col in df_cols:
            result.append(col)
        else:
            opts = [col.replace('retweet_timestamp', 'retweet_timestampe')]
            for suffix in ('_x', '_y'):
                if col.endswith(suffix):
                    opts.append(col[:-len(suffix)])
                    opts.append(
                        col[:-len(suffix)].replace('retweet_timestamp', 'retweet_timestampe'))
            for opt in opts:
                if opt in df_cols:
                    result.append(opt)
                    break
            else:
                raise ValueError(f"wasn't able to find a substitute for {col}")
    return result


def prep_tsv_columns(df):
    media = df['present_media'].fillna('')
    df['has_photo'] = media.str.contains('Photo').astype('int32')
    df['has_video'] = media.str.contains('Video').astype('int32')
    df['has_gif'] = media.str.contains('GIF').astype('int32')
    return df


def prep_text_columns(df):
    if "has_rt" in Settings.feature_list:
        df['has_rt'] = texts.apply(get_is_rt)
    with Timer('load tweet_word_bucketid.parquet'):
        word_dict_df = pd.read_parquet(f"{Settings.stored_dir}/tweet_word_bucketid.parquet")
        word = word_dict_df['dict_col'].to_list()
        id = word_dict_df['bucket_id'].to_list()
        word_dict = dict(zip(word, id))
    with Timer('load mention_with_bucketid.parquet'):
        mention_df = pd.read_parquet(f"{Settings.stored_dir}/mention_with_bucketid.parquet")
        word = mention_df['mention'].to_list()
        id = mention_df['bucket_id'].to_list()
        count = mention_df['count'].to_list()
        mention_dict = dict(zip(word, zip(id, count)))

    with Timer('process text word'):
        v1s, v2s, v3s, v4s, v5s = [], [], [], [], []
        print("        Start to process text token, total len is %d" % len(df['text_tokens']))
        total_len_by_100 = int(len(df['text_tokens']) / 100)
        processed = 0
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
         
        def decode_and_clean_tweet_text(x):
            x = tokenizer.decode([int(n) for n in x.split('\t')])
            x = x.replace(
                'https : / / t. co / ', 'https://t.co/').replace('@ ', '@')
            return x

        def get_word_idx(x):
            bucket = {}
            for v in x.split(' '):
                if v not in ['','[',']','.','!','@','_','#'] and v in word_dict:
                    id = word_dict[v]
                    if id not in bucket:
                        bucket[id] = 0
                    bucket[id] += 1
            after_sort = sorted(bucket.items(), key=lambda x: x[1], reverse=True)
            return [x[0] for x in after_sort]

        def get_mention_bucket_id(x):
            if '@' not in x:
                return []
            l_re = re.findall(r"RT\s@(\S+)", x)
            l_at = re.findall(r"\s@(\S+)", x)
            l = [item for item in l_at if item not in l_re]
            if len(l) == 0:
                return []
            if l[0] not in mention_dict:
                return []
            ret = mention_dict[l[0]]
            return [ret[0], ret[1]]

        for index, token in df['text_tokens'].items():
            value = decode_and_clean_tweet_text(token)
            ret = get_word_idx(value)
            v1s.append(ret[0] if (len(ret)>0) else None)
            v2s.append(ret[1] if (len(ret)>1) else None)
            ret = get_mention_bucket_id(value)
            v3s.append(True if (len(ret)>0) else False)
            v4s.append(ret[0] if (len(ret)>0) else None)
            v5s.append(ret[1] if (len(ret)>0) else 0)
            processed += 1
            if processed % total_len_by_100 == 0:
                print("        Processed tweet text completed %d %%" % (processed / total_len_by_100))
        df['most_used_word_bucket_id'] = pd.Series(v1s, dtype=float)
        df['second_used_word_bucket_id'] = pd.Series(v2s, dtype=float)
        df['has_mention'] = pd.Series(v3s, dtype=bool)
        df['mentioned_bucket_id'] = pd.Series(v4s, dtype=float)
        df['mentioned_count'] = pd.Series(v5s, dtype=int)
        print("Has Mention count is ", len(df[(df['has_mention'] == 1)]))

        del word_dict_df
        del word_dict
        del mention_df
        del mention_dict
    
    return df


def get_rates(df):
    df['engaged_user_rate'] = df["engaged_with_user_following_count"] / \
        df["engaged_with_user_follower_count"]
    df['engaging_user_rate'] = df["engaging_user_following_count"] / \
        df["engaging_user_follower_count"]
    if 'a_ff_rate' in Settings.feature_list:
        df['a_ff_rate'] =  df['engaged_user_rate'].astype('float32')
    if 'b_ff_rate' in Settings.feature_list:
        df['b_ff_rate'] = df['engaging_user_rate'].astype('float32')
    return df


def prep_bool_cols(df):
    bool_columns = ['engagee_follows_engager',
                    "engaged_with_user_is_verified", 'engaging_user_is_verified']
    for c in bool_columns:
        df[f'{c}_indicator'] = df[c].astype('int8')

    return df


def prep_datetime_columns(df):
    df['tweet_datetime'] = pd.to_datetime(df['tweet_timestamp'], unit='s')
    df['tweet_hour'] = df['tweet_datetime'].dt.hour.astype('int8')
    df['tweet_dow'] = df['tweet_datetime'].dt.dayofweek.astype('int8')
    df['dt_hour'] = df['tweet_hour']
    df['dt_dow'] = df['tweet_dow']
    df['dt_minute'] = df['tweet_datetime'].dt.minute.astype('int8')
    df['dt_second'] = df['tweet_datetime'].dt.second.astype('int8')
    return df


def feature_generation(df):
    df['both_verified'] = (df["engaging_user_is_verified"]
                           & df["engaged_with_user_is_verified"]).astype('int8')
    return df


def prep_response(df):
    cols = ['reply_timestamp', 'retweet_timestamp',
            'retweet_with_comment_timestamp', 'like_timestamp']
    cols = _fix_columns(df.columns, cols)
    for c in cols:
        df[f'{c}_indicator'] = df[c].notnull().astype('int8')
    return df

# add new features
def prep_countencoder(df, estimators):
    for c in df.columns:
        if '_norm' in c:
            encoder = FrequencyEncoder()
            src_cols = get_join_cols_from_model(c, estimators)
            if len(src_cols) == 0:
                raise ModuleNotFoundError("can't find %s in model_list" % c)
            df = encoder.fit_transform(df, src_cols, c)
            del encoder
        elif 'CE_' in c:
            encoder = CountEncoder()
            src_cols = get_join_cols_from_model(c, estimators)
            if len(src_cols) == 0:
                raise ModuleNotFoundError("can't find %s in model_list" % c)
            df = encoder.fit_transform(df, src_cols, c)
            del encoder
    return df


def do_final_align_to_train(df, estimators):
    df = prep_countencoder(df, estimators)
    ind_df = df[['tweet_id', 'engaging_user_id']]
    df = df[Settings.feature_list]
    df['has_mention'] = df['has_mention'].astype('int32')

    return ind_df, df

# ending add new features


def all_prep(df):
    with Timer('prep_text_columns'):
        df = prep_text_columns(df) # TODO: need to add tw_word and hash
    with Timer('prep_tsv_columns'):
        df = prep_tsv_columns(df)
    with Timer('prep_bool_cols'):
        df = prep_bool_cols(df)
    with Timer('feature_generation'):
        df = feature_generation(df)
    with Timer('prep_datetime_columns'):
        df = prep_datetime_columns(df)
    with Timer('get_rates'):
        df = get_rates(df)
    #df = language_indicator(df)
    #with Timer('prep_response'):
    #    df = prep_response(df)
    return df


def load_estimators(estdir):
    Settings.lazy_load = True
    Settings.stored_dir = estdir
    Settings.suffix = '-numeric'

    with Timer('loading features'):
        with open(f'{estdir}/features.json') as inp:
            feature_list = json.load(inp)

    with Timer('loading uid'):
        with open(f'{estdir}/uid.pkl', 'rb') as inp:
            uid = pickle.load(inp)

    estimators = []
    for metapath in glob.glob(f'{estdir}/*-meta.json'):
        with open(metapath) as inp:
            meta = json.load(inp)
        estimators.append(get_estimator(meta))
    estimators.sort(key=lambda e: os.path.getsize(
        f'{estdir}/{e.name}-numeric.parquet'), reverse=True)

    return uid, feature_list, estimators


def load_data(in_name, uid, format="parquet"):
    with Timer(f'load {in_name}'):
        if format == "parquet":
            df = pd.read_parquet(in_name)
        else:
            if in_name.endswith('.parquet'):
                df = pd.read_parquet(in_name)
            else:
                df = load_tsv(in_name)

    with Timer('recompute categories'):
        existing_ids = set(uid)
        new_ids = set()
        for col, dtype in df.dtypes.items():
            if str(dtype) == 'object' and col.endswith('_user_id'):
                new_ids |= set(df[col]) - existing_ids
    if new_ids:
        uid2 = pd.Index(np.concatenate((uid, np.array(list(new_ids)))))
    else:
        uid2 = uid

    with Timer('update input categories'):
        for col, dtype in df.dtypes.items():
            if str(dtype) == 'object' and col.endswith('_user_id'):
                df[col] = pd.Categorical(df[col], categories=uid2).codes

    return df, uid2


def compute_train_data(estimators, model_dir, feature_list, df):
    # added by chendi
    Settings.using_lgbm = True
    Settings.feature_list = feature_list

    # call transformer code
    with Timer('compute indicators'):
        df_transformed = all_prep(df)

    # now perform target embedding lookup
    with Timer('augment by features'):
        feature_engineered_df = df_transformed

        for m in estimators:
            with Timer(f'adding feature {m.name}'):
                feature_engineered_df = m.transform(feature_engineered_df)
    ind_df, feature_engineered_df = do_final_align_to_train(feature_engineered_df, estimators) # TODO: check if all features matches with train

    with Timer('get needed columns'):
        ycols = ['reply_timestamp', 'retweet_timestamp',
                 'retweet_with_comment_timestamp', 'like_timestamp']
        ymodels = []
        for model_name in ycols:
            for model in glob.glob(f'{model_dir}/{model_name}*'):
                ymodels.append(model)
        xcols = Settings.feature_list
        xcols = _fix_columns(feature_engineered_df.columns, xcols)

    with Timer('split data'):
        X = feature_engineered_df.loc[:, xcols].replace(np.inf, np.nan)

    return X, ymodels, ind_df


def predict(X, ymodels):
    outcome, result = {}, {}
    target_columns = ['reply_timestamp', 'retweet_timestamp',
                 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list_pred = []
    for i in range(len(target_columns)):
        with open(f'{Settings.stored_dir}/feature_list_{target_columns[i]}.json') as inp:
            tmp_list = json.load(inp)
            feature_list_pred.append(tmp_list)
        del tmp_list
    if Settings.using_lgbm:
        with Timer(f'inferring for {len(target_columns)} columns'):
            k = 0
            for response, model in zip(target_columns, ymodels):
                with Timer(f'inference for {response}'):
                    lgbm = lgb.Booster(model_file=f'{model}')
                    test = X[feature_list_pred[k]]
                    total_len = len(test)
                    remaining = total_len
                    total_len_by_10 = int(total_len / 10)
                    start = 0
                    to_concat = []
                    while remaining > 0:
                        step = total_len_by_10 if (remaining > total_len_by_10) else remaining
                        end = start + step
                        print(start, end, step)
                        sliced_test = test.iloc[start:end,:]
                        pred = lgbm.predict(sliced_test)
                        to_concat.append(pred)
                        start += step
                        remaining -= step
                    result[response] = np.concatenate(to_concat, axis=None)
                    k = k+1
                    del test
        return result
    else:
        with Timer(f'inferring for {len(target_columns)} columns'):
            for response, model in zip(target_columns, ymodels):
                with Timer(f'inference for {response}'):
                    xgb = XGBClassifier(max_depth=6, n_estimators=250, learning_rate=0.1, n_jobs=8, num_parallel_tree=1, 
                                        tree_method='hist', subsample=0.8, reg_alpha=0.1, reg_lambda=0.01, colsample_bytree=0.7)
                    xgb.load_model(f'{model}')
                    inf_start = timeit.default_timer()
                    pred = xgb.predict(X)
                    inf_stop = timeit.default_timer()
                    # outcome[response] = (len(pred)/(inf_stop - inf_start)), average_precision_score(ytest, pred)
                    result[response] = pred
        
        # for response in target_columns:
        #     rate, ap_score = outcome[response]
        #     print(f'Inference rate on {response}: {rate} samples/sec')
        #     print(f'AP score of {response}: {ap_score}')
        
        return result  


def compose_output(ind_df, predicts, uid, out_name):
    ycols = ['reply_timestamp', 'retweet_timestamp',
             'retweet_with_comment_timestamp', 'like_timestamp']
    ycols = _fix_columns(predicts.keys(), ycols)
    with Timer('compose prediction df'):
        pred_df = pd.DataFrame(predicts, columns=ycols)
    with Timer('compose answer df'):
        ind_df['engaging_user_id'] = ind_df['engaging_user_id'].apply(lambda x: uid[x])
        df = pd.concat(
            [ind_df.loc[:, ['tweet_id', 'engaging_user_id']], pred_df], axis=1)  # .dropna()
    with Timer('write resulting df'):
        df.to_csv(out_name, header=False, index=False, chunksize=200_000)


def main(estdir, model_dir, in_name, out_name):
    uid, feature_list, estimators = load_estimators(estdir)
    with Timer(f'loading {in_name}'):
        df, uid = load_data(in_name, uid, format="csv")
    with Timer('compute X-Y'):
        X, ymodels, ind_df = compute_train_data(estimators, model_dir, feature_list, df)
    predicts = predict(X, ymodels)
    with Timer('generate output'):
        compose_output(ind_df, predicts, uid, out_name)


if __name__ == '__main__':
    try:
        estdir, model_dir, in_name, out_name = sys.argv[1:5]
    except ValueError:
        print("input sys.args are incorrect, using default values, inputs are ", sys.argv[1:5])
        #sys.exit(f'Usage: {sys.argv[0]} estimators-dir model-dir df-in.parquet|df-in.tsv answer.csv')
        estdir = 'data/model_compressed'
        model_dir = 'data/model_0_3'
        in_name = 'test'
        out_name = 'results.csv'
    main(estdir, model_dir, in_name, out_name)
