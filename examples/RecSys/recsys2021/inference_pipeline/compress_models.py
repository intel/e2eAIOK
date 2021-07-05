import pandas as pd
import collections
import pickle
import sys
import os
import sys
import json
import shutil

from utils import Timer, Model, CMBD_Model, CATE_Model, CPD_Model, Settings


def load_models(drop_dups=False):
    Settings.drop_duplicates = drop_dups
    Settings.lazy_load = False

    # combined_features
    model_list = [
        CMBD_Model(join_cols=['engaged_with_user_id'],lookup_table='lookup_table_engaged_with_user_id_fold'),
        CMBD_Model(join_cols=['engaged_with_user_id','language'],lookup_table='lookup_table_engaged_with_user_id_fold_language'),
        CMBD_Model(join_cols=['engaged_with_user_id','tweet_dow'],lookup_table='lookup_table_engaged_with_user_id_fold_tweet_dow'),
        CMBD_Model(join_cols=['engaged_with_user_id','tweet_hour'],lookup_table='lookup_table_engaged_with_user_id_fold_tweet_hour'),
        CMBD_Model(join_cols=['engaged_with_user_id','tweet_type'],lookup_table='lookup_table_engaged_with_user_id_fold_tweet_type'),
        CMBD_Model(join_cols=['engaging_user_id','tweet_hour'],lookup_table='lookup_table_fold_engaging_user_id_tweet_hour'),
        CMBD_Model(join_cols=['language'],lookup_table='lookup_table_fold_language'),
        CMBD_Model(join_cols=['engaging_user_id','language'],lookup_table='lookup_table_fold_language_engaging_user_id'),
        CMBD_Model(join_cols=['tweet_dow'],lookup_table='lookup_table_fold_tweet_dow'),
        CMBD_Model(join_cols=['engaging_user_id','tweet_dow'],lookup_table='lookup_table_fold_tweet_dow_engaging_user_id'),
        CMBD_Model(join_cols=['tweet_type'],lookup_table='lookup_table_fold_tweet_type'),
        CMBD_Model(join_cols=['engaging_user_id','tweet_type'],lookup_table='lookup_table_fold_tweet_type_engaging_user_id'),
        CMBD_Model(join_cols=['has_mention','engaging_user_id'],lookup_table='lookup_table_has_mention_engaging_user_id'),
        CMBD_Model(join_cols=['mentioned_bucket_id'],lookup_table='lookup_table_mentioned_bucket_id'),
        CMBD_Model(join_cols=['mentioned_bucket_id','engaging_user_id'],lookup_table='lookup_table_mentioned_bucket_id_engaging_user_id'),
        CMBD_Model(join_cols=['mentioned_count'],lookup_table='lookup_table_mentioned_count'),
        CMBD_Model(join_cols=['mentioned_count','engaging_user_id'],lookup_table='lookup_table_mentioned_count_engaging_user_id'),
        CMBD_Model(join_cols=['most_used_word_bucket_id'],lookup_table='lookup_table_most_used_word_bucket_id'),
        CMBD_Model(join_cols=['second_used_word_bucket_id'],lookup_table='lookup_table_second_used_word_bucket_id')
    ]

    return model_list


def compute_categories(model_list):
    categs = collections.defaultdict(set)
    for m in model_list:
        for colname, dtype in m.df_agg.dtypes.items():
            if dtype == 'object':
                try:
                    categs[colname] |= set(m.df_agg[colname])
                except TypeError:  # unhashable, won't be able to categorize
                    continue

    uid = set()
    for catname, catval in categs.items():
        if catname.endswith('_user_id'):
            uid |= catval
    uid = pd.Index(uid)

    categs2 = {k: uid if k.endswith('_user_id') else pd.Index(
        v) for k, v in categs.items() if v}
    return uid, categs2


def compress(df_agg, categories):
    for colname, dtype in df_agg.dtypes.items():
        if str(dtype).startswith('int'):
            colvalue = df_agg[colname]
            if colvalue.min() >= -128 and colvalue.max() <= 127:
                df_agg[colname] = df_agg[colname].astype('int8')
        elif str(dtype).startswith('float'):
            df_agg[colname] = df_agg[colname].astype('float32')
        elif colname == "text_tokens":
            df_agg[colname] = df_agg[colname].astype(str)
        elif colname in categories:
            coldata = pd.Categorical(
                df_agg[colname], categories=categories[colname])
            if colname.endswith('_user_id'):
                coldata = coldata.codes
            df_agg[colname] = coldata
            # df.take() consolidates blocks in block manager
    df_agg.take([0, 0])


def main(outdir='compressed-models', drop_dups=False):
    with Timer('loading features'):
        with open(f'features.json') as inp:
            Settings.feature_list = json.load(inp)
    with Timer('loading models'):
        model_list = load_models(drop_dups)
    with Timer('compute categories'):
        uid, categs = compute_categories(model_list)
    with Timer('compress models'):
        for m in model_list:
            with Timer(f'compress {m.name} model'):
                compress(m.df_agg, categs)
    os.makedirs(outdir, exist_ok=True)

    with Timer('store compressed'):
        with Timer('store uid categories'):
            with open(f'{outdir}/uid.pkl', 'wb') as out:
                pickle.dump(uid.values, out, pickle.HIGHEST_PROTOCOL)
        for m in model_list:
            with Timer(f'saving {m.name}'):
                with open(f'{outdir}/{m.name}-meta.json', 'w') as out:
                    json.dump(m.get_meta(), out)
                m.df_agg.to_parquet(f'{outdir}/{m.name}-numeric.parquet')
                shutil.copyfile('features.json', f'{outdir}/features.json')


if __name__ == '__main__':
    try:
        outdir = sys.argv[1]
        Settings.stored_dir = sys.argv[2]
        drop_dups = '--drop-dups' in sys.argv
    except IndexError:
        # sys.exit(f'Usage: {sys.argv[0]} compressed-models-dir stored_dir [--drop-dups]')
        outdir = '/mnt/nvme2/chendi/BlueWhale/sample_0_3/model_compressed'
        Settings.stored_dir = '/mnt/nvme2/chendi/BlueWhale/sample_0_3/inference_pipeline'
    main(outdir)
