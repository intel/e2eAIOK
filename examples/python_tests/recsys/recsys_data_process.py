import pyspark.sql.functions as f
import init

import os
import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark import *
from timeit import default_timer as timer
import logging
from RecsysSchema import RecsysSchema
from pyrecdp.data_processor import *
from pyrecdp.utils import *
from transformers import BertTokenizer
import hashlib


def categorifyAllFeatures(df, proc, output_name="categorified", gen_dict=False):
    dict_dfs = []
    if gen_dict:
        # only call below function when target dicts were not pre-prepared
        op_multiItems = GenerateDictionary(
            ['present_domains', 'present_links', 'hashtags'], doSplit=True)
        op_singleItems = GenerateDictionary(['tweet_id', 'language', {'src_cols': [
                                            'engaged_with_user_id', 'enaging_user_id'], 'col_name': 'user_id'}])
        proc.reset_ops([op_multiItems, op_singleItems])
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        # or we can simply load from pre-gened
        dict_names = ['hashtags', 'language', 'present_domains',
                      'present_links', 'tweet_id', 'user_id']
        dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in dict_names]

    # Below codes are for inference
    # op_multiItems = GenerateDictionary(
    #    ['present_domains', 'present_links', 'hashtags'], doSplit=True)
    # op_singleItems = GenerateDictionary(['tweet_id', 'language', {'src_cols': [
    #                                     'engaged_with_user_id', 'enaging_user_id'], 'col_name': 'user_id'}])
    # proc.reset_ops([op_multiItems, op_singleItems])
    # t1 = timer()
    # dict_dfs = proc.merge_dicts(df, dict_dfs)
    # t2 = timer()
    # print("Merge Dictionary took %.3f" % (t2 - t1))
    # ###############################

    # pre-defined dict
    # pre-define
    media = {
        '': 0,
        'GIF': 1,
        'GIF_GIF': 2,
        'GIF_Photo': 3,
        'GIF_Video': 4,
        'Photo': 5,
        'Photo_GIF': 6,
        'Photo_Photo': 7,
        'Photo_Video': 8,
        'Video': 9,
        'Video_GIF': 10,
        'Video_Photo': 11,
        'Video_Video': 12
    }

    tweet_type = {'Quote': 0, 'Retweet': 1, 'TopLevel': 2}

    media_df = proc.spark.createDataFrame(convert_to_spark_dict(media))
    tweet_type_df = proc.spark.createDataFrame(
        convert_to_spark_dict(tweet_type))

    dict_dfs.append({'col_name': 'present_media', 'dict': media_df})
    dict_dfs.append({'col_name': 'tweet_type', 'dict': tweet_type_df})

    for i in dict_dfs:
        dict_name = i['col_name']
        dict_df = i['dict']
        print("%s has numRows as %d" % (dict_name, dict_df.count()))

    ###### 2. define operations and append them to data processor ######

    # 1. define operations
    # 1.1 fill na and features
    op_fillna_str = FillNA(
        ['present_domains', 'present_links', 'hashtags', 'present_media', 'tweet_id'], "")
    op_feature_modification_type_convert = FeatureModification(cols=['tweet_timestamp',
                                                                     'engaged_with_user_follower_count',
                                                                     'engaged_with_user_following_count',
                                                                     'engaged_with_user_account_creation',
                                                                     'enaging_user_follower_count',
                                                                     'enaging_user_following_count',
                                                                     'enaging_user_account_creation'], op='toInt')
    op_feature_modification_present_media_replace = FeatureModification(
        cols={'present_media': "f.concat_ws('_', f.slice(f.split(f.col('present_media'),'\t'), 1, 2))"}, op='inline')
    op_feature_add_len_hashtags = FeatureAdd(
        cols={'len_hashtags': "f.when(f.col('hashtags') == '', f.lit(0)).otherwise(f.size(f.split(f.col('hashtags'), '\t')))"}, op='inline')
    op_feature_add_len_domains = FeatureAdd(
        cols={'len_domains': "f.when(f.col('present_domains') == '', f.lit(0)).otherwise(f.size(f.split(f.col('present_domains'), '\t')))"}, op='inline')
    op_feature_add_len_links = FeatureAdd(
        cols={'len_links': "f.when(f.col('present_links') == '', f.lit(0)).otherwise(f.size(f.split(f.col('present_links'), '\t')))"}, op='inline')
    op_new_feature_dt_dow = FeatureAdd(cols={
        "dt_dow": "f.dayofweek(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
        "dt_hour": "f.hour(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
        "dt_minute": "f.minute(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
        "dt_second": "f.second(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())"}, op='inline')
    op_new_feature_origin = FeatureAdd(
        cols={"origin_tweet_id": "f.col('tweet_id')", "origin_engaging_user_id": "f.col('enaging_user_id')"}, op='inline')
    op_fillna_num = FillNA(['tweet_timestamp'], -1)
    ops = [op_fillna_str,
           op_feature_modification_type_convert, op_feature_modification_present_media_replace,
           op_feature_add_len_hashtags, op_feature_add_len_domains, op_feature_add_len_links,
           op_new_feature_dt_dow, op_new_feature_origin, op_fillna_num]
    proc.reset_ops(ops)

    # 1.3 categorify
    # since language dict is small, we may use udf to make partition more even
    op_categorify_multi = Categorify(
        ['present_domains', 'present_links', 'hashtags'], dict_dfs=dict_dfs, doSplit=True, keepMostFrequent=True)
    op_categorify = Categorify(['language', 'tweet_id', 'present_media', 'tweet_type', {
        'engaged_with_user_id': 'user_id'}, {'enaging_user_id': 'user_id'}], dict_dfs=dict_dfs)

    op_fillna_for_categorified = FillNA(['present_domains', 'present_links', 'hashtags', 'language',
                                         'tweet_id', 'present_media', 'tweet_type', 'engaged_with_user_id', 'enaging_user_id'], -1)
    ops_1 = [op_categorify_multi, op_categorify, op_fillna_for_categorified]
    proc.append_ops(ops_1)

    ##### 3. do data transform(data frame materialize) #####
    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()
    print("Data Process and udf categorify took %.3f" % (t2 - t1))
    return df


def categorifyTweetText(df, proc, output_name="tweet_text_categorified_20days", gen_dict=False):
    dict_dfs = []
    if gen_dict:
        # only call below function when target dicts were not pre-prepared
        op_multiItems = GenerateDictionary(
            ['tweet'], doSplit=True, withCount=True, sep=' ')
        proc.reset_ops([op_multiItems])
        ##### transform #####
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        # or we can simply load from pre-gened
        name = "tweet"
        tweet_dict_df = proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))
        dict_dfs = [{'col_name': 'tweet', 'dict': tweet_dict_df}]

    tweet_dict_df = dict_dfs[0]['dict']
    freqRange = [2, 100000]
    tweet_dict_df = tweet_dict_df.filter((f.col('count') <= f.lit(
        freqRange[1])) & (f.col('count') >= f.lit(freqRange[0])))
    op_fillNA = FillNA(['tweet'], "")
    op_rename = FeatureAdd(
        cols={"original_tweet": "f.col('tweet')"}, op='inline')
    op_categorify = Categorify(
        ['tweet'], dict_dfs=dict_dfs, doSplit=True, sep=' ', doSortForArray=True)
    proc.reset_ops([op_fillNA, op_rename, op_categorify])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Categorify tweet took %.3f" % (t2 - t1))
    return df


def categorifyTweetHash(df, proc, output_name="tweet_text_processed_20days", gen_dict=False):
    dict_dfs = []
    if gen_dict:
        # only call below function when target dicts were not pre-prepared
        op_gen_dict = GenerateDictionary(['tw_hash'])
        proc.reset_ops([op_gen_dict])
        ##### transform #####
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        # or we can simply load from pre-gened
        name = "tw_hash"
        tw_hash_dict_df = proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))
        dict_dfs = [{'col_name': 'tw_hash', 'dict': tw_hash_dict_df}]

    op_categorify = Categorify(['tw_hash'], dict_dfs=dict_dfs)
    proc.reset_ops([op_categorify])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("Categorify tw_hash took %.3f" % (t2 - t1))
    return df


def decodeBertTokenizer(df, proc, output_name="data_all_with_text"):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=False)

    # define UDF
    tokenizer_decode = f.udf(lambda x: tokenizer.decode(
        [int(n) for n in x.split('\t')]))
    format_url = f.udf(lambda x: x.replace(
        'https : / / t. co / ', 'https://t.co/').replace('@ ', '@'))

    # define operations
    op_feature_modification_tokenizer_decode = FeatureAdd(
        cols={'tweet': 'text_tokens'}, udfImpl=tokenizer_decode)
    op_feature_modification_format_url = FeatureModification(
        cols=['tweet'], udfImpl=format_url)

    # execute
    proc.reset_ops([op_feature_modification_tokenizer_decode,
                    op_feature_modification_format_url])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("BertTokenizer decode and format took %.3f" % (t2 - t1))

    return df


def tweetFeatureEngineer(df, proc, output_name="tweet_feature_engineer"):

    def extract_hash(text, split_text='@', no=0):
        text = text.lower()
        uhash = ''
        text_split = text.split('@')
        if len(text_split) > (no+1):
            text_split = text_split[no+1].split(' ')
            cl_loop = True
            uhash += clean_text(text_split[0])
            while cl_loop:
                if len(text_split) > 1:
                    if text_split[1] in ['_']:
                        uhash += clean_text(text_split[1]) + \
                            clean_text(text_split[2])
                        text_split = text_split[2:]
                    else:
                        cl_loop = False
                else:
                    cl_loop = False
        hash_object = hashlib.md5(uhash.encode('utf-8'))
        return hash_object.hexdigest()

    def clean_text(text):
        if len(text) > 1:
            if text[-1] in ['!', '?', ':', ';', '.', ',']:
                return(text[:-1])
        return(text)

    # features upon tweet
    to_notsign = f.udf(lambda x: x.replace('\[CLS\] RT @', ''))
    count_space = f.udf(lambda x: x.count(' '))
    count_text_length = f.udf(lambda x: len(x))
    user_defined_hash = f.udf(
        lambda x: extract_hash(x, split_text='RT @', no=0))
    # features upon tweet_nortsign
    count_at = f.udf(lambda x: x.count('@'))
    user_define_hash_1 = f.udf(lambda x: extract_hash(x))
    user_define_hash_2 = f.udf(lambda x: extract_hash(x, no=1))

    # features upon tweet
    op_fillna_for_tweet = FillNA(['original_tweet'], "")
    op_feature_add_tweet_nortsign = FeatureAdd(
        cols={'tweet_nortsign': 'original_tweet'}, udfImpl=to_notsign)
    op_feature_add_count_words = FeatureAdd(
        cols={'count_words': 'original_tweet'}, udfImpl=count_space)
    op_feature_add_count_char = FeatureAdd(
        cols={'count_char': 'original_tweet'}, udfImpl=count_text_length)
    op_feature_add_tw_uhash = FeatureAdd(
        cols={'tw_uhash': 'original_tweet'}, udfImpl=user_defined_hash)
    op_feature_add_tw_hash = FeatureAdd(
        cols={'tw_hash': "f.hash(f.col('original_tweet'))%1000000000"}, op='inline')
    # features upon tweet_nortsign
    op_feature_add_count_at = FeatureAdd(
        cols={'count_ats': 'tweet_nortsign'}, udfImpl=count_at)
    op_feature_add_tw_uhash0 = FeatureAdd(
        cols={'tw_hash0': 'tweet_nortsign'}, udfImpl=user_define_hash_1)
    op_feature_add_tw_uhash1 = FeatureAdd(
        cols={'tw_hash1': 'tweet_nortsign'}, udfImpl=user_define_hash_2)

    # execute
    proc.reset_ops([op_feature_add_tweet_nortsign, op_feature_add_count_words, op_feature_add_count_char,
                    op_feature_add_tw_uhash, op_feature_add_tw_hash,
                    op_feature_add_count_at, op_feature_add_tw_uhash0, op_feature_add_tw_uhash1])
    t1 = timer()
    df = proc.transform(df, output_name)
    t2 = timer()
    print("Adding Feature upon tweet and tweet_nortsign column took %.3f" % (t2 - t1))
    # expect to spend about 1000secs
    return df


def tweetFeatureEngineer(df, proc, output_name="tweet_feature_engineer_20days"):
    op_fillna_for_tweet = FillNA(['tweet'], "")
    op_feature_add_tw_hash = FeatureAdd(
        cols={'tw_hash': "f.hash(f.col('original_tweet'))%1000000000"}, op='inline')
    op_feature_add_tw_first_word = FeatureAdd(
        {'tw_first_word': "f.col('tweet').getItem(0)"}, op='inline')
    op_feature_add_tw_second_word = FeatureAdd(
        {'tw_second_word': "f.col('tweet').getItem(1)"}, op='inline')
    op_feature_add_tw_last_word = FeatureAdd(
        {'tw_last_word': "f.col('tweet').getItem(f.size(f.col('tweet')) - 1)"}, op='inline')
    op_feature_add_tw_second_last_word = FeatureAdd(
        {'tw_llast_word': "f.col('tweet').getItem(f.size(f.col('tweet')) - 1)"}, op='inline')
    op_feature_add_tw_word_len = FeatureAdd(
        {'tw_len': "f.size(f.col('tweet'))"}, op='inline')
    op_feature_modification_fillna = FillNA(
        ['tw_hash', 'tw_first_word', 'tw_second_word', 'tw_last_word', 'tw_llast_word', 'tw_len'], -1)

    proc.reset_ops([op_fillna_for_tweet, op_feature_add_tw_hash, op_feature_add_tw_first_word, op_feature_add_tw_second_word,
                    op_feature_add_tw_last_word, op_feature_add_tw_second_last_word, op_feature_add_tw_word_len,
                    op_feature_modification_fillna])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("feature engineering upon Frequency encoded tweet column took %.3f" % (t2 - t1))
    return df


def get_train_data_with_amount_of_days(df, proc, num_of_day=20):
    categorified_with_text_df = df
    categorified_with_text_df.cache()
    # 1.1 get timestamp range
    import datetime
    min_timestamp = categorified_with_text_df.select('tweet_timestamp').agg(
        {'tweet_timestamp': 'min'}).collect()[0]['min(tweet_timestamp)']
    max_timestamp = categorified_with_text_df.select('tweet_timestamp').agg(
        {'tweet_timestamp': 'max'}).collect()[0]['max(tweet_timestamp)']
    seconds_in_day = 3600 * 24

    print(
        "min_timestamp is %s, max_timestamp is %s, %d days max is %s" % (
            datetime.datetime.fromtimestamp(
                min_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.datetime.fromtimestamp(
                max_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            num_of_day,
            datetime.datetime.fromtimestamp(
                min_timestamp + num_of_day * seconds_in_day).strftime('%Y-%m-%d %H:%M:%S')
        ))

    time_range_split = {
        'target': (min_timestamp, seconds_in_day * num_of_day + min_timestamp)
    }

    print(time_range_split)

    # 1.2 save ranged data for train
    # filtering out train range data and save
    train_start, train_end = time_range_split['target']
    df = categorified_with_text_df.filter(
        (f.col('tweet_timestamp') >= f.lit(train_start)) & (f.col('tweet_timestamp') < f.lit(train_end)))
    output_path = "%s/%s/data_splitted_by_%ddays" % (
        proc.path_prefix, proc.current_path, num_of_day)
    df.write.format('parquet').mode('overwrite').save(output_path)
    return proc.spark.read.parquet(output_path)


def main(current_path):
    current_path = "/recsys2021/1day"
    path_prefix = "hdfs://"
    original_folder = "/recsys2021/decompress"
    dicts_folder = "recsys_dicts/"
    recsysSchema = RecsysSchema()

    ##### 1. Start spark and initialize data processor #####
    t0 = timer()
    spark = SparkSession.builder.master('yarn')\
        .appName("Recsys2021_data_process")\
        .getOrCreate()

    schema = recsysSchema.toStructType()

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix,
                         current_path=current_path, dicts_path=dicts_folder)

    # ===============================================
    # basic: Do categorify for all columns for xgboost
    # df = spark.read.schema(schema).option('sep', '\x01').csv(path_prefix + original_folder)
    # df = get_train_data_with_amount_of_days(df, proc, 5) # 5 days will use 1 day for train dataset
    df = spark.read.parquet("%s/data_splitted_by_5days/" % current_path)
    df = categorifyAllFeatures(df, proc, gen_dict=False)

    # ===============================================
    # optional: do bert decode
    # df = spark.read.parquet("%s/categorfied/" % current_path)
    # df = decodeBertTokenizer(df, proc)

    # ===============================================
    # optional: do tweet text feature engineering
    # step1: categorify tweet text
    # df = spark.read.parquet("%s/processed_for_20days/" % current_path)
    # df = categorifyTweetText(df, proc, gen_dict=False)

    # step2: add new feature with categorified tweet
    # df = spark.read.parquet("%s/tweet_text_categorified_20days/" % current_path)
    # df = tweetFeatureEngineer(df, proc)

    # step3: categorify tweet hash
    # df = spark.read.parquet("%s/tweet_feature_engineer_20days/" % current_path)
    # df = categorifyTweetHash(df, proc, gen_dict=True)

    # ===============================================


if __name__ == "__main__":
    main(current_path="/recsys2021/")
