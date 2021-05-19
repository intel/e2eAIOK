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


def main():
    path_prefix = "hdfs://"
    folder = "/recsys2021/decompress/"
    file = "/recsys2021/decompress/part-00036"
    #path = os.path.join(path_prefix, folder)
    path = os.path.join(path_prefix, file)
    recsysSchema = RecsysSchema()

    # DataFrame[
    # *  text_tokens: string,
    # *  hashtags: string,
    # *  tweet_id: string,
    # *  present_media: string,
    # *  present_links: string,
    # *  present_domains: string,
    # *  tweet_type: string,
    # *  language: string,
    # *  tweet_timestamp: int,
    # *  engaged_with_user_id: string,
    # *  engaged_with_user_follower_count: int,
    # *  engaged_with_user_following_count: int,
    # *  engaged_with_user_is_verified: boolean,
    # *  engaged_with_user_account_creation: int,
    # *  enaging_user_id: string,
    # *  enaging_user_follower_count: int,
    # *  enaging_user_following_count: int,
    # *  enaging_user_is_verified: boolean,
    # *  enaging_user_account_creation: int,
    # *  engagee_follows_engager: boolean,
    # *  reply_timestamp: float,
    # *  retweet_timestamp: float,
    # *  retweet_with_comment_timestamp: float,
    # *  like_timestamp: float
    #
    # ]

    ##### 1. Start spark and initialize data processor #####
    t0 = timer()
    spark = SparkSession.builder.master('yarn').appName(
        "Recsys2021_DATA_PROCESS").getOrCreate()
    # Enable Arrow-based columnar data transfers
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    schema = recsysSchema.toStructType()
    df = spark.read.schema(schema).option('sep', '\x01').csv(path)

    proc = DataProcessor(spark)

    # basic: Do categorify for all columns for xgboost
    df = categorifyAllFeatures(df, proc)

    # optional: do bert decode
    #df = decodeBertTokenizer(df, proc)

    # optional: adding new features upon tweet text
    #df = decodeBertTokenizer(df, proc)


def categorifyAllFeatures(df, proc):

    ###### 2. First phase data engineering(part 1) ######
    # pre-define
    # 0.1 define udfs
    replace = udf(lambda x:  '_'.join(
        x.split('\t')[:2]) if x else "", StringType())
    count = udf(lambda x: str(x).count('\t')+1 if x else 0, LongType())
    # 0.2 define dictionary
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

    # 1. define operations

    # 1.1 fill na
    op_fillna_num = FillNA(['reply_timestamp', 'retweet_timestamp',
                            'retweet_with_comment_timestamp', 'like_timestamp'], 0)
    op_fillna_str = FillNA(
        ['present_domains', 'present_links', 'hashtags', 'present_media'], "")

    # 1.2 feature modify and add
    op_feature_modification_type_convert = FeatureModification(cols=['tweet_timestamp',
                                                                     'engaged_with_user_follower_count',
                                                                     'engaged_with_user_following_count',
                                                                     'engaged_with_user_account_creation',
                                                                     'enaging_user_follower_count',
                                                                     'enaging_user_following_count',
                                                                     'enaging_user_account_creation', 'reply_timestamp',
                                                                     'retweet_timestamp',
                                                                     'retweet_with_comment_timestamp',
                                                                     'like_timestamp'], op='toInt')
    op_feature_modification_present_media_replace = FeatureModification(
        cols={'present_media': "concat_ws('_', split(col('present_media'),'\t'))"}, op='inline')
    # op_feature_modification_present_media_replace = FeatureModification(
    #    cols=['present_media'], udfImpl=replace)

    # op_feature_add_len = FeatureAdd(
    #    cols={'len_hashtags': 'hashtags', 'len_domains': 'present_domains', 'len_links': 'present_links'}, udfImpl=count)
    op_feature_add_len_hashtags = FeatureAdd(
        cols={'len_hashtags': "when(col('hashtags') == '', lit(0)).otherwise(size(split(col('hashtags'), '\t')))"}, op='inline')
    op_feature_add_len_domains = FeatureAdd(
        cols={'len_domains': "when(col('present_domains') == '', lit(0)).otherwise(size(split(col('present_domains'), '\t')))"}, op='inline')
    op_feature_add_len_links = FeatureAdd(
        cols={'len_links': "when(col('present_links') == '', lit(0)).otherwise(size(split(col('present_links'), '\t')))"}, op='inline')
    op_feature_add_engage_time = FeatureAdd(
        cols={'engage_time': "least(col('reply_timestamp'), col('retweet_timestamp'), col('retweet_with_comment_timestamp'), col('like_timestamp'))"}, op='inline')
    op_new_feature_dt_dow = FeatureAdd(cols={
        "dt_dow": "dayofweek(from_unixtime(col('tweet_timestamp'))).cast(IntegerType())",
        "dt_hour": "hour(from_unixtime(col('tweet_timestamp'))).cast(IntegerType())",
        "dt_minute": "minute(from_unixtime(col('tweet_timestamp'))).cast(IntegerType())",
        "dt_second": "second(from_unixtime(col('tweet_timestamp'))).cast(IntegerType())"}, op='inline')

    op_feature_change = FeatureModification(cols={
        "reply_timestamp": "when(col('reply_timestamp') > 0, 1).otherwise(0)",
        "retweet_timestamp": "when(col('retweet_timestamp') > 0, 1).otherwise(0)",
        "retweet_with_comment_timestamp": "when(col('retweet_with_comment_timestamp') > 0, 1).otherwise(0)",
        "like_timestamp": "when(col('like_timestamp') > 0, 1).otherwise(0)"}, op='inline')

    ops = [op_fillna_num, op_fillna_str,
           op_feature_modification_type_convert, op_feature_modification_present_media_replace,
           op_feature_add_len_hashtags, op_feature_add_len_domains, op_feature_add_len_links,
           op_feature_add_engage_time, op_new_feature_dt_dow, op_feature_change]
    proc.reset_ops(ops)

    # 1.3 categorify
    # udf took lots of memory, process in advance
    op_categorifyMultiItems = CategorifyMultiItems(
        ['present_domains', 'present_links', 'hashtags'])
    op_categorify_present_media = CategorifyWithDictionary(
        ['present_media'], media)
    op_categorify_tweet_type = CategorifyWithDictionary(
        ['tweet_type'], tweet_type)
    # since language dict is small, we may use udf to make partition more even
    op_categorify_language = Categorify(['language'], hint='udf')

    ops_1 = [op_categorifyMultiItems, op_categorify_present_media,
             op_categorify_tweet_type, op_categorify_language]
    proc.append_ops(ops_1)
    t1 = timer()
    df = proc.transform(df)
    t2 = timer()
    print("Data Process and udf categorify took %.3f" % (t2 - t1))

    ###### 3. First phase data engineering(part 2) ######
    # since we observed extremely high mem footage to
    # do below joins, split each run to save memory

    op_categorify_tweet_id = Categorify(['tweet_id'])
    proc.reset_ops([op_categorify_tweet_id])
    t1 = timer()
    df = proc.transform(df)
    t2 = timer()

    op_categorify_user_id = Categorify(['engaged_with_user_id', 'enaging_user_id'], src_cols=[
        'engaged_with_user_id', 'enaging_user_id'])
    proc.reset_ops([op_categorify_user_id])
    t5 = timer()
    df = proc.transform(df)
    t6 = timer()

    print("Categorify w/join took %.3f %.3f" % ((t2 - t1), (t6 - t5)))

    df.write.format('parquet').mode(
        'overwrite').save("/recsys2021/categorified/")
    return df


def decodeBertTokenizer(df, proc):
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=False)

    # define UDF
    tokenizer_decode = udf(lambda x: tokenizer.decode(
        [int(n) for n in x.split('\t')]))
    format_url = udf(lambda x: x.replace(
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
    df = proc.transform(df)
    t2 = timer()
    print("BertTokenizer decode and format took %.3f" % (t2 - t1))

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
    to_notsign = udf(lambda x: x.replace('\[CLS\] RT @', ''))
    count_space = udf(lambda x: x.count(' '))
    count_text_length = udf(lambda x: len(x))
    user_defined_hash = udf(lambda x: extract_hash(x, split_text='RT @', no=0))
    # features upon tweet_nortsign
    count_at = udf(lambda x: x.count('@'))
    user_define_hash_1 = udf(lambda x: extract_hash(x))
    user_define_hash_2 = udf(lambda x: extract_hash(x, no=1))

    # features upon tweet
    op_feature_add_tweet_nortsign = FeatureAdd(
        cols={'tweet_nortsign': 'tweet'}, udfImpl=to_notsign)
    op_feature_add_count_words = FeatureAdd(
        cols={'count_words': 'tweet'}, udfImpl=count_space)
    op_feature_add_count_char = FeatureAdd(
        cols={'count_char': 'tweet'}, udfImpl=count_text_length)
    op_feature_add_tw_uhash = FeatureAdd(
        cols={'tw_uhash': 'tweet'}, udfImpl=user_defined_hash)
    op_feature_add_tw_hash = FeatureAdd(
        cols={'tw_hash': "hash(col('tweet'))%1000000000"}, op='inline')
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
    df = proc.transform(df)
    t2 = timer()
    print("Adding Feature upon tweet and tweet_nortsign column took %.3f" % (t2 - t1))
    # expect to spend about 1000secs

    df.write.format('parquet').mode(
        'overwrite').save("/recsys2021/bertTokenizeDecoded/")
    return df


def tweetFreatureEngineer(df, proc):
    proc = DataProcessor(spark)
    op_fillna_for_tweet = FillNA(['tweet'], "")
    op_categorify_multiple_tweet = CategorifyMultiItems(
        ['tweet'], strategy=1, sep=' ', skipList=['', '[', ']', '.', '!', '@', '_', '#'])
    proc.reset_ops([op_fillna_for_tweet, op_categorify_multiple_tweet])
    t1 = timer()
    df = proc.transform(df)
    t2 = timer()
    print("Frequency encode tweet column took %.3f" % (t2 - t1))

    op_feature_add_tw_freq_hash = FeatureAdd(
        {'tw_freq_hash': "col('tw_hash')"}, op='inline')
    op_feature_add_tw_first_word = FeatureAdd(
        {'tw_first_word': "col('tweet').getItem(0)"}, op='inline')
    op_feature_add_tw_second_word = FeatureAdd(
        {'tw_second_word': "col('tweet').getItem(1)"}, op='inline')
    op_feature_add_tw_last_word = FeatureAdd(
        {'tw_last_word': "col('tweet').getItem(size(col('tweet')) - 1)"}, op='inline')
    op_feature_add_tw_second_last_word = FeatureAdd(
        {'tw_llast_word': "col('tweet').getItem(size(col('tweet')) - 1)"}, op='inline')
    op_feature_add_tw_word_len = FeatureAdd(
        {'tw_len': "size(col('tweet'))"}, op='inline')
    op_feature_modification_fillna = FillNA(
        ['tw_freq_hash', 'tw_first_word', 'tw_second_word', 'tw_last_word', 'tw_llast_word', 'tw_len'], -1)

    proc.reset_ops([op_feature_add_tw_freq_hash, op_feature_add_tw_first_word, op_feature_add_tw_second_word,
                    op_feature_add_tw_last_word, op_feature_add_tw_second_last_word, op_feature_add_tw_word_len,
                    op_feature_modification_fillna])
    t1 = timer()
    df = proc.transform(df)
    t2 = timer()
    print("feature engineering upon Frequency encoded tweet column took %.3f" % (t2 - t1))

    df.write.format('parquet').mode(
        'overwrite').save("/recsys2021/tweet_feature_engineer/")
    return df


if __name__ == "__main__":
    main()
