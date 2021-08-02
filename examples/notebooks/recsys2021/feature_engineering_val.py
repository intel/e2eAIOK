#!/env/bin/python

import init

import findspark
findspark.init()

import os
import pandas as pd
import numpy as np
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
import pyspark.sql.types as t
from timeit import default_timer as timer
import logging
from RecsysSchema import RecsysSchema
from pyrecdp.data_processor import *
from pyrecdp.encoder import *
from pyrecdp.utils import *
import hashlib

target_list = [
 'reply_timestamp',
 'retweet_timestamp',
 'retweet_with_comment_timestamp',
 'like_timestamp'
]

final_feature_list = [
 'engaged_with_user_follower_count',
 'engaged_with_user_following_count',
 'engaged_with_user_is_verified',
 'engaging_user_follower_count',
 'engaging_user_following_count',
 'engaging_user_is_verified',
 'has_photo',
 'has_video',
 'has_gif',
 'a_ff_rate',
 'b_ff_rate', 
 'dt_hour',
 'dt_dow',
 'has_mention',  
 'mentioned_bucket_id',    
 'mentioned_count',    
 'most_used_word_bucket_id',
 'second_used_word_bucket_id',
 'TE_tweet_type_reply_timestamp',
 'TE_tweet_type_retweet_timestamp',
 'TE_dt_dow_retweet_timestamp',
 'TE_most_used_word_bucket_id_reply_timestamp',
 'TE_most_used_word_bucket_id_retweet_timestamp',
 'TE_most_used_word_bucket_id_retweet_with_comment_timestamp',
 'TE_most_used_word_bucket_id_like_timestamp',
 'TE_second_used_word_bucket_id_reply_timestamp',
 'TE_second_used_word_bucket_id_retweet_timestamp',
 'TE_second_used_word_bucket_id_retweet_with_comment_timestamp',
 'TE_second_used_word_bucket_id_like_timestamp',
 'TE_mentioned_bucket_id_retweet_timestamp',
 'TE_mentioned_bucket_id_retweet_with_comment_timestamp',
 'TE_mentioned_bucket_id_like_timestamp',
 'TE_mentioned_bucket_id_reply_timestamp',
 'TE_language_reply_timestamp',
 'TE_language_retweet_timestamp',
 'TE_language_retweet_with_comment_timestamp',
 'TE_language_like_timestamp',
 'TE_mentioned_count_reply_timestamp',
 'TE_mentioned_count_retweet_timestamp',
 'TE_mentioned_count_retweet_with_comment_timestamp',
 'TE_mentioned_count_like_timestamp',
 'TE_engaged_with_user_id_reply_timestamp',
 'TE_engaged_with_user_id_retweet_timestamp',
 'TE_engaged_with_user_id_retweet_with_comment_timestamp',
 'TE_engaged_with_user_id_like_timestamp',
 'GTE_language_engaged_with_user_id_reply_timestamp',
 'GTE_language_engaged_with_user_id_retweet_timestamp',
 'GTE_language_engaged_with_user_id_retweet_with_comment_timestamp',
 'GTE_language_engaged_with_user_id_like_timestamp',
 'GTE_tweet_type_engaged_with_user_id_reply_timestamp',
 'GTE_tweet_type_engaged_with_user_id_retweet_timestamp',
 'GTE_tweet_type_engaged_with_user_id_retweet_with_comment_timestamp',
 'GTE_tweet_type_engaged_with_user_id_like_timestamp',
 'GTE_has_mention_engaging_user_id_reply_timestamp',
 'GTE_has_mention_engaging_user_id_retweet_timestamp',
 'GTE_has_mention_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_has_mention_engaging_user_id_like_timestamp',
 'GTE_mentioned_bucket_id_engaging_user_id_reply_timestamp',
 'GTE_mentioned_bucket_id_engaging_user_id_retweet_timestamp',
 'GTE_mentioned_bucket_id_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_mentioned_bucket_id_engaging_user_id_like_timestamp',
 'GTE_language_engaging_user_id_reply_timestamp',
 'GTE_language_engaging_user_id_retweet_timestamp',
 'GTE_language_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_language_engaging_user_id_like_timestamp',
 'GTE_tweet_type_engaging_user_id_reply_timestamp',
 'GTE_tweet_type_engaging_user_id_retweet_timestamp',
 'GTE_tweet_type_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_tweet_type_engaging_user_id_like_timestamp',
 'GTE_dt_dow_engaged_with_user_id_reply_timestamp',
 'GTE_dt_dow_engaged_with_user_id_retweet_timestamp',
 'GTE_dt_dow_engaged_with_user_id_retweet_with_comment_timestamp',
 'GTE_dt_dow_engaged_with_user_id_like_timestamp',
 'GTE_mentioned_count_engaging_user_id_reply_timestamp',
 'GTE_mentioned_count_engaging_user_id_retweet_timestamp',
 'GTE_mentioned_count_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_mentioned_count_engaging_user_id_like_timestamp',
 'GTE_dt_hour_engaged_with_user_id_reply_timestamp',
 'GTE_dt_hour_engaged_with_user_id_retweet_timestamp',
 'GTE_dt_hour_engaged_with_user_id_retweet_with_comment_timestamp',
 'GTE_dt_hour_engaged_with_user_id_like_timestamp',
 'GTE_dt_dow_engaging_user_id_reply_timestamp',
 'GTE_dt_dow_engaging_user_id_retweet_timestamp',
 'GTE_dt_dow_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_dt_dow_engaging_user_id_like_timestamp',
 'GTE_dt_hour_engaging_user_id_reply_timestamp',
 'GTE_dt_hour_engaging_user_id_retweet_timestamp',
 'GTE_dt_hour_engaging_user_id_retweet_with_comment_timestamp',
 'GTE_dt_hour_engaging_user_id_like_timestamp'
]

def decodeBertTokenizerAndExtractFeatures(df, proc, output_name):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=False)

    # define UDF
    tokenizer_decode = f.udf(lambda x: tokenizer.decode(
        [int(n) for n in x.split('\t')]))
    format_url = f.udf(lambda x: x.replace(
        'https : / / t. co / ', 'https://t.co/').replace('@ ', '@'))

    # define decode udf operations
    op_feature_modification_tokenizer_decode = FeatureAdd(
        cols={'tweet': 'text_tokens'}, udfImpl=tokenizer_decode)
    op_feature_modification_format_url = FeatureModification(
        cols=['tweet'], udfImpl=format_url)
    
    op_feature_target_classify = FeatureModification(cols={
        "reply_timestamp": "f.when(f.col('reply_timestamp') > 0, 1).otherwise(0)",
        "retweet_timestamp": "f.when(f.col('retweet_timestamp') > 0, 1).otherwise(0)",
        "retweet_with_comment_timestamp": "f.when(f.col('retweet_with_comment_timestamp') > 0, 1).otherwise(0)",
        "like_timestamp": "f.when(f.col('like_timestamp') > 0, 1).otherwise(0)"}, op='inline')
    
    # define new features
    op_feature_from_original = FeatureAdd(
        cols={"has_photo": "f.col('present_media').contains('Photo').cast(t.IntegerType())",
              "has_video": "f.col('present_media').contains('Vedio').cast(t.IntegerType())",
              "has_gif": "f.col('present_media').contains('GIF').cast(t.IntegerType())",             
              "a_ff_rate": "f.col('engaged_with_user_following_count')/f.col('engaged_with_user_follower_count')",
              "b_ff_rate": "f.col('engaging_user_following_count') /f.col('engaging_user_follower_count')",
              "dt_dow": "f.dayofweek(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
              "dt_hour": "f.hour(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",           
              "mention": "f.regexp_extract(f.col('tweet'), r'[^RT]\s@(\S+)', 1)",
              "has_mention": "(f.col('mention')!= '').cast(t.IntegerType())"
        }, op='inline')

    # execute
    proc.reset_ops([op_feature_modification_tokenizer_decode,
                    op_feature_modification_format_url,
                    op_feature_target_classify,
                    op_feature_from_original])
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("BertTokenizer decode and feature extacting took %.3f" % (t2 - t1))

    return df

def categorifyFeatures(df, proc, output_name, gen_dict, sampleRatio=1):
    # 1. prepare dictionary
    dict_dfs = []
    if gen_dict:
        # only call below function when target dicts were not pre-prepared
        op_gen_dict_multiItems = GenerateDictionary(['tweet'], doSplit=True, sep=' ', bucketSize=100)
        op_gen_dict_singleItems = GenerateDictionary(['mention'], bucketSize=100)
        proc.reset_ops([op_gen_dict_multiItems, op_gen_dict_singleItems])
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        dict_names = ['tweet', 'mention']
        dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in dict_names]
    # 2. since we need both mentioned_bucket_id and mentioned_count, add two mention id dict_dfs
    for dict_df in dict_dfs:
        if dict_df['col_name'] == 'mention':
            dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
            dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop('dict_col_id').withColumnRenamed('count', 'dict_col_id')})
    op_feature_add = FeatureAdd({"mentioned_bucket_id": "f.col('mention')", "mentioned_count": "f.col('mention')"}, op='inline')
    
    # 3. categorify
    op_categorify_multiItems = Categorify([{'bucketized_tweet_word': 'tweet'}], dict_dfs=dict_dfs, doSplit=True, sep=' ')
    op_categorify_singleItem = Categorify(['mentioned_bucket_id', 'mentioned_count'], dict_dfs=dict_dfs)
    proc.reset_ops([op_feature_add, op_categorify_multiItems, op_categorify_singleItem])
    
    # 4. get most and second used bucketized_tweet_word
    op_feature_add_sorted_bucketized_tweet_word = FeatureAdd(
        cols={'sorted_bucketized_tweet_word': "f.expr('sortIntArrayByFrequency(bucketized_tweet_word)')"}, op='inline')
    op_feature_add_convert = FeatureAdd(
        cols={'most_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>0, f.col('sorted_bucketized_tweet_word').getItem(0)).otherwise(np.nan)",
             'second_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>1, f.col('sorted_bucketized_tweet_word').getItem(1)).otherwise(np.nan)"}, op='inline')
    proc.append_ops([op_feature_add_sorted_bucketized_tweet_word, op_feature_add_convert])

    # 5. transform
    t1 = timer()
    if sampleRatio != 1:
        df = df.sample(sampleRatio)
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("categorify and getMostAndSecondUsedWordBucketId took %.3f" % (t2 - t1))
    return (df, dict_dfs)


def encodingFeatures(df, proc, output_name, gen_dict, sampleRatio=1):   
    targets = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    y_mean_all = []
    
    t1 = timer()
    if gen_dict:
        for tgt in targets:
            tmp = df.groupBy().mean(tgt).collect()[0]
            y_mean = tmp[f"avg({tgt})"]
            y_mean_all.append(y_mean)
        schema = t.StructType([t.StructField(tgt, t.FloatType(), True) for tgt in targets])
        y_mean_all_df = proc.spark.createDataFrame([tuple(y_mean_all)], schema)
        y_mean_all_df.write.format("parquet").mode("overwrite").save(
            "%s/%s/%s/targets_mean" % (proc.path_prefix, proc.current_path, proc.dicts_path))
    y_mean_all_df = proc.spark.read.parquet(
        "%s/%s/%s/targets_mean" % (proc.path_prefix, proc.current_path, proc.dicts_path))

    features = [
            'engaged_with_user_id',
            'language',
            'dt_dow',
            'tweet_type',
            'most_used_word_bucket_id',
            'second_used_word_bucket_id',
            'mentioned_count',
            'mentioned_bucket_id',
            ['has_mention', 'engaging_user_id'],
            ['mentioned_count', 'engaging_user_id'],
            ['mentioned_bucket_id', 'engaging_user_id'],
            ['language', 'engaged_with_user_id'],
            ['language', 'engaging_user_id'],
            ['dt_dow', 'engaged_with_user_id'],
            ['dt_dow', 'engaging_user_id'],
            ['dt_hour', 'engaged_with_user_id'],
            ['dt_hour', 'engaging_user_id'],
            ['tweet_type', 'engaged_with_user_id'],
            ['tweet_type', 'engaging_user_id']
    ]
    excludes = {'dt_dow': ['reply_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp'],
               'tweet_type': ['like_timestamp', 'retweet_with_comment_timestamp']
              }

    te_train_dfs = []
    te_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        if str(c) in excludes:
            target_tmp = []
            for tgt in targets:
                if tgt not in excludes[c]:
                    target_tmp.append(tgt)
        out_col_list = []
        for tgt in target_tmp:
            if isinstance(c, list):
                out_col_list.append('GTE_'+'_'.join(c)+'_'+tgt)
                out_name = 'GTE_'+'_'.join(c)
            else:
                out_col_list.append(f'TE_{c}_{tgt}')
                out_name = f'TE_{c}'
        if gen_dict:
            start = timer()
            encoder = TargetEncoder(proc, c, target_tmp, out_col_list, out_name, out_dtype=t.FloatType(), y_mean_list=y_mean_all)
            te_train_df, te_test_df = encoder.transform(df)
            te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': te_train_df})
            te_test_dfs.append({'col_name': c, 'dict': te_test_df})
            print(f"generating target encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))
        else:
            te_train_path = "%s/%s/%s/train/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
            te_test_path = "%s/%s/%s/test/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)               
            te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': proc.spark.read.parquet(te_train_path)})
            te_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(te_test_path)})
    t2 = timer()
    print("Generate encoding feature totally took %.3f" % (t2 - t1))

    # merge dicts to original table
    op_merge_to_train = ModelMerge(te_train_dfs)
    proc.reset_ops([op_merge_to_train])
    
    # select features
    op_select = SelectFeature(target_list + final_feature_list)
    proc.append_ops([op_select])

    t1 = timer()
    if sampleRatio != 1:
        df = df.sample(sampleRatio)
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("encodingFeatures took %.3f" % (t2 - t1))
    
    return (df, te_train_dfs, te_test_dfs, y_mean_all_df)


def splitByDate(df, proc, train_output, test_output, numFolds=5):
    # 1.1 get timestamp range
    import datetime
    min_timestamp = df.select('tweet_timestamp').agg({'tweet_timestamp': 'min'}).collect()[0]['min(tweet_timestamp)']
    max_timestamp = df.select('tweet_timestamp').agg({'tweet_timestamp': 'max'}).collect()[0]['max(tweet_timestamp)']
    seconds_in_day = 3600 * 24

    print(
        "min_timestamp is %s, max_timestamp is %s, 20 days max is %s" % (
            datetime.datetime.fromtimestamp(min_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.datetime.fromtimestamp(max_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            datetime.datetime.fromtimestamp(min_timestamp + 20 * seconds_in_day).strftime('%Y-%m-%d %H:%M:%S')
        ))

    time_range_split = {
        'train': (min_timestamp, seconds_in_day * 18 + min_timestamp),
        'test': (seconds_in_day * 18 + min_timestamp, max_timestamp)
    }

    print(time_range_split)

    # 1.2 save ranged data for train
    # filtering out train range data and save
    train_start, train_end = time_range_split['train']
    test_start, test_end = time_range_split['test']
    t1 = timer()
    train_df = df.filter(
        (f.col('tweet_timestamp') >= f.lit(train_start)) & (f.col('tweet_timestamp') < f.lit(train_end)))
    # train_df = train_df.withColumn("fold", f.round(f.rand(seed=42)*numFolds))
    train_df = train_df.withColumn("fold", f.round(f.rand(seed=42)*(numFolds-1)).cast("int"))
    train_df.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + train_output)
    t2 = timer()
    print("split to train took %.3f" % (t2 - t1))
    
    t1 = timer()
    test_df = df.filter(
        (f.col('tweet_timestamp') >= f.lit(test_start)) & (f.col('tweet_timestamp') < f.lit(test_end)))
    test_df.write.format('parquet').mode('overwrite').save(proc.path_prefix + proc.current_path + test_output)
    t2 = timer()
    print("split to test took %.3f" % (t2 - t1))
    
    return (proc.spark.read.parquet(proc.path_prefix + proc.current_path + train_output),
            proc.spark.read.parquet(proc.path_prefix + proc.current_path + test_output))


def mergeFeaturesToTest(df, dict_dfs, te_test_dfs, y_mean_all_df, proc, output_name):
    # categorify test data with train generated dictionary
    # 1. since we need both mentioned_bucket_id and mentioned_count, add two mention id dict_dfs
    for dict_df in dict_dfs:
        if dict_df['col_name'] == 'mention':
            dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
            dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop('dict_col_id').withColumnRenamed('count', 'dict_col_id')})
    op_feature_add = FeatureAdd({"mentioned_bucket_id": "f.col('mention')", "mentioned_count": "f.col('mention')"}, op='inline')
    
    # 2. categorify
    op_categorify_multiItems = Categorify([{'bucketized_tweet_word': 'tweet'}], dict_dfs=dict_dfs, doSplit=True, sep=' ')
    op_categorify_singleItem = Categorify(['mentioned_bucket_id', 'mentioned_count'], dict_dfs=dict_dfs)
    proc.reset_ops([op_feature_add, op_categorify_multiItems, op_categorify_singleItem])
    
    # 3. get most and second used bucketized_tweet_word
    op_feature_add_sorted_bucketized_tweet_word = FeatureAdd(
        cols={'sorted_bucketized_tweet_word': "f.expr('sortIntArrayByFrequency(bucketized_tweet_word)')"}, op='inline')
    op_feature_add_convert = FeatureAdd(
        cols={'most_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>0, f.col('sorted_bucketized_tweet_word').getItem(0)).otherwise(np.nan)",
             'second_used_word_bucket_id': "f.when(f.size(f.col('sorted_bucketized_tweet_word'))>1, f.col('sorted_bucketized_tweet_word').getItem(1)).otherwise(np.nan)"}, op='inline')
    proc.append_ops([op_feature_add_sorted_bucketized_tweet_word, op_feature_add_convert])
    
    # 4. merge dicts to original table
    op_merge_to_test = ModelMerge(te_test_dfs)
    proc.append_ops([op_merge_to_test])
        
    # 5. set null in encoding features to y_mean
    y_mean_all = y_mean_all_df.collect()[0]
    for tgt in target_list:
        to_fill_list = []
        for feature in final_feature_list:
            if 'TE_' in feature and tgt in feature:
                to_fill_list.append(feature)
        op_fill_na = FillNA(to_fill_list, y_mean_all[tgt])
        proc.append_ops([op_fill_na])
    
    # select features
    op_select = SelectFeature(target_list + final_feature_list)
    proc.append_ops([op_select])

    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("mergeFeaturesToTest took %.3f" % (t2 - t1))


def get_encoding_features_dicts(proc):
    targets = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    y_mean_all = []
    y_mean_all_df = proc.spark.read.parquet(
        "%s/%s/%s/targets_mean" % (proc.path_prefix, proc.current_path, proc.dicts_path))
    features = [
            'engaged_with_user_id',
            'language',
            'dt_dow',
            'tweet_type',
            'most_used_word_bucket_id',
            'second_used_word_bucket_id',
            'mentioned_count',
            'mentioned_bucket_id',
            ['has_mention', 'engaging_user_id'],
            ['mentioned_count', 'engaging_user_id'],
            ['mentioned_bucket_id', 'engaging_user_id'],
            ['language', 'engaged_with_user_id'],
            ['language', 'engaging_user_id'],
            ['dt_dow', 'engaged_with_user_id'],
            ['dt_dow', 'engaging_user_id'],
            ['dt_hour', 'engaged_with_user_id'],
            ['dt_hour', 'engaging_user_id'],
            ['tweet_type', 'engaged_with_user_id'],
            ['tweet_type', 'engaging_user_id']
    ]
    excludes = {'dt_dow': ['reply_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp'],
               'tweet_type': ['like_timestamp', 'retweet_with_comment_timestamp']
              }

    te_train_dfs = []
    te_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        if str(c) in excludes:
            target_tmp = []
            for tgt in targets:
                if tgt not in excludes[c]:
                    target_tmp.append(tgt)
        out_col_list = []
        for tgt in target_tmp:
            if isinstance(c, list):
                out_col_list.append('GTE_'+'_'.join(c)+'_'+tgt)
                out_name = 'GTE_'+'_'.join(c)
            else:
                out_col_list.append(f'TE_{c}_{tgt}')
                out_name = f'TE_{c}'
        te_train_path = "%s/%s/%s/train/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
        te_test_path = "%s/%s/%s/test/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
        te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': proc.spark.read.parquet(te_train_path)})
        te_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(te_test_path)})
        
    return (te_train_dfs, te_test_dfs, y_mean_all_df)

def main():
    path_prefix = "hdfs://localhost:9000/"
    current_path = "/recsys2021/feateng/"
    original_folder = "/recsys2021/oridata/valid/"
    dicts_folder = "recsys_dicts/"
    recsysSchema = RecsysSchema()

    ##### 1. Start spark and initialize data processor #####
    scala_udf_jars = "/home/vmagent/app/script/test/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    t0 = timer()
    spark = SparkSession.builder.master('local[64]')\
        .appName("Recsys2021_data_process")\
        .config("spark.driver.memory", '450g')\
        .config("spark.worker.memory", "450g")\
        .config("spark.local.dir", "/home/vmagent/app/data")\
        .config("spark.sql.broadcastTimeout", "7200")\
        .config("spark.cleaner.periodicGC.interval", "15min")\
        .config("spark.executorEnv.HF_DATASETS_OFFLINE", "1")\
        .config("spark.executorEnv.TRANSFORMERS_OFFLINE", "1")\
        .config("spark.executor.memory", "20g")\
        .config("spark.executor.memoryOverhead", "10g")\
        .config("spark.executor.cores", "64")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .config("spark.executor.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()

    schema = recsysSchema.toStructType()

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix,
                        current_path=current_path, dicts_path=dicts_folder, shuffle_disk_capacity="1500GB",spark_mode='local')
    df = spark.read.schema(schema).option('sep', '\x01').csv(path_prefix + original_folder)
    df = df.withColumnRenamed('enaging_user_follower_count', 'engaging_user_follower_count')
    df = df.withColumnRenamed('enaging_user_is_verified', 'engaging_user_is_verified')
    df = df.withColumnRenamed('enaging_user_following_count', 'engaging_user_following_count')
    df = df.withColumnRenamed('enaging_user_account_creation', 'engaging_user_account_creation')
    print("data loaded!")

    # # fast test, comment for full dataset
    # df.sample(0.01).write.format("parquet").mode("overwrite").save(path_prefix+"%s/sample_0_0_1" % current_path)
    # df = spark.read.parquet(path_prefix+"%s/sample_0_0_1" % current_path)
    # print("data sampled!")

    # ===============================================
    # decode tweet_tokens
    df = decodeBertTokenizerAndExtractFeatures(df, proc, output_name="decoded_with_extracted_features_val")
    print("data decoded!")

    # ===============================================
    # splitting and sampling
    # df, test_df = splitByDate(df, proc, train_output="train", test_output="test", numFolds=5)
    # print("data splited!")

    # ===============================================
    # generate dictionary for categorify indexing
    # df, dict_dfs = categorifyFeatures(df, proc, output_name="train_with_categorified_features", gen_dict=True, sampleRatio=1)
    # print("data categorified!")

    # ===============================================
    # encoding features
    # df, te_train_dfs, te_test_dfs, y_mean_all_df = encodingFeatures(df, proc, output_name="train_with_features_sample_0_0_3", gen_dict=True, sampleRatio=0.03)
    # print("data encoded!")

    # ===============================================
    # adding features to test
    ### Below codes is used to prepare for mergeFeaturesToTest for test separately ###
    # df = spark.read.parquet(path_prefix+current_path+"decoded_with_extracted_features_val")
    dict_names = ['tweet', 'mention']
    dict_dfs = [{'col_name': name, 'dict': spark.read.parquet(
        "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in dict_names]
    te_train_dfs, te_val_dfs, y_mean_all_df = get_encoding_features_dicts(proc)
    ##################################################################################
    val_df = mergeFeaturesToTest(df, dict_dfs, te_val_dfs, y_mean_all_df, proc, output_name="val_with_features")
    print("val data merged!")

if __name__ == "__main__":
    main()
    