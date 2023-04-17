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
from timeit import default_timer as timer
import logging
from RecsysSchema import RecsysSchema
from pyrecdp.data_processor import *
from pyrecdp.encoder import *
from pyrecdp.utils import *
import hashlib


def decodeBertTokenizerAndExtractFeatures(df, proc, output_name):
    # modification on original feature
    op_fillna_str = FillNA(
        ['present_domains', 'present_links', 'hashtags', 'present_media', 'tweet_id'], "")
    op_fillna_num = FillNA(['reply_timestamp', 'retweet_timestamp',
                        'retweet_with_comment_timestamp', 'like_timestamp'], 0)
    
    op_feature_modification_type_convert = FeatureModification(cols=['tweet_timestamp',
                                                                     'engaged_with_user_follower_count',
                                                                     'engaged_with_user_following_count',
                                                                     'engaged_with_user_account_creation',
                                                                     'engaging_user_follower_count',
                                                                     'engaging_user_following_count',
                                                                     'engaging_user_account_creation',
                                                                     'reply_timestamp',
                                                                     'retweet_timestamp',
                                                                     'retweet_with_comment_timestamp',
                                                                     'like_timestamp'], op='toInt')
    
    op_feature_target_classify = FeatureModification(cols={
        "reply_timestamp": "f.when(f.col('reply_timestamp') > 0, 1).otherwise(0)",
        "retweet_timestamp": "f.when(f.col('retweet_timestamp') > 0, 1).otherwise(0)",
        "retweet_with_comment_timestamp": "f.when(f.col('retweet_with_comment_timestamp') > 0, 1).otherwise(0)",
        "like_timestamp": "f.when(f.col('like_timestamp') > 0, 1).otherwise(0)"}, op='inline')
    
    # adding new features
    op_feature_to_be_categorified = FeatureAdd(
        cols={"present_domains_indicator": "f.col('present_domains')",\
              "present_links_indicator": "f.col('present_links')",\
              "hashtags_indicator": "f.col('hashtags')",\
              "language_indicator": "f.col('language')",\
              "tweet_id_indicator": "f.col('tweet_id')",\
              "present_media_indicator": "f.col('present_media')",\
              "tweet_type_indicator": "f.col('tweet_type')",\
              "engaged_with_user_id_indicator": "f.col('engaged_with_user_id')",\
              "engaging_user_id_indicator": "f.col('engaging_user_id')"
             }, 
        op='inline')
    
    op_feature_from_original = FeatureAdd(
        cols={"has_photo": "f.col('present_media').contains('Photo').cast(t.IntegerType())",
              "has_video": "f.col('present_media').contains('Vedio').cast(t.IntegerType())",
              "has_gif": "f.col('present_media').contains('GIF').cast(t.IntegerType())",             
              "a_ff_rate": "f.col('engaged_with_user_following_count')/f.col('engaged_with_user_follower_count')",
              "b_ff_rate": "f.col('engaging_user_following_count') /f.col('engaging_user_follower_count')",
              "dt_dow": "f.dayofweek(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
              "dt_hour": "f.hour(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",  
              "dt_minute": "f.minute(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
              "dt_second": "f.second(f.from_unixtime(f.col('tweet_timestamp'))).cast(t.IntegerType())",
              'present_domains_indicator': "f.concat_ws('_', f.slice(f.split(f.col('present_domains_indicator'),'\t'), 1, 2))",
              'len_hashtags': "f.when(f.col('hashtags') == '', f.lit(0)).otherwise(f.size(f.split(f.col('hashtags'), '\t')))",
              'len_domains': "f.when(f.col('present_domains') == '', f.lit(0)).otherwise(f.size(f.split(f.col('present_domains'), '\t')))",
              'len_links': "f.when(f.col('present_links') == '', f.lit(0)).otherwise(f.size(f.split(f.col('present_links'), '\t')))",
              'engage_time': "f.least(f.col('reply_timestamp'), f.col('retweet_timestamp'), f.col('retweet_with_comment_timestamp'), f.col('like_timestamp'))"   
        }, op='inline')
    op_fillna_tweet_timestamp = FillNA(['tweet_timestamp'], -1)
    ops = [op_fillna_str, op_fillna_num,
           op_feature_modification_type_convert, op_feature_target_classify,
           op_feature_to_be_categorified, op_feature_from_original, op_fillna_tweet_timestamp]

    ########## Tweet token feature engineering ##########
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased', do_lower_case=False)

    # define UDF
    tokenizer_decode = f.udf(lambda x: tokenizer.decode(
        [int(n) for n in x.split('\t')]))
    format_url = f.udf(lambda x: x.replace(
        'https : / / t. co / ', 'https://t.co/').replace('@ ', '@'))

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
        hash_object = hashlib.md5(uhash.encode('utf-8')) #nosec
        return hash_object.hexdigest()

    def clean_text(text):
        if len(text) > 1:
            if text[-1] in ['!', '?', ':', ';', '.', ',']:
                return(text[:-1])
        return(text)
    
    # udf defines upon tweet
    to_notsign = f.udf(lambda x: x.replace('\[CLS\] RT @', ''))
    count_space = f.udf(lambda x: x.count(' '))
    count_text_length = f.udf(lambda x: len(x))
    user_defined_hash = f.udf(lambda x: extract_hash(x, split_text='RT @', no=0))
    count_at = f.udf(lambda x: x.count('@'))
    user_define_hash_1 = f.udf(lambda x: extract_hash(x))
    user_define_hash_2 = f.udf(lambda x: extract_hash(x, no=1))

    # decode
    op_feature_modification_tokenizer_decode = FeatureAdd(cols={'tweet': 'text_tokens'}, udfImpl=tokenizer_decode)
    op_feature_modification_format_url = FeatureModification(cols=['tweet'], udfImpl=format_url)

    # adding new features
    op_feature_add_tweet_indicator = FeatureAdd(cols={"tweet_indicator": "f.col('tweet')"}, op='inline')
    op_feature_add_tweet_nortsign = FeatureAdd(cols={'tweet_nortsign': 'tweet'}, udfImpl=to_notsign)
    op_feature_add_count_words = FeatureAdd(cols={'count_words': 'tweet'}, udfImpl=count_space)
    op_feature_add_count_char = FeatureAdd(cols={'count_char': 'tweet'}, udfImpl=count_text_length)
    op_feature_add_tw_uhash = FeatureAdd(cols={'tw_uhash': 'tweet'}, udfImpl=user_defined_hash)
    op_feature_add_tw_hash = FeatureAdd(cols={'tw_hash': "f.hash(f.col('tweet'))%1000000000"}, op='inline')
    # features upon tweet_nortsign
    op_feature_add_count_at = FeatureAdd(cols={'count_ats': 'tweet_nortsign'}, udfImpl=count_at)
    op_feature_add_tw_uhash0 = FeatureAdd(cols={'tw_hash0': 'tweet_nortsign'}, udfImpl=user_define_hash_1)
    op_feature_add_tw_uhash1 = FeatureAdd(cols={'tw_hash1': 'tweet_nortsign'}, udfImpl=user_define_hash_2)

    ops += [op_feature_modification_tokenizer_decode, op_feature_modification_format_url,
            op_feature_add_tweet_indicator,
            op_feature_add_tweet_nortsign, op_feature_add_count_words, op_feature_add_count_char,
            op_feature_add_tw_uhash, op_feature_add_tw_hash, op_feature_add_count_at,
            op_feature_add_tw_uhash0, op_feature_add_tw_uhash1]
    proc.reset_ops(ops)

    # execute
    t1 = timer()
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("BertTokenizer decode and feature extacting took %.3f" % (t2 - t1))

    return df

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
    train_df = train_df.withColumn("fold", f.round(f.rand(seed=42)*numFolds))
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


def categorifyFeatures(df, proc, output_name="train_with_categorified_features", gen_dict=True, sampleRatio=1):
    if gen_dict:
        # only call below function when target dicts were not pre-prepared
        op_multiItems = GenerateDictionary(
            ['present_domains_indicator', 'present_links_indicator', 'hashtags_indicator'], doSplit=True)
        op_singleItems = GenerateDictionary(
            ['tweet_id_indicator', 'language_indicator', 
             {'src_cols': ['engaged_with_user_id_indicator', 'engaging_user_id_indicator'], 'col_name': 'user_id'}])
        op_tweet = GenerateDictionary(
            ['tweet_indicator'], doSplit=True, withCount=True, sep=' ')
        
        proc.reset_ops([op_multiItems, op_singleItems, op_tweet])
        t1 = timer()
        dict_dfs = proc.generate_dicts(df)
        t2 = timer()
        print("Generate Dictionary took %.3f" % (t2 - t1))
    else:
        # or we can simply load from pre-gened
        dict_names = ['hashtags_indicator', 'language_indicator', 'present_domains_indicator',
                      'present_links_indicator', 'tweet_id_indicator', 'user_id_indicator', 'tweet_indicator']
        dict_dfs = [{'col_name': name, 'dict': proc.spark.read.parquet(
            "%s/%s/%s/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, name))} for name in dict_names]

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

    dict_dfs.append({'col_name': 'present_media_indicator', 'dict': media_df})
    dict_dfs.append({'col_name': 'tweet_type_indicator', 'dict': tweet_type_df})

    for i in dict_dfs:
        dict_name = i['col_name']
        dict_df = i['dict']
        print("%s has numRows as %d" % (dict_name, dict_df.count()))

    ###### 2. define operations and append them to data processor ######

    # 1. define operations
    # 1.1 filter on tweet dict
    i = 0
    for dict_df in dict_dfs:
        if dict_df['col_name'] == 'tweet':
            tweet_dict_df = dict_df['dict']
            df_cnt = tweet_dict_df.count()
            freqRange = [2, df_cnt * 0.9]
            tweet_dict_df = tweet_dict_df.filter((f.col('count') <= f.lit(freqRange[1])) & (f.col('count') >= f.lit(freqRange[0])))
            dict_dfs[i]['dict'] = tweet_dict_df
        i += 1      

    # 1.3 categorify
    # since language dict is small, we may use udf to make partition more even
    op_categorify_1 = Categorify(
        ['present_domains_indicator', 'present_links_indicator', 'hashtags_indicator'], dict_dfs=dict_dfs, doSplit=True, keepMostFrequent=True)
    op_categorify_2 = Categorify(['language_indicator', 'present_media_indicator', 'tweet_type_indicator'], dict_dfs=dict_dfs)
    op_categorify_3 = Categorify([{'engaged_with_user_id_indicator': 'user_id'}, {'engaging_user_id_indicator': 'user_id'}], dict_dfs=dict_dfs)
    op_categorify_4 = Categorify(['tweet_id_indicator'], dict_dfs=dict_dfs)
    op_categorify_5 = Categorify(['tweet_indicator'], dict_dfs=dict_dfs, doSplit=True, sep=' ', doSortForArray=True)
    #### below are features upon categorified tweet
    op_feature_add_tw_word = FeatureAdd({
        'tw_first_word': "f.col('tweet_indicator').getItem(0)",
        'tw_second_word': "f.col('tweet_indicator').getItem(1)",
        'tw_last_word': "f.col('tweet_indicator').getItem(f.size(f.col('tweet_indicator')) - 1)",
        'tw_llast_word': "f.col('tweet_indicator').getItem(f.size(f.col('tweet_indicator')) - 1)",
        'tw_len': "f.size(f.col('tweet_indicator'))"
    }, op='inline')
    ops = [op_categorify_1, op_categorify_2, op_categorify_3, op_categorify_4, op_categorify_5, op_feature_add_tw_word]
    proc.reset_ops(ops)

    t1 = timer()
    if sampleRatio < 1 and sampleRatio > 0:
        df = df.sample(sampleRatio)
    df = proc.transform(df, name=output_name)
    t2 = timer()
    print("categorify took %.3f" % (t2 - t1))    
    return df, dict_dfs


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

    te_features = [
        'present_media',
        'tweet_type',
        'language',
        'engaged_with_user_id',
        'engaging_user_id',
        ['present_domains','language','engagee_follows_engager','tweet_type','present_media','engaged_with_user_is_verified'],
        ['engaged_with_user_id','tweet_type','language'],
        ['tw_first_word','tweet_type','language'],
        ['tw_last_word','tweet_type','language'],
        ['tw_hash0','tweet_type','language'],
        ['tw_hash1','tweet_type','language'],
        ['tw_uhash','tweet_type','language'],
        ['tw_hash'],
        ['present_media','tweet_type','language','engaged_with_user_is_verified','engaging_user_is_verified','engagee_follows_engager'],
        ['present_domains','present_media','tweet_type','language'],
        ['present_links','present_media','tweet_type','language'],
        ['hashtags','present_media','tweet_type','language']
        
    ]
    ce_features = ['present_media', 'tweet_type', 'language', 'engaged_with_user_id', 'engaging_user_id']
    fe_features = ['present_media', 'tweet_type', 'language', 'engaged_with_user_id', 'engaging_user_id']
    encoding_features = [("TE", te_features), ("CE", ce_features), ("FE", fe_features)]

    te_train_dfs = []
    te_test_dfs = []
    ce_train_dfs = []
    ce_test_dfs = []
    fe_train_dfs = []
    fe_test_dfs = []
    for feature_type, features in encoding_features:
        for c in features:
            target_tmp = targets
            out_name = ""
            out_col_list = []
            for tgt in target_tmp:
                if isinstance(c, list):
                    out_col_list.append(f'G{feature_type}_' + '_'.join(c) + f'_{tgt}')
                    out_name = f'G{feature_type}_' + '_'.join(c)
                else:
                    out_col_list.append(f'{feature_type}_{c}_{tgt}')
                    out_name = f'{feature_type}_{c}'
            if gen_dict:
                start = timer()
                if feature_type == 'TE':
                    encoder = TargetEncoder(proc, c, target_tmp, out_col_list, out_name, out_dtype=t.FloatType(), y_mean_list=y_mean_all)
                    te_train_df, te_test_df = encoder.transform(df)
                    te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': te_train_df})
                    te_test_dfs.append({'col_name': c, 'dict': te_test_df})
                    print(f"generating target encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))

                elif feature_type == 'CE':
                    encoder = CountEncoder(proc, c, target_tmp, out_col_list, out_name)  
                    ce_train_df, ce_test_df = encoder.transform(df)
                    ce_train_dfs.append({'col_name': c if isinstance(c, list) else [c], 'dict': ce_train_df})
                    ce_test_dfs.append({'col_name': c, 'dict': ce_test_df})  
                    print(f"generating count encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))

                elif feature_type == 'FE':
                    # For frequency encoding, we don't need to merge with train data
                    encoder = FrequencyEncoder(proc, c, target_tmp, out_col_list, out_name) 
                    fe_train_df, fe_test_df = encoder.transform(df)
                    fe_train_dfs.append({'col_name': c if isinstance(c, list) else [c], 'dict': fe_train_df})
                    print(f"generating frequency encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))

            else:
                te_train_path = "%s/%s/%s/train/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name)
                te_test_path = "%s/%s/%s/test/%s" % (proc.path_prefix, proc.current_path, proc.dicts_path, out_name) 
                if feature_type == 'TE':
                    te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': proc.spark.read.parquet(te_train_path)})
                    te_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(te_test_path)})
                if feature_type == 'CE':
                    ce_train_dfs.append({'col_name': c if isinstance(c, list) else [c], 'dict': proc.spark.read.parquet(te_train_path)})
                    ce_test_dfs.append({'col_name': c, 'dict': proc.spark.read.parquet(te_test_path)})
                if feature_type == 'FE':
                    # For frequency encoding, we don't need to merge with train data
                    fe_train_dfs.append({'col_name': c if isinstance(c, list) else [c], 'dict': proc.spark.read.parquet(te_train_path)})

    t2 = timer()
    print("Generate encoding feature totally took %.3f" % (t2 - t1))

    # merge dicts to original table
    if sampleRatio < 1 and sampleRatio > 0:
        df = df.sample(sampleRatio)
    i = 3
    for train_dfs in [te_train_dfs, ce_train_dfs, fe_train_dfs]:
        op_merge_to_train = ModelMerge(train_dfs)
        proc.reset_ops([op_merge_to_train])
        i -= 1
        _output_name = output_name if i == 0 else f"{output_name}_{i}"

        t1 = timer()
        df = proc.transform(df, name=_output_name)
        t2 = timer()
        print("encodingFeatures took %.3f" % (t2 - t1))
    
    return (df, te_train_dfs, te_test_dfs, y_mean_all_df)

def main():
    path_prefix = "hdfs://"
    current_path = "/recsys2020_example/"
    original_folder = "/recsys2021_0608/"
    dicts_folder = "recsys_dicts/"
    recsysSchema = RecsysSchema()

    ##### 1. Start spark and initialize data processor #####
    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    t0 = timer()
    spark = SparkSession.builder.master('local[80]')\
        .appName("Recsys2020_data_process")\
        .config("spark.driver.memory", "480G")\
        .config("spark.executor.cores", "80")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .config("spark.sql.broadcastTimeout", "7200")\
        .config("spark.cleaner.periodicGC.interval", "10min")\
        .getOrCreate()

    schema = recsysSchema.toStructType()

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix,
                        current_path=current_path, dicts_path=dicts_folder, shuffle_disk_capacity="1200GB")
    df = spark.read.parquet(path_prefix + original_folder)
    df = df.withColumnRenamed('enaging_user_following_count', 'engaging_user_following_count')
    df = df.withColumnRenamed('enaging_user_is_verified', 'engaging_user_is_verified')

    # fast test, comment for full dataset
    # df.sample(0.01).write.format("parquet").mode("overwrite").save("%s/sample_0_0_1" % current_path)
    # df = spark.read.parquet("%s/sample_0_0_1" % current_path)

    # ===============================================
    # decode tweet_tokens
    df = decodeBertTokenizerAndExtractFeatures(df, proc, output_name="decoded_with_extracted_features")

    # ===============================================
    # splitting and sampling
    df, test_df = splitByDate(df, proc, train_output="train", test_output="test", numFolds=5)

    # ===============================================
    # generate dictionary for categorify indexing
    df, dict_dfs = categorifyFeatures(df, proc, output_name="train_with_categorified_features", gen_dict=True, sampleRatio=0.03)

    # ===============================================
    # encoding features
    df, te_train_dfs, te_test_dfs, y_mean_all_df = encodingFeatures(df, proc, output_name="train_with_features_sample_0_0_3", gen_dict=True, sampleRatio=0.03)
    
if __name__ == "__main__":
    main()