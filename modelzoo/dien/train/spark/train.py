import init

import os
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
from pyspark.sql.types import *
from timeit import default_timer as timer
import logging
import pickle as pkl
from pyrecdp.data_processor import *
from pyrecdp.utils import *

import tensorflow as tf
from train_utils import *



def categorify_dien_data(proc, train, test, user_df, asin_df, cat_df):
    dict_dfs = []
    dict_dfs.append({'col_name': 'reviewer_id', 'dict': user_df})
    dict_dfs.append({'col_name': 'asin', 'dict': asin_df})
    dict_dfs.append({'col_name': 'category', 'dict': cat_df})
    dict_dfs.append({'col_name': 'hist_asin', 'dict': asin_df})
    dict_dfs.append({'col_name': 'hist_category', 'dict': cat_df})

    for dict_df in dict_dfs:
        print(dict_df['dict'])
    op_categorify = Categorify(['reviewer_id', 'asin', 'category'], dict_dfs=dict_dfs)
    op_categorify_2 = Categorify(['hist_asin', 'hist_category'], dict_dfs=dict_dfs, doSplit=True, sep='\x02')
    proc.reset_ops([op_categorify, op_categorify_2])

    train = proc.apply(train)
    test = proc.apply(test)
    return train, test, user_df, asin_df, cat_df

def unicode_to_utf8(d):
    return dict((str(key), int(value)) for (key,value) in d.items())

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return unicode_to_utf8(pkl.load(f))

def load_dien_data(proc, data_dir, read_from_parquet = False):
    if not read_from_parquet:
        label_field = StructField('label', IntegerType())
        review_id_field = StructField('reviewer_id', StringType())
        asin_field = StructField('asin', StringType())
        category_field = StructField('category', StringType())
        hist_asin_field = StructField('hist_asin', StringType())
        hist_category_field = StructField('hist_category', StringType())
        csv_schema = StructType(
            [label_field, review_id_field, asin_field, category_field, hist_asin_field, hist_category_field])
    
        train = proc.spark.read.schema(csv_schema).option('sep', '\t').csv(proc.path_prefix + data_dir + "/local_train_splitByUser")
        test = proc.spark.read.schema(csv_schema).option('sep', '\t').csv(proc.path_prefix + data_dir + "/local_test_splitByUser")
        user_dict = load_pkl(data_dir + "/uid_voc.pkl")
        item_dict = load_pkl(data_dir + "/mid_voc.pkl")
        cat_dict = load_pkl(data_dir + "/cat_voc.pkl")
        user = convert_to_spark_df(user_dict, proc.spark)
        asin= convert_to_spark_df(item_dict, proc.spark)
        cat= convert_to_spark_df(cat_dict, proc.spark)
        train, test, user, asin, cat = categorify_dien_data(proc, train, test, user, asin, cat)
    else:
        train = proc.spark.read.parquet(f"{proc.path_prefix}{data_dir}/local_train_splitByUser.parquet")
        test = proc.spark.read.parquet(f"{proc.path_prefix}{data_dir}/local_test_splitByUser.parquet")
        user = proc.spark.read.parquet(f"{proc.path_prefix}{data_dir}/uid_voc.parquet")
        asin = proc.spark.read.parquet(f"{proc.path_prefix}{data_dir}/mid_voc.parquet")
        cat = proc.spark.read.parquet(f"{proc.path_prefix}{data_dir}/cat_voc.parquet")
    n_uid = user.count()
    n_mid = asin.count()
    n_cat = cat.count()
    return train, test, n_uid, n_mid, n_cat

def main():
    path_prefix = "file://"
    current_path = "/home/vmagent/app/frameworks.bigdata.bluewhale/examples/dien/train/spark/"
    data_dir = "/home/vmagent/app/recdp/examples/python_tests/dien/output/"
    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"
    time_start = timer()

    spark = SparkSession.builder.master('local[104]')\
        .appName("dien_training")\
        .config("spark.driver.memory", "480G")\
        .config("spark.driver.memoryOverhead", "20G")\
        .config("spark.executor.cores", "104")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()

    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')
    train_data, test_data, n_uid, n_mid, n_cat = load_dien_data(proc, data_dir, read_from_parquet = True)
    train_data.show()
    test_data.show()
    print(f"n_uid is {n_uid}, n_mid is {n_mid}, n_cat is {n_cat}")

    model = build_model('DIEN', n_uid, n_mid, n_cat, batch_size = 256)
    [inputs, feature_cols] = align_input_features(model)

    
#
#    for i in range(args.epochs):
#        estimator.fit(train_data, epochs=1, batch_size=args.batch_size, feature_cols=feature_cols,
#                  label_cols=['label'], validation_data=test_data)
#
#        result = estimator.evaluate(test_data, args.batch_size, feature_cols=feature_cols,
#                            label_cols=['label'])
#        print('test result:', result)
#        prediction_df = estimator.predict(test_data, feature_cols=feature_cols)
#        transform_label = udf(lambda x: int(x[1]), "int")
#        prediction_df = prediction_df.withColumn('label_t', transform_label(col('label')))
#        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
#                                          labelCol="label_t",
#                                          metricName="areaUnderROC")
#        auc = evaluator.evaluate(prediction_df)
#        print("test AUC score is: ", auc)
#
#    cpkts_dir = os.path.join(args.model_dir, 'cpkts/')
#    if not exists(cpkts_dir): makedirs(cpkts_dir)
#    snapshot_path = cpkts_dir + "cpkt_noshuffle_" + args.model_type
#    estimator.save_tf_checkpoint(snapshot_path)
#    time_end = timer()
#    print('perf { total time: %f }' % (time_end - time_start))

if __name__ == '__main__':
    main()