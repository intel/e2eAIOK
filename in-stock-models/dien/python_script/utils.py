from .init_spark import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import pandas
import pickle
import os.path
import random
from time import time
import pandas as pd
import numpy as np

def load_csv(spark, local_prefix_src):
    review_id_field = StructField('review_id', StringType())
    movie_id_field = StructField('movie_id', StringType())
    overall_field = StructField('overall', FloatType())
    unix_time_field = StructField('unix_review_time', IntegerType())
    reviews_info_schema = StructType(
        [review_id_field, movie_id_field, overall_field, unix_time_field])

    category_field = StructField('category', StringType())
    item_info_schema = StructType([movie_id_field, category_field])

    reviews_info_df = spark.read.schema(reviews_info_schema).option(
        'sep', '\t').csv(local_prefix_src + "/reviews-info")
    item_info_df = spark.read.schema(item_info_schema).option(
        'sep', '\t').csv(local_prefix_src + "/item-info")

    return reviews_info_df, item_info_df


def load_neg_mid(neg_file_path):
    mid_list = []
    with open(neg_file_path, "r") as f:
        for line in f.readlines():
            mid_list.append(line.strip())
    return mid_list


def list_dir(path):
    source_path_dict = {}
    dirs = os.listdir(path)
    for files in dirs:
        try:
            sub_dirs = os.listdir(path + "/" + files)
            for file_name in sub_dirs:
                if (file_name.endswith('parquet') or file_name.endswith('csv')):
                    source_path_dict[files] = os.path.join(
                        path, files, file_name)
        except:
            source_path_dict[files] = os.path.join(path, files)
    return source_path_dict


def to_pydict(df):
    if df.shape[1] == 2:
        keys = df.keys()
        l1 = df[keys[0]].to_list()
        l2 = df[keys[1]].to_list()
        return dict(zip(l1, l2))
    return {}


def result_rename_or_convert(fpath):
    source_path_dict = list_dir(fpath)
    fix = "_spark"
    #os.rename(source_path_dict["local_test_splitByUser" + fix], fpath + '/local_test_splitByUser')
    os.rename(source_path_dict["local_train_splitByUser" +
                               fix], fpath + '/local_train_splitByUser')
    uid_voc = to_pydict(pandas.read_parquet(source_path_dict["uid_voc" + fix]))
    mid_voc = to_pydict(pandas.read_parquet(source_path_dict["mid_voc" + fix]))
    cat_voc = to_pydict(pandas.read_parquet(source_path_dict["cat_voc" + fix]))

    pickle.dump(uid_voc, open(fpath + '/uid_voc.pkl', "wb"), protocol=0)
    pickle.dump(mid_voc, open(fpath + '/mid_voc.pkl', "wb"), protocol=0)
    pickle.dump(cat_voc, open(fpath + '/cat_voc.pkl', "wb"), protocol=0)
