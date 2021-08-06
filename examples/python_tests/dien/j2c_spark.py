import init

import os
import pandas as pd
import numpy as np
from pyspark.sql import *
from pyspark import *
import pyspark.sql.functions as f
from pyspark.sql.types import *
from timeit import default_timer as timer
import logging
from pyrecdp.data_processor import *
from pyrecdp.utils import *
import shutil

def load_csv(spark, path):
    review_id_field = StructField('reviewerID', StringType())
    asin_field = StructField('asin', StringType())
    overall_field = StructField('overall', FloatType())
    unix_time_field = StructField('unixReviewTime', IntegerType())
    reviews_info_schema = StructType(
        [review_id_field, asin_field, overall_field, unix_time_field])

    category_field = StructField('categories', StringType())
    item_info_schema = StructType([asin_field, category_field])

    reviews_info_df = spark.read.schema(reviews_info_schema).option('sep', '\t').csv(path + "/reviews-info")
    item_info_df = spark.read.schema(item_info_schema).option('sep', '\t').csv(path + "/item-info")

    return reviews_info_df, item_info_df

def process_meta(file, o_path):
    fi = open(file, "r")
    fo = open("%s/item-info" % o_path, "w")
    for line in fi:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat, file=fo)


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


def result_rename_or_convert(fpath):
    source_path_dict = list_dir(fpath)
    fix = "-spark"
    try:
        os.rename(source_path_dict["reviews-info" + fix], fpath + 'reviews-info')
        shutil.rmtree(source_path_dict["reviews-info" + fix], ignore_errors=True)
        # os.rename(source_path_dict["item-info" + fix], fpath + 'item-info')
    except:
        pass

def compare_with_expected(spark, path, records_df, item_info_df):
    records_expected_df, item_info_expected_df = load_csv(spark, path)
    cmp_res_records = records_expected_df.join(records_df, ['reviewerID', 'asin', 'overall', 'unixReviewTime'], 'anti')
    cmp_res_items = item_info_expected_df.join(item_info_df, ['categories', 'asin'], 'anti')
    error_parsed_records_len = cmp_res_records.count()
    error_parsed_items_len = cmp_res_items.count()
    print(f"records_df error_parsed_len is {error_parsed_records_len}, example as below:")
    cmp_res_records.show()
    print(f"item_info_df error_parsed_len is {error_parsed_items_len}, example as below:")
    cmp_res_items.show()


def main():
    path_prefix = "file://"
    current_path = "/home/vmagent/app/recdp/examples/python_tests/dien/output/"
    original_folder = "/home/vmagent/app/recdp/examples/python_tests/dien/"

    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    ##### 1. Start spark and initialize data processor #####
    t0 = timer()
    spark = SparkSession.builder.master('local[104]')\
        .appName("dien_data_process")\
        .config("spark.driver.memory", "480G")\
        .config("spark.driver.memoryOverhead", "20G")\
        .config("spark.executor.cores", "104")\
        .config("spark.driver.extraClassPath", f"{scala_udf_jars}")\
        .getOrCreate()

    # 1.1 prepare dataFrames
    # 1.2 create RecDP DataProcessor
    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')

    records_df = spark.read.json("%s/%s/raw_data/reviews_Books.json" % (path_prefix, original_folder))
    records_df = records_df.select('reviewerID', 'asin', 'overall', 'unixReviewTime')
    records_df.repartition(1).write.format("csv").option('sep', '\t').mode("overwrite").save("%s/%s/j2c_test/reviews-info-spark" % (path_prefix, original_folder))
    process_meta('%s/raw_data/meta_Books.json' % original_folder, "%s/j2c_test" % original_folder)
    result_rename_or_convert("%s/j2c_test/" % (original_folder))

    #item_info_df = spark.read.json("%s/%s/raw_data/meta_Books.json" % (path_prefix, original_folder))
    #item_info_df = item_info_df.withColumn("categories", f.expr("categories[0][size(categories[0]) - 1] as categories"))
    #item_info_df.write.format("csv").option('sep', '\t').mode("overwrite").save(original_folder + "/j2c_test/item-info")
    #compare_with_expected(spark, path_prefix + original_folder, records_df, item_info_df)
    t1 = timer()

    print(f"Total process time is {(t1 - t0)} secs")

    ####################################
    
if __name__ == "__main__":
    main()
