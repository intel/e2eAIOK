import init

from pyrecdp.data_processor import *
from pyrecdp.utils import *

import logging
from timeit import default_timer as timer
import os
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd

def main():
    path_prefix = "file://"
    current_path = "/home/vmagent/app/recdp/examples/python_tests/haystack_sod/"
    data_folder = "stack-overflow/"
    #data_folder = ""

    scala_udf_jars = "/home/vmagent/app/recdp/ScalaProcessUtils/target/recdp-scala-extensions-0.1.0-jar-with-dependencies.jar"

    ##### 1. Start spark and initialize data processor #####
    spark = SparkSession.builder.master('local[*]')\
        .appName("HAYSTACK_SOD")\
        .config("spark.driver.memory", "400G")\
        .config("spark.driver.memoryOverhead", "80G")\
        .config("spark.executor.cores", "80")\
        .getOrCreate()

    #files = ["day_%d" % i for i in range(0, 24)]
    SO_FILE = {'question': 'Questions.csv', 'answer': 'Answers.csv'}
    file_names = dict((key, path_prefix + os.path.join(current_path, data_folder, filename)) for key, filename in SO_FILE.items())

    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')
    t0 = timer()
    print(file_names['answer'])
    answers_pdf = pd.read_csv(file_names['answer'], encoding="ISO-8859-1")
    answer_df = spark.createDataFrame(answers_pdf)
    answer_df.printSchema()
    #answer_df.show(1, truncate=False, vertical=True)
    t1 = timer()
    print(f"Load Answer.csv took {(t1 - t0)} secs")

    # Process answer
    t0 = timer()
    answer_df = answer_df.drop('OwnerUserId').drop('CreationDate').drop('Id')
    top_answers = answer_df.groupby("ParentId").agg(f.max('Score').alias('Score'))
top_answers.reset_index(drop=True, inplace=True)
top_answers.rename(columns={"ParentId": "Id", "Body":"TopAnswer-html"}, inplace=True)
top_answers.drop(columns=['Score'], inplace=True)
    t1 = timer()
    print(f"Load Answer.csv took {(t1 - t0)} secs")

if __name__ == "__main__":
    main()