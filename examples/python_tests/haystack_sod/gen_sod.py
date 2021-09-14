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
from bs4 import BeautifulSoup

MODE = 0  # 0: actual, 1: debug
SO_PATH = './stack-overflow/'

def main():
    path_prefix = "file://"
    current_path = "/home/vmagent/app/recdp/examples/python_tests/haystack_sod/"
    data_folder = "stack-overflow/"
    #data_folder = ""

    if not os.path.isdir(f"{current_path}/stack-overflow/"):
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('stackoverflow/stacksample', path=SO_PATH, unzip=True, quiet=False)

    ##### 1. Start spark and initialize data processor #####
    spark = SparkSession.builder.master('local[*]')\
        .appName("HAYSTACK_SOD")\
        .config("spark.driver.memory", "400G")\
        .config("spark.driver.memoryOverhead", "80G")\
        .config("spark.executor.cores", "80")\
        .config("spark.driver.maxResultSize", "16G")\
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "32768")\
        .getOrCreate()

    #files = ["day_%d" % i for i in range(0, 24)]
    SO_FILE = {'question': 'Questions.csv', 'answer': 'Answers.csv'}
    file_names = dict((key, path_prefix + os.path.join(current_path, data_folder, filename)) for key, filename in SO_FILE.items())

    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')

    ############## Process answer ###############
    t0 = timer()
    if not os.path.isdir(f"{current_path}/answer.parquet"):
        answers_pdf = pd.read_csv(file_names['answer'], encoding="ISO-8859-1")
        answers_pdf.drop(columns=['OwnerUserId', 'CreationDate', 'Id'], inplace=True)
        answer_df = spark.createDataFrame(answers_pdf)
        answer_df = proc.transform(answer_df, name="answer.parquet")        
    else:
        answer_df = spark.read.parquet(f"{path_prefix}{current_path}/answer.parquet")
    t1 = timer()
    print(f"Load Answer.csv took {(t1 - t0)} secs")


    t0 = timer()
    answer_df = answer_df.withColumnRenamed("ParentId", "Id").withColumnRenamed("Body", "TopAnswer-html")
    op_collapse = CollapseByHist(by="Id", orderBy="Score")
    proc.reset_ops([op_collapse])
    top_answers_df = proc.apply(answer_df)
    top_answers_df = top_answers_df.drop('Score')

    top_answers_df = proc.transform(top_answers_df, name="top_answer_processed.parquet")
    #top_answers_df.show()
    t1 = timer()
    print(f"Process Answer.csv took {(t1 - t0)} secs")

    ############## Process question ###############
    t0 = timer()
    if not os.path.isdir(f"{current_path}/question.parquet"):
        questions_pdf = pd.read_csv(file_names['question'], encoding="ISO-8859-1")
        questions_pdf.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate', 'Score'], inplace=True)
        question_df = spark.createDataFrame(questions_pdf)
        question_df = proc.transform(question_df, name="question.parquet")        
    else:
        question_df = spark.read.parquet(f"{path_prefix}{current_path}/question.parquet")
    t1 = timer()
    print(f"Load Question.csv took {(t1 - t0)} secs")

    t0 = timer()
    convert_udf = f.udf(lambda x: BeautifulSoup(x, "lxml").get_text(), StringType())
    question_df = question_df.withColumnRenamed("Title", "text").withColumnRenamed("Body", "Question-html")
    op_merge = ModelMerge([{'col_name': ['Id'], 'dict': top_answers_df}])
    proc.reset_ops([op_merge])
    question_df = proc.apply(question_df)
    question_df = question_df.filter(f.col("TopAnswer-html").isNotNull())
    question_df = question_df.drop('Id')
    question_df = question_df.dropDuplicates()
    question_df = question_df.na.fill({'Question-html': '', 'TopAnswer-html': ''})
    question_df = question_df.withColumn("question-body", convert_udf(f.col("Question-html")))
    question_df = question_df.withColumn("answer", convert_udf(f.col("TopAnswer-html")))
    question_df = question_df.drop("Question-html").drop("TopAnswer-html")

    question_df = proc.transform(question_df, name="question_processed.parquet")
    question_df.show()
    t1 = timer()
    print(f"Process Question.csv took {(t1 - t0)} secs")

if __name__ == "__main__":
    main()