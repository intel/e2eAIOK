import init
import init_haystack        
from haystack.retriever.dense import DensePassageRetriever
from haystack.utils import print_answers

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

SO_PATH = './stack-overflow/'
EMBEDDING_PARALLELISM = 2
NUMCORES_PER_SOCKET = 52
EMBEDDING_BATCH_SIZE = 32768

def main():
    path_prefix = "file://"
    current_path = os.getcwd()
    data_folder = "stack-overflow/"

    if not os.path.isdir(f"{current_path}/stack-overflow/"):
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('stackoverflow/stacksample', path=SO_PATH, unzip=True, quiet=False)

    ##### 1. Start spark and initialize data processor #####
    spark = SparkSession.builder.master('spark://sr602:7077')\
        .appName("HAYSTACK_SOD")\
        .config("spark.driver.memory", "60G")\
        .config("spark.executor.memory", "250G")\
        .config("spark.executor.instances", EMBEDDING_PARALLELISM)\
        .config("spark.executor.cores", NUMCORES_PER_SOCKET)\
        .config("spark.driver.maxResultSize", "16G")\
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", EMBEDDING_BATCH_SIZE)\
        .config("spark.cleaner.periodicGC.interval", "7200min")\
        .getOrCreate()

#    SO_FILE = {'question': 'Questions.csv', 'answer': 'Answers.csv'}
#    file_names = dict((key, path_prefix + os.path.join(current_path, data_folder, filename)) for key, filename in SO_FILE.items())
#
#    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')
#
#    ############## Process answer ###############
#    t0 = timer()
#    if not os.path.isdir(f"{current_path}/answer.parquet"):
#        answers_pdf = pd.read_csv(file_names['answer'], encoding="ISO-8859-1")
#        answers_pdf.drop(columns=['OwnerUserId', 'CreationDate', 'Id'], inplace=True)
#        answer_df = spark.createDataFrame(answers_pdf)
#        answer_df = proc.transform(answer_df, name="answer.parquet")        
#    else:
#        answer_df = spark.read.parquet(f"{path_prefix}{current_path}/answer.parquet")
#    t1 = timer()
#    print(f"Load Answer.csv took {(t1 - t0)} secs")
#
#
#    t0 = timer()
#    answer_df = answer_df.withColumnRenamed("ParentId", "Id").withColumnRenamed("Body", "TopAnswer-html")
#    op_collapse = CollapseByHist(by="Id", orderBy="Score")
#    proc.reset_ops([op_collapse])
#    top_answers_df = proc.apply(answer_df)
#    top_answers_df = top_answers_df.drop('Score')
#
#    top_answers_df = proc.transform(top_answers_df, name="top_answer_processed.parquet")
#    t1 = timer()
#    print(f"Process Answer.csv took {(t1 - t0)} secs")
#
#    ############## Process question ###############
#    t0 = timer()
#    if not os.path.isdir(f"{current_path}/question.parquet"):
#        questions_pdf = pd.read_csv(file_names['question'], encoding="ISO-8859-1")
#        questions_pdf.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate', 'Score'], inplace=True)
#        question_df = spark.createDataFrame(questions_pdf)
#        question_df = proc.transform(question_df, name="question.parquet")        
#    else:
#        question_df = spark.read.parquet(f"{path_prefix}{current_path}/question.parquet")
#    t1 = timer()
#    print(f"Load Question.csv took {(t1 - t0)} secs")
#
#    t0 = timer()
#    convert_udf = f.udf(lambda x: BeautifulSoup(x, "lxml").get_text(), StringType())
#    question_df = question_df.withColumnRenamed("Title", "text").withColumnRenamed("Body", "Question-html")
#    op_merge = ModelMerge([{'col_name': ['Id'], 'dict': top_answers_df}])
#    proc.reset_ops([op_merge])
#    question_df = proc.apply(question_df)
#    question_df = question_df.filter(f.col("TopAnswer-html").isNotNull())
#    question_df = question_df.drop('Id')
#    question_df = question_df.dropDuplicates()
#    question_df = question_df.na.fill({'Question-html': '', 'TopAnswer-html': ''})
#    question_df = question_df.withColumn("question-body", convert_udf(f.col("Question-html")))
#    question_df = question_df.withColumn("answer", convert_udf(f.col("TopAnswer-html")))
#    question_df = question_df.drop("Question-html").drop("TopAnswer-html")
#
#    question_df = proc.transform(question_df, name="question_processed.parquet")
#    question_df.show()
#    t1 = timer()
#    print(f"Process Question.csv took {(t1 - t0)} secs")

    ############# Embedding and write to DocumentStore ###############
    question_df = spark.read.parquet(f"{path_prefix}{current_path}/question_processed.parquet")
    question_df = question_df.repartition(EMBEDDING_PARALLELISM)

    import init_haystack
    from haystack.document_store import MilvusDocumentStore
    from haystack.utils import launch_milvus

    from haystack.utils import print_answers
    from haystack.utils import launch_milvus
    print('launching milvus & retriever...')

    # get embeddings for question
    from typing import Iterator
    @pandas_udf(ArrayType(DoubleType()))
    def pandas_udf_embed_queries(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
        print(f"start launcu_milvus")
        launch_milvus()
        document_store = MilvusDocumentStore(host="localhost", username="", password="",
                                                index="flat")
        print(f"start create retriever")
        retriever = DensePassageRetriever(document_store=document_store,
                                    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                    use_gpu=False,batch_size=512,
                                    embed_title=True)
        for df in iterator:
            print(df.head())
            document_store.write_documents(df.to_dict(orient="records"))
            document_store.update_embeddings(document_store)
            yield df

    t0 = timer()
    print(f"Start to do EmbeddingRetriever embed_queries ...")
    debug_df = pandas_udf_embed_queries(f.struct(f.col("text"), f.col("question-body"), f.col("answer")))
    t1 = timer()
    print(f'EmbeddingRetriever embed_queries took {(t1 - t0)} secs')


if __name__ == "__main__":
    main()

