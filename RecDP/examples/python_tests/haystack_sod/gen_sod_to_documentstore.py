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

SO_PATH = './stack-overflow/'
EMBEDDING_PARALLELISM = 8
NUMCORES_PER_SOCKET = 64
EMBEDDING_BATCH_SIZE = 262144
ELASTIC_SERVER = "dr8s30"

def main():
    path_prefix = "hdfs://dr8s30:9000/"
    current_local_path = "/home/vmagent/app/recdp/examples/python_tests/haystack_sod/"
    current_path = "/haystack_sod/"
    data_folder = "stack-overflow/"

    #if not os.path.isdir(f"{current_path}/stack-overflow/"):
    #    import kaggle
    #    kaggle.api.authenticate()
    #    kaggle.api.dataset_download_files('stackoverflow/stacksample', path=SO_PATH, unzip=True, quiet=False)

    ##### 1. Start spark and initialize data processor #####
    spark = SparkSession.builder.master(f'spark://{ELASTIC_SERVER}:7077')\
        .appName("HAYSTACK_SOD")\
        .config("spark.driver.memory", "60G")\
        .config("spark.executor.memory", "250G")\
        .config("spark.executor.instances", EMBEDDING_PARALLELISM)\
        .config("spark.executor.cores", NUMCORES_PER_SOCKET)\
        .config("spark.driver.maxResultSize", "16G")\
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", EMBEDDING_BATCH_SIZE)\
        .config("spark.cleaner.periodicGC.interval", "7200min")\
        .config("spark.cleaner.referenceTracking", "false")\
        .config("spark.sql.files.minPartitionNum", "1")\
        .config("spark.executor.heartbeatInterval", "10000000")\
        .config("spark.network.timeout", "10000000")\
        .getOrCreate()

    #.config("spark.sql.files.maxPartitionBytes", "162MB")\
    SO_FILE = {'question': 'Questions.csv', 'answer': 'Answers.csv'}
    file_names = dict((key, os.path.join(current_local_path, data_folder, filename)) for key, filename in SO_FILE.items())

    proc = DataProcessor(spark, path_prefix, current_path=current_path, shuffle_disk_capacity="1200GB", spark_mode='local')

    ############## Process answer ###############
    t0 = timer()
    print("load ", file_names['answer'])
    answers_pdf = pd.read_csv(file_names['answer'], encoding="ISO-8859-1")
    answers_pdf.drop(columns=['OwnerUserId', 'CreationDate', 'Id'], inplace=True)
    answer_df = spark.createDataFrame(answers_pdf)
    answer_df = proc.transform(answer_df, name="answer.parquet")        
    t1 = timer()
    print(f"Load Answer.csv took {(t1 - t0)} secs")


    t0 = timer()
    answer_df = answer_df.withColumnRenamed("ParentId", "Id").withColumnRenamed("Body", "TopAnswer-html")
    op_collapse = CollapseByHist(by="Id", orderBy="Score")
    proc.reset_ops([op_collapse])
    top_answers_df = proc.apply(answer_df)
    top_answers_df = top_answers_df.drop('Score')

    top_answers_df = proc.transform(top_answers_df, name="top_answer_processed.parquet")
    t1 = timer()
    print(f"Process Answer.csv took {(t1 - t0)} secs")

    ############## Process question ###############
    t0 = timer()
    print("load ", file_names['question'])
    questions_pdf = pd.read_csv(file_names['question'], encoding="ISO-8859-1")
    questions_pdf.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate', 'Score'], inplace=True)
    question_df = spark.createDataFrame(questions_pdf)
    question_df = proc.transform(question_df, name="question.parquet")        
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

    ############# Embedding and write to DocumentStore ###############
    question_df = question_df.repartition(EMBEDDING_PARALLELISM)

    import init_haystack
    from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

    from haystack.retriever.dense import EmbeddingRetriever
    from haystack.utils import print_answers
    from haystack.utils import launch_es
    print('launching es & retriever...')

    # get embeddings for question
    from typing import Iterator
    @pandas_udf(ArrayType(DoubleType()))
    def pandas_udf_embed_queries(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
        launch_es()
        document_store = ElasticsearchDocumentStore(host=ELASTIC_SERVER, username="", password="",
                                                index="document",
                                                embedding_field="question_emb",
                                                embedding_dim=768,
                                                excluded_meta_data=["question_emb"])

        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=False)
        for df in iterator:
            df["question_emb"] = retriever.embed_queries(texts=list(df['text'].values))
            try:
                document_store.write_documents(df.to_dict(orient="records"))
            except:
                pass
            yield df["question_emb"]

    total_len = question_df.count()
    print('generating embeddings len = %d ...' % (total_len))
    t0 = timer()
    print(f"Start to do EmbeddingRetriever embed_queries ...")
    question_df = question_df.withColumn("question_emb", pandas_udf_embed_queries(f.struct(f.col("text"), f.col("question-body"), f.col("answer"))))
    question_df = proc.transform(question_df, name = "question_with_embed.parquet")
    question_df.show()
    t1 = timer()
    print(f'EmbeddingRetriever embed_queries took {(t1 - t0)} secs')

    print('!!!ALL DONE!!!')

    question_df = question_df.withColumn("emb_size", f.size(f.col("question_emb")))
    question_df.show()    

if __name__ == "__main__":
    main()
