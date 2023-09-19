import os

from profanity_check import predict

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, BooleanType

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor as SparkDataProcessor


@F.udf(returnType=BooleanType())
def predict_profanity(text):
    scores = predict([text])
    return not bool(scores[0])


def read_json(data_dir, spark):
    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    first = True
    schema = StructType([
        StructField("text", StringType(), True),
        StructField("meta", StringType(), True)
    ])
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        df = spark.read.text(filepath)
        df = df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*")
        if first:
            first = False
            source_df = df
        else:
            source_df = source_df.union(df)

    return source_df


def profanity_filter(data_dir, out_dir, enable_ray=False):
    if enable_ray:
        rdp = SparkDataProcessor(spark_mode='ray')
    else:
        rdp = SparkDataProcessor()
    spark = rdp.spark
    try:
        with Timer("Load data from josnl file"):
            source_df = read_json(data_dir, spark)
            total_data_num = source_df.count()

        with Timer("Filter out data containing profanity"):
            filtered_df = source_df.filter(predict_profanity(source_df.text))
            remain_data_num = filtered_df.count()

        with Timer("Save data"):
            outfile_path = os.path.join(out_dir, "profanity_filtered")
            filtered_df.write.mode("overwrite").json(outfile_path)
        print(f"Completed!!")
        print(f"    Load total {total_data_num} documents")
        print(f"    Removed {total_data_num - remain_data_num} documents with profanity")

    except Exception as e:
        spark.stop()
        print("Failed", e)

