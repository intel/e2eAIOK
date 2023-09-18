import os

from pyspark.sql import DataFrame

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.llmutils.utils import get_llmutils_home
from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor as SparkDataProcessor

import pyspark.sql.functions as F
from urllib.parse import urlparse
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
import urllib.error

BLACKLIST_URL = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"
BLACKLIST_STORE_PATH = "/tmp"
BLACKLIST_CATEGORIES = ["adult", "phishing", "dating", "gambling", "filehosting", "ddos", "agressif", "chat",
                        "mixed_adult",
                        "arjel"]


def prepare_blacklist():
    blacklist_tar_path = "/tmp/blacklists.tar.gz"
    if not os.path.exists(blacklist_tar_path):
        try:
            import wget
            wget.download(BLACKLIST_URL, out=blacklist_tar_path)
        except urllib.error.HTTPError:
            print("Failed to download blacklists. Please check your network.")
            exit(1)
    unzip_cmd = f"tar -zxf {blacklist_tar_path} -C {BLACKLIST_STORE_PATH}"
    os.system(unzip_cmd)


def load_blacklist(spark):
    data_schema = StructType([
        StructField('domain', StringType()),
    ])
    blacklist_df: DataFrame = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=data_schema)
    for category in BLACKLIST_CATEGORIES:
        domain_file = os.path.join(BLACKLIST_STORE_PATH, "blacklists", category, "domains")
        df = spark.read.text(domain_file)
        df = df.withColumnRenamed("value", "domain")
        blacklist_df = blacklist_df.union(df)
    return blacklist_df


def read_json(data_dir, spark):
    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    schema = StructType([
        StructField("text", StringType(), True),
        StructField("meta", StringType(), True)
    ])
    first = True
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        df = spark.read.text(filepath)
        df = df.withColumn('jsonData', F.from_json(F.col('value'), schema)).select("jsonData.*")
        df = df.select(F.col("text"), F.col("meta"), F.json_tuple(F.col("meta"), "url"))
        df = df.toDF("text", "meta", "url")
        if first:
            first = False
            source_df = df
        else:
            source_df = source_df.union(df)

    return source_df


# define how to do parallel here
def filter_by_blocklist(data_dir, out_dir):
    rdp = SparkDataProcessor()
    spark = rdp.spark
    try:
        with Timer("Download and load blocklist"):
            prepare_blacklist()
            blacklist_df = load_blacklist(spark)
            total_blocked_domain_num = blacklist_df.count()

        with Timer("Load data from josnl file"):
            source_df = read_json(data_dir, spark)
            total_data_num = source_df.count()

        with Timer("Filter out data according to blocked domains"):
            with_url_df = source_df.filter(~F.isnull("url"))
            filtered_df = source_df.filter(F.isnull("url")).select(F.col("text"), F.col("meta"))

            rdd = with_url_df.rdd.map(
                lambda x: (x['text'], x['meta'], x['url'], urlparse(x['url']).hostname if x['url'] else ""))
            with_domain_df = spark.createDataFrame(rdd, ["text", "meta", "url", 'domain'])
            left_anti_df = with_domain_df.join(blacklist_df, on='domain', how='left_anti')
            filtered_df = filtered_df.union(left_anti_df.select(F.col("text"), F.col("meta")))
            remain_data_num = filtered_df.count()
        os.makedirs(out_dir, exist_ok=True)
        outfile_path = os.path.join(out_dir, "filtered")
        filtered_df.write.mode("overwrite").json(outfile_path)

        print(f"Completed!!")
        print(f"    Load total {total_data_num} documents")
        print(f"    Load total {total_blocked_domain_num} blocked domains")
        print(f"    Removed {total_data_num - remain_data_num} documents according to blacklist")

    except Exception as e:
        spark.stop()
        print("Failed", e)


def filter_by_bad_words(data_dir, out_dir, language="en"):
    rdp = SparkDataProcessor()
    spark = rdp.spark
    try:
        with Timer("Load bad words list and create pattern"):
            llmutils_path = get_llmutils_home()
            bad_words_lists_path = os.path.join(llmutils_path, "bad_words_lists", language)
            with open(bad_words_lists_path, "r") as f:
                lines = f.readlines()
            bad_words_list = [s.replace('\n', '') for s in lines]
            total_bad_words_num = len(bad_words_list)
            bad_words_pattern = "|".join(bad_words_list)

        with Timer("Load data from josnl file"):
            source_df = read_json(data_dir, spark)
            total_data_num = source_df.count()

        with Timer("Filter out data according to bad words"):
            filtered_df = source_df.filter(source_df.text.rlike(bad_words_pattern))
            remain_data_num = filtered_df.count()
        os.makedirs(out_dir, exist_ok=True)
        outfile_path = os.path.join(out_dir, "bad_words_filtered")
        filtered_df.write.mode("overwrite").json(outfile_path)

        print(f"Completed!!")
        print(f"    Load total {total_data_num} documents")
        print(f"    Load total {total_bad_words_num} blocked domains")
        print(f"    Removed {total_data_num - remain_data_num} documents according to blacklist")

    except Exception as e:
        spark.stop()
        print("Failed", e)


def filter_by_length(data_dir, out_dir, minimum_length=100, maximum_length=10000):
    rdp = SparkDataProcessor()
    spark = rdp.spark
    try:

        with Timer("Load data from josnl file"):
            source_df = read_json(data_dir, spark)
            total_data_num = source_df.count()

        with Timer("Filter out data according to length limit"):
            @F.udf(returnType=BooleanType())
            def check_length(text):
                if len(text) < minimum_length or len(text) > maximum_length:
                    return False
                else:
                    return True
            filtered_df = source_df.filter(check_length(source_df.text))
            remain_data_num = filtered_df.count()
        os.makedirs(out_dir, exist_ok=True)
        outfile_path = os.path.join(out_dir, "length_filtered")
        filtered_df.write.mode("overwrite").json(outfile_path)

        print(f"Completed!!")
        print(f"    Load total {total_data_num} documents")
        print(f"    Removed {total_data_num - remain_data_num} documents according to length limit")

    except Exception as e:
        spark.stop()
        print("Failed", e)
