from functools import lru_cache

from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.base import LLMOPERATORS, statistics_decorator
from pyspark.sql import DataFrame
import os
import pyspark.sql.functions as F

BLACKLIST_URL = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"
BLACKLIST_CATEGORIES = ["adult", "phishing", "dating", "gambling", "filehosting", "ddos", "agressif", "chat",
                        "mixed_adult","arjel"]


def prepare_blacklist():
    from pyrecdp.datasets.download import download
    BLACKLIST_URL = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"

    blacklist_tar_path = "blacklists"
    downloader = download(name=blacklist_tar_path, url=BLACKLIST_URL, unzip=True)
    return downloader.saved_path

@lru_cache
def load_blacklist_spark_df(spark):
    saved_path = prepare_blacklist()
    from pyspark.sql.types import StructType, StructField, StringType
    data_schema = StructType([
        StructField('domain', StringType()),
    ])
    blacklist_df: DataFrame = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=data_schema)
    for category in BLACKLIST_CATEGORIES:
        domain_file = os.path.join(saved_path, category, "domains")
        df = spark.read.text(domain_file)
        df = df.withColumnRenamed("value", "domain")
        blacklist_df = blacklist_df.union(df)
    return blacklist_df

@lru_cache
def load_blacklist_set():
    saved_path = prepare_blacklist()
    print("*********************_______________________________________")
    blacklist = []
    for category in BLACKLIST_CATEGORIES:
        domain_file = os.path.join(saved_path, category, "domains")
        with open(domain_file, "r") as f:
            lines = f.readlines()
        blacklist.extend(lines)
    return set(blacklist)


def get_url_from_meta(x):
    import json
    try:
        meta_obj = json.loads(x)
        if 'url' in meta_obj:
            return meta_obj['url']
        else:
            return "Not Provided"
    except:
        return "Not Provided"


def get_domain(x):
    from urllib.parse import urlparse
    domain = urlparse(x).hostname
    if not domain:
        domain = "Not Provided"
    return domain


class URLFilter(BaseFilter):
    def __init__(self):
        """
            Keeps samples according to URLs based on blacklists https://dsi.ut-capitole.fr/blacklists/
        """
        super().__init__()
        self.text_key = "meta"
        self.blacklist = load_blacklist_set()

    @statistics_decorator
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        get_url_from_meta_udf = F.udf(get_url_from_meta)
        get_domain_udf = F.udf(get_domain)
        if self.inplace:
            # remove unwanted text row inplace
            blacklist_df = load_blacklist_spark_df(spark)
            source_df = spark_df
            with_domain_df = source_df.withColumn('domain', get_domain_udf(get_url_from_meta_udf('meta')))
            left_anti_df = with_domain_df.join(blacklist_df, on='domain', how='left_anti')
            return left_anti_df
        else:
            raise NotImplementedError("We only support inplace modification for URLFilter.")

    def get_compute_func(self, *args, **kwargs):
        def compute(sample) -> bool:
            url = get_url_from_meta(sample)
            domain = get_domain(url)
            if domain in self.blacklist:
                return False
            else:
                return True
        return compute


LLMOPERATORS.register(URLFilter)


