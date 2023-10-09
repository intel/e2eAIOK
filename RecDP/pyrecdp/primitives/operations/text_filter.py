from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame
import os
from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE
import pyspark.sql.functions as F


def prepare_func_filter_by_length(minimum_length=100, maximum_length=10000):
    def check_length(text):
        if len(text) < minimum_length or (maximum_length != -1 and len(text) > maximum_length):
            return False
        else:
            return True

    return check_length


def prepare_func_filter_by_badwords(language):
    from pyrecdp.primitives.llmutils.utils import get_llmutils_home
    import re
    llmutils_path = get_llmutils_home()
    bad_words_lists_path = os.path.join(llmutils_path, "bad_words_lists", language)
    with open(bad_words_lists_path, "r") as f:
        lines = f.readlines()
    bad_words_list = [s.replace('\n', '') for s in lines]
    bad_words_pattern = " | ".join(bad_words_list)

    def check_badwords(text):
        found = re.search(bad_words_pattern, text)
        if found:
            return False
        else:
            return True

    return check_badwords


def prepare_func_filter_by_profanity():
    from profanity_check import predict
    def check_profanity(text):
        scores = predict([text])
        ret = not bool(scores[0])
        return ret

    return check_profanity


class LengthFilter(BaseLLMOperation):
    def __init__(self, text_key='text', minimum_length=100, maximum_length=-1):
        settings = {'text_key': text_key, 'minimum_length': minimum_length, 'maximum_length': maximum_length}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = True
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        self.check_length = None
        self.support_ray = True
        self.support_spark = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.check_length is None:
            self.check_length = prepare_func_filter_by_length(self.minimum_length, self.maximum_length)
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: self.check_length(x[self.text_key]))
        else:
            raise NotImplementedError("We only support inplace modification for LengthFilter.")

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.types as T
        check_length_udf = F.udf(prepare_func_filter_by_length(self.minimum_length, self.maximum_length),
                                 T.BooleanType())
        return spark_df.filter(check_length_udf(F.col(self.text_key)))


LLMOPERATORS.register(LengthFilter)


class BadwordsFilter(BaseLLMOperation):
    def __init__(self, text_key='text', language='en'):
        settings = {'text_key': text_key, 'language': language}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = True
        self.language = language
        self.check_badwords = None
        self.support_ray = True
        self.support_spark = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.check_badwords is None:
            self.check_badwords = prepare_func_filter_by_badwords(self.language)
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: self.check_badwords(x[self.text_key]))
        else:
            raise NotImplementedError("We only support inplace modification for BadwordsFilter.")

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.types as T
        check_badwords_udf = F.udf(prepare_func_filter_by_badwords(self.language), T.BooleanType())
        return spark_df.filter(check_badwords_udf(F.col(self.text_key)))


LLMOPERATORS.register(BadwordsFilter)


class ProfanityFilter(BaseLLMOperation):
    def __init__(self, text_key='text', inplace=True):
        settings = {'text_key': text_key}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = inplace
        self.check_profanity = None
        self.support_ray = True
        self.support_spark = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.check_profanity is None:
            self.check_profanity = prepare_func_filter_by_profanity()
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: self.check_profanity(x[self.text_key]))
        else:
            raise NotImplementedError("We only support inplace modification for ProfanityFilter.")

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.types as T
        check_profanity_udf = F.udf(prepare_func_filter_by_profanity(), T.BooleanType())
        return spark_df.filter(check_profanity_udf(F.col(self.text_key)))


LLMOPERATORS.register(ProfanityFilter)


BLACKLIST_URL = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"
BLACKLIST_STORE_PATH = "/tmp"
BLACKLIST_CATEGORIES = ["adult", "phishing", "dating", "gambling", "filehosting", "ddos", "agressif", "chat",
                        "mixed_adult",
                        "arjel"]


def prepare_blacklist():
    from pyrecdp.datasets.download import download
    BLACKLIST_URL = "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"

    blacklist_tar_path = "blacklists"
    downloader = download(name=blacklist_tar_path, url=BLACKLIST_URL, unzip=True)
    return downloader.saved_path


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


def load_blacklist_set():
    saved_path = prepare_blacklist()
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


class URLFilter(BaseLLMOperation):
    def __init__(self, text_key='text'):
        settings = {'text_key': text_key}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = True
        self.support_spark = True
        self.support_ray = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        blacklist = load_blacklist_set()
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: get_url_from_meta(x) not in blacklist)
        else:
            raise NotImplementedError("We only support inplace modification for URLFilter.")

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


LLMOPERATORS.register(URLFilter)
