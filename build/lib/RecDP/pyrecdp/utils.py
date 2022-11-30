from pyrecdp.init_spark import *
import os
import re
from pyspark import *
from pyspark.sql import *
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList, JavaMap
import pyspark.sql.types as spk_type
import pyspark.sql.functions as spk_func

def convert_to_spark_dict(orig_dict, schema=['dict_col', 'dict_col_id']):
    ret = []
    for row_k, row_v in orig_dict.items():
        ret.append({schema[0]: row_k, schema[1]: row_v})
    return ret


def convert_to_spark_df(orig_dict, spark):
    df = spark.createDataFrame(convert_to_spark_dict(orig_dict))
    df = df.withColumn('dict_col_id', spk_func.col('dict_col_id').cast(spk_type.IntegerType()))
    return df

def list_dir(path, only_get_one = True):
    source_path_dict = {}
    dirs = os.listdir(path)
    for files in dirs:
        try:
            sub_dirs = os.listdir(path + "/" + files)
            if not only_get_one:
                source_path_dict[files] = []
            for file_name in sub_dirs:
                if (file_name.endswith('parquet') or file_name.endswith('csv')):
                    if not only_get_one:
                        source_path_dict[files].append(os.path.join(
                            path, files, file_name))
                    else:
                        source_path_dict[files] = os.path.join(
                            path, files, file_name)
        except:
            if not only_get_one:
                source_path_dict[files] = [os.path.join(path, files)]
            else:
                source_path_dict[files] = os.path.join(path, files)
    return source_path_dict


def parse_size(size):
    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    size = size.upper()
    number, unit = re.findall('(\d+)(\w*)', size)[0]
    return int(float(number)*units[unit])


def parse_cores_num(local_str):
    number = re.findall('local\[(\d+)\]', local_str)
    if len(number) > 0:
        return int(number[0])
    else:
        return os.cpu_count()


def get_estimate_size_of_dtype(dtype_name):
    units = {'byte': 1, 'short': 2, 'int': 4, 'long': 8, 'float': 4, 'double': 8, 'string': 10}
    return units[dtype_name] if dtype_name in units else 4


def get_dtype(df, colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]


def _j4py(spark, r):
    sc = spark.sparkContext
    if isinstance(r, JavaObject):
        clsName = r.getClass().getSimpleName()
        # convert RDD into JavaRDD
        if clsName != 'JavaRDD' and clsName.endswith("RDD"):
            r = r.toJavaRDD()
            clsName = 'JavaRDD'
        elif clsName == 'JavaRDD':
            jrdd = sc._jvm.SerDe.javaToPython(r)
            return RDD(jrdd, sc)
        elif clsName == 'DataFrame' or clsName == 'Dataset':
            return DataFrame(r, SQLContext.getOrCreate(sc))
        else:
             raise NotImplementedError(f"can't convert {clsName} {r} to java")
    else:
        raise NotImplementedError(f"can't convert {r} to java")
    return r


def _py4j(obj, gateway = None):
    """ Convert Python object into Java """
    if isinstance(obj, DataFrame):
        obj = obj._jdf
    elif isinstance(obj, SparkContext):
        obj = obj._jsc
    elif isinstance(obj, (list, tuple)):
        obj = ListConverter().convert([_py4j(x) for x in obj],
                                      gateway._gateway_client)
    elif isinstance(obj, JavaObject):
        pass
    elif isinstance(obj, (int, float, bool, bytes, str)):
        pass
    else:
        raise NotImplementedError(f"can't convert {str(obj)} to python")
    return obj    

def jvm_categorify(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.Categorify
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

def jvm_categorify_for_array(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.CategorifyForArray
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

def jvm_categorify_for_multi_level_array(spark, df, col_name, dict_df, sep_list):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.CategorifyForMultiLevelArray
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df), _py4j(sep_list, gateway)))

def jvm_categorify_by_freq_for_array(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.CategorifyByFreqForArray
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

def jvm_get_negative_sample(spark, df, col_name, dict_df, neg_cnt):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.NegativeSample
    return _j4py(spark, _jcls.add(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df), _py4j(neg_cnt)))

def jvm_get_negative_feature(spark, df, tgt_name, src_name, dict_df, neg_cnt):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.NegativeFeature
    return _j4py(spark, _jcls.add(_py4j(spark.sparkContext), _py4j(df), _py4j(tgt_name), _py4j(src_name), _py4j(dict_df), _py4j(neg_cnt)))

def jvm_get_negative_feature_for_array(spark, df, tgt_name, src_name, dict_df, neg_cnt):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.NegativeFeatureForArray
    return _j4py(spark, _jcls.add(_py4j(spark.sparkContext), _py4j(df), _py4j(tgt_name), _py4j(src_name), _py4j(dict_df), _py4j(neg_cnt)))

def jvm_scala_df_test_int(spark, df, col_name):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.ScalaDFTest
    return _j4py(spark, _jcls.scala_test_int(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name)))

def jvm_scala_df_test_str(spark, df, col_name):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.ScalaDFTest
    return _j4py(spark, _jcls.scala_test_str(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name)))

def jvm_scala_df_test_broadcast(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.ScalaDFTest
    return _j4py(spark, _jcls.scala_test_broadcast(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))
