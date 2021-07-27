from .init_spark import *
import re
from pyspark import *
from pyspark.sql import *
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import ListConverter, JavaArray, JavaList, JavaMap

def convert_to_spark_dict(orig_dict, schema=['dict_col', 'dict_col_id']):
    ret = []
    for row_k, row_v in orig_dict.items():
        ret.append({schema[0]: row_k, schema[1]: row_v})
    return ret


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


def parse_size(size):
    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    size = size.upper()
    number, unit = re.findall('(\d+)(\w*)', size)[0]
    return int(float(number)*units[unit])


def get_estimate_size_of_dtype(dtype_name):
    units = {'byte': 1, 'short': 2, 'int': 4, 'long': 8, 'float': 4, 'double': 8, 'string': 10}
    return units[dtype_name] if dtype_name in units else 4


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


def _py4j(obj):
    """ Convert Python object into Java """
    if isinstance(obj, DataFrame):
        obj = obj._jdf
    elif isinstance(obj, SparkContext):
        obj = obj._jsc
    elif isinstance(obj, (list, tuple)):
        obj = ListConverter().convert([_py4j(x) for x in obj],
                                      sc._gateway._gateway_client)
    elif isinstance(obj, JavaObject):
        pass
    elif isinstance(obj, (int, float, bool, bytes, str)):
        pass
    else:
        raise NotImplementedError(f"can't convert {obj} to python")
    return obj    

def jvm_categorify(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.Categorify
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

def jvm_categorify_for_array(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.CategorifyForArray
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

def jvm_categorify_by_freq_for_array(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.CategorifyByFreqForArray
    return _j4py(spark, _jcls.categorify(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

def jvm_get_negative_samples(spark, df, col_name, dict_df):
    gateway = spark.sparkContext._gateway
    _jcls = gateway.jvm.org.apache.spark.sql.api.NegativeSample
    return _j4py(spark, _jcls.add(_py4j(spark.sparkContext), _py4j(df), _py4j(col_name), _py4j(dict_df)))

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