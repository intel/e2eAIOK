import uuid
from pyrecdp.utils import *
from pyspark.ml.feature import *
import pandas as pd
import numpy as np
import pyspark.sql.types as spk_type
import pyspark.sql.types as t
import pyspark.sql.functions as spk_func
import pyspark.sql.functions as f
from pyspark.sql import *
from pyspark import *
from pyspark.ml.feature import *
import os
import sys
from timeit import default_timer as timer
import logging
import shutil
import random


class Operation:
    '''
    Base abstract class of operation

    Args:

        cols (list): columns used to generate new features or modify in place

    Attributes:

        op_name (str): operation name
        cols (list): columns
    '''

    def __init__(self, cols):
        self.op_name = "AbstractOperation"
        self.cols = cols

    def describe(self):
        return "%s(%s)" % (self.op_name, ','.join(self.cols))

    def collect(self, df):
        raise NotImplementedError(self.op_name + "doesn't support collect")

    def to_dict_dfs(self, df, spark = None, enable_gazelle=False):
        raise NotImplementedError(self.op_name + "doesn't support to_dict_dfs")

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        return df

    # some common method
    def find_dict(self, name, dict_dfs):
        for i in dict_dfs:
            if isinstance(i, tuple):
                for v in i:
                    if isinstance(v, str):
                        dict_name = v
                    else:
                        dict_df = v
                if str(dict_name) == str(name):
                    return dict_df
            else:
                dict_df = i['dict']
                dict_name = i['col_name']
                if str(dict_name) == str(name):
                    return dict_df
        return None

    def get_colname_dict_as_tuple(self, i):
        if isinstance(i, tuple):
            for v in i:
                if isinstance(v, str):
                    dict_name = v
                else:
                    dict_df = v
            return (dict_name, dict_df)
        else:
            dict_df = i['dict']
            dict_name = i['col_name']
            return (dict_name, dict_df)

    def generate_dict_dfs(self, df, spark=None, cache_path=None, withCount=True, isParquet=True, enable_gazelle=False):
        # at least we can cache here to make read much faster
        # handle multiple columns issue
        if isParquet == False and spark != None and cache_path != None:
            cols = self.cols
            # let's raise NotImplemented exception for union case and doSplit case
            if self.doSplit:
                raise NotImplementedError("csv optimization is not supporting for doSplit yet.")
            for i in self.cols:
                if isinstance(i, dict):
                    raise NotImplementedError("csv optimization is not supporting for generate dictionary for multiple columns yet.")
            df = (df
                .select(spk_func.posexplode(spk_func.array(*cols)))
                .withColumnRenamed('pos', 'column_id')
                .withColumnRenamed('col', 'dict_col')
                .filter('dict_col is not null')
                .groupBy('column_id', 'dict_col')
                .count())
            windowed = Window.partitionBy('column_id').orderBy(spk_func.desc('count'))
            df = df.withColumn('dict_col_id', spk_func.row_number().over(windowed))
            if enable_gazelle:
                df.write.format('arrow').mode('overwrite').save(cache_path)
                df = spark.read.format("arrow").load(cache_path)
            else:
                df.write.format("parquet").mode("overwrite").save(cache_path)
                df = spark.read.parquet(cache_path)
            i = 0
            for col_name in cols:
                dict_df = df.filter('column_id == %d' % i).drop('column_id')
                if withCount:
                    self.dict_dfs.append({'col_name': col_name, 'dict': dict_df})
                else:
                    self.dict_dfs.append(
                        {'col_name': col_name, 'dict': dict_df.drop('count')})
                i += 1
            return self.dict_dfs
        # below is when input is parquet
        for i in self.cols:
            col_name = ""
            if isinstance(i, dict):
                # This is more those need to union columns
                if 'col_name' in i:
                    col_name = i['col_name']
                else:
                    col_name = 'union_dict'
                src_cols = []
                if 'src_cols' in i:
                    src_cols.extend(i['src_cols'])
                else:
                    raise ValueError(
                        "Union columns must has argument 'src_cols'")
                first = True
                dict_df = df
                for _i in src_cols:
                    if first:
                        dict_df = df.select(spk_func.col(_i).alias('dict_col'))
                        first = False
                    else:
                        dict_df = dict_df.union(
                            df.select(spk_func.col(_i).alias('dict_col')))
            else:  # if cols are simply col_name
                col_name = i
                dict_df = df.select(spk_func.col(i).alias('dict_col'))
                if self.doSplit:
                    dict_df = dict_df.select(
                        spk_func.explode(spk_func.split(spk_func.col('dict_col'), self.sep))).withColumn('dict_col', spk_func.col('col'))
            if self.bucketSize == -1:
                if self.id_by_python_dict:
                    dict_df = dict_df.distinct()
                    dict_data = dict((row['dict_col'], 1) for row in dict_df.collect())
                    for i, x in enumerate(dict_data):
                        dict_data[x] = i
                    dict_df = convert_to_spark_df(dict_data, df.spark)
                elif self.id_by_count:
                    dict_df = dict_df.groupBy('dict_col').count()
                    dict_df = dict_df.withColumn('dict_col_id', spk_func.row_number().over(
                        Window.orderBy(spk_func.desc('count')))).withColumn('dict_col_id', spk_func.col('dict_col_id') - 1).select('dict_col', 'dict_col_id', 'count')
                else:
                    dict_df = dict_df.withColumn("monotonically_increasing_id", spk_func.monotonically_increasing_id())
                    dict_df = dict_df.groupBy('dict_col').agg(spk_func.min("monotonically_increasing_id").alias("monotonically_increasing_id"), spk_func.count("*").alias("count"))
                    dict_df = dict_df.withColumn('dict_col_id', spk_func.row_number().over(
                    Window.orderBy(spk_func.col('monotonically_increasing_id')))).withColumn('dict_col_id', spk_func.col('dict_col_id') - 1).select('dict_col', 'dict_col_id', 'count')

            else:
                # when user set a bucketSize, we will quantileDiscretizer in this case
                dict_df = dict_df.groupBy('dict_col').count()
                qd = QuantileDiscretizer(numBuckets=self.bucketSize, inputCol="count", outputCol='dict_col_id')
                dict_df  = qd.fit(dict_df).transform(dict_df).withColumn('dict_col_id', spk_func.col('dict_col_id').cast(spk_type.IntegerType()))
            if withCount:
                self.dict_dfs.append({'col_name': col_name, 'dict': dict_df})
            else:
                self.dict_dfs.append(
                    {'col_name': col_name, 'dict': dict_df.drop('count')})
        return self.dict_dfs


    def categorify_strategy_decision_maker(self, dict_dfs, df, df_cnt, per_core_memory_size, flush_threshold, enable_gazelle,estimated_bytes=20):
        small_cols = []
        long_cols = []
        huge_cols = []
        smj_cols = []
        udf_cols = []
        total_small_cols_num_rows = 0
        total_estimated_shuffled_size = 0
        df_estimzate_size_per_row = sum([get_estimate_size_of_dtype(dtype) for _, dtype in df.dtypes])
        df_estimated_size = df_estimzate_size_per_row * df_cnt
        # Below threshold is maximum BHJ numRows, 4 means maximum 25% memory to cache Broadcast data, and 20 means we estimate each row has 20 bytes.
        if enable_gazelle:
            threshold = per_core_memory_size / estimated_bytes
            threshold_per_bhj = threshold if threshold <= 100000000 else 100000000
        else:
            threshold = per_core_memory_size / 4 / estimated_bytes
            threshold_per_bhj = threshold if threshold <= 30000000 else 30000000
        flush_threshold = flush_threshold * 0.8
        # print("[DEBUG] bhj total threshold is %.3f M rows, one bhj threshold is %.3f M rows, flush_threshold is %.3f GB" % (threshold / 1000000, threshold_per_bhj/1000000, flush_threshold / 2**30))
        dict_dfs_with_cnt = []
        for i in dict_dfs:
            col_name, dict_df = self.get_colname_dict_as_tuple(i)
            found = False
            if self.cols == None:
                found = True
            else:
                for j in self.cols:
                    _, col_name_from_cols = self.get_col_tgt_src(j)
                    if col_name_from_cols == col_name:
                        found = True
                        break
            if found == True:
              dict_df_cnt = dict_df.count()
              dict_dfs_with_cnt.append((col_name, dict_df, dict_df_cnt))
        sorted_dict_dfs_with_cnt = [(col_name, dict_df, dict_df_cnt) for col_name, dict_df, dict_df_cnt in sorted(dict_dfs_with_cnt, key=lambda pair: pair[2])]
        # for to_print in sorted_dict_dfs_with_cnt:
        #    print(to_print)

        sorted_cols = []
        for (col_name, dict_df, dict_df_cnt) in sorted_dict_dfs_with_cnt:
            sorted_cols.append(col_name)
            if self.doSplit:
                threshold_per_bhj = 30000000
            if (dict_df_cnt > threshold_per_bhj or (total_small_cols_num_rows + dict_df_cnt) > threshold):
                print(f"for {col_name} current dict_length {dict_df_cnt} exceeded threshold_per_bhj {threshold_per_bhj} or accumulated BHJ length {total_small_cols_num_rows + dict_df_cnt} exceeded estimated threshold {threshold}, will go smj")
                if ((total_estimated_shuffled_size + df_estimated_size) > flush_threshold):
                    # print("etstimated_to_shuffle_size for %s is %.3f GB, will do smj and spill to disk" % (str(col_name), df_estimated_size / 2**30))
                    huge_cols.append(col_name)
                    smj_cols.append(col_name)
                    long_cols.append(col_name)
                    total_estimated_shuffled_size = 0
                else:
                    # print("etstimated_to_shuffle_size for %s is %.3f GB, will do smj" % (str(col_name), df_estimated_size / 2**30))
                    smj_cols.append(col_name)
                    long_cols.append(col_name)
                    # if accumulate shuffle capacity may exceed maximum shuffle disk size, we should use hdfs instead
                    total_estimated_shuffled_size += df_estimated_size
            else:
                if self.doSplit:
                    # print("%s will do udf" % (col_name))
                    total_small_cols_num_rows += dict_df_cnt
                    udf_cols.append(col_name)
                else:
                    # print("%s will do bhj" % (col_name))
                    total_small_cols_num_rows += dict_df_cnt
                    small_cols.append(col_name)
            df_estimated_size -= df_cnt * 6
        return (sorted_cols, {'short_dict': small_cols, 'long_dict': long_cols, 'huge_dict': huge_cols, 'udf': udf_cols, 'smj_dict': smj_cols})

    def check_scala_extension(self, spark):
        driverClassPath = spark.sparkContext.getConf().get('spark.driver.extraClassPath')
        if driverClassPath == None or 'recdp' not in driverClassPath:
            return False
        else:
            return True

class FeatureModification(Operation):
    '''
    Operation to modify feature column in-place, support build-in function and udf

    Args:

        cols (list or dict): columns which are be modified
        op (str): define modify function name
        udfImpl (udf): if op is udf, udfImpl will be applied to specified cols
    '''

    def __init__(self, cols, op='udf', udfImpl=None):
        self.op_name = "FeatureModification"
        self.cols = cols
        self.op = op
        self.udf_impl = udfImpl

    def describe(self):
        if self.op == 'udf':
            f_cols = ["udf(%s)" % x for x in self.cols]
            return "%s(%s)" % (self.op_name, ','.join(f_cols))
        elif self.op == 'inline':
            f_cols = ["modify(%s, %s)" % (x, y)
                      for (x, y) in self.cols.items()]
            return "%s(%s)" % (self.op_name, ','.join(f_cols))
        else:
            f_cols = ["%s(%s)" % (self.op, x) for x in self.cols]
            return "%s(%s)" % (self.op_name, ','.join(f_cols))

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.op == 'udf':
            for i in self.cols:
                df = df.withColumn(i, self.udf_impl(spk_func.col(i)))
        elif self.op == 'inline':
            for i, func in self.cols.items():
                df = df.withColumn(i, eval(func))
        elif self.op == 'toInt':
            for i in self.cols:
                df = df.withColumn(i, spk_func.col(
                    i).cast(spk_type.IntegerType()))
        return df


class FeatureAdd(Operation):
    '''
    Operation to add new feature column based on current columns, support build-in function and udf

    Args:

        cols (dict): new_col: col or inline_func create new features
        op (str): define modify function name
        udfImpl (udf): if op is udf, udfImpl will be applied to specified cols
    '''

    def __init__(self, cols, op='udf', udfImpl=None):
        self.op_name = "FeatureAdd"
        self.cols = cols
        self.op = op
        self.udf_impl = udfImpl

    def describe(self):
        if self.op == 'udf':
            f_cols = ["new_feature(%s, udf(%s))" % (x, y)
                      for (x, y) in self.cols.items()]
            return "%s(%s)" % (self.op_name, ','.join(f_cols))
        elif self.op == 'inline':
            f_cols = ["new_feature(%s, %s)" % (x, y)
                      for (x, y) in self.cols.items()]
            return "%s(%s)" % (self.op_name, ','.join(f_cols))
        else:
            f_cols = ["new_feature(%s, %s(%s))" % (x, self.op, y)
                      for (x, y) in self.cols.items()]
            return "%s(%s)" % (self.op_name, ','.join(f_cols))

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.op == 'udf':
            for after, before in self.cols.items():
                df = df.withColumn(after, self.udf_impl(spk_func.col(before)))
        elif self.op == 'inline':
            for after, before in self.cols.items():
                try:
                    df = df.withColumn(after, eval(before))
                except Exception as e:
                    print("[ERROR]: inline script is %s" % before)
                    raise e
        elif self.op == 'toInt':
            for after, before in self.cols.items():
                df = df.withColumn(after, spk_func.col(
                    before).cast(spk_type.IntegerType()))
        return df


class FillNA(Operation):
    '''
    Operation to fillna to columns

    Args:

        cols (list): columns which are be modified
        default: filled default value
    '''

    def __init__(self, cols, default):
        self.op_name = "FillNA"
        self.cols = cols
        self.default = default

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        return df.fillna(self.default, [i for i in self.cols])


class DropFeature(Operation):
    '''
    Operation to fillna to columns

    Args:

        cols (list): columns which are be modified
        default: filled default value
    '''

    def __init__(self, cols):
        self.op_name = "DropFeature"
        self.cols = cols

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        for i in self.cols:
            df = df.drop(i)
        return df


class Distinct(Operation):
    '''
    Operation to perform distinct to columns

    Args:

        cols (list): columns which are be modified
        default: filled default value
    '''

    def __init__(self):
        self.op_name = "Distinct"

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        return df.distinct()

class SelectFeature(Operation):
    '''
    Operation to select and rename columns

    Args:

        cols (list): columns which are be modified
        default: filled default value
    '''

    def __init__(self, cols=[]):
        self.op_name = "SelectFeature"
        self.cols = cols

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        to_select = []
        for i in self.cols:
            if isinstance(i, str):
                to_select.append(i)
            elif isinstance(i, tuple):
                to_select.append("%s as %s" % (i[0], i[1]))

        return df.selectExpr(*to_select)


class Sort(Operation):
    def __init__(self, args = []):
        self.args = args

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        return df.orderBy(*self.args)

class Categorify(Operation):
    '''
    Operation to categorify columns by distinct id

    Args:

        cols (list): columns which are be modified
    '''

    def __init__(self, cols, dict_dfs=None, gen_dicts=True, hint='auto', doSplit=False, sep='\t', doSortForArray=False, keepMostFrequent=False, saveTmpToDisk=False, multiLevelSplit=False, multiLevelSep=[],estimated_bytes=20):
        self.op_name = "Categorify"
        self.cols = cols
        self.dict_dfs = dict_dfs
        self.gen_dicts = gen_dicts
        self.hint = hint
        self.doSplit = doSplit
        self.sep = sep
        self.doSortForArray = doSortForArray
        self.keepMostFrequent = keepMostFrequent
        self.saveTmpToDisk = saveTmpToDisk
        self.save_path_id = 0
        self.multi_level_split = multiLevelSplit
        if self.multi_level_split:
            self.hint = 'udf'
        self.multi_level_sep = multiLevelSep
        self.strategy_type = {
            'udf': 'udf', 'broadcast_join': 'short_dict', 'shuffle_join': 'huge_dict'}
        self.estimated_bytes = estimated_bytes

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        strategy = {}
        sorted_cols = []
        if self.hint != "auto" and self.hint in self.strategy_type:
            strategy[self.strategy_type[self.hint]] = []
            for i in self.dict_dfs:
                col_name, dict_df = self.get_colname_dict_as_tuple(i)
                strategy[self.strategy_type[self.hint]].append(col_name)
                sorted_cols.append(col_name)
        else:
            sorted_cols, strategy = self.categorify_strategy_decision_maker(self.dict_dfs, df, df_cnt, per_core_memory_size, flush_threshold, enable_gazelle,self.estimated_bytes)
        
        sorted_cols_pair = []
        pri_key_loaded = []
        for cn in sorted_cols:
            for i in self.cols:
                col_name, src_name = self.get_col_tgt_src(i)
                if src_name == cn:
                    if col_name not in pri_key_loaded:
                        pri_key_loaded.append(col_name)
                        sorted_cols_pair.append({col_name: src_name})


        last_method = 'bhj'
        # For now, we prepared dict_dfs and strategy
        for i in sorted_cols_pair:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            if col_name != src_name and col_name not in df.columns:
                df = df.withColumn(col_name, spk_func.col(src_name))
            # for udf type, we will do udf to do user-defined categorify
            if 'udf' in strategy and src_name in strategy['udf']:
                last_method = 'udf'
                if not enable_scala and not self.multi_level_split:
                    df = self.categorify_with_udf(df, dict_df, col_name, spark)
                else:
                    df = self.categorify_with_scala_udf(df, dict_df, col_name, spark)
        for i in sorted_cols_pair:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            if col_name != src_name and col_name not in df.columns:
                df = df.withColumn(col_name, spk_func.col(src_name))
            # for short dict, we will do bhj
            if 'short_dict' in strategy and src_name in strategy['short_dict']:
                last_method = 'bhj'
                df = self.categorify_with_join(df, dict_df, col_name, spark, save_path, method="bhj")
        for i in sorted_cols_pair:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            if col_name != src_name and col_name not in df.columns:
                df = df.withColumn(col_name, spk_func.col(src_name))
            # for long dict, we will do shj all along
            if 'long_dict' in strategy and src_name in strategy['long_dict']:
                if 'huge_dict' in strategy and src_name in strategy['huge_dict']:
                    if 'smj_dict' in strategy and src_name in strategy['smj_dict']:
                        last_method = 'smj'
                        df = self.categorify_with_join(df, dict_df, col_name, spark, save_path, method="smj", saveTmpToDisk=True, enable_gazelle=enable_gazelle)
                    else:
                        last_method = 'shj'
                        df = self.categorify_with_join(df, dict_df, col_name, spark, save_path, method="shj", saveTmpToDisk=True, enable_gazelle=enable_gazelle)
                else:
                    if 'smj_dict' in strategy and src_name in strategy['smj_dict']:
                        last_method = 'smj'
                        df = self.categorify_with_join(df, dict_df, col_name, spark, save_path, method="smj", saveTmpToDisk=False, enable_gazelle=enable_gazelle)
                    else:
                        last_method = 'shj'
                        df = self.categorify_with_join(df, dict_df, col_name, spark, save_path, method="shj", saveTmpToDisk=False, enable_gazelle=enable_gazelle)
        # when last_method is BHJ, we should add a separator for spark wscg optimization
        if last_method == 'bhj' and not enable_gazelle and enable_scala:
            vspark = str(spark.version)
            if not (vspark.startswith("3.1") or vspark.startswith("3.0")):
                return df
            #print("Adding a CodegenSeparator to pure BHJ WSCG case")
            found = False
            for dname, dtype in df.dtypes:
                if dtype == "string":
                    df = df.withColumn(dname, spk_func.expr(f"CodegenSeparator1({dname})"))
                    found = True
                    break
                if dtype == "int":
                    df = df.withColumn(dname, spk_func.expr(f"CodegenSeparator0({dname})"))
                    found = True
                    break
                if dtype == "bigint":
                    df = df.withColumn(dname, spk_func.expr(f"CodegenSeparator2({dname})"))
                    found = True
                    break
            if found == False:
                df = df.withColumn("CodegenSeparator", spk_func.expr(f"CodegenSeparator())"))
    
        return df

    def get_col_tgt_src(self, i):
        if isinstance(i, str):
            col_name = i
            src_name = i
            return (col_name, src_name)
        elif isinstance(i, dict):
            col_name = next(iter(i.keys()))
            src_name = next(iter(i.values()))
            return (col_name, src_name)

    def get_mapping_udf(self, broadcast_data, spark, default=None):
        broadcast_dict = spark.sparkContext.broadcast(
            broadcast_data)
        sep = self.sep
        doSortForArray = self.doSortForArray
        def largest_freq_encode(x):
            broadcast_data = broadcast_dict.value
            min_val = None
            if x != '':
                x_l = x.split(sep) if not isinstance(x, list) else x
                for v in x_l:
                    if v != '' and v in broadcast_data and (min_val == None or broadcast_data[v] < min_val):
                        min_val = broadcast_data[v]
            return min_val

        def freq_encode(x):
            broadcast_data = broadcast_dict.value
            val = []
            if x != '':
                x_l = x.split(sep) if not isinstance(x, list) else x
                for v in x_l:
                    if v != '' and v in broadcast_data:
                        val.append(broadcast_data[v])
                if doSortForArray and len(val) > 0:
                    val.sort()
            return val

        if self.keepMostFrequent:
            return spk_func.udf(largest_freq_encode, spk_type.IntegerType())
        else:
            return spk_func.udf(freq_encode, spk_type.ArrayType(spk_type.IntegerType()))

    def categorify_with_udf(self, df, dict_df, i, spark):
        #print("do %s to %s" % ("python_udf", i))
        dict_data = dict((row['dict_col'], row['dict_col_id']) for row in dict_df.collect())
        udf_impl = self.get_mapping_udf(dict_data, spark)
        df = df.withColumn(i, udf_impl(spk_func.col(i)))
        return df

    def categorify_with_scala_udf(self, df, dict_df, i, spark):
        #print("do %s to %s" % ("scala_udf", i))
        # call java to broadcast data and set broadcast handler to udf
        if not self.check_scala_extension(spark):
            raise ValueError("RecDP need to enable recdp-scala-extension to run categorify_with_udf, please config spark.driver.extraClassPath and spark.executor.extraClassPath for spark")

        if self.doSplit == False:
            df = jvm_categorify(spark, df, i, dict_df)
        else:
            if get_dtype(df, i) == 'string':
                df = df.withColumn(i, spk_func.split(spk_func.col(i), self.sep))
            if self.keepMostFrequent:
                df = jvm_categorify_by_freq_for_array(spark, df, i, dict_df)
            else:
                if self.multi_level_split:
                    df = jvm_categorify_for_multi_level_array(spark, df, i, dict_df, self.multi_level_sep)
                else:
                    df = jvm_categorify_for_array(spark, df, i, dict_df)
                if self.doSortForArray:
                    df = df.withColumn(i, spk_func.array_sort(spk_func.col(i)))
        return df

    def categorify_with_join(self, df, dict_df, i, spark, save_path, method='shj', saveTmpToDisk=False, enable_gazelle=False):
        saveTmpToDisk = self.saveTmpToDisk or saveTmpToDisk
        hint_type = "shuffle_hash"
        if method == "bhj":
            hint_type = "broadcast"
        elif method == "smj":
            hint_type = "merge"
        #print("do %s to %s" % (method, i))
        to_select = df.columns + ['dict_col_id']
        if self.doSplit == False:
            df = df.join(dict_df.hint(hint_type), spk_func.col(i) == dict_df.dict_col, 'left')\
                .select(*to_select).drop(i).withColumnRenamed('dict_col_id', i)
        else:
            df = df.withColumn(
                'row_id', spk_func.monotonically_increasing_id())
            if get_dtype(df, i) == 'string':
                df = df.withColumn(i, spk_func.split(spk_func.col(i), self.sep))
            tmp_df = df.select('row_id', i).withColumn(i, spk_func.explode(spk_func.col(i)))
            tmp_df = tmp_df.join(
                dict_df.hint("shuffle_hash"),
                spk_func.col(i) == dict_df.dict_col,
                'left').filter(spk_func.col('dict_col_id').isNotNull())
            tmp_df = tmp_df.select(
                'row_id', spk_func.col('dict_col_id'))
            if self.keepMostFrequent:
                tmp_df = tmp_df.groupby('row_id').agg(spk_func.array_sort(
                    spk_func.collect_list(spk_func.col('dict_col_id'))).getItem(0).alias('dict_col_id'))
            elif self.doSortForArray:
                tmp_df = tmp_df.groupby('row_id').agg(spk_func.array_sort(
                    spk_func.collect_list(spk_func.col('dict_col_id'))).alias('dict_col_id'))
            else:
                tmp_df = tmp_df.groupby('row_id').agg(
                    spk_func.collect_list(spk_func.col('dict_col_id')).alias('dict_col_id'))

            df = df.join(tmp_df.hint(hint_type),
                         'row_id', 'left').select(*to_select).drop(i).withColumnRenamed('dict_col_id', i)
        # if self.saveTmpToDisk is True, we will save current df to hdfs or localFS instead of replying on Shuffle
        if saveTmpToDisk:
            cur_save_path = "%s_%d" % (save_path, self.save_path_id)
            self.save_path_id += 1
            if enable_gazelle:
                df.write.format('arrow').mode('overwrite').save(save_path)
                df = spark.read.format("arrow").load(save_path)
            else:
                df.write.format('parquet').mode('overwrite').save(cur_save_path)
                df = spark.read.parquet(cur_save_path)
        return df

    def categorify_with_bhj(self, df, dict_df, i, spark):
        if self.doSplit == False:
            df = df.join(dict_df.hint('broadcast'), spk_func.col(i) == dict_df.dict_col, 'left')\
                .withColumn(i, dict_df.dict_col_id).drop("dict_col_id", "dict_col")
        else:
            raise NotImplementedError(
                "We should use udf to handle withSplit + small dict scenario")
        return df


class GenerateDictionary(Operation):
    '''
    Operation to generate dictionary based on single or multi-columns by distinct id

    Args:

        cols (list): columns which are be modified
        doSplit (bool): If we need to split data
        sep (str): split seperator
    '''
    # TODO: We should add an optimization for csv input

    def __init__(self, cols, withCount=True, doSplit=False, sep='\t', isParquet=True, bucketSize=-1, id_by_count = True, id_by_python_dict = False):
        self.op_name = "GenerateDictionary"
        self.cols = cols
        self.doSplit = doSplit
        self.sep = sep
        self.withCount = withCount
        self.dict_dfs = []
        self.isParquet = isParquet
        self.bucketSize = bucketSize
        self.id_by_count = id_by_count
        self.id_by_python_dict = id_by_python_dict

    def merge_dict(self, dict_dfs, to_merge_dict_dfs):
        self.dict_dfs = []
        for i in dict_dfs:
            dict_df = i['dict']
            dict_name = i['col_name']
            to_merge = self.find_dict(dict_name, to_merge_dict_dfs)
            if to_merge == None:
                raise ValueError(
                    "Expect '%s' in merge_dicts.to_merge_dict_dfs, while find none")
            max_id_of_to_merge_row = to_merge.agg(
                {'dict_col_id': 'max'}).collect()[0]
            max_id_of_to_merge = max_id_of_to_merge_row['max(dict_col_id)']
            fwd_dict_df = dict_df.join(to_merge, 'dict_col', 'anti').withColumn('dict_col_id', spk_func.row_number().over(
                Window.orderBy('dict_col_id')) + max_id_of_to_merge)
            self.dict_dfs.append(
                {'col_name': dict_name, 'dict': to_merge.union(fwd_dict_df)})
        return self.dict_dfs

    def to_dict_dfs(self, df, spark = None, cache_path = None, enable_gazelle=False):
        return self.generate_dict_dfs(df, spark, cache_path, self.withCount, self.isParquet, enable_gazelle=enable_gazelle)


class ModelMerge(Operation):
    '''
    Operation to Merge Pre-Generated Model

    Args:
      example - dicts
      [{'col_name': ['Id'], 'dict': top_answers_df}]
    '''
    # TODO: We should add an optimization for csv input

    def __init__(self, dicts, saveTmpToDisk=False,estimated_bytes=20):
        self.op_name = "ModelMerge"
        self.dict_dfs = dicts
        self.saveTmpToDisk = saveTmpToDisk
        self.save_path_id = 0
        self.cols = None
        self.doSplit = False
        self.estimated_bytes=estimated_bytes

    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.dict_dfs == None:
            raise NotImplementedError("process %s ")
        sorted_cols, strategy = self.categorify_strategy_decision_maker(self.dict_dfs, df, df_cnt, per_core_memory_size, flush_threshold, enable_gazelle,self.estimated_bytes)

        # For now, we prepared dict_dfs and strategy
        for col_name in sorted_cols:
            dict_df = self.find_dict(col_name, self.dict_dfs) 
            # for short dict, we will do bhj
            if 'short_dict' in strategy and col_name in strategy['short_dict']:
                df = self.merge_with_join(df, dict_df, col_name, spark, save_path, saveTmpToDisk=False, method='bhj', enable_gazelle=enable_gazelle)
        for col_name in sorted_cols:
            dict_df = self.find_dict(col_name, self.dict_dfs) 
            # for huge dict, we will do shj seperately
            if 'long_dict' in strategy and col_name in strategy['long_dict']:
                if 'huge_dict' in strategy and col_name in strategy['huge_dict']:
                    if 'smj_dict' in strategy and col_name in strategy['smj_dict']:
                        df = self.merge_with_join(df, dict_df, col_name, spark, save_path, saveTmpToDisk=True, method='smj', enable_gazelle=enable_gazelle)
                    else:
                        df = self.merge_with_join(df, dict_df, col_name, spark, save_path, saveTmpToDisk=True, method='shj', enable_gazelle=enable_gazelle)
                else:
                    if 'smj_dict' in strategy and col_name in strategy['smj_dict']:
                        df = self.merge_with_join(df, dict_df, col_name, spark, save_path, saveTmpToDisk=False, method='smj', enable_gazelle=enable_gazelle)
                    else:
                        df = self.merge_with_join(df, dict_df, col_name, spark, save_path, saveTmpToDisk=False, method='shj', enable_gazelle=enable_gazelle)
        return df

    def merge_with_join(self, df, dict_df, i, spark, save_path, saveTmpToDisk=False, method='shj', enable_gazelle=False):
        saveTmpToDisk = self.saveTmpToDisk or saveTmpToDisk
        hint_type = "shuffle_hash"
        if method == "bhj":
            hint_type = "broadcast"
        elif method == "smj":
            hint_type = "merge"
        #print("do %s to %s" % (method, i))
        df = df.join(dict_df.hint(hint_type), i, how='left')
        # if self.saveTmpToDisk is True, we will save current df to hdfs or localFS instead of replying on Shuffle
        if saveTmpToDisk:
            cur_save_path = "%s_%d" % (save_path, self.save_path_id)
            self.save_path_id += 1
            if enable_gazelle:
                df.write.format('arrow').mode('overwrite').save(save_path)
                df = spark.read.format("arrow").load(save_path)
            else:
                df.write.format('parquet').mode('overwrite').save(cur_save_path)
                df = spark.read.parquet(cur_save_path)
        return df


class NegativeSample(Operation):
    def __init__(self, cols, dicts, negCnt = 1):
        self.op_name = "NegativeSample"
        self.cols = cols
        self.dict_dfs = dicts
        self.neg_cnt = negCnt
        if len(cols) != len(dicts):
            raise ValueError("NegativeSample expects input dicts has same size cols for mapping")
        self.num_cols = len(cols)

    def get_negative_sample_udf(self, spark, broadcast_data):
        broadcast_movie_id_list = spark.sparkContext.broadcast(
            broadcast_data)

        def get_random_id(asin):
            item_list = broadcast_movie_id_list.value
            asin_total_len = len(item_list)
            asin_neg = asin
            while True:
                asin_neg_index = random.randint(0, asin_total_len - 1)
                asin_neg = item_list[asin_neg_index]
                if asin_neg == None or asin_neg == asin:
                    continue
                else:
                    break
            return [asin_neg, asin]

        return spk_func.udf(get_random_id, spk_type.ArrayType(spk_type.StringType()))


    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if enable_scala:
            for i in range(0, self.num_cols):
                df = jvm_get_negative_sample(spark, df, self.cols[i], self.get_colname_dict_as_tuple(self.dict_dfs[i])[1], self.neg_cnt)
        else:
            udf_list = []
            for i in range(0, self.num_cols):
                broadcasted_data = [row['dict_col'] for row in self.get_colname_dict_as_tuple(self.dict_dfs[i])[1].select("dict_col").collect()]
                negative_sample = self.get_negative_sample_udf(spark, broadcasted_data)
                udf_list.append(negative_sample)
            for i in range(0, self.num_cols):
                df = df.withColumn(self.cols[i], udf_list[i](spk_func.col(self.cols[i])))
                df = df.select(spk_func.col("*"), spk_func.posexplode(spk_func.col(self.cols[i]))).drop(self.cols[i]).withColumnRenamed("col", self.cols[i])
        return df


class NegativeFeature(Operation):
    def __init__(self, cols, dicts, doSplit=False, sep='\t', negCnt = 1):
        self.op_name = "NegativeFeature"
        self.cols = cols
        self.dict_dfs = dicts
        self.doSplit = doSplit
        self.neg_cnt = negCnt
        self.sep = sep
        if len(cols) != len(dicts):
            raise ValueError("NegativeFeature expects input dicts has same size cols for mapping")
        self.num_cols = len(cols)

    def get_negative_feature_udf(self, spark, broadcast_data):
        broadcast_movie_id_list = spark.sparkContext.broadcast(
            broadcast_data)

        if self.doSplit:
            def get_random_id(hist_asin):
                item_list = broadcast_movie_id_list.value
                asin_total_len = len(item_list)
                res = []
                asin_list = hist_asin.split(self.sep) if not isinstance(hist_asin, list) else hist_asin
                for asin in asin_list:
                    asin_neg = asin
                    while True:
                        asin_neg_index = random.randint(0, asin_total_len - 1)
                        asin_neg = item_list[asin_neg_index]
                        if asin_neg == None or asin_neg == asin:
                            continue
                        else:
                            res.append(asin_neg)
                            break
                return res

            return spk_func.udf(get_random_id, spk_type.ArrayType(spk_type.StringType()))
        else:
            def get_random_id(asin):
                item_list = broadcast_movie_id_list.value
                asin_total_len = len(item_list)
                asin_neg = asin
                while True:
                    asin_neg_index = random.randint(0, asin_total_len - 1)
                    asin_neg = item_list[asin_neg_index]
                    if asin_neg == None or asin_neg == asin:
                        continue
                    else:
                        break
                return asin_neg

            return spk_func.udf(get_random_id, spk_type.StringType())


    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if enable_scala:
            for tgt, src in self.cols.items():
                dict_df = self.find_dict(src, self.dict_dfs)
                if self.doSplit:
                    src_type = get_dtype(df, src)
                    if src_type == 'string':
                        df = df.withColumn(tgt, spk_func.split(spk_func.col(src), self.sep))
                        df = jvm_get_negative_feature_for_array(spark, df, tgt, tgt, dict_df, self.neg_cnt)
                    else:
                        df = jvm_get_negative_feature_for_array(spark, df, tgt, src, dict_df, self.neg_cnt)
                else:
                    df = jvm_get_negative_feature(spark, df, tgt, src, dict_df, self.neg_cnt)
        else:
            for tgt, src in self.cols.items():
                broadcasted_data = [row['dict_col'] for row in self.find_dict(src, self.dict_dfs)[1].select("dict_col").collect()]
                negative_feature = self.get_negative_feature_udf(spark, broadcasted_data)
                df = df.withColumn(tgt, negative_feature(spk_func.col(src)))
        return df


class ScalaDFTest(Operation):
    def __init__(self, cols, dicts=None, method="int"):
        self.op_name = "ScalaDFTest"
        self.cols = cols
        self.dict_dfs = dicts
        if dicts != None and len(cols) != len(dicts):
            raise ValueError("ScalaDFTest expects input dicts has same size cols for mapping")
        self.num_cols = len(cols)
        self.method = method


    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.method == "int":
            for i in range(0, self.num_cols):
                df = jvm_scala_df_test_int(spark, df, self.cols[i])
        elif self.method == "str":
            for i in range(0, self.num_cols):
                df = jvm_scala_df_test_str(spark, df, self.cols[i])
        elif self.method == "broadcast" and self.dict_dfs != None:
            for i in range(0, self.num_cols):
                df = jvm_scala_df_test_broadcast(spark, df, self.cols[i], self.get_colname_dict_as_tuple(self.dict_dfs[i])[1])
        return df


class CollapseByHist(Operation):
    def __init__(self, cols = [], by = None, orderBy = None, minNumHist = 0, maxNumHist = -1):
        self.op_name = "CollapseByHist"
        self.cols = cols
        self.by = by
        self.orderBy = orderBy
        self.minNumHist = minNumHist
        self.maxNumHist = maxNumHist
        if by == None or (isinstance(by, list) and len(by) == 0):
            raise ValueError("CollapseByHist expects input by should not None or Empty")


    def process(self, df, spark, df_cnt, enable_scala=True, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        w = Window.partitionBy(self.by)
        last_collapse_col = ""
        if self.orderBy != None:
            w = w.orderBy(self.orderBy)
        df = df.withColumn('row_id', spk_func.row_number().over(w))
        for c in self.cols:
            df = df.withColumn(f'hist_{c}', spk_func.collect_list(c).over(Window.partitionBy(self.by)))
            df = df.withColumn(f'hist_{c}', spk_func.when(spk_func.col(f'hist_{c}').isNull(), spk_func.array()).otherwise(spk_func.col(f'hist_{c}')))
            last_collapse_col = f'hist_{c}'
        if last_collapse_col == "":
            if self.orderBy != None:
                df = df.withColumn('row_max', spk_func.max(self.orderBy).over(Window.partitionBy(self.by)))
                df = df.filter((df[self.orderBy] == df.row_max))
                df = df.drop('row_id').drop('row_max')
            else:
                df = df.withColumn('row_cnt', spk_func.max('row_id').over(Window.partitionBy(self.by)))
                df = df.filter((df.row_id == df.row_cnt) & (df.row_cnt > spk_func.lit(self.minNumHist)))
                df = df.drop('row_id')
        else:
            df = df.withColumn('row_cnt', spk_func.size(spk_func.col(last_collapse_col)))
            df = df.withColumn('row_cnt', spk_func.col('row_cnt').cast(spk_type.IntegerType()))
            df = df.filter((df.row_id == df.row_cnt) & (df.row_cnt > spk_func.lit(self.minNumHist)))
            for c in self.cols:
                if self.maxNumHist != -1:
                    df = df.withColumn('max_hist_len', spk_func.when(spk_func.col('row_cnt') > self.maxNumHist, self.maxNumHist).otherwise(spk_func.col('row_cnt')))
                else:
                    df = df.withColumn('max_hist_len', spk_func.col('row_cnt'))
                df = df.withColumn(f'hist_{c}', spk_func.expr(f"slice(hist_{c}, 1, max_hist_len - 1)"))
            df = df.drop('row_id').drop('row_cnt').drop('max_hist_len')
        return df


class DataProcessor:
    def __init__(self, spark, path_prefix="hdfs://", current_path="", shuffle_disk_capacity="unlimited", dicts_path="dicts", spark_mode='local', enable_gazelle=False):
        self.ops = []
        self.spark = spark
        self.uuid = uuid.uuid1()
        self.tmp_id = 0
        self.path_prefix = path_prefix
        self.current_path = current_path
        self.dicts_path = dicts_path
        self.tmp_materialzed_list = []
        self.per_core_memory_size = 0
        self.enable_gazelle = enable_gazelle
        self.spark_mode = spark_mode
        fail_to_parse = True
        if spark_mode == 'yarn' or spark_mode == 'standalone':
            try:
                memory_size = parse_size(spark.sparkContext.getConf().get('spark.executor.memory'))
                memory_size += parse_size(spark.sparkContext.getConf().get('spark.executor.memoryOverhead'))
                numCores = int(spark.sparkContext.getConf().get('spark.executor.cores'))
                self.per_core_memory_size = memory_size / numCores
                fail_to_parse = False
            except:
                pass
        elif spark_mode == 'local' or fail_to_parse:
            memory_size = parse_size(spark.sparkContext.getConf().get('spark.driver.memory'))
            numCores = int(parse_cores_num(spark.sparkContext.getConf().get('spark.master')))
            self.per_core_memory_size = memory_size / numCores
            self.spark_mode = 'local'
        if shuffle_disk_capacity == "unlimited":
            self.flush_threshold = (2**63 - 1)
        else:
            self.flush_threshold = parse_size(shuffle_disk_capacity)
        self.gateway = spark.sparkContext._gateway
        self.enable_scala = self.registerScalaUDFs()
        print("per core memory size is %.3f GB and shuffle_disk maximum capacity is %.3f GB" % (self.per_core_memory_size * 1.0/(2**30), self.flush_threshold * 1.0/(2**30)))

    def __del__(self):
        for tmp_file in self.tmp_materialzed_list:
            shutil.rmtree(tmp_file, ignore_errors=True)

    def registerScalaUDFs(self):
        driverClassPath = self.spark.sparkContext.getConf().get('spark.driver.extraClassPath')
        execClassPath = self.spark.sparkContext.getConf().get('spark.executor.extraClassPath')
        if self.spark_mode == 'local':
            if driverClassPath == None or 'recdp' not in driverClassPath:
                return False
        else: 
            if driverClassPath == None or execClassPath == None or 'recdp' not in driverClassPath or 'recdp' not in execClassPath:
                return False
        print("recdp-scala-extension is enabled")
        self.spark.udf.registerJavaFunction("sortStringArrayByFrequency","com.intel.recdp.SortStringArrayByFrequency")
        self.spark.udf.registerJavaFunction("sortIntArrayByFrequency","com.intel.recdp.SortIntArrayByFrequency")
        self.spark._jsparkSession.udf().register("CodegenSeparator", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator())
        self.spark._jsparkSession.udf().register("CodegenSeparator0", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator0())
        self.spark._jsparkSession.udf().register("CodegenSeparator1", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator1())
        self.spark._jsparkSession.udf().register("CodegenSeparator2", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator2())
        return True

    def describe(self):
        description = []
        for op in self.ops:
            description.append(op.describe())
        print("DataProcessor current worflow is \n    ",
              '\n    ->'.join(description))

    def append_ops(self, ops):
        self.ops += ops

    def reset_ops(self, ops):
        self.ops = ops

    def get_tmp_cache_path(self):
        tmp_id = self.tmp_id
        self.tmp_id += 1
        save_path = "%s/%s/tmp/%s-%s-%d" % (
            self.path_prefix, self.current_path, "materialized_tmp", self.uuid, tmp_id)
        self.tmp_materialzed_list.append(save_path)
        return save_path

    def materialize(self, df, df_name="materialized_tmp", method=1):
        if method == 0:
            return df.cache()
        else:
            save_path = ""
            if df_name == "materialized_tmp":
                save_path = self.get_tmp_cache_path()
                self.tmp_materialzed_list.append(save_path)
            else:
                save_path = "%s/%s/%s" % (self.path_prefix,
                                          self.current_path, df_name)
            print(f"save data to {save_path}")
            if self.enable_gazelle:
                #df.write.format('parquet').mode('overwrite').save(save_path)
                df.write.format('arrow').mode('overwrite').save(save_path)
                return self.spark.read.format("arrow").load(save_path)
            else:
                df.write.format('parquet').mode('overwrite').save(save_path)
                return self.spark.read.parquet(save_path)

    def refine_op(self, op, df, save_path):
        if isinstance(op, Categorify) and op.dict_dfs == None :
            dict_dfs = []
            if not op.gen_dicts:
                dict_names = op.cols
                dict_dfs = [{'col_name': name, 'dict': self.spark.read.parquet(
                    "%s/%s/%s/%s" % (self.path_prefix, self.current_path, self.dicts_path, name))} for name in dict_names]
            if op.gen_dicts:
                dfs = []
                op_gen_dicts = GenerateDictionary(op.cols, doSplit=op.doSplit, sep=op.sep)
                dfs.extend(op_gen_dicts.to_dict_dfs(df, self.spark, save_path, self.enable_gazelle))
                for dict_df in dfs:
                    dict_dfs.append({'col_name': dict_df['col_name'], 'dict': self.materialize(
                        dict_df['dict'], "%s/%s" % (self.dicts_path, dict_df['col_name']))})
            op.dict_dfs = dict_dfs
        return op

    def apply(self, df, df_cnt = None):
        if len(self.ops) > 0 and df_cnt == None:
            df_cnt = df.count()
        for op in self.ops:
            save_path = self.get_tmp_cache_path()
            op = self.refine_op(op, df, save_path)
            df = op.process(df, self.spark, df_cnt, self.enable_scala, save_path=save_path, per_core_memory_size = self.per_core_memory_size, flush_threshold = self.flush_threshold, enable_gazelle=self.enable_gazelle)
        self.ops = []
        return df

    def transform(self, df, name="materialized_tmp", df_cnt = None):
        return self.materialize(self.apply(df, df_cnt), df_name=name)

    def generate_dicts(self, df):
        # flat ops to dfs
        dfs = []
        materialized_dfs = []
        for op in self.ops:
            if op.op_name != "GenerateDictionary":
                raise NotImplementedError(
                    "We haven't support apply generate_dict to not GenerateDictionary operator yet.")
            save_path = self.get_tmp_cache_path()
            dfs.extend(op.to_dict_dfs(df, self.spark, save_path, self.enable_gazelle))
        for dict_df in dfs:
            materialized_dfs.append({'col_name': dict_df['col_name'], 'dict': self.materialize(
                dict_df['dict'], "%s/%s" % (self.dicts_path, dict_df['col_name']))})
        return materialized_dfs

    def merge_dicts(self, df, to_merge_dict_dfs):
        # flat ops to dfs
        dfs = []
        materialized_dfs = []
        for op in self.ops:
            if op.op_name != "GenerateDictionary":
                raise NotImplementedError(
                    "We haven't support apply generate_dict to not GenerateDictionary operator yet.")
            save_path = get_tmp_cache_path()
            dfs.extend(op.merge_dict(op.to_dict_dfs(df, self.spark, save_path, self.enable_gazelle), to_merge_dict_dfs))
        for dict_df in dfs:
            materialized_dfs.append({'col_name': dict_df['col_name'], 'dict': self.materialize(
                dict_df['dict'], "%s/%s_merged" % (self.dicts_path, dict_df['col_name']))})
        return materialized_dfs

    def get_sample(self, df, df_cnt = None, vertical = False, truncate = 50):
        if df_cnt == None:
            df_cnt = df.count()
        for op in self.ops:
            save_path = self.get_tmp_cache_path()
            op = self.refine_op(op, df, save_path)
            df = op.process(df, self.spark, df_cnt, self.enable_scala, save_path=save_path, per_core_memory_size = self.per_core_memory_size, flush_threshold = self.flush_threshold, enable_gazelle=self.enable_gazelle)
        df.show(vertical = vertical, truncate = truncate)
