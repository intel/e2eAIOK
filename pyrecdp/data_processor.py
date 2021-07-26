import uuid
from .utils import *
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
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
                df.write.format('arrow').mode('overwrite').save(save_path)
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
            dict_df = dict_df.groupBy('dict_col').count()
            if self.bucketSize == -1:
                dict_df = dict_df.withColumn('dict_col_id', spk_func.row_number().over(
                    Window.orderBy(spk_func.desc('count')))).withColumn('dict_col_id', spk_func.col('dict_col_id') - 1).select('dict_col', 'dict_col_id', 'count')
            else:
                # when user set a bucketSize, we will quantileDiscretizer in this case
                qd = QuantileDiscretizer(numBuckets=self.bucketSize, inputCol="count", outputCol='dict_col_id')
                dict_df  = qd.fit(dict_df).transform(dict_df).withColumn('dict_col_id', spk_func.col('dict_col_id').cast(spk_type.IntegerType()))
            if withCount:
                self.dict_dfs.append({'col_name': col_name, 'dict': dict_df})
            else:
                self.dict_dfs.append(
                    {'col_name': col_name, 'dict': dict_df.drop('count')})
        return self.dict_dfs


    def categorify_strategy_decision_maker(self, dict_dfs, df, df_cnt, per_core_memory_size, flush_threshold):
        small_cols = []
        long_cols = []
        huge_cols = []
        smj_cols = []
        udf_cols = []
        total_small_cols_num_rows = 0
        total_estimated_shuffled_size = 0
        df_estimzate_size_per_row = sum([get_estimate_size_of_dtype(dtype) for _, dtype in df.dtypes])
        df_estimated_size = df_estimzate_size_per_row * df_cnt
        # Below threshold is maximum BHJ numRows, 4 means maximum 50% memory to cache Broadcast data, and 20 means we estimate each row has 20 bytes.
        threshold = per_core_memory_size / 4 / 20
        threshold_per_bhj = threshold if threshold <= 30000000 else 30000000
        flush_threshold = flush_threshold * 0.8
        print("bhj total threshold is %.3f M rows, one bhj threshold is %.3f M rows, flush_threshold is %.3f GB" % (threshold / 1000000, threshold_per_bhj/1000000, flush_threshold / 2**30))
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
        for to_print in sorted_dict_dfs_with_cnt:
            print(to_print)

        sorted_cols = []
        for (col_name, dict_df, dict_df_cnt) in sorted_dict_dfs_with_cnt:
            sorted_cols.append(col_name)
            if self.doSplit:
                threshold_per_bhj = 30000000
            if (dict_df_cnt > threshold_per_bhj or (total_small_cols_num_rows + dict_df_cnt) > threshold):
                if ((total_estimated_shuffled_size + df_estimated_size) > flush_threshold):
                    print("etstimated_to_shuffle_size for %s is %.3f GB, will do smj and spill to disk" % (str(col_name), df_estimated_size / 2**30))
                    huge_cols.append(col_name)
                    smj_cols.append(col_name)
                    long_cols.append(col_name)
                    total_estimated_shuffled_size = 0
                else:
                    print("etstimated_to_shuffle_size for %s is %.3f GB, will do smj" % (str(col_name), df_estimated_size / 2**30))
                    smj_cols.append(col_name)
                    long_cols.append(col_name)
                    # if accumulate shuffle capacity may exceed maximum shuffle disk size, we should use hdfs instead
                    total_estimated_shuffled_size += df_estimated_size
            else:
                if self.doSplit:
                    print("%s will do udf" % (col_name))
                    total_small_cols_num_rows += dict_df_cnt
                    udf_cols.append(col_name)
                else:
                    print("%s will do bhj" % (col_name))
                    total_small_cols_num_rows += dict_df_cnt
                    small_cols.append(col_name)
            df_estimated_size -= df_cnt * 6
        return (sorted_cols, {'short_dict': small_cols, 'long_dict': long_cols, 'huge_dict': huge_cols, 'udf': udf_cols, 'smj_dict': smj_cols})

    def check_scala_extension(self, spark):
        driverClassPath = spark.sparkContext.getConf().get('spark.driver.extraClassPath')
        execClassPath = spark.sparkContext.getConf().get('spark.executor.extraClassPath')
        if driverClassPath == None or execClassPath == None or 'recdp' not in driverClassPath or 'recdp' not in execClassPath:
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.op == 'udf':
            for after, before in self.cols.items():
                df = df.withColumn(after, self.udf_impl(spk_func.col(before)))
        elif self.op == 'inline':
            for after, before in self.cols.items():
                #print("[DEBUG]: inline script is %s" % before)
                df = df.withColumn(after, eval(before))
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
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

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        to_select = []
        for i in self.cols:
            if isinstance(i, str):
                to_select.append(i)
            elif isinstance(i, tuple):
                to_select.append("%s as %s" % (i[0], i[1]))

        return df.selectExpr(*to_select)


class Categorify(Operation):
    '''
    Operation to categorify columns by distinct id

    Args:

        cols (list): columns which are be modified
    '''

    def __init__(self, cols, dict_dfs=None, hint='auto', doSplit=False, sep='\t', doSortForArray=False, keepMostFrequent=False, saveTmpToDisk=False):
        self.op_name = "Categorify"
        self.cols = cols
        self.dict_dfs = dict_dfs
        self.hint = hint
        self.doSplit = doSplit
        self.sep = sep
        self.doSortForArray = doSortForArray
        self.keepMostFrequent = keepMostFrequent
        self.saveTmpToDisk = saveTmpToDisk
        self.save_path_id = 0
        self.strategy_type = {
            'udf': 'udf', 'broadcast_join': 'short_dict', 'shuffle_join': 'huge_dict'}

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.dict_dfs == None:
            # We should do categorify upon same column
            self.dict_dfs = self.to_dict_dfs(df, spark, save_path, enable_gazelle)
            for i in self.dict_dfs:
                col_name, dict_df = self.get_colname_dict_as_tuple(i)
        strategy = {}
        sorted_cols = []
        if self.hint != "auto" and self.hint in self.strategy_type:
            strategy[self.strategy_type[self.hint]] = []
            for i in self.dict_dfs:
                col_name, dict_df = self.get_colname_dict_as_tuple(i)
                strategy[self.strategy_type[self.hint]].append(col_name)
                sorted_cols.append(col_name)
        else:
            sorted_cols, strategy = self.categorify_strategy_decision_maker(self.dict_dfs, df, df_cnt, per_core_memory_size, flush_threshold)
        
        sorted_cols_pair = []
        for cn in sorted_cols:
            for i in self.cols:
                col_name, src_name = self.get_col_tgt_src(i)
                if src_name == cn:
                    sorted_cols_pair.append({col_name: src_name})
                    break

        last_method = 'bhj'
        # For now, we prepared dict_dfs and strategy
        for i in sorted_cols_pair:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            if col_name != src_name:
                df = df.withColumn(col_name, spk_func.col(src_name))
            # for udf type, we will do udf to do user-defined categorify
            if 'udf' in strategy and src_name in strategy['udf']:
                last_method = 'udf'
                #df = self.categorify_with_udf(df, dict_df, col_name, spark)
                df = self.categorify_with_scala_udf(df, dict_df, col_name, spark)
        for i in sorted_cols_pair:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            if col_name != src_name:
                df = df.withColumn(col_name, spk_func.col(src_name))
            # for short dict, we will do bhj
            if 'short_dict' in strategy and src_name in strategy['short_dict']:
                last_method = 'bhj'
                df = self.categorify_with_join(df, dict_df, col_name, spark, save_path, method="bhj")
        for i in sorted_cols_pair:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            if col_name != src_name:
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
        if last_method == 'bhj':
            print("Adding a CodegenSeparator to pure BHJ WSCG case")
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
                for v in x.split(sep):
                    if v != '' and v in broadcast_data and (min_val == None or broadcast_data[v] < min_val):
                        min_val = broadcast_data[v]
            return min_val

        def freq_encode(x):
            broadcast_data = broadcast_dict.value
            val = []
            if x != '':
                for v in x.split(sep):
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
        dict_data = dict((row['dict_col'], row['dict_col_id']) for row in dict_df.collect())
        udf_impl = self.get_mapping_udf(dict_data, spark)
        df = df.withColumn(i, udf_impl(spk_func.col(i)))
        return df

    def categorify_with_scala_udf(self, df, dict_df, i, spark):
        # call java to broadcast data and set broadcast handler to udf
        if not self.check_scala_extension(spark):
            raise ValueError("RecDP need to enable recdp-scala-extension to run categorify_with_udf, please config spark.driver.extraClassPath and spark.executor.extraClassPath for spark")
        gateway = spark.sparkContext._gateway
        categorify_broadcast_handler = gateway.jvm.org.apache.spark.sql.api.CategorifyBroadcast.broadcast(spark.sparkContext._jsc, dict_df._jdf)

        if self.doSplit == False:
            spark._jsparkSession.udf().register(f"Categorify_{i}", gateway.jvm.org.apache.spark.sql.api.Categorify(categorify_broadcast_handler))
            df = df.withColumn(i, spk_func.expr(f"Categorify_{i}({i})"))
        else:
            df = df.withColumn(i, spk_func.split(spk_func.col(i), self.sep))
            if self.keepMostFrequent:
                spark._jsparkSession.udf().register(f"CategorifyByFreqForArray_{i}", gateway.jvm.org.apache.spark.sql.api.CategorifyByFreqForArray(categorify_broadcast_handler))
                df = df.withColumn(i, spk_func.expr(f"CategorifyByFreqForArray_{i}({i})"))
            else:
                spark._jsparkSession.udf().register(f"CategorifyForArray_{i}", gateway.jvm.org.apache.spark.sql.api.CategorifyForArray(categorify_broadcast_handler))
                df = df.withColumn(i, spk_func.expr(f"CategorifyForArray_{i}({i})"))
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
        print("do %s to %s" % (method, i))
        to_select = df.columns + ['dict_col_id']
        if self.doSplit == False:
            df = df.join(dict_df.hint(hint_type), spk_func.col(i) == dict_df.dict_col, 'left')\
                .select(*to_select).drop(i).withColumnRenamed('dict_col_id', i)
        else:
            df = df.withColumn(
                'row_id', spk_func.monotonically_increasing_id())
            tmp_df = df.select('row_id', i).withColumn(
                i, f.explode(f.split(f.col(i), self.sep)))
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

    


class CategorifyMultiItems(Operation):
    '''
    Operation to categorify columns contains multiple id in one item

    Args:

        cols (list): columns which are be modified
        strategy (int): 1. return id with biggest freq_cnt in list - used by recsys; 2. TBD ...
        sep (str): separator, default is '\t'
    '''

    def __init__(self, cols, strategy=0, sep='\t', skipList=[], freqRange=[2, 100000]):
        self.op_name = "CategorifyMultiItems"
        self.cols = cols
        self.sep = sep
        self.strategy = strategy
        self.skipList = skipList
        self.freqRange = freqRange

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        for i in self.cols:
            sorted_data_df = df.select(spk_func.explode(spk_func.split(spk_func.col(i), self.sep))).groupBy('col').count().orderBy(
                spk_func.desc('count'), 'col').select('col', 'count')
            if self.strategy == 0:
                sorted_data = sorted_data_df.collect()
                dict_data = dict((id['col'], idx) for (id, idx) in zip(
                    sorted_data, range(len(sorted_data))))
                udf_impl = self.get_mapping_udf_0(
                    dict_data, spark, self.sep, self.skipList)
                df = df.withColumn(i, udf_impl(spk_func.col(i)))
            elif self.strategy == 1:
                sorted_data = sorted_data_df.collect()
                dict_data = dict((id['col'], [id['count'], idx]) for (id, idx) in zip(
                    sorted_data, range(len(sorted_data))))
                udf_impl = self.get_mapping_udf_1(
                    dict_data, spark, self.sep, self.skipList, self.freqRange)
                df = df.withColumn(i, udf_impl(spk_func.col(i)))

            else:
                raise NotImplementedError(
                    "CategorifyMultiItems only supports strategy as 0")
        return df

    def get_mapping_udf_0(self, broadcast_data, spark, sep, skipList, default=None):
        # check if we support this type
        first_value = next(iter(broadcast_data.values()))
        if not isinstance(first_value, int) and not isinstance(first_value, str) and not isinstance(first_value, float):
            raise NotImplementedError

        # numPrint = 0
        # for key, value in broadcast_data.items():
        #    print(key, value)
        #    numPrint += 1
        #    if (numPrint > 20):
        #        break

        broadcast_dict = spark.sparkContext.broadcast(
            broadcast_data)

        def largest_freq_encode(x):
            broadcast_data = broadcast_dict.value
            if x != '':
                val = []
                for v in x.split(sep):
                    if v != '' and v in broadcast_data:
                        val.append(broadcast_data[v])
                if len(val) > 0:
                    val.sort()
                    return val[0]
                else:
                    return 0
            else:
                return 0
        # switch return type
        if isinstance(first_value, int):
            return spk_func.udf(largest_freq_encode, spk_type.IntegerType())
        if isinstance(first_value, str):
            return spk_func.udf(largest_freq_encode, spk_type.StringType())
        if isinstance(first_value, float):
            return spk_func.udf(largest_freq_encode, spk_type.FloatType())

    def get_mapping_udf_1(self, broadcast_data, spark, sep, skipList, freqRange, default=None):
        broadcast_dict = spark.sparkContext.broadcast(
            broadcast_data)

        def frequence_encode(x):
            dict_data = broadcast_dict.value
            li = []
            for v in x.split(sep):
                if v not in skipList:
                    f, i = dict_data[v]
                    if f < freqRange[1] and f > freqRange[0]:
                        li.append(i)
            return sorted(li, reverse=True)
        return spk_func.udf(frequence_encode, spk_type.ArrayType(spk_type.IntegerType()))


class CategorifyWithDictionary(Operation):
    '''
    Operation to categorify columns by pre-defined dictionary

    Args:

        cols (list): columns which are be modified
        dictData (dict): pre-defined dictionary to map column data to corresponding id
    '''

    def __init__(self, cols, dictData):
        self.op_name = "CategorifyWithDictionary"
        self.cols = cols
        self.dict_data = dictData

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if len(self.dict_data) == 0:
            for i in self.cols:
                df = df.withColumn(i, spk_func.lit(None))
        else:
            udf_impl = self.get_mapping_udf(self.dict_data, spark)
            for i in self.cols:
                df = df.withColumn(i, udf_impl(spk_func.col(i)))
        return df

    def get_mapping_udf(self, broadcast_data, spark, default=None):
        # check if we support this type
        first_value = next(iter(broadcast_data.values()))
        if not isinstance(first_value, int) and not isinstance(first_value, str) and not isinstance(first_value, float):
            raise NotImplementedError

        broadcast_dict = spark.sparkContext.broadcast(
            broadcast_data)

        def get_mapped(x):
            map_dict = broadcast_dict.value
            if x in map_dict:
                return map_dict[x]
            else:
                return default
        # switch return type
        if isinstance(first_value, int):
            return spk_func.udf(get_mapped, spk_type.IntegerType())
        if isinstance(first_value, str):
            return spk_func.udf(get_mapped, spk_type.StringType())
        if isinstance(first_value, float):
            return spk_func.udf(get_mapped, spk_type.FloatType())


class GenerateDictionary(Operation):
    '''
    Operation to generate dictionary based on single or multi-columns by distinct id

    Args:

        cols (list): columns which are be modified
        doSplit (bool): If we need to split data
        sep (str): split seperator
    '''
    # TODO: We should add an optimization for csv input

    def __init__(self, cols, withCount=True, doSplit=False, sep='\t', isParquet=True, bucketSize=-1):
        self.op_name = "GenerateDictionary"
        self.cols = cols
        self.doSplit = doSplit
        self.sep = sep
        self.withCount = withCount
        self.dict_dfs = []
        self.isParquet = isParquet
        self.bucketSize = bucketSize

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
    '''
    # TODO: We should add an optimization for csv input

    def __init__(self, dicts, saveTmpToDisk=False):
        self.op_name = "ModelMerge"
        self.dict_dfs = dicts
        self.saveTmpToDisk = saveTmpToDisk
        self.save_path_id = 0
        self.cols = None
        self.doSplit = False

    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        if self.dict_dfs == None:
            raise NotImplementedError("process %s ")
        sorted_cols, strategy = self.categorify_strategy_decision_maker(self.dict_dfs, df, df_cnt, per_core_memory_size, flush_threshold)

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
        print("do %s to %s" % (method, i))
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
    def __init__(self, cols, dicts):
        self.op_name = "NegativeSample"
        self.cols = cols
        self.dict_dfs = dicts
        if len(cols) != len(dicts):
            raise ValueError("NegativeSample expects input dicts has same size cols for mapping")
        self.num_cols = len(cols)


    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        for i in range(0, self.num_cols):
            df = jvm_get_negative_samples(spark, df, self.cols[i], self.get_colname_dict_as_tuple(self.dict_dfs[i])[1])
        return df


class CollapseByHist(Operation):
    def __init__(self, cols, by, orderBy = None, minNumHist = 0):
        self.op_name = "CollapseByHist"
        self.cols = cols
        self.by = by
        self.orderBy = orderBy
        self.minNumHist = minNumHist
        if cols == None or by == None or len(cols) == 0 or (isinstance(by, list) and len(by) == 0):
            raise ValueError("CollapseByHist expects input cols and by are not None or Empty")


    def process(self, df, spark, df_cnt, save_path="", per_core_memory_size=0, flush_threshold = 0, enable_gazelle=False):
        w = Window.partitionBy(self.by)
        if self.orderBy != None:
            w = w.orderBy(self.orderBy)
        for c in self.cols:
            df = df.withColumn(f'hist_{c}', spk_func.collect_list(c).over(w))        
        df = df.withColumn('row_id', spk_func.row_number().over(w))
        df = df.withColumn('row_cnt', spk_func.count(spk_func.lit(1)).over(Window.partitionBy(self.by)))
        df = df.withColumn('row_cnt', spk_func.col('row_cnt').cast(spk_type.IntegerType()))
        df = df.filter((df.row_id == df.row_cnt) & (df.row_cnt > spk_func.lit(self.minNumHist)))
        for c in self.cols:
            df = df.withColumn(f'hist_{c}', spk_func.expr(f"slice(hist_{c}, 1, row_cnt - 1)"))
        df = df.drop('row_id').drop('row_cnt')
        return df


#class RandomIndex(Operation):
#   def rand_ordinal_n(self, df, n, name='ordinal'):
#       return df.withColumn(name, (rand() * n).cast("int"))


class DataProcessor:
    def __init__(self, spark, path_prefix="hdfs://", current_path="", shuffle_disk_capacity="unlimited", dicts_path="dicts", spark_mode='yarn', enable_gazelle=False):
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
        if spark_mode == 'yarn' or spark_mode == 'standalone':
            memory_size = parse_size(spark.sparkContext.getConf().get('spark.executor.memory'))
            numCores = int(spark.sparkContext.getConf().get('spark.executor.cores'))
            self.per_core_memory_size = memory_size / numCores
        elif spark_mode == 'local':
            memory_size = parse_size(spark.sparkContext.getConf().get('spark.driver.memory'))
            numCores = int(spark.sparkContext.getConf().get('spark.executor.cores'))
            self.per_core_memory_size = memory_size / numCores
        if shuffle_disk_capacity == "unlimited":
            self.flush_threshold = (2**63 - 1)
        else:
            self.flush_threshold = parse_size(shuffle_disk_capacity)
        self.gateway = spark.sparkContext._gateway
        self.registerScalaUDFs()
        print("per core memory size is %.3f GB and shuffle_disk maximum capacity is %.3f GB" % (self.per_core_memory_size * 1.0/(2**30), self.flush_threshold * 1.0/(2**30)))

    def __del__(self):
        for tmp_file in self.tmp_materialzed_list:
            shutil.rmtree(tmp_file, ignore_errors=True)

    def registerScalaUDFs(self):
        driverClassPath = self.spark.sparkContext.getConf().get('spark.driver.extraClassPath')
        execClassPath = self.spark.sparkContext.getConf().get('spark.executor.extraClassPath')
        if driverClassPath == None or execClassPath == None or 'recdp' not in driverClassPath or 'recdp' not in execClassPath:
            return
        print("recdp-scala-extension is enabled")
        self.spark.udf.registerJavaFunction("sortStringArrayByFrequency","com.intel.recdp.SortStringArrayByFrequency")
        self.spark.udf.registerJavaFunction("sortIntArrayByFrequency","com.intel.recdp.SortIntArrayByFrequency")
        self.spark._jsparkSession.udf().register("CodegenSeparator", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator())
        self.spark._jsparkSession.udf().register("CodegenSeparator0", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator0())
        self.spark._jsparkSession.udf().register("CodegenSeparator1", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator1())
        self.spark._jsparkSession.udf().register("CodegenSeparator2", self.gateway.jvm.org.apache.spark.sql.api.CodegenSeparator2())

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
            if self.enable_gazelle:
                #df.write.format('parquet').mode('overwrite').save(save_path)
                df.write.format('arrow').mode('overwrite').save(save_path)
                return self.spark.read.format("arrow").load(save_path)
            else:
                df.write.format('parquet').mode('overwrite').save(save_path)
                return self.spark.read.parquet(save_path)

    def transform(self, df, name="materialized_tmp", df_cnt = None):
        if df_cnt == None:
            df_cnt = df.count()
        for op in self.ops:
            save_path = self.get_tmp_cache_path()
            df = op.process(df, self.spark, df_cnt, save_path=save_path, per_core_memory_size = self.per_core_memory_size, flush_threshold = self.flush_threshold, enable_gazelle=self.enable_gazelle)
        return self.materialize(df, df_name=name)

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

    def collect(self, df):
        for op in self.ops[:-1]:
            df = op.process(df, self.spark)
        return self.ops[-1].collect(df)

    def py2Java(self, df, func_name, javaClassName, *args):
        self.spark._jsparkSession.udf().register(func_name, javaClassName)
