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

    def to_dict_dfs(self, df):
        raise NotImplementedError(self.op_name + "doesn't support to_dict_dfs")

    def process(self, df, spark, save_path=""):
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
                if dict_name == name:
                    return dict_df
            else:
                dict_df = i['dict']
                dict_name = i['col_name']
                if dict_name == name:
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

    def generate_dict_dfs(self, df, withCount=False):
        # at least we can cache here to make read much faster
        # handle multiple columns issue
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
            dict_df = dict_df.withColumn('dict_col_id', spk_func.row_number().over(
                Window.orderBy(spk_func.desc('count')))).withColumn('dict_col_id', spk_func.col('dict_col_id') - 1).select('dict_col', 'dict_col_id', 'count')
            if withCount:
                print("generate_dict_dfs withCount = True")
                self.dict_dfs.append({'col_name': col_name, 'dict': dict_df})
            else:
                print("generate_dict_dfs withCount = False")
                self.dict_dfs.append(
                    {'col_name': col_name, 'dict': dict_df.drop('count')})
        return self.dict_dfs


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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
        if self.dict_dfs == None:
            # We should do categorify upon same column
            self.dict_dfs = self.to_dict_dfs(df)
            for i in self.dict_dfs:
                col_name, dict_df = self.get_colname_dict_as_tuple(i)
        strategy = {}
        if self.hint != "auto" and self.hint in self.strategy_type:
            strategy[self.strategy_type[self.hint]] = []
            for i in self.dict_dfs:
                col_name, dict_df = self.get_colname_dict_as_tuple(i)
                strategy[self.strategy_type[self.hint]].append(col_name)
        else:
            strategy = self.categorify_strategy_decision_maker(self.dict_dfs)

        # For now, we prepared dict_dfs and strategy
        for i in self.cols:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            # for udf type, we will do udf to do user-defined categorify
            if 'udf' in strategy and src_name in strategy['udf']:
                dict_data = dict((row['dict_col'], row['dict_col_id'])
                                 for row in dict_df.collect())
                df = self.categorify_with_udf(df, dict_data, col_name, spark)
        for i in self.cols:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            # for short dict, we will do bhj
            if 'short_dict' in strategy and src_name in strategy['short_dict']:
                df = self.categorify_with_bhj(df, dict_df, col_name, spark)
        for i in self.cols:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            # for long dict, we will do shj all along
            if 'long_dict' in strategy and src_name in strategy['long_dict']:
                df = self.categorify_with_bhj(df, dict_df, col_name, spark)
        for i in self.cols:
            col_name, src_name = self.get_col_tgt_src(i)
            dict_df = self.find_dict(src_name, self.dict_dfs)
            # for huge dict, we will do shj seperately
            if 'huge_dict' in strategy and src_name in strategy['huge_dict']:
                df = self.categorify_with_join(
                    df, dict_df, col_name, spark, save_path)
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

    def categorify_with_udf(self, df, dict_data, i, spark):
        udf_impl = self.get_mapping_udf(dict_data, spark)
        df = df.withColumn(i, udf_impl(spk_func.col(i)))
        return df

    def categorify_with_join(self, df, dict_df, i, spark, save_path):
        if self.doSplit == False:
            df = df.join(dict_df.hint('shuffle_hash'), spk_func.col(i) == dict_df.dict_col, 'left')\
                .withColumn(i, dict_df.dict_col_id).drop("dict_col_id", "dict_col")
        else:
            df = df.withColumn(
                'row_id', spk_func.monotonically_increasing_id())
            tmp_df = df.select('row_id', i).withColumn(
                i, f.explode(f.split(f.col(i), self.sep)))
            tmp_df = tmp_df.join(
                dict_df.hint('shuffle_hash'),
                spk_func.col(i) == dict_df.dict_col,
                'left').filter(spk_func.col('dict_col_id').isNotNull())
            tmp_df = tmp_df.select(
                'row_id', spk_func.col('dict_col_id'))
            if self.doSortForArray:
                if self.keepMostFrequent:
                    tmp_df = tmp_df.groupby('row_id').agg(spk_func.array_sort(
                        spk_func.collect_list(spk_func.col('dict_col_id'))).getItem(0).alias('dict_col_id'))
                else:
                    tmp_df = tmp_df.groupby('row_id').agg(spk_func.array_sort(
                        spk_func.collect_list(spk_func.col('dict_col_id'))).alias('dict_col_id'))
            else:
                tmp_df = tmp_df.groupby('row_id').agg(
                    spk_func.collect_list(spk_func.col('dict_col_id')).alias('dict_col_id'))
            df = df.join(tmp_df.hint('shuffle_hash'),
                         'row_id', 'left').drop('row_id').drop(i).withColumnRenamed('dict_col_id', i)
        # if self.saveTmpToDisk is True, we will save current df to hdfs or localFS instead of replying on Shuffle
        if self.saveTmpToDisk:
            cur_save_path = "%s_%d" % (save_path, self.save_path_id)
            self.save_path_id += 1
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

    def categorify_strategy_decision_maker(self, dict_dfs):
        small_cols = []
        long_cols = []
        huge_cols = []
        udf_cols = []
        for i in dict_dfs:
            col_name, dict_df = self.get_colname_dict_as_tuple(i)
            if (dict_df.count() > 30000000):
                huge_cols.append(col_name)
            elif (dict_df.count() > 30000000):
                long_cols.append(col_name)
            else:
                if self.doSplit:
                    udf_cols.append(col_name)
                else:
                    small_cols.append(col_name)
        return {'short_dict': small_cols, 'long_dict': long_cols, 'huge_dict': huge_cols, 'udf': udf_cols}


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

    def process(self, df, spark, save_path=""):
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

    def process(self, df, spark, save_path=""):
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

    def __init__(self, cols, withCount=False, doSplit=False, sep='\t'):
        self.op_name = "GenerateDictionary"
        self.cols = cols
        self.doSplit = doSplit
        self.sep = sep
        self.withCount = withCount
        self.dict_dfs = []

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

    def to_dict_dfs(self, df):
        return self.generate_dict_dfs(df, self.withCount)


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

    def process(self, df, spark, save_path=""):
        if self.dict_dfs == None:
            raise NotImplementedError("process %s ")
        strategy = self.strategy_decision_maker(self.dict_dfs)

        # For now, we prepared dict_dfs and strategy
        for out_col, i_dict in self.dict_dfs.items():
            col_name, dict_df, default_v = i_dict
            # for short dict, we will do bhj
            if 'short_dict' in strategy and out_col in strategy['short_dict']:
                df = self.merge_with_bhj(
                    df, dict_df, col_name, default_v, out_col, spark)
        for out_col, i_dict in self.dict_dfs.items():
            col_name, dict_df, default_v = i_dict
            # for huge dict, we will do shj seperately
            if 'huge_dict' in strategy and out_col in strategy['huge_dict']:
                df = self.merge_with_join(
                    df, dict_df, col_name, default_v, out_col, spark, save_path)
        return df

    def merge_with_join(self, df, dict_df, i, default_v, out_col, spark, save_path):
        df = df.join(dict_df.hint('shuffle_hash'), i, how='left')
        # df = df.fillna(default_v, out_col)
        # if self.saveTmpToDisk is True, we will save current df to hdfs or localFS instead of replying on Shuffle
        if self.saveTmpToDisk:
            cur_save_path = "%s_%d" % (save_path, self.save_path_id)
            self.save_path_id += 1
            df.write.format('parquet').mode('overwrite').save(cur_save_path)
            df = spark.read.parquet(cur_save_path)
        return df

    def merge_with_bhj(self, df, dict_df, i, default_v, out_col, spark):
        df = df.join(spk_func.broadcast(dict_df), i, how='left')
        # df = df.fillna(default_v, out_col)
        return df

    def strategy_decision_maker(self, dict_dfs):
        small_cols = []
        long_cols = []
        huge_cols = []
        udf_cols = []
        small_cols_total = 0
        for out_col, i_dict in self.dict_dfs.items():
            col_name, dict_df, default_v = i_dict
            dict_df_count = dict_df.count()
            if (dict_df_count > 10000000):
                huge_cols.append(out_col)
            else:
                if small_cols_total < 50000000 or dict_df_count < 1000000:
                    small_cols_total += dict_df_count
                    small_cols.append(out_col)
                else:
                    huge_cols.append(out_col)
        return {'short_dict': small_cols, 'huge_dict': huge_cols}

# class NegativeSample(Operation):
#    def get_negative_sample_udf(self, broadcast_data):
#        broadcast_movie_id_list = self.spark.sparkContext.broadcast(
#            broadcast_data)
#
#        def get_random_id(asin):
#            item_list = broadcast_movie_id_list.value
#            asin_total_len = len(item_list)
#            asin_neg = asin
#            while True:
#                asin_neg_index = random.randint(0, asin_total_len - 1)
#                asin_neg = item_list[asin_neg_index]
#                if asin_neg == None or asin_neg == asin:
#                    continue
#                else:
#                    break
#            return asin_neg
#        return udf(get_random_id, StringType())


# class RandomIndex(Operation):
#    def rand_ordinal_n(self, df, n, name='ordinal'):
#        return df.withColumn(name, (rand() * n).cast("int"))


class DataProcessor:
    def __init__(self, spark, path_prefix="hdfs://", current_path="", dicts_path=""):
        self.ops = []
        self.spark = spark
        self.uuid = uuid.uuid1()
        self.tmp_id = 0
        self.path_prefix = path_prefix
        self.current_path = current_path
        self.dicts_path = dicts_path
        self.tmp_materialzed_list = []

    def __del__(self):
        for tmp_file in self.tmp_materialzed_list:
            shutil.rmtree(tmp_file, ignore_errors=True)

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

    def materialize(self, df, df_name="materialized_tmp", method=1):
        if method == 0:
            return df.cache()
        else:
            tmp_id = self.tmp_id
            self.tmp_id += 1
            save_path = ""
            if df_name == "materialized_tmp":
                save_path = "%s/%s/tmp/%s-%s-%d" % (
                    self.path_prefix, self.current_path, df_name, self.uuid, tmp_id)
                self.tmp_materialzed_list.append(save_path)
            else:
                save_path = "%s/%s/%s" % (self.path_prefix,
                                          self.current_path, df_name)
            df.write.format('parquet').mode('overwrite').save(save_path)
            return self.spark.read.parquet(save_path)

    def transform(self, df, name="materialized_tmp"):
        tmp_id = self.tmp_id
        self.tmp_id += 1
        save_path = "%s/%s/tmp/%s-%s-%d" % (
            self.path_prefix, self.current_path, "materialized_tmp", self.uuid, tmp_id)
        self.tmp_materialzed_list.append(save_path)
        for op in self.ops:
            df = op.process(df, self.spark, save_path=save_path)
        return self.materialize(df, df_name=name)

    def generate_dicts(self, df):
        # flat ops to dfs
        dfs = []
        materialized_dfs = []
        for op in self.ops:
            if op.op_name != "GenerateDictionary":
                raise NotImplementedError(
                    "We haven't support apply generate_dict to not GenerateDictionary operator yet.")
            dfs.extend(op.to_dict_dfs(df))
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
            dfs.extend(op.merge_dict(op.to_dict_dfs(df), to_merge_dict_dfs))
        for dict_df in dfs:
            materialized_dfs.append({'col_name': dict_df['col_name'], 'dict': self.materialize(
                dict_df['dict'], "%s/%s_merged" % (self.dicts_path, dict_df['col_name']))})
        return materialized_dfs

    def collect(self, df):
        for op in self.ops[:-1]:
            df = op.process(df, self.spark)
        return self.ops[-1].collect(df)
