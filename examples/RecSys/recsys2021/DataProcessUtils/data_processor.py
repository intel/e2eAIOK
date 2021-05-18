import logging
from timeit import default_timer as timer
import os
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd
from pyspark.ml.feature import *
from .utils import *
import uuid


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

    def process(self, df, spark):
        return df


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

    def process(self, df, spark):
        if self.op == 'udf':
            for i in self.cols:
                df = df.withColumn(i, self.udf_impl(col(i)))
        elif self.op == 'inline':
            for i, func in self.cols.items():
                df = df.withColumn(i, eval(func))
        elif self.op == 'toInt':
            for i in self.cols:
                df = df.withColumn(i, col(i).cast(IntegerType()))

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

    def process(self, df, spark):
        if self.op == 'udf':
            for after, before in self.cols.items():
                df = df.withColumn(after, self.udf_impl(col(before)))
        elif self.op == 'inline':
            for after, before in self.cols.items():
                df = df.withColumn(after, eval(before))
        elif self.op == 'toInt':
            for after, before in self.cols.items():
                df = df.withColumn(after, col(before).cast(IntegerType()))
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

    def process(self, df, spark):
        return df.fillna(self.default, [i for i in self.cols])


class Categorify(Operation):
    '''
    Operation to categorify columns by distinct id

    Args:

        cols (list): columns which are be modified
    '''

    def __init__(self, cols, src_cols=None, hint='join'):
        self.op_name = "Categorify"
        self.cols = cols
        self.src_cols = src_cols
        self.hint = hint

    def process(self, df, spark):
        if self.hint == 'udf':
            return self.process_with_udf(df, spark)
        else:
            return self.process_with_join(df, spark)

    def process_with_udf(self, df, spark):
        if self.src_cols == None:
            for i in self.cols:
                sorted_data = df.select(i).groupBy(i).count().orderBy(
                    desc('count'), i).select(i).collect()
                dict_data = dict((id[i], idx) for (id, idx) in zip(
                    sorted_data, range(0, len(sorted_data))))
                udf_impl = self.get_mapping_udf(dict_data, spark)
                df = df.withColumn(i, udf_impl(col(i)))
        else:
            first = True
            dict_df = df
            for i in self.cols:
                if first:
                    dict_df = df.select(col(i).alias('dict_col'))
                    first = False
                else:
                    dict_df = dict_df.union(
                        df.select(col(i).alias('dict_col')))
            sorted_data = dict_df.groupBy('dict_col').count().orderBy(
                desc('count'), 'dict_col').select('dict_col').collect()
            dict_data = dict((id['dict_col'], idx) for (id, idx) in zip(
                sorted_data, range(0, len(sorted_data))))
            udf_impl = self.get_mapping_udf(dict_data, spark)
            for i in self.cols:
                df = df.withColumn(i, udf_impl(col(i)))
        return df

    def get_mapping_udf(self, broadcast_data, spark, default=None):
        # check if we support this type
        first_value = next(iter(broadcast_data.values()))
        if not isinstance(first_value, int) and not isinstance(first_value, str) and not isinstance(first_value, float):
            raise NotImplementedError

        broadcast_dict = spark.sparkContext.broadcast(
            broadcast_data)

        def get_mapped(x):
            broadcast_data = broadcast_dict.value
            if x in broadcast_data:
                return broadcast_data[x]
            else:
                return default

        # switch return type
        if isinstance(first_value, int):
            return udf(get_mapped, IntegerType())
        if isinstance(first_value, str):
            return udf(get_mapped, StringType())
        if isinstance(first_value, float):
            return udf(get_mapped, FloatType())

    def process_with_join(self, df, spark):
        if self.src_cols == None:
            for i in self.cols:
                dict_df = df.select(col(i)).groupBy(i).count()
                dict_df = dict_df.withColumn('dict_col_id', row_number().over(
                    Window.orderBy(desc('count')))).withColumn('dict_col_id', col('dict_col_id') - 1).select(i, 'dict_col_id')
                df = df.join(dict_df.hint('shuffle_hash'), i, 'left').withColumn(
                    i, col('dict_col_id')).drop('dict_col_id')
        else:
            first = True
            dict_df = df
            for i in self.cols:
                if first:
                    dict_df = df.select(col(i).alias('dict_col'))
                    first = False
                else:
                    dict_df = dict_df.union(
                        df.select(col(i).alias('dict_col')))
            dict_df = dict_df.groupBy('dict_col').count()
            dict_df = dict_df.withColumn('dict_col_id', row_number().over(
                Window.orderBy(desc('count')))).withColumn('dict_col_id', col('dict_col_id') - 1).select('dict_col', 'dict_col_id')
            for i in self.cols:
                df = df.join(dict_df.hint('shuffle_hash'), col(i) == col('dict_col'), 'left').withColumn(
                    i, col('dict_col_id')).drop('dict_col_id', 'dict_col')

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

    def process(self, df, spark):
        for i in self.cols:
            sorted_data_df = df.select(explode(split(col(i), self.sep))).groupBy('col').count().orderBy(
                desc('count'), 'col').select('col', 'count')
            if self.strategy == 0:
                sorted_data = sorted_data_df.collect()
                dict_data = dict((id['col'], idx) for (id, idx) in zip(
                    sorted_data, range(len(sorted_data))))
                udf_impl = self.get_mapping_udf_0(
                    dict_data, spark, self.sep, self.skipList)
                df = df.withColumn(i, udf_impl(col(i)))
            elif self.strategy == 1:
                sorted_data = sorted_data_df.collect()
                dict_data = dict((id['col'], [id['count'], idx]) for (id, idx) in zip(
                    sorted_data, range(len(sorted_data))))
                udf_impl = self.get_mapping_udf_1(
                    dict_data, spark, self.sep, self.skipList, self.freqRange)
                df = df.withColumn(i, udf_impl(col(i)))

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
            return udf(largest_freq_encode, IntegerType())
        if isinstance(first_value, str):
            return udf(largest_freq_encode, StringType())
        if isinstance(first_value, float):
            return udf(largest_freq_encode, FloatType())

    def get_mapping_udf_1(self, broadcast_data, spark, sep, skipList, freqRange, default=None):
        broadcast_dict = spark.sparkContext.broadcast(
            broadcast_data)
# original
#        def frequence_encode(x):
#            dict_data = broadcast_dict.value
#            li = []
#            lf = []
#            for v in x.split(sep):
#                if v not in skipList:
#                    f, i = dict_data[v]
#                    if f < freqRange[1] and f > freqRange[0]:
#                        li.append(str(i))
#                        lf.append(f)
#            # li will sort according to lf
#            return ' '.join(list((np.array(li)[np.argsort(lf)])))

# optimized since idx is orderBy count
        def frequence_encode(x):
            dict_data = broadcast_dict.value
            li = []
            for v in x.split(sep):
                if v not in skipList:
                    f, i = dict_data[v]
                    if f < freqRange[1] and f > freqRange[0]:
                        li.append(i)
            return sorted(li, reverse=True)
        return udf(frequence_encode, ArrayType(IntegerType()))


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

    def process(self, df, spark):
        if len(self.dict_data) == 0:
            for i in self.cols:
                df = df.withColumn(i, lit(None))
        else:
            udf_impl = self.get_mapping_udf(self.dict_data, spark)
            for i in self.cols:
                df = df.withColumn(i, udf_impl(col(i)))
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
            return udf(get_mapped, IntegerType())
        if isinstance(first_value, str):
            return udf(get_mapped, StringType())
        if isinstance(first_value, float):
            return udf(get_mapped, FloatType())


class GenerateDictionary(Operation):
    '''
    Operation to generate dictionary based on single or multi-columns by distinct id

    Args:

        cols (list): columns which are be modified
        doSplit (bool): If we need to split data
        sep (str): split seperator
    '''

    def __init__(self, cols, withCount=False, doSplit=False, sep='\t'):
        self.op_name = "GenerateDictionary"
        self.cols = cols
        self.doSplit = doSplit
        self.sep = sep
        self.withCount = withCount

    def process(self, df, spark):
        return df

    def collect(self, df):
        first = True
        dict_df = df
        # handle multiple columns issue
        for i in self.cols:
            singular_df = df.select(col(i).alias('dict_col'))
            if self.doSplit:
                singular_df = singular_df.select(
                    explode(split(col('dict_col'), self.sep))).withColumn('dict_col', col('col'))
            if first:
                first = False
                dict_df = singular_df
            else:
                dict_df = dict_df.union(singular_df)
        sorted_data = dict_df.groupBy('dict_col').count().orderBy(
            desc('count'), 'dict_col').select('dict_col', 'count').collect()
        if self.withCount:
            return dict((id['dict_col'], [idx, id['count']]) for (id, idx) in zip(
                sorted_data, range(0, len(sorted_data))))
        else:
            return dict((id['dict_col'], idx) for (id, idx) in zip(
                sorted_data, range(0, len(sorted_data))))


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
    def __init__(self, spark):
        self.ops = []
        self.spark = spark
        self.uuid = uuid.uuid1()
        self.tmp_id = 0

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
            df.write.format('parquet').mode(
                'overwrite').save("/tmp/" + "%s-%s-%d" % (df_name, self.uuid, tmp_id))
            return self.spark.read.parquet("/tmp/" + "%s-%s-%d" % (df_name, self.uuid, tmp_id))

    def transform(self, df):
        for op in self.ops:
            df = op.process(df, self.spark)
        return self.materialize(df)

    def collect(self, df):
        for op in self.ops[:-1]:
            df = op.process(df, self.spark)
        return self.ops[-1].collect(df)
