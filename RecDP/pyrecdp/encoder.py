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
import os
import sys
from timeit import default_timer as timer
import logging
import shutil


class Encoder:
    def __init__(self, proc):
        self.op_name = "Encoder"
        self.uuid = proc.uuid
        self.tmp_id = proc.tmp_id
        self.path_prefix = proc.path_prefix
        self.current_path = proc.current_path
        self.dicts_path = proc.dicts_path
        self.spark = proc.spark
        self.tmp_materialzed_list = []

    def transform(self, train, valid, train_only=True):
        raise NotImplementedError("This is base Encoder class")

    def materialize(self, df, df_name="materialized_tmp"):
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


class TargetEncoder(Encoder):
    def __init__(self, proc, x_col_list, y_col_list, out_col_list, out_name, out_dtype=None, y_mean_list=None, smooth=20, seed=42,threshold=0):
        super().__init__(proc)
        self.op_name = "TargetEncoder"
        self.x_col_list = x_col_list
        self.y_col_list = y_col_list
        self.out_col_list = out_col_list
        self.out_dtype = out_dtype
        self.out_name = out_name
        self.y_mean_list = y_mean_list
        self.seed = seed
        self.smooth = smooth
        self.expected_list_size = len(y_col_list)
        self.threshold = threshold
        if len(self.out_col_list) < self.expected_list_size:
            raise ValueError("TargetEncoder __init__, input out_col_list should be same size as y_col_list")      
        if y_mean_list != None and len(self.y_mean_list) < self.expected_list_size:
            raise ValueError("TargetEncoder __init__, input y_mean_list should be same size as y_col_list")        

    def transform(self, df):
        x_col = self.x_col_list
        cols = ['fold', x_col] if isinstance(x_col, str) else ['fold'] + x_col
        agg_per_fold = df.groupBy(cols)
        agg_all = df.groupBy(x_col)

        per_fold_list = []
        all_list = []

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            per_fold_list.append(f.count(y_col).alias(f'count_{y_col}'))
            per_fold_list.append(f.sum(y_col).alias(f'sum_{y_col}'))
            all_list.append(f.count(y_col).alias(f'count_all_{y_col}'))
            all_list.append(f.sum(y_col).alias(f'sum_all_{y_col}'))

        agg_per_fold = agg_per_fold.agg(*per_fold_list)
        agg_all = agg_all.agg(*all_list)
        agg_per_fold = agg_per_fold.join(agg_all, x_col, 'left')

        if self.threshold > 0:
            agg_all = agg_all.where(f.col(f'count_all_{self.y_col_list[0]}')>self.threshold)

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = self.out_col_list[i]
            out_dtype = self.out_dtype
            y_mean = self.y_mean_list[i] if self.y_mean_list != None else None
            
            if y_mean is None:
                y_mean = np.array(df.groupBy().mean(y_col).collect())[0][0]
            mean = float(y_mean)
            smooth = self.smooth

            # print(agg_per_fold.dtypes)

            # prepare for agg_per_fold
            agg_per_fold = agg_per_fold.withColumn(
                f'count_all_{y_col}', f.col(f'count_all_{y_col}')-f.col(f'count_{y_col}'))
            agg_per_fold = agg_per_fold.withColumn(
                f'sum_all_{y_col}', f.col(f'sum_all_{y_col}')-f.col(f'sum_{y_col}'))
            agg_per_fold = agg_per_fold.withColumn(
                out_col,
                (f.col(f'sum_all_{y_col}') + f.lit(mean) * f.lit(smooth))/(f.col(f'count_all_{y_col}') + f.lit(smooth)))
            agg_all = agg_all.withColumn(
                out_col,
                (f.col(f'sum_all_{y_col}') + f.lit(mean) * f.lit(smooth))/(f.col(f'count_all_{y_col}') + f.lit(smooth)))
            if out_dtype is not None:
                agg_per_fold = agg_per_fold.withColumn(
                    out_col, f.col(out_col).cast(out_dtype))
                agg_all = agg_all.withColumn(
                    out_col, f.col(out_col).cast(out_dtype))
            agg_per_fold = agg_per_fold.drop(
                f'count_all_{y_col}', f'count_{y_col}', f'sum_all_{y_col}', f'sum_{y_col}')
            agg_all = agg_all.drop(f'count_all_{y_col}', f'sum_all_{y_col}')
        return (self.materialize(agg_per_fold, "%s/train/%s" % (self.dicts_path, self.out_name)),
                self.materialize(agg_all, "%s/test/%s" % (self.dicts_path, self.out_name)))


class CountEncoder(Encoder):
    def __init__(self, proc, x_col_list, y_col_list, out_col_list, out_name, train_generate=True):
        super().__init__(proc)
        self.op_name = "CountEncoder"
        self.x_col_list = x_col_list
        self.y_col_list = y_col_list
        self.out_col_list = out_col_list
        self.out_name = out_name        
        self.expected_list_size = len(y_col_list)
        self.train_generate = train_generate
        if len(self.out_col_list) < self.expected_list_size:
            raise ValueError("CountEncoder __init__, input out_col_list should be same size as y_col_list")

    def transform(self, df):
        x_col = self.x_col_list
        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all = df.groupby(cols)

        all_list = []

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = self.out_col_list[i]
            all_list.append(f.count(y_col).alias(f'{out_col}'))

        agg_all = agg_all.agg(*all_list)
        for i in range(0, self.expected_list_size):
            out_col = self.out_col_list[i]
            agg_all = agg_all.withColumn(out_col, f.col(out_col).cast(spk_type.IntegerType()))

        if self.train_generate:
            return (self.materialize(agg_all, "%s/train/%s" % (self.dicts_path, self.out_name)),
                    self.materialize(agg_all, "%s/test/%s" % (self.dicts_path, self.out_name)))
        else:
            return (self.materialize(agg_all, "%s/test/%s" % (self.dicts_path, self.out_name)))



class FrequencyEncoder(Encoder):
    def __init__(self, proc, x_col_list, y_col_list, out_col_list, out_name):
        super().__init__(proc)
        self.op_name = "FrequencyEncoder"
        self.x_col_list = x_col_list
        self.y_col_list = y_col_list
        self.out_col_list = out_col_list
        self.out_name = out_name        
        self.expected_list_size = len(y_col_list)
        if len(self.out_col_list) < self.expected_list_size:
            raise ValueError("FrequencyEncoder __init__, input out_col_list should be same size as y_col_list")

    def transform(self, df):
        length_df = df.count()
        x_col = self.x_col_list
        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all = df.groupby(cols)

        all_list = []

        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = self.out_col_list[i]
            all_list.append(f.count(y_col).alias(out_col))
        agg_all = agg_all.agg(*all_list)

        for i in range(0, self.expected_list_size):
            out_col = self.out_col_list[i]
            agg_all = agg_all.withColumn(out_col, (f.col(out_col)*1.0/length_df).cast(spk_type.FloatType()))
        return (self.materialize(agg_all, "%s/train/%s" % (self.dicts_path, self.out_name)),
                None)