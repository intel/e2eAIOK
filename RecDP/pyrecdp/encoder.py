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

    def transform(self, train, valid, train_only=False):
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
    def __init__(self, proc, x_col, y_col, out_col, y_mean=None, out_dtype=None, folds=5, smooth=20, seed=42):
        super().__init__(proc)
        self.op_name = "TargetEncoder"
        self.x_col = x_col
        self.y_col = y_col
        self.out_col = out_col
        self.out_dtype = out_dtype
        self.y_mean = y_mean
        self.folds = folds
        self.seed = seed
        self.smooth = smooth

    def transform(self, train, valid, train_only=False):
        self.mean = float(self.y_mean)
        x_col = self.x_col
        y_col = self.y_col
        out_col = self.out_col
        if self.y_mean is None:
            self.y_mean = np.array(train.groupBy().mean(y_col).collect())[0][0]
        mean = self.y_mean
        smooth = self.smooth
        out_dtype = self.out_dtype

        cols = ['fold', x_col] if isinstance(x_col, str) else ['fold']+x_col
        agg_per_fold = train.groupBy(cols).agg(
            f.count(y_col).alias('count_y'), f.sum(y_col).alias('sum_y'))
        agg_per_fold = agg_per_fold.withColumn('count_y_all', f.sum('count_y').over(Window.partitionBy(
            x_col))).withColumn('sum_y_all', f.sum('sum_y').over(Window.partitionBy(x_col)))

        agg_per_fold.cache()
        if train_only == False:
            agg_all = agg_per_fold

        # prepare for agg_per_fold
        agg_per_fold = agg_per_fold.withColumn(
            'count_y_all', f.col('count_y_all')-f.col('count_y'))
        agg_per_fold = agg_per_fold.withColumn(
            'sum_y_all', f.col('sum_y_all')-f.col('sum_y'))
        agg_per_fold = agg_per_fold.withColumn(
            out_col,
            (f.col('sum_y_all')+f.lit(mean)*f.lit(smooth))/(f.col('count_y_all')+f.lit(smooth)))
        if out_dtype is not None:
            agg_per_fold = agg_per_fold.withColumn(
                out_col, f.col(out_col).cast(out_dtype))
        agg_per_fold = agg_per_fold.drop(
            'count_y_all', 'count_y', 'sum_y_all', 'sum_y')

        if train_only == False:
            # prepare for agg_all
            agg_all = agg_all.drop('fold').drop(
                'count_y').drop('sum_y').distinct()
            agg_all = agg_all.withColumn(
                out_col,
                (f.col('sum_y_all')+f.lit(mean)*f.lit(smooth))/(f.col('count_y_all')+f.lit(smooth)))
            if out_dtype is not None:
                agg_all = agg_all.withColumn(
                    out_col, f.col(out_col).cast(out_dtype))
            to_select = [x_col, out_col] if isinstance(
                x_col, str) else x_col + [out_col]
            agg_all = agg_all.select(*to_select)

        train_out = (cols, self.materialize(
            agg_per_fold, "train/%s" % out_col), mean)
        if train_only == False:
            valid_out = (x_col, self.materialize(
                agg_all, "valid/%s" % out_col), mean)
        else:
            valid_out = ()
        return (train_out, valid_out)


class CountEncoder(Encoder):
    def __init__(self, proc, x_col, out_col, seed=42):
        super().__init__(proc)
        self.op_name = "CountEncoder"
        self.x_col = x_col
        self.out_col = out_col
        self.seed = seed

    def transform(self, train, valid, train_only=False):
        x_col = self.x_col
        out_col = self.out_col

        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all = train.groupby(cols).count(
        ).withColumnRenamed('count', out_col)
        agg_test = valid.groupby(cols).count().withColumnRenamed(
            'count', out_col+'_valid')

        agg_test_size = agg_test.count()
        if agg_test_size > 30000000:
            agg_all = agg_all.join(agg_test.hint(
                'shuffle_hash'), cols, how='left')
        else:
            agg_all = agg_all.join(f.broadcast(agg_test), cols, how='left')
        agg_all = agg_all.fillna(0, out_col+'_valid')
        agg_all = agg_all.withColumn(
            out_col, f.col(out_col)+f.col(out_col+'_valid'))
        agg_all = agg_all.drop(out_col+'_valid')
        agg_all.cache()

        train_out = (cols, self.materialize(
            agg_all, "train/%s" % out_col), 0)
        if train_only == False:
            valid_out = (cols, self.materialize(
                agg_all, "valid/%s" % out_col), 0)
        else:
            valid_out = ()
        return (train_out, valid_out)


class FrequencyEncoder(Encoder):
    def __init__(self, proc, x_col, out_col, seed=42):
        super().__init__(proc)
        self.op_name = "FrequencyEncoder"
        self.x_col = x_col
        self.out_col = out_col
        self.seed = seed

    def transform(self, train, valid, train_only=False):
        x_col = self.x_col
        out_col = self.out_col
        length_train = train.count()
        length_valid = valid.count()

        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all_train = train.groupby(cols).count(
        ).withColumnRenamed('count', out_col)
        agg_all_train = agg_all_train.withColumn(out_col, f.col(
            out_col).cast(spk_type.IntegerType()))
        agg_all_train = agg_all_train.withColumn(
            out_col, f.col(out_col)*1.0/length_train)
        agg_all_train = agg_all_train.withColumn(out_col, f.col(
            out_col).cast(spk_type.FloatType()))

        agg_all_valid = valid.groupby(cols).count(
        ).withColumnRenamed('count', out_col)
        agg_all_valid = agg_all_valid.withColumn(out_col, f.col(
            out_col).cast(spk_type.IntegerType()))
        agg_all_valid = agg_all_valid.withColumn(
            out_col, f.col(out_col)*1.0/length_valid)
        agg_all_valid = agg_all_valid.withColumn(out_col, f.col(
            out_col).cast(spk_type.FloatType()))

        train_out = (cols, self.materialize(
            agg_all_train, "train/%s" % out_col), 0)
        if train_only == False:
            valid_out = (cols, self.materialize(
                agg_all_valid, "valid/%s" % out_col), 0)
        else:
            valid_out = ()
        return (train_out, valid_out)
