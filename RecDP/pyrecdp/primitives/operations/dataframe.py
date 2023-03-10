import numpy as np
import time
import pandas as pd            
import math
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import RDD as SparkRDD

class SparkDataFrameToRDDConverter:
    @staticmethod
    def get_function(rdp):
        print("append SparkDataFrameToRDDConverter")
        def convert(df):
            def transform(rdd_):
                rows = (row_.asDict() for row_ in rdd_)
                pdf = pd.DataFrame(rows)
                return [pdf]
            return df.rdd.mapPartitions(transform)            
        return convert
   
class RDDToSparkDataFrameConverter:
    @staticmethod
    def get_function(rdp):
        print("append RDDToSparkDataFrameConverter")
        def convert(rdd):
            df = rdd.flatMap(lambda pdf: pdf.to_dict('records')).toDF()
            return df
            
        return convert

class DataFrameToRDDConverter:
    @staticmethod
    def get_function(rdp):
        print("append DataFrameToRDDConverter")
        def convert(df):
            # convert pandas to spark
            N_PARTITIONS = 200
            num_rows = df.shape[0]
            start_time = time.time()
            permuted_indices = np.array(list(range(num_rows)))
            dfs = []
            steps = math.ceil(num_rows / N_PARTITIONS)
            end = 0
            start = 0
            while end < num_rows:
                end = start + steps if start + steps < num_rows else num_rows
                rows = permuted_indices[start : end]
                start += steps
                dfs.append(df.iloc[rows])
            rdds = rdp.spark.sparkContext.parallelize(dfs)
            end_time = time.time()
            print(f"DataframeConvert partition pandas dataframe to spark RDD took {(end_time - start_time):.3f} secs")
            return rdds
            
        return convert

class RDDToDataFrameConverter:
    @staticmethod
    def get_function(rdp):
        print("append RDDToDataFrameConverter")
        def convert(rdd):
            start_time = time.time()
            pdfs = rdd.collect()
            end_time = time.time()
            # id_pdfs = [tpl[1] for tpl in pdfs]
            # nr_pdfs = [tpl[0].shape[0] for tpl in pdfs]
            # ordered_pdfs = [tpl[0] for tpl in sorted(pdfs, key = lambda x: x[1])]
            nr_pdfs = [tpl.shape[0] for tpl in pdfs]
            ordered_pdfs = pdfs
            print(f"DataframeTransform took {(end_time - start_time):.3f} secs, processed {sum(nr_pdfs)} rows with num_partitions as {len(pdfs)}")
            start_time = time.time()
            combined = pd.concat(ordered_pdfs)
            end_time = time.time()
            print(f"DataframeTransform combine to one pandas dataframe took {(end_time - start_time):.3f} secs")
            return combined
            
        return convert

class DataFrameToSparkDataFrameConverter:
    def get_function(rdp):
        print("append DataFrameToSparkDataFrameConverter")
        def convert(pdf):
            return rdp.spark.createDataFrame(pdf) 
        return convert
    
class SparkDataFrameToDataFrameConverter:
    @staticmethod
    def get_function(rdp):
        print("append SparkDataFrameToDataFrameConverter")
        def convert(df):
            return df.to_pandas()          
        return convert