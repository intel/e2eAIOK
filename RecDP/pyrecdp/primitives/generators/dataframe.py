from .base import BaseFeatureGenerator as super_class
import numpy as np
import time

class DataframeConvertFeatureGenerator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return True
    
    def fit_prepare(self, pa_schema):
        return pa_schema

    def get_function_pd(self):
        def convert_df(df):
            return df
        return convert_df
    
    def get_function_spark(self, rdp):
        def convert_spark(df):
            import pandas as pd            
            import math
            if not isinstance(df, pd.DataFrame):
               return df
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
            rdds = rdp.spark.sparkContext.parallelize(dfs).zipWithIndex()
            end_time = time.time()
            print(f"DataframeConvert partition pandas dataframe to spark RDD took {(end_time - start_time):.3f} secs")
            return rdds
            
        return convert_spark
    
class DataframeTransformFeatureGenerator:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, pa_schema):
        return True
    
    def fit_prepare(self, pa_schema):
        return pa_schema

    def get_function_pd(self):
        def transform_df(df):
            return df
        return transform_df
    
    def get_function_spark(self, rdp):
        import pandas as pd
        def save(iter, id):
            pdfs = []
            for x in iter:
                pdfs.append(x)
            partition_pdf = pd.concat(pdfs)
            partition_pdf.to_parquet(f"DataframeConvertFeatureGenerator.{id}")
            
        def transform_df(df):
            # df = df.mapPartitions(lambda x: save(x, df.id))
            # transformed_line = df.count()
            # print(f"transformed_line is {transformed_line}")
            if isinstance(df, pd.DataFrame):
               return df
            start_time = time.time()
            pdfs = df.collect()
            end_time = time.time()
            id_pdfs = [tpl[1] for tpl in pdfs]
            nr_pdfs = [tpl[0].shape[0] for tpl in pdfs]
            ordered_pdfs = [tpl[0] for tpl in sorted(pdfs, key = lambda x: x[1])]
            print(f"DataframeTransform took {(end_time - start_time):.3f} secs, processed {sum(nr_pdfs)} rows with num_partitions as {len(id_pdfs)}")
            start_time = time.time()
            combined = pd.concat(ordered_pdfs)
            end_time = time.time()
            print(f"DataframeTransform combine to one pandas dataframe took {(end_time - start_time):.3f} secs")
            return combined
            
        return transform_df