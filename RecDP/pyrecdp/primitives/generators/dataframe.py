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
            #if not isinstance(df, pd.DataFrame):
            #    return df
            # convert pandas to spark
            N_PARTITIONS = 200
            num_rows = df.shape[0]
            start_time = time.time()
            permuted_indices = np.random.permutation(num_rows)
            dfs = []
            for i in range(N_PARTITIONS):
                dfs.append(df.iloc[permuted_indices[i::N_PARTITIONS]])
            rdds = rdp.spark.sparkContext.parallelize(dfs)
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
            start_time = time.time()
            pdfs = df.collect()
            end_time = time.time()
            nr_pdfs = [pdf.shape[0] for pdf in pdfs]
            print(f"DataframeTransform took {(end_time - start_time):.3f} secs, processed {sum(nr_pdfs)} rows")
            start_time = time.time()
            combined = pd.concat(pdfs)
            end_time = time.time()
            print(f"DataframeTransform combine to one pandas dataframe took {(end_time - start_time):.3f} secs")
            return combined
            
        return transform_df