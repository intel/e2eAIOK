import pandas as pd

class DataFrameAPI:
    @staticmethod
    def instiate(df):
        if isinstance(df, str):
            # open the path depends on the size
            if df.endswith('.csv'):
                df = pd.read_csv(df)
            elif df.endswith('.parquet'):
                df = pd.read_parquet(df)
            else:
                raise NotImplementedError("Instiate DataFrame based on path only support csv and parquet")
        if not isinstance(df, pd.DataFrame):
            raise NotImplementedError("Instiate DataFrame based on non-pandas dataframe is not implemented yet.")
        
        from pandas_flavor import register_dataframe_method
        @register_dataframe_method
        def may_sample(df, nrows = 100000):
            if df.shape[0] > nrows:
                return df.sample(n=nrows, random_state = 123)
            else:
                return df
        return df