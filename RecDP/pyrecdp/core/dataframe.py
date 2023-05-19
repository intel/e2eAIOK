import pandas as pd

class DataFrameAPI:
    @staticmethod
    def instiate(df):
        if isinstance(df, pd.DataFrame):
            from pandas_flavor import register_dataframe_method
            @register_dataframe_method
            def may_sample(df, nrows = 100000):
                if df.shape[0] > nrows:
                    return df.sample(n=nrows, random_state = 123)
                else:
                    return df
            return df
        elif isinstance(df, str):
            # open the path depends on the size
            raise NotImplementedError("Instiate DataFrame based on path is not implemented yet.")
        raise NotImplementedError("Instiate DataFrame based on non-pandas dataframe is not implemented yet.")