import pandas as pd

class PandasDataFrame(pd.DataFrame):
    def __init__(self, baseObject):
        self.__setattr__(self, '_wrapped_obj', baseObject)
        self.__setattr__(self, '_wrapped_obj', baseObject)

        #self._wrapped_obj = baseObject
        #self.sample_indices = None
        

    def may_sample(self):
        if self._wrapped_obj.shape[0] > 10000:
            return PandasDataFrame(self._wrapped_obj.sample(n=10000, random_state = 123))
        else:
            return self
    
    def __getitem__(self, key):
        if isinstance(key, list):
            return PandasDataFrame(self._wrapped_obj[key])
        else:
            return self._wrapped_obj[key]

class DataFrameAPI:
    @staticmethod
    def instiate(df):
        if isinstance(df, pd.DataFrame):
            from pandas_flavor import register_dataframe_method
            @register_dataframe_method
            def may_sample(df):
                if df.shape[0] > 100000:
                    return df.sample(n=100000, random_state = 123)
                else:
                    return df
            return df
        elif isinstance(df, str):
            # open the path depends on the size
            raise NotImplementedError("Instiate DataFrame based on path is not implemented yet.")
        raise NotImplementedError("Instiate DataFrame based on non-pandas dataframe is not implemented yet.")