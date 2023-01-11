import inspect

class BaseFeatureGenerator:
    def __init__(self):
        pass
    
    def is_useful(self, pa_schema):
        return True
    
    def fit_prepare(self, pa_schema):
        return

    def get_function_pd(self):
        def base_feature_generator(df):
            return df
        return base_feature_generator
    
    def get_function_spark(self, rdp):
        actual_func = self.get_function_pd()
        def transform(iter, *args):
            for x in iter:
                yield actual_func(x[0], *args), x[1]
        def base_spark_feature_generator(df):
            return df.mapPartitions(transform)
        return base_spark_feature_generator

    def fit_transform(self, df):
        return self.get_function_pd()(df)
    
    def dump_codes(self):
        return inspect.getsource(self.get_function_pd())