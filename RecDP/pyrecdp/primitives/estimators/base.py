from pyrecdp.primitives.operations.base import BaseOperation
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from pyrecdp.core.utils import callable_string_fix

class BaseEstimator(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config

    def get_func_train(self):
        raise NotImplementedError("BaseEstimator is an abstract class")

    def get_func_predict(self):
        raise NotImplementedError("BaseEstimator is an abstract class")

    def get_function_pd(self):
        return self.get_func_train()
        
    def get_evaluate_func(self, metric):
        if metric == 'rmse':
            def rmse_func(ground_truth, pred):
                return np.sqrt(mean_squared_error(ground_truth, pred))
            return rmse_func
        if metric == 'auc':
            def auc_func(ground_truth, pred):
                return roc_auc_score(ground_truth, pred)
            return auc_func

    def get_splitter_func(self, splitter_func_str):
        if splitter_func_str is not None:
            if callable(splitter_func_str):
                return splitter_func_str
            elif isinstance(splitter_func_str, str):
                splitter_func_str = callable_string_fix(splitter_func_str)
                import re
                func_name = ''.join(re.findall('def (\S+)\(', splitter_func_str.split('\n')[0]))
                print(f"Detect splitter provided, function name is {func_name}")
                print(splitter_func_str)
                exec(splitter_func_str, globals())
                return eval(func_name)
            else:
                raise NotImplementedError(f"Unable to inteprete {splitter_func_str}as train_test_splitter")
        else:
            def splitter_func(df):
                total_len = df.shape[0]
                test_len = int(total_len * 0.1)
                test_sample = df.iloc[-test_len:]
                train_sample = df.drop(test_sample.index)
                return train_sample, test_sample
            return splitter_func