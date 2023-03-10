from pyrecdp.primitives.operations.base import BaseOperation
import numpy as np
from sklearn.metrics import mean_squared_error

class BaseEstimator(BaseOperation):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.config = self.op.config
        if 'method' in self.config:
            self.method = self.config['method']
        else:
            self.method = 'predict'

    def get_func_train(self):
        raise NotImplementedError("BaseEstimator is an abstract class")

    def get_func_predict(self):
        raise NotImplementedError("BaseEstimator is an abstract class")

    def get_function_pd(self):
        if self.method == 'train':
            return self.get_func_train()
        else:
            return self.get_func_predict()
        
    def get_evaluate_func(self, metric):
        if metric == 'rmse':
            def rmse_func(ground_truth, pred):
                return np.sqrt(mean_squared_error(ground_truth, pred))
            return rmse_func