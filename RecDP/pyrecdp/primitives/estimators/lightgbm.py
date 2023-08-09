from .base import BaseEstimator
import pandas as pd
import lightgbm as lgbm

class LightGBM(BaseEstimator):
    def __init__(self, op_base):
        super().__init__(op_base)
        self.support_spark_dataframe = False
        self.support_spark_rdd = False

    def get_func_train(self):
        label = self.config['label']
        objective = self.config['objective']
        if 'metrics' in self.config:
            metrics = self.config['metrics']
        else:
            metrics = None
        train_test_splitter = self.get_splitter_func(self.config['train_test_splitter'])
        def train(df):
            train_sample, test_sample = train_test_splitter(df)
            x_train = train_sample.drop(columns=[label])
            y_train = train_sample[label].values
            lgbm_train = lgbm.Dataset(x_train, y_train)
            x_val = test_sample.drop(columns=[label])
            y_val = test_sample[label].values
            lgbm_val = lgbm.Dataset(x_val, y_val)
            
            params = {
                'boosting_type':'gbdt',
                'objective': objective,
                'num_leaves': 31,
                'learning_rate': 0.01,
                'seed':0,
                'verbose': 1
            }
            if not isinstance(metrics, type(None)):
                params['metrics'] = metrics
            model = lgbm.train(params=params, train_set=lgbm_train, valid_sets=lgbm_val, verbose_eval=100)
            #model.save_model(model_file, num_iteration=model.best_iteration)

            f_imp = model.feature_importance(importance_type='split').tolist()
            f_names = model.feature_name()
            ret = dict((fn, fi) for fn, fi in zip(f_names, f_imp))
            ret = sorted(ret.items(), key = lambda x:x[1], reverse = True)
            return ret
        return train
        
    def get_func_predict(self):
        label = self.config['label']
        metrics = self.config['metrics']
        model_file = self.config['model_file']
        evalute_func = self.get_evaluate_func(metrics)
        model = lgbm.Booster(model_file = model_file)
        def predict(test_df):
            contain_label = False
            if label in test_df.columns:
                x_val = test_df.drop(columns=[label])
                contain_label = True
            pred = model.predict(x_val)
            if contain_label:
                y_val = test_df[label].values
                score = evalute_func(y_val, pred)
                print(f"This test_data contains label, {metrics} is {score}")

            test_df['predict_value'] = pd.Series(pred)
            return test_df
    
        return predict