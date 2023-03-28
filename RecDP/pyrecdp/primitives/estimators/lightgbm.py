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
        metrics = self.config['metrics']
        objective = self.config['objective']
        model_file = self.config['model_file']
        train_test_splitter = self.get_splitter_func(self.config['train_test_splitter'])
        def train(df):
            train_sample, test_sample = train_test_splitter(df)
            
            x_train = train_sample.drop(columns=[label])
            y_train = train_sample[label].values

            x_val = test_sample.drop(columns=[label])
            y_val = test_sample[label].values

            lgbm_train = lgbm.Dataset(x_train, y_train, silent=False)
            lgbm_val = lgbm.Dataset(x_val, y_val, silent=False)
            
            params = {
                'boosting_type':'gbdt',
                'objective': objective,
                'nthread': 4,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'max_depth': -1,
                'subsample': 0.8,
                'bagging_fraction' : 1,
                'max_bin' : 5000 ,
                'bagging_freq': 20,
                'colsample_bytree': 0.6,
                'metric': metrics,
                'min_split_gain': 0.5,
                'min_child_weight': 1,
                'min_child_samples': 10,
                'scale_pos_weight':1,
                'zero_as_missing': True,
                'seed':0,
                'num_rounds':2000,
                'num_boost_round': 2000,
                'early_stopping_rounds': 50
            }
            model = lgbm.train(params=params, train_set=lgbm_train, valid_sets=lgbm_val, verbose_eval=100)
            model.save_model(model_file, num_iteration=model.best_iteration) 
            return model
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