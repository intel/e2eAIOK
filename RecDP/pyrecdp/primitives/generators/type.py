from autogluon.features.generators.identity import IdentityFeatureGenerator as super_class
import pandas as pd
from pandas.api import types as pdt
import numpy as np
from collections import OrderedDict

class TypeInferFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        self.obj = None
        super().__init__(**kwargs)
        self.lazy = False
    
    def _fit_transform(self, X, **kwargs):
        new_df_list = OrderedDict()
        for feature_name in X.columns:
            new_df_list[feature_name] = self._infer_type(X[feature_name], self.lazy)
        
        return pd.DataFrame(new_df_list)
    
    def _infer_type(self, s, lazy = False):
        def try_category(s):
            if pdt.is_categorical_dtype(s) and not pdt.is_bool_dtype(s):
                return s
            n_unique = s.nunique()
            threshold = (n_unique / 5) if (n_unique / 5) < 1000 else 1000
            if 1 <= n_unique <= threshold:
                s = pd.Categorical(s)
            return s
            
        def try_datetime(s):
            if pdt.is_datetime64_any_dtype(s):
                return s
            if not pdt.is_string_dtype(s):
                return s
            if s.isnull().all():
                return s
            try:
                if len(s) > 500:
                    # Sample to speed-up type inference
                    result = s.sample(n=500, random_state=0)
                result = pd.to_datetime(result, errors='coerce')
                if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
                    return s
                # Now We can think this s can be datetime
                s = pd.to_datetime(s, errors='coerce')
            except:
                return s
            return s

        def try_text(s):
            if not pdt.is_string_dtype(s):
                return s
            if len(s) > 500:
                # Sample to speed-up type inference
                result = s.sample(n=500, random_state=0)
            try:
                avg_words = pd.Series(result).str.split().str.len().mean()
                if avg_words > 1:
                    # possible to use nlp method
                    s = pd.Series(result).str.split()
            except:
                return s
            return s
        
        s = try_category(s)
        s = try_datetime(s)
        s = try_text(s)
        return s

    def is_useful(self, df):
        return False

    def update_feature_statistics(self, X, state_dict):
        length = X.shape[0]
        for feature_name in X.columns:
            feature = X[feature_name]
            unique_list = feature.unique()
            desc_info = dict((k, v) for k, v in feature.describe().to_dict().items() if k not in ['count'])
            stat = {'type': feature.dtype.name, 'unique': {"u": len(unique_list), "m": length}, 'quantile':desc_info}
            if feature_name not in state_dict:
                state_dict[feature_name] = stat
        return state_dict
