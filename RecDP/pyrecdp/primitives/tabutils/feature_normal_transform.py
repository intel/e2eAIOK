"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import numpy as np
from scipy.stats import yeojohnson
from scipy.stats import kstest
from sklearn.preprocessing import QuantileTransformer
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def diagnostic_plots(df_series, col_name, plot = False):
    if plot:
        try:
            fig = plt.figure(figsize=(15,4))
            ax = fig.add_subplot(121)
            df_series.hist(bins=30)
            ax = fig.add_subplot(122)
            res = stats.probplot(df_series, plot = plt)
        except:
            res = stats.probplot(df_series)
    else:
        res = stats.probplot(df_series)
    if plot:
        plt.xlabel(col_name)
        plt.show()
    return abs(np.log(res[1][0]))

def log_transform(x):
    if (x.min() <= 0):
        return np.log1p(x - x.min() + 1e-16)
    else:
        return np.log1p(x)

def sqrt_transform(x):
    if (x.min() <= 0):
        return (x - x.min() + 1e-16)**(1/2)
    else:
        return x**(1/2)
    
def exp_transform(x):
    return np.exp(x)

def reciprocal_transform(x):
    return 1/(x + 1e-9)

def yeojohnson_transform(x):
    y, _ = yeojohnson(x)
    return pd.Series(y)

def quantile_transform(x):
    quantile_trans = QuantileTransformer(output_distribution='normal')
    data_transformed = quantile_trans.fit_transform(x.to_numpy().reshape(-1, 1))
    return pd.Series(data_transformed[:, 0])


transform_dict = {
    'log_transform': log_transform,
    'exp_transform': exp_transform,
    'sqrt_transform': sqrt_transform,
    'reciprocal_transform': reciprocal_transform,
    'yeojohnson_transform': yeojohnson_transform,
    'quantile_transform': quantile_transform,
}

def normal_transformation(df, col_name, plot = False):
    print('Finding transform for : ', col_name)
    
    if len(df[col_name].dropna()) == 0:
        print("All Nan values")
        return None

    if kstest(df[col_name].dropna(), 'norm').pvalue >= 0.05:
        print("Already Normal")
        return None
    
    if plot:
        print(f'{col_name} Raw value plot')
        print("Absolute log of slope = ", diagnostic_plots(df[col_name].dropna(), col_name, plot = plot), "\n\n")


    res = dict()

    try:
        log_target = log_transform(df[col_name])
        if plot:
            print(f'{col_name} logarithmic plot')
        res['log_transform'] = diagnostic_plots(log_target.dropna(), col_name, plot)
        if plot:
            print("Absolute log of slope = ", res['log_transform'], "\n\n")
    except:
        pass

    try:
        exp_target = exp_transform(df[col_name])
        # print(f"Exp transform kstest pvalue {kstest(exp_target.dropna(), 'norm').pvalue}")
        # print(kstest(exp_target.dropna(), 'norm'))
        if plot:
            print(f'{col_name} Exponential plot')
        res['exp_transform'] = diagnostic_plots(exp_target.dropna().mask(exp_target == np.inf, 1e12, inplace = False), col_name, plot)
        if plot:
            print("Absolute log of slope = ", res['exp_transform'], "\n\n")
    except:
        pass

    # print(f"Sqrt transform kstest pvalue {kstest(sqrt_target.dropna(), 'norm').pvalue}")
    # print(kstest(sqrt_target.dropna(), 'norm'))
    try:
        sqrt_target = sqrt_transform(df[col_name])
        if plot:
            print(f'{col_name} Square Root plot')
        res['sqrt_transform'] = diagnostic_plots(sqrt_target.dropna(), col_name, plot)
        if plot:
            print("Absolute log of slope = ", res['sqrt_transform'], "\n\n")
    except:
        pass

    try:
        reciprocal_target = reciprocal_transform(df[col_name])
        # print(f"Reciprocal transform kstest pvalue {kstest(reciprocal_target.dropna(), 'norm').pvalue}")
        # print(kstest(reciprocal_target.dropna(), 'norm'))
        if plot:
            print(f'{col_name} Reciprocal plot')
        res['reciprocal_transform'] = diagnostic_plots(reciprocal_target.dropna(), col_name, plot)
        # bcx_target, lam = boxcox(df[col_name])
        if plot:
            print("Absolute log of slope = ", res['reciprocal_transform'], "\n\n")
    except:
        pass

    try:
        yf_target = yeojohnson_transform(df[col_name])
        # print(f"Yeo Johnson transform kstest pvalue {kstest(yf_target, 'norm').pvalue}")
        if plot:
            print(f'{col_name} Yeo Johnson plot')
        res['yeojohnson_transform'] = diagnostic_plots(yf_target.dropna(), col_name, plot)
        if plot:
            print("Absolute log of slope = ", res['yeojohnson_transform'], "\n\n")
    except:
        pass
    # quantile_trans = QuantileTransformer(output_distribution='normal')
    # data_transformed = quantile_trans.fit_transform(df[col_name].to_numpy().reshape(-1, 1))
    # # print(f"Quantile transform kstest pvalue {kstest(data_transformed.dropna(), 'norm').pvalue}")
    # print(f'{col_name} Quantile Transformer plot')
    # res['quantile_transform'] = diagnostic_plots(pd.Series(data_transformed[:, 0]).dropna(), col_name)

    ret_key = min(res, key = res.get)
    print(f"The best transform is: {ret_key}, with absolute log of slope {res[ret_key]}")
    return ret_key

def find_best_normal_transform(df, included_cols = None, excluded_cols = None, plot = False, save_dict = False):
    best_transform_dict = dict()
    for col in df.columns:
        if df[col].dtype == 'object':
            continue
        if excluded_cols is not None and col in excluded_cols:
            continue
        if included_cols is not None and col not in included_cols:
            continue
        v = normal_transformation(df, col, plot)
        if v is not None:
            best_transform_dict[col] = v
    if type(save_dict) == str:
        with open(save_dict, 'wb') as handle:
            pickle.dump(best_transform_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    elif save_dict:
        with open('best_transform_dict.pickle', 'wb') as handle:
            pickle.dump(best_transform_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        pass
    return best_transform_dict

def apply_best_normal_transform(df, inplace = True, best_transform_dict = None, included_cols = None, excluded_cols = None, plot = False):
    if type(best_transform_dict) == str:
        with open(best_transform_dict, 'rb') as handle:
            best_transform_dict = pickle.load(handle)
    if best_transform_dict is None:
        best_transform_dict = find_best_normal_transform(df, plot = plot, included_cols = included_cols, excluded_cols = excluded_cols)
    if inplace:
        for col in df.columns:
            if df[col].dtype == 'object' or col not in best_transform_dict:
                continue
            if excluded_cols is not None and col in excluded_cols:
                continue
            if included_cols is not None and col not in included_cols:
                continue
            df[col] = transform_dict[best_transform_dict[col]](df[col])
        return df
    else:
        new_df = pd.DataFrame(columns = df.columns)
        for col in df.columns:
            if df[col].dtype == 'object' or col not in best_transform_dict:
                new_df[col] = df[col]
                continue
            if excluded_cols is not None and col in excluded_cols:
                new_df[col] = df[col]
                continue
            if included_cols is not None and col not in included_cols:
                new_df[col] = df[col]
                continue
            new_df['transformed_' + col] = transform_dict[best_transform_dict[col]](df[col])
        return new_df
    
# import pandas as pd
# df = pd.read_csv('../data/Dataset.csv')
# print(normal_transformation(df, 'MAP'))