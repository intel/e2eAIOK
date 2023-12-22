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

import pandas as pd
import numpy as np
import numba
numba.set_num_threads(16)

def masking(df, columns, exclude_columns=[]):
    """
    Creating a mask features where feature present
    would be zero and nan feature would be zero
    """
    columns = [col for col in columns if col not in exclude_columns]
    return (~df[columns].isna()).astype("int").add_prefix("mask_")

def linear_impute(df, id_col_name, limit_direction = 'both'):

    '''
    df: Entire Dataset Dataframe
    id_col_name: Column name representing patient ids
    Expected output
    linear_impute(df, 'id')
    id    a     b    c  label
    0   1  1.0  10.0  4.0      0
    1   1  2.0  10.0  5.0      1
    2   1  3.0  10.0  6.0      1
    3   2  4.0  12.0  NaN      0
    4   2  4.0  16.0  NaN      1
    5   2  4.0  20.0  NaN      0
    '''
    df_grouped = df.groupby(id_col_name)
    return df_grouped.apply(
        lambda df: df.interpolate(method="linear", limit_direction=limit_direction)
    ).reset_index(drop=True)


def forward_fill(df, column_name):
    """
    df: Entire Dataset frame
    column_name: Groupby column
    df = pd.DataFrame(
     {
     'id': [1, 1, 1, 2, 2, 2],
     'a': [1, np.nan, 3, np.nan, 4, np.nan],
     'b': [np.nan, np.nan, 10, 12, np.nan, 20],
     'c': [4, 5, 6, np.nan, np.nan, np.nan],
     label': [0,1,1,0,1,0]
     }
    )
    forward_impute(df,'id')
    Expected output
    id    a     b    c  label
    0   1  1.0   NaN  4.0      0
    1   1  1.0   NaN  5.0      1
    2   1  3.0  10.0  6.0      1
    3   2  NaN  12.0  NaN      0
    4   2  4.0  12.0  NaN      1
    5   2  4.0  20.0  NaN      0 
    """
    gb_col = df[column_name]
    df = df.groupby(column_name).ffill()
    df = pd.concat([gb_col, df], axis=1)
    return df

def delta(df, gb_col, exlude_columns = [],  window=1):
    """Finds the difference between present hour and window hour
    import random
    data = {
    "ID" :  [10001]*6 + [10002]*4 + [10003]*7,
    "f1": random.sample(range(1, 50), 17),
    "f2": random.sample(range(1, 50), 17),
    "label": [0]*10 + [1]*7
    }

    data_features = pd.DataFrame(data)
    delta_features = delta(data_features,"ID")
    print(pd.concat([data_features, delta_features], axis=1))
    """
    df_delta = df.loc[:, ~df.columns.isin(exlude_columns)] 
    df_new = pd.DataFrame()
    df_new = df_delta.groupby(gb_col).diff(window)
    df_new = df_new.fillna(0).add_prefix("Delta_" + str(window))
    return df_new


def CalcEnergy(x):
    result = (x * x).sum()
    return result/len(x)

def CalcSlope(x):
    length = len(x)
    if length < 2:
        return np.nan
    slope = (x[-1] - x[0])/(length -1)
    return slope

def statistics(df, stats, feature_list , gb_col, window=6):
    df = df[feature_list + [gb_col]]
    window = df.groupby(gb_col, as_index=False).rolling(min_periods=1, window=window)
    #window_2 = df.groupby(gb_col, as_index=False).rolling(min_periods=2, window=6)
    df_c = {}
    i = 0
    new_names = []
    if "var" in stats:
        df_var = pd.DataFrame()
        df_var = window.var(engine="numba", engine_kwargs={"parallel": True})
        df_var = df_var.add_prefix("Var_").fillna(0)
        df_c[i] = df_var
        new_names = new_names + df_var.keys().tolist()
        i = i + 1
    if "mean" in stats:
        df_mean = pd.DataFrame()
        df_mean = window.mean(engine="numba", engine_kwargs={"parallel": True})
        df_c[i] = df_mean.add_prefix("Mean_")
        new_names = new_names + df_c[i].keys().tolist()
        i = i + 1
    if "min" in stats:
        df_min = pd.DataFrame()
        df_min = window.min(engine="numba", engine_kwargs={"parallel": True})
        df_c[i] = df_min.add_prefix("Min_")
        new_names = new_names + df_c[i].keys().tolist()
        i = i + 1
    if "max" in stats:
        df_max = pd.DataFrame()
        df_max = window.max(engine="numba", engine_kwargs={"parallel": True})
        df_c[i] = df_max.add_prefix("Max_")
        new_names = new_names + df_c[i].keys().tolist()
        i = i + 1
    if "median" in stats:
        df_median = pd.DataFrame()
        df_median = window.median(engine="numba", engine_kwargs={"parallel": True})
        df_c[i] = df_median.add_prefix("Median_").drop(columns=["Median_" + gb_col])
        new_names = new_names + df_c[i].keys().tolist()
        i = i + 1
    if "energy" in stats:
        df_energy = pd.DataFrame()
        df_energy = window.apply(CalcEnergy, raw = True, engine="numba", engine_kwargs={"parallel": True})
        df_c[i] = df_energy.add_prefix("Energy_").drop(columns=["Energy_" + gb_col])
        new_names = new_names + df_c[i].keys().tolist()
        i = i + 1
    new_df = pd.concat(df_c.values(), axis = 1, ignore_index=True)
    names = dict(zip(new_df.keys(),new_names))
    new_df.rename(names, axis = 1, inplace=True)
    return new_df

def get_slope_stats(df, include_features, gb_col, window_size = 6):
    df = df[include_features + [gb_col]]
    tmp = df.groupby(gb_col, as_index=False).rolling(min_periods=2, window=window_size).apply(CalcSlope, raw = True, engine="numba", engine_kwargs={"parallel": True})
    tmp = tmp.add_prefix("Slope_").drop(columns=["Slope_" + gb_col])
    return tmp

def binning(df, col_name, range_a, range_b, new_col, binned_value):
    """
    Binning of a column data
    """
    df.loc[df[col_name].between(range_a, range_b, 'right'), new_col] = int(binned_value)

def age_binning(df, col_name, interval):
    """
    Binning of age into buckets
    """
    step_count = 1
    for i in range(1,100, interval):
        if( i < 20):
            step_count = 1
       # print(i, i+interval, step_count)
        binning(df, col_name, i, i + interval, "Binned_Age", step_count)
        if(i < 90):
            step_count = step_count + 1
    return df
        
def one_hot_encoding(df, col_name, prefix_name):
    """
    Do one hot encoding for the column
    """
    dummies = pd.get_dummies(df[col_name], prefix=prefix_name)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns = [col_name])
    return df

def shannon_entropy(df, column):
    Wl = 3
    l_z = len(df[column])
    x = df[column].to_numpy().reshape(-1, 1)
    if l_z - Wl < 0:
        l_z_1 = Wl - l_z
        stack11 = np.zeros((l_z_1, x.shape[1]))
        for iter34 in range(l_z_1):
            stack11[iter34,:] = x[0,:]
        x = np.concatenate((stack11, x), axis= 0)
        del stack11, l_z_1, l_z
    
    eps = 2.220446049250313e-16
    pA1 = None
    if np.var(x) != 0:
        kde = stats.gaussian_kde(x[:,0])
        pA1 = kde.pdf(x[:,0])
    else:
        pA1 = np.ones((1, len(x)))
    pA1 = pA1 + eps
    return -np.sum(pA1*np.log2(pA1))

def qSOFA(df, SBP_col, Resp_col):
    """
    Calculates qSOFA
    #Add Test Function
    """
    SBP = df[SBP_col]
    Resp = df[Resp_col]
    # if not (SBP.notna().all() and Resp.notna().all()):
    #     print("Error !!! NaN in the columns {}, {}".format(SBP_col, Resp_col))
    #     return df
    qSOFA  = SBP.map(lambda x:1 if x<=100 else 0) + Resp.map(lambda x:1 if x>=22 else 0)
    qSOFA_df = pd.DataFrame(qSOFA, columns=["qSOFA"])
    df = pd.concat([df, qSOFA_df], axis=1)
    return df

def SIRS(df, Temp_col, Heart_Rate_col, Resp_col, WBC_col=None):
    """
    Calculates SIRS condition
    Temp : Celcius
    Heart Rate : bpm
    WBC : k/Uml
    Resp : Rates per min
    #Sample Test Input
    import random 
    import numpy as np
    HR = random.sample(range(10, 120), 5)
    Temp = random.sample(range(30, 40), 5)
    Resp = np.random.uniform(low=10.0, high=30.0, size=5)
    WBC = random.sample(range(2, 20), 5)
    data = {"HR" : HR,"Temp": Temp, "Resp": Resp, "WBC": WBC}
    df = pd.DataFrame(data)
    df = SIRS(df, "Temp", "HR", "Resp", "WBC")
    print(df)
    """
    Heart_Rate = df[Heart_Rate_col]
    Temp = df[Temp_col]
    Resp = df[Resp_col]
    WBC = df[WBC_col]
    SIRS  = Heart_Rate.map(lambda x:1 if x>=90 else 0) + Resp.map(lambda x:1 if x>=22 else 0) \
            + WBC.map(lambda x:0 if (x == 0) else (1 if ((x>=12 or x <=4))  else 0)) \
            + Temp.map(lambda x:0 if (x == 0) else (1 if ((x>=38 or x <=36))  else 0))
    SIRS_df = pd.DataFrame(SIRS,columns=["SIRS"])
    df = pd.concat([df, SIRS_df], axis=1)
    return df

def SOFA(df, SaO2_FiO2 = None, MAP = None, Liver = None, Creatinine = None, Platelets = None):
    """
    import modin.pandas as pd
    random.seed(10)
    np.random.seed(10)
    SaO2_FiO2 = np.random.uniform(low=50.0, high=500.0, size=10)
    Map = random.sample(range(50, 100), 10)
    Bilirubin = np.random.uniform(low=1.0, high=20.0, size=10)
    Cr = np.random.uniform(low=1.0, high=10.0, size=10)
    Platelates = random.sample(range(0, 500), 10)
    data = {
    "SaO2_FiO2" : SaO2_FiO2,
    "MAP" : Map,
    "Bilirubin" : Bilirubin,
    "Cr" : Cr,
    "Plts" : Platelates,
    "label": [0]*3 + [1]*7
    }
    df = pd.DataFrame(data)
    df = SOFA_new(df, SaO2_FiO2 = "SaO2_FiO2", Liver="Bilirubin", Platelets="Plts" ,Creatinine="Cr")
    print(df)
        SaO2_FiO2  MAP  Bilirubin        Cr  Plts  label  SOFA
    0  397.094289   86  14.021837  5.882899   334      0     8
    1   59.338377   52  19.114474  2.279530   415      0    10
    2  335.141706   77   1.075017  4.360067    82      0     5
    3  386.961747   80  10.731653  7.067203    17      1    11
    4  274.328156   99  16.439798  4.976499   266      1     9
    5  151.158490   50  12.637995  4.906126   250      1    10
    6  139.128289   63  14.713351  6.559903   167      1    11
    7  392.238820   79   6.545645  5.618244    38      1    10
    8  126.099876   81  18.437708  6.853575   127      1    12
    9   89.752916   67  14.576940  6.409351   487      1    11
    """   
    sofa_score = 0
    if (SaO2_FiO2):
        Resp = df[SaO2_FiO2]
        sofa_score = sofa_score + Resp.map(lambda x:
                                  1 if ((x < 302) & (x > 221)) else
                                  2 if ((x < 221) & (x >142))  else 
                                  3 if ((x < 142)  & (x > 67)) else
                                  4 if ((x < 67) & (x > 0)) else 0)
    if (MAP):
        MAP = df[MAP]
        sofa_score = sofa_score + MAP.map(lambda x:
                                    1 if ((x < 70) & (x > 0)) else 0)
    if (Creatinine):
        Cr = df[Creatinine]
        sofa_score = sofa_score + Cr.map(lambda x:
                                  1 if ((x >= 1.2) & (x <= 1.9)) else
                                  2 if ((x > 1.9) & (x <= 3.4))  else 
                                  3 if ((x > 3.4)  & (x <= 4.9)) else
                                  4 if (x > 4.9)  else 0)
    if (Liver):
        Bilirubin = df[Liver]
        sofa_score = sofa_score + Bilirubin.map(lambda x:
                                    1 if ((x >= 1.2) & (x <= 1.9)) else
                                    2 if ((x > 1.9) & (x <= 5.9))  else 
                                    3 if ((x > 5.9)  & (x <= 11.9)) else
                                    4 if (x > 11.9)  else 0)
    if (Platelets):
        Plts = df[Platelets]
        sofa_score = sofa_score + Plts.map(lambda x:
                                    1 if ((x >= 100) & (x < 150)) else
                                    2 if ((x >= 50) & (x < 100))  else 
                                    3 if ((x >= 20)  & (x <= 50)) else
                                    4 if ((x < 20) & (x > 0)) else 0)
    SOFA_df = pd.DataFrame(sofa_score,columns=["SOFA"])
    df = pd.concat([df, SOFA_df], axis=1)
    return df

def MEWS(df, Resp = None, HR = None, Temp_C = None, Temp_F = None, SBP = None):
    
    mews_score = 0
    if (Resp):
        Resp = df[Resp]
        mews_score = mews_score + Resp.map(lambda x:
                                  1 if((x > 15) & (x < 20)) else
                                  2 if ((x < 9)  & (x > 0)) else
                                  2 if ((x < 29) & (x >20))  else 
                                  3 if (x > 30) else 0)
    if (Temp_C):
       Temp = df[Temp_C]
       mews_score = mews_score + Temp.map(lambda x:
                                 2 if(((x < 35) & (x > 0))) or (x > 38.5) else 0)
    if (Temp_F):
       Temp = df[Temp_F]
       mews_score = mews_score + Temp.map(lambda x:
                                 2 if(((x < 95) & (x > 0))) or (x > 101.3) else 0)
    if (HR):
        HR = df[HR]
        mews_score = mews_score + HR.map(lambda x:
                                  1 if (((x >= 41) & (x <= 50)) or ((x >= 101) & (x <= 110))) else
                                  2 if (((x <= 40)  & (x >=0))  or ((x >= 111) & (x <= 129))) else
                                  3 if (x >= 130) else 0)
    if (SBP):
        SBP = df[SBP]
        mews_score = mews_score + SBP.map(lambda x:
                                   2 if ((x >= 71) & (x <= 80)) or ((x >= 200)) else 
                                   1 if ((x >= 81) & (x <= 100)) else
                                   3 if ((x <= 70) & (x > 0)) else 0)

    MEWS_df = pd.DataFrame(mews_score,columns=["MEWS"])
    df = pd.concat([df, MEWS_df], axis=1)
    return df

def sliding_window(X_df, lenf, gb_col, window=6):
    """Creates a sliding window of the features
    import random
    random.seed(12)
    data = {
    "ID" :  [10001]*6 + [10002]*4,
    "Temp": random.sample(range(1, 50), 10),
    "HR": random.sample(range(1, 20), 10),
    "label": [0,1,1,1,0,1,0,1,1,1]
    }
    import pandas as pd
    data_features = pd.DataFrame(data)
    patient_id = "ID"
    window_length = 4
    print(data_features)
    df_subset = data_features.drop(columns = ["label"])
    df = sliding_window_new(df_subset,len(data_features),patient_id,window_length)
    data_features = pd.concat([data_features[patient_id], df, data_features.drop(columns=df_subset)],axis=1)
    print(data_features)
        ID  Temp-3  HR-3  Temp-2  HR-2  Temp-1  HR-1  Temp-0  HR-0  label
    0  10001      31    16      31    16      31    16      31    16      0
    1  10001      31    16      31    16      31    16      18     9      1
    2  10001      31    16      31    16      18     9      43    15      1
    3  10001      31    16      18     9      43    15      34     8      1
    4  10001      18     9      43    15      34     8      47    18      0
    5  10001      43    15      34     8      47    18      23     1      1
    6  10002      10    11      10    11      10    11      10    11      0
    7  10002      10    11      10    11      10    11      25    10      1
    8  10002      10    11      10    11      25    10       1     3      1
    9  10002      10    11      25    10       1     3      24    19      1
    """
    win = window
    s_window = X_df.groupby(gb_col, as_index=False).rolling(min_periods=1, window = win)
    i = 0
    sl_x = [0] * lenf
    num_features = len(X_df.columns) - 1
    feature_length = win * num_features
    for x in s_window:
      sl_x[i] = (x.values.reshape(1,-1))[0].tolist()
      if(len(sl_x[i]) < feature_length):
         first_window = sl_x[i][:num_features]
         missing_feature_length = feature_length - len(sl_x[i])
         factor = int(missing_feature_length/len(first_window))
         repeat_feature = first_window * factor
         sl_x[i] = repeat_feature + sl_x[i]
         if(len(sl_x[i]) != feature_length):
            print("Still feature lenghth is less")
      i = i + 1
    new_x = sl_x
    new_labels = ((X_df.drop(columns=gb_col)).columns).tolist() * win
    # setup dummy DataFrame with repeated columns
    df = pd.DataFrame(data=new_x, columns=new_labels)
    identifier = []
    old_labels_length = len(X_df.columns) - 1
    # create unique identifier for each repeated column
    i = 0
    for i in range(win, 0, -1):
        for j in range(old_labels_length):
            identifier.append("-" + str(i-1))
      # rename columns with the new identifiers
    df.columns = df.columns.astype('string') + identifier
    return df
