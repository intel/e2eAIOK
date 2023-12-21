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
import seaborn as sns
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import missingno as msno

def remove_spaces_in_colnames(df):
    """
    Removes spaces in column names
    """
    new_colnames = []
    for col in df.columns:
        col = ' '.join(col.split()).replace(' ', '_' )
        new_colnames.append(col)
    df.columns = new_colnames
    return df 


def drop_patients_with_low_entries(df, grpby_col, min_record_per_patient = 5):
    """
    Drop patients with low entries with min records
    """
    t = df.groupby(grpby_col).size().to_frame('Count').reset_index()
    selected_patients = t[t.Count >= min_record_per_patient][grpby_col]
    res = df[df[grpby_col].isin(selected_patients)]
    return res


def get_list_of_septic_non_septic_patients(df, patient_id, label):
    """
    """
    patients_affected = df[df[label] == 1][patient_id].unique()
    non_affected_patients = df[~ df[patient_id].isin(patients_affected)][patient_id].unique()
    return patients_affected, non_affected_patients

def create_dendrogram(df_subset):
    """
    Create a Denodrogram for a dataframe
    """
    fig = plt.figure(figsize=(20, 8))
    print("Total number of columns on which linkage matrix is drawn = ", len(df_subset.columns))
    corr = df_subset.corr()
    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1)
    corr = corr.fillna(0)
    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.linkage(distance_matrix, 'ward')
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=df_subset.columns.tolist(), leaf_rotation=90
    )
    # Plot axis labels
    plt.xlabel('Features')
    plt.ylabel('distance (Ward)')
    plt.show()
    return corr, dist_linkage



def select_feature_with_linkage_matrix(dist_linkage, distance, df_subset):
    """
    Select features from clusters at distance 1
    """
    cluster_ids = hierarchy.fcluster(dist_linkage, distance, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [df_subset.columns[v[0]] for v in cluster_id_to_feature_ids.values()]
    print("Total number of features selected = ", len(selected_features))
    return selected_features

def nan_percentage_in_df(df):
    """
    Percentage of nan values in dataframe
    """
    print("dataframe dimension: ", df.shape)
    total_entries = df.shape[0] * df.shape[1]
    print("Total entries:  ", total_entries)
    num_nan_entries = df.isnull().sum().sum()
    print("Total null entries: ", num_nan_entries)
    nan_percentage = (num_nan_entries / total_entries) * 100
    print ("Percentage of null entries: ", nan_percentage)  
    return nan_percentage

def histogram_boxplot(data, xlabel, ylabel, sample_size = 1000, title = None, font_scale=2, figsize=(16,8), bins = None):
    """
    data: 1-d data array
    xlabel: xlabel 
    ylabel: the class name 
    sample_size: 
    title: title of the plot
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    sns.set(style="darkgrid")
    sns.set(rc={'figure.figsize':(16,8)})

    # creating a figure composed of 3 matplotlib.Axes objects
    f, (ax_box1, ax_box2, ax_hist) = plt.subplots(3, sharex=True, gridspec_kw={"height_ratios": (.15, .15, .85)})
    colours = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
    # assigning a graph to each ax

    sep_df = data[data[ylabel]==1]#.sample(sample_size)
    sns.boxplot(x=sep_df[xlabel], ax=ax_box1, color="#ea4335")
    sns.histplot(sep_df, x=xlabel, ax=ax_hist, kde=True, color="#ea4335")

    non_sep_df = data[data[ylabel]==0]#.sample(sample_size)
    sns.boxplot(x=non_sep_df[xlabel], ax=ax_box2, color='#4285f4')
    sns.histplot(non_sep_df, x=xlabel, ax=ax_hist, kde=True, color='#4285f4')
    
    # Remove x axis name for the boxplots
    ax_box1.set(xlabel='')
    ax_box1.set(title = title)
    ax_box2.set(xlabel='')


    plt.legend(title='', loc=2, labels=['Patients-affected', 'Patients not affected'],bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.show()

def describe_data(df, patient_id, label):
    print("The shape of the Raw input data is: ", df.shape)
    patients_affected, non_affected_patients = get_list_of_septic_non_septic_patients(df, patient_id, label)
    print("\nTotal number of patients: ", len(patients_affected) + len(non_affected_patients))
    print("Total patients affected: {} \n  Total  Patients not affected : {}".format(len(patients_affected), len(non_affected_patients)))
    print("Ratio of patients affected vs Non affected patients : ", len(patients_affected) / (len(non_affected_patients)))
    print("Patients Affected Prevalence as  %Percentage : ", len(patients_affected) / (len(patients_affected) + len(non_affected_patients)) * 100)
    print("\n----------Following Features are present in the data -----------")
    print(df.columns)
    print("\n----------The data types of the Features given below -----------")
    print(df.info())
    print("\n----------Labels count are given as under  -----------")
    print(df[label].value_counts())
    return patients_affected, non_affected_patients

def plot_missing_values(df):
    msno.matrix(df, figsize=(20,10), fontsize=12, color=(0.38, 0.35, 1), sparkline=False)
    plt.show()

    fig = plt.figure(figsize=(20,20))
    msno.bar(df,   color="dodgerblue", sort="ascending", figsize=(10,15), fontsize=12);
    plt.show()


#Example to test the function
#example 1
# example_df = pd.DataFrame({'Glucose_min':range(1,10), 
#                    'Glucose_max':range(11, 20),  
#                    'feature3':range(21, 30),
#                    'age':range(31, 40)
#                    })
# example_df.loc[3, 'Glucose_max'] = np.nan
# example_df.loc[4, 'Glucose_min'] = np.nan
# example_df.loc[4, 'Glucose_max'] = np.nan
# example_df.loc[6, 'Glucose_min'] = np.nan
# example_df.loc[8, :] = np.nan
# example_df

#example 2
# tdf = pd.DataFrame({'a':[1, np.nan, np.nan, 4], 
#                    'b':[2, 3, np.nan, np.nan]}
#                     )
# x = tdf[['a', 'b']]
# x = x.fillna(axis=1, method='ffill').fillna(axis=1, method='bfill')
# x