from .base import BaseFeatureGenerator as super_class
from pyrecdp.primitives.utils import SeriesSchema, is_text_series
import pandas as pd
import pyarrow as pa
import numpy as np

class StatisticsFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def is_useful(self, df):
        return True
    
    def fit_prepare(self, pa_schema):
        return
    
    def update_feature_statistics(self, X, y):
        overview_info = {}
        overview_detail = {}
        overview_info['Number of Features'] = X.shape[1]
        overview_info['Number of Rows'] = X.shape[0]
        interactions_detail = self.get_interactive_plot(X, y)
        
        length = X.shape[0]
        for feature_name in X.columns:
            feature = X[feature_name]
            desc_info = dict((k, v) for k, v in feature.describe().to_dict().items() if k not in ['count'])
            if 'unique' in desc_info:
                n_unique = desc_info['unique']
            else:
                n_unique = feature.nunique()

            feature_type = SeriesSchema(feature).dtype_str
            feature_type = "text" if is_text_series(feature) else feature_type
            
            stat = {'type': feature_type, 'unique': {"u": n_unique, "m": length}, 'quantile':desc_info}
            if feature_name not in overview_detail:
                overview_detail[feature_name] = stat
        
        data_stats = {}
        data_stats["overview"] = (overview_info, overview_detail)
        data_stats['interactions']=(dict(), interactions_detail)
        
        return data_stats
    
    def get_interactive_plot(self, feature_data, y):
        from plotly.subplots import make_subplots
        import plotly.graph_objs as go
        from plotly.offline import plot
        row_height = 300
        n_plot_per_row = 2
        n_feat = len(feature_data.columns)
        n_row = int(n_feat / n_plot_per_row) + 1
        n_col = n_plot_per_row if n_feat > n_plot_per_row else n_feat

        subplot_titles = feature_data.columns
        fig_list = make_subplots(rows=n_row, cols=n_col, subplot_titles  = subplot_titles, y_title = y.name)
        ret = {}
        indices = y.notna()
        if indices.size > 1000:
            frac = 1000/indices.size
            mask = pd.Series(np.random.choice(a=[False, True], size=(indices.size), p=[1-frac, frac]))
            indices = indices & mask

        for idx, c_name in enumerate(feature_data.columns):
            feature = feature_data[c_name]
            # if string, wrap when too long
            sch = SeriesSchema(feature)
            is_feature_string = True if sch.is_string or sch.is_categorical_and_string else False
            
            if is_feature_string:
                tmp = feature[indices].str.slice(0, 12, 1)
                fig = go.Scatter(x=tmp, y=y[indices], mode='markers', name=c_name, showlegend=False)
            else:
                fig = go.Scatter(x=feature[indices], y=y[indices], mode='markers', name=c_name, showlegend=False)

            fig_list.add_trace(fig, row = int(idx / n_plot_per_row) + 1, col = ((idx % n_plot_per_row) + 1))

        fig_list.update_layout(height=row_height * n_row, width=400 * n_col)

        ret = {"error": False}
        ret['html'] = plot(fig_list, output_type='div')
        return ret
