import unittest
import sys
import pandas as pd
from pathlib import Path
pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)
from IPython.display import display
from pyrecdp.core.schema import DataFrameSchema
from pyrecdp.core.di_graph import DiGraph
from pyrecdp.primitives.operations import Operation


class TestUnitMethod(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_parquet(f"{pathlib}/tests/data/recsys2023_train.parquet")
        self.fraud_detect_df = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")
        self.pipeline = DiGraph()
        self.pipeline[0] = Operation(
            0, None, output = DataFrameSchema(self.df), op = 'DataFrame', config = 'main')

    def test_distribution_infer(self):
        from pyrecdp.primitives.profilers import DistributionInferFeatureProfiler
        DistributionInferFeatureProfiler().fit_prepare(self.pipeline, [0], 0, self.df, y = None)
        
    def test_categorify(self):
        from pyrecdp.primitives.operations.category import CategorifyOperation
        feature = ['f_2']
        train_df = self.df[self.df['f_1'] < 66]
        test_df = self.df[self.df['f_1'] == 66]
        df_x_1 = train_df[feature]
        df_x_2 = test_df[feature]
        dict_path = "f_2_cat.pkl"
        feature_out = "f_2_idx"
        item = (feature, df_x_1, dict_path, feature_out)
        ret_train = pd.DataFrame()
        ret_train['f_1'] = train_df['f_1']
        ret_train['f_2'] = train_df['f_2']
        ret_train['f_2_idx'] = CategorifyOperation.label_encode(item)
        
        item = (feature, df_x_2, dict_path, feature_out)
        ret_test = pd.DataFrame()
        ret_test['f_1'] = test_df['f_1']
        ret_test['f_2'] = test_df['f_2']
        ret_test['f_2_idx'] = CategorifyOperation.label_encode_transform(item)
        display(ret_train)
        display(ret_test)
        
    def test_Time_categorify(self):
        from pyrecdp.primitives.operations.category import CategorifyOperation
        feature = ['Time']
        df = self.fraud_detect_df
        df['Time'] = pd.to_datetime(df['Time'])
        train_df = df[df['Year'] < 2018].reset_index(drop=True)
        test_df = df[df['Year'] > 2018].reset_index(drop=True)
        df_x_1 = train_df[feature]
        df_x_2 = test_df[feature]
        dict_path = f"{feature[0]}_cat.pkl"
        feature_out = f"{feature[0]}_idx"
        item = (feature, df_x_1, dict_path, feature_out)
        ret_train = pd.DataFrame()
        ret_train['Time'] = train_df['Time']
        ret_train[feature_out] = CategorifyOperation.label_encode(item)
        
        item = (feature, df_x_2, dict_path, feature_out)
        ret_test = pd.DataFrame()
        ret_test['Time'] = test_df['Time']
        ret_test[feature_out] = CategorifyOperation.label_encode_transform(item)
        display(ret_train)
        display(ret_test)
        
    def test_group_categorify(self):
        from pyrecdp.primitives.operations.category import GroupCategorifyOperation
        grouped_features = ['f_1', 'f_2']
        train_df = self.df[self.df['f_1'] < 66]
        test_df = self.df[self.df['f_1'] == 66]
        df_x_1 = train_df[grouped_features]
        df_x_2 = test_df[grouped_features]
        dict_path = "f_1_f_2_gc.parquet"
        feature_out = "f_1_f_2_idx"
        item = (grouped_features, df_x_1, dict_path, feature_out)
        ret_train = pd.DataFrame()
        ret_train['f_1'] = train_df['f_1']
        ret_train['f_2'] = train_df['f_2']
        ret_train['f_1_f_2_idx'] = GroupCategorifyOperation.group_label_encode(item)
        
        item = (grouped_features, df_x_2, dict_path, feature_out)
        ret_test = pd.DataFrame()
        ret_test['f_1'] = test_df['f_1']
        ret_test['f_2'] = test_df['f_2']
        ret_test['f_1_f_2_idx'] = GroupCategorifyOperation.group_label_encode_transform(item)
        display(ret_train)
        display(ret_test)
        
    def test_TE(self):
        from pyrecdp.primitives.operations.encode import TargetEncodeOperation
        feature = 'f_2'
        target_label = 'is_installed'
        train_df = self.df[self.df['f_1'] < 66]
        test_df = self.df[self.df['f_1'] == 66]
        df_x_1 = train_df[feature]
        df_y = train_df[target_label]
        df_x_2 = test_df[feature]
        dict_path = "f_2_is_installed_te.pkl"
        feature_out = "f_2_is_installed_te_idx"
        item = (feature, df_x_1, df_y, dict_path, feature_out)
        ret_train = pd.DataFrame()
        ret_train['f_1'] = train_df['f_1']
        ret_train['f_2'] = train_df['f_2']
        ret_train['f_2_TE'] = TargetEncodeOperation.target_encode(item)
        
        item = (feature, df_x_2, None, dict_path, feature_out)
        ret_test = pd.DataFrame()
        ret_test['f_1'] = test_df['f_1']
        ret_test['f_2'] = test_df['f_2']
        ret_test['f_2_CE'] = TargetEncodeOperation.target_encode_transform(item)
        display(ret_train)
        display(ret_test)
        
    def test_CE(self):
        from pyrecdp.primitives.operations.encode import CountEncodeOperation
        feature = 'f_2'
        train_df = self.df[self.df['f_1'] < 66]
        test_df = self.df[self.df['f_1'] == 66]
        df_x_1 = train_df[feature]
        df_x_2 = test_df[feature]
        dict_path = "f_2_is_installed_ce.pkl"
        feature_out = "f_2_is_installed_ce_idx"

        item = (feature, df_x_1, dict_path, feature_out)
        ret_train = pd.DataFrame()
        ret_train['f_1'] = train_df['f_1']
        ret_train['f_2'] = train_df['f_2']
        ret_train['f_2_CE'] = CountEncodeOperation.count_encode(item)

        item = (feature, df_x_2, dict_path, feature_out)
        ret_test = pd.DataFrame()
        ret_test['f_1'] = test_df['f_1']
        ret_test['f_2'] = test_df['f_2']
        ret_test['f_2_CE'] = CountEncodeOperation.count_encode_transform(item)
        display(ret_train)
        display(ret_test)