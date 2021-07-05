import timeit
import pandas as pd
import numpy as np
import pickle
import re
import os
import sys
import glob

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class Settings:
    drop_duplicates = False
    lazy_load = True
    show_timer = True
    stored_dir = '.'
    suffix = '_fold'
    feature_list = []
    using_lgbm = False

class Timer:
    level = 0

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if Settings.show_timer:
            print(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')


class Model():
    def __init__(self, name=None, join_cols=None):
        self.name = name
        self.join_cols = join_cols
        if not Settings.lazy_load:
            self.df_agg = self.load_data()

    def load_data(self):
        with Timer(f'load {self.name}.parquet'):
            for filename in glob.glob(f'{Settings.stored_dir}/{self.name}{Settings.suffix}.parquet'):
                df_agg = pd.read_parquet(f"{Settings.stored_dir}/{self.name}{Settings.suffix}.parquet")
            for filename in glob.glob(f'{Settings.stored_dir}/{self.name}'):
                df_agg = pd.read_parquet(f"{Settings.stored_dir}/{self.name}")
            try:
                df_agg
            except NameError:
                raise FileNotFoundError(f"Can't find Model file with name {Settings.stored_dir}/{self.name} or {Settings.stored_dir}/{self.name}{Settings.suffix}.parquet")
            if Settings.drop_duplicates:
                df_agg = df_agg.drop_duplicates()
        return df_agg

    def transform(self, data):
        df_agg = self.load_data() if Settings.lazy_load else self.df_agg
        try:
            return pd.merge(data, df_agg, on=self.join_cols, how='left')
        except:
            print(f"[ERROR]Pandas merge join failed on {self.name}")
            e = sys.exc_info()[0]
            write_to_page( "<p>Error: %s</p>" % e )

    def get_meta(self):
        return {'name': self.name, 'join_cols': self.join_cols}

    @classmethod
    def from_meta(cls, dct):
        self = cls()
        self.name = dct['name']
        self.join_cols = dct['join_cols']
        return self


class CATE_Model(Model):
    def __init__(self, cate_name=None, join_cols=None, file_name=None):
        self.cate_name = cate_name
        self.join_cols = join_cols
        if file_name == None:
            self.file_name = "%s_%s" % (join_cols, cate_name)
        else:
            self.file_name = file_name
        self.df_agg = self.load_data()

    def load_data(self):
        with Timer(f'load {self.file_name}'):
            df_agg = pd.read_parquet(
                f"{Settings.stored_dir}/{self.file_name}")
        return df_agg

    def transform(self, data, left_col=None):
        df_agg = self.df_agg
        if left_col != None:
            df_agg_tmp = df_agg.rename(columns={join_cols: left_col})
            return pd.merge(data, df_agg_tmp, on=left_col, how='left')
        else:
            return pd.merge(data, df_agg, on=self.join_cols, how='left')


class CMBD_Model(Model):
    def __init__(self, name=None, join_cols=None, lookup_table=None):
        if name == None:
            self.name = lookup_table
        else:
            self.name = name
        self.join_cols = join_cols
        self.lookup_table = lookup_table
        self.feature_list = []
        if not Settings.lazy_load:
            self.df_agg = self.load_data()

    def load_data(self):
        with Timer(f'load {self.lookup_table}'):
            to_read_file_name = ""
            for filename in glob.glob(f'{Settings.stored_dir}/{self.name}{Settings.suffix}.parquet'):
                to_read_file_name = filename
            for filename in glob.glob(f'{Settings.stored_dir}/{self.name}'):
                to_read_file_name = filename
            if to_read_file_name == "":
                # this case is for compress_models to load from valid_lookup
                to_read_file_name = f"{Settings.stored_dir}/valid_lookup/{self.lookup_table}"
            if self.name != self.lookup_table:
                columns = self.name + join_cols
                df_agg = pd.read_parquet(to_read_file_name, columns=columns)
            else:
                df_agg = pd.read_parquet(to_read_file_name)
            if Settings.drop_duplicates:
                df_agg = df_agg.drop_duplicates()
        self.feature_list = [c for c in df_agg.columns if c not in self.join_cols]
        return df_agg

    def transform(self, data):
        df_agg = self.load_data() if Settings.lazy_load else self.df_agg
        try:
            return pd.merge(data, df_agg, on=self.join_cols, how='left')
        except:
            print(f"[ERROR]Pandas merge join failed on {self.name}")
            print(data[self.join_cols].info())
            print(df_agg[self.join_cols].info())
            raise e


    def get_meta(self):
        return {'name': self.name, 'join_cols': self.join_cols, 'lookup_table': self.lookup_table}

    @classmethod
    def from_meta(cls, dct):
        self = cls()
        self.name = dct['name']
        self.join_cols = dct['join_cols']
        self.lookup_table = dct['lookup_table']
        return self


class CPD_Model(Model):
    def __init__(self, name=None, inputCol=None, outputCol=None, join_cols=None):
        super().__init__(name, join_cols)
        self.inputCol = inputCol
        self.outputCol = outputCol

    def load_data(self):
        with Timer(f'load {self.name}.parquet'):
            # cannot drop duplicates here - too complex items; make sure to have *NO* dupes in source data
            df_agg = pd.read_parquet(
                f"{Settings.stored_dir}/{self.name}{Settings.suffix}.parquet")
        return df_agg

    @staticmethod
    def intersect_cols(x1, x2):
        values = np.zeros(len(x1), dtype=np.uint8)
        mask = pd.notna(x1) & pd.notna(x2)
        for idx in mask.index[mask]:
            xset = set(x1[idx])
            yset = set(x2[idx].split('\t'))
            values[idx] = len(xset.intersection(yset))
        return values

    def transform(self, data):
        df_agg = self.load_data() if Settings.lazy_load else self.df_agg
        alldf = pd.merge(data, df_agg, on=self.join_cols, how='left')
        alldf[f'CPD_{self.inputCol}_{self.outputCol}'] = self.intersect_cols(
            alldf[f"{self.inputCol}_{self.inputCol}_intersection_prescence_unique"], alldf[self.inputCol])
        return alldf

    def get_meta(self):
        return {'name': self.name, 'join_cols': self.join_cols, 'inputCol': self.inputCol, 'outputCol': self.outputCol}

    @classmethod
    def from_meta(cls, dct):
        self = cls()
        self.name = dct['name']
        self.join_cols = dct['join_cols']
        self.inputCol = dct['inputCol']
        self.outputCol = dct['outputCol']
        return self


def get_estimator(meta):
    if 'lookup_table' in meta:
        return CMBD_Model.from_meta(meta)
    if 'inputCol' in meta:
        if 'outputCol' in meta:
            return CPD_Model.from_meta(meta)
        raise ValueError(f'unexpected columns: {" ".join(meta.keys())}')
    if 'outputCol' not in meta:
        return Model.from_meta(meta)
    raise ValueError(f'unexpected columns: {" ".join(meta.keys())}')


def get_vocab():
    with open(f"{SCRIPT_DIR}/vocab.pkl", 'rb') as fp:
        vocab_token = pickle.load(fp)

    joint, separate = {}, {}
    for vocab, token in vocab_token.items():
        if vocab.startswith('##'):
            joint[token] = vocab[2:]
        else:
            separate[token] = ' ' + vocab

    return joint, separate


def get_tweet_text(s, joint_tv, separate_tv):
    result = []
    for itoken in s.split('\t'):
        idx = int(itoken)
        try:
            result.append(separate_tv[idx])
        except KeyError:
            result.append(joint_tv[idx])
    return ''.join(result)


def get_is_rt(text):
    if 'RT' not in text:
        return 0
    if re.search(r"^RT\s+|\sRT\s+", text):
        return 1
    return 0


if os.environ.get('DO_DEBUG'):
    import pdb
    import sys
    import traceback
    print('... registering debug post mortem')

    def do_debug(type, value, tb):
        traceback.print_exception(type, value, tb)
        pdb.pm()
    sys.excepthook = do_debug


class CountEncoder:

    def __init__(self, seed=42):
        self.seed = seed

    def fit_transform(self, test, x_col, out_col):
        c_col = 'dummy'
        test[c_col] = 1

        cols = [x_col] if isinstance(x_col, str) else x_col

        agg_test = test[cols + [out_col]]
        agg_test = test.groupby(cols).agg({out_col: 'last', c_col: 'count'}).reset_index()
        agg_test[out_col] = agg_test[out_col].fillna(0)
        agg_test[out_col] = agg_test[out_col] + agg_test[c_col]
        agg_test = agg_test.drop(c_col, axis=1)
        test = test.drop(out_col, axis=1)
        test = test.merge(agg_test, on=cols, how='left')
        return test


class FrequencyEncoder:

    def __init__(self, seed=42):
        self.seed = seed

    def fit_transform(self, test, x_col, out_col):
        c_col = 'dummy'
        test[c_col] = 1
        drop = True

        cols = [x_col] if isinstance(x_col, str) else x_col

        agg_test = test.groupby(cols).agg({c_col: 'count'}).reset_index()
        agg_test[out_col] = (agg_test[c_col]*1.0 / len(test)).astype('float32')
        agg_test = agg_test.drop(c_col, axis=1)
        test = test.drop(out_col, axis=1)
        test = test.merge(agg_test, on=cols, how='left')
        return test

def get_join_cols_from_model(c, model_list):
    for m in model_list:
        if isinstance(m, CMBD_Model):
            if c in m.feature_list:
                return m.join_cols
    return []
