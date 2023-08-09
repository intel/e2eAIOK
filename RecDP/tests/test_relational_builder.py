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
from pyrecdp.autofe import RelationalBuilder, FeatureWrangler

class TestRelationalBuilder(unittest.TestCase):
    def test_outbrain(self):
        train_data = {
            'clicks': "clicks_train.csv",
            'documents_categories': "documents_categories.csv",
            'documents_entities': "documents_entities.csv",
            'documents_meta': "documents_meta.csv",
            'documents_topics': "documents_topics.csv",
            'events': "events.csv",
            'page_views': "page_views_sample.csv",
            'promoted_content': "promoted_content.csv"}
        dir_path = f"{pathlib}/tests/data/outbrain/"
        train_data = dict((f_name, pd.read_csv(f"{dir_path}/{f_path}")) for f_name, f_path in train_data.items())
        relation_pipeline = RelationalBuilder(dataset=train_data, label="clicked")
        pipeline = FeatureWrangler(data_pipeline=relation_pipeline)
        #print(pipeline.export())
        ret_df = pipeline.fit_transform(engine_type = 'pandas')
        # test with shape
        #self.assertEqual(ret_df.shape[0], 10000)
        #print(ret_df.dtypes)
        #self.assertTrue(ret_df.shape[1] >= 12)

    def test_outbrain_path(self):
        train_data = {
            'clicks': "clicks_train.csv",
            'documents_categories': "documents_categories.csv",
            'documents_entities': "documents_entities.csv",
            'documents_meta': "documents_meta.csv",
            'documents_topics': "documents_topics.csv",
            'events': "events.csv",
            'page_views': "page_views_sample.csv",
            'promoted_content': "promoted_content.csv"}
        dir_path = f"{pathlib}/tests/data/outbrain/"
        train_data = dict((f_name, f"{dir_path}/{f_path}") for f_name, f_path in train_data.items())
        relation_pipeline = RelationalBuilder(dataset=train_data, label="clicked")
        pipeline = FeatureWrangler(data_pipeline=relation_pipeline)
        ret_df = pipeline.fit_transform(engine_type = 'spark')