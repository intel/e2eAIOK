from .base_api import base_api

class outbrain(base_api):
    def __init__(self):
        super().__init__()
        file_list = {
            'clicks': "clicks_train.csv",
            'documents_categories': "documents_categories.csv",
            'documents_entities': "documents_entities.csv",
            'documents_meta': "documents_meta.csv",
            'documents_topics': "documents_topics.csv",
            'events': "events.csv",
            'page_views': "page_views_sample.csv",
            'promoted_content': "promoted_content.csv"
        }

        self.saved_path = dict((f_name, self.download_url(f_path, f"https://outbrain-sampled.s3.us-west-2.amazonaws.com/{f_path}")) for f_name, f_path in file_list.items())

    def to_pandas(self, nrows = None):
        import pandas as pd
        return dict((f_name, pd.read_csv(f_path)) for f_name, f_path in self.saved_path.items())
         