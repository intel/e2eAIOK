from .base_api import base_api

class download(base_api):
    def __init__(self, name, url, unzip = False):
        super().__init__()      
        self.saved_path = self.download_url(name, url, unzip = unzip)
        print(f"Data is downloaded, use {self.saved_path} to open")
         