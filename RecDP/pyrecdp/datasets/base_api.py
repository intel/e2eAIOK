import os, requests
from tqdm import tqdm
import shutil
from pathlib import Path

libpath = str(Path(__file__).parent.resolve())

class base_api:
    def __init__(self):        
        self.cache_dir = f"{libpath}/dataset_cache"
        os.makedirs(self.cache_dir, exist_ok = True)
        
        
    def download(self, name, url):
        to_download = url
        to_save = f"{self.cache_dir}/{name}"
        # check if miniconda exsists
        if os.path.exists(to_save):
            return to_save

        with requests.get(to_download, stream=True) as r:
            # check header to get content length, in bytes
            total_length = int(r.headers.get("Content-Length"))
            
            # implement progress bar via tqdm
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw: 
                # save the output to a file
                with open(f"{to_save}", 'wb')as output:
                    shutil.copyfileobj(raw, output)
        return to_save