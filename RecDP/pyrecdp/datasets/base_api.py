import os, requests
from tqdm import tqdm
import shutil
from pathlib import Path
import tarfile, gzip, zipfile

libpath = str(Path(__file__).parent.resolve())

class base_api:
    def __init__(self):        
        self.cache_dir = f"{libpath}/dataset_cache"
        os.makedirs(self.cache_dir, exist_ok = True)
       
    # def download_s3(self, bucket, filename):
    #     to_save = f"{self.cache_dir}/{filename}"
    #     # check if miniconda exsists
    #     if os.path.exists(to_save):
    #         return to_save
    #     s3r = boto3.resource('s3', aws_access_key_id=a1.replace("\\", ""),
    #         aws_secret_access_key=a2.replace("\\", ""))
    #     buck = s3r.Bucket(bucket)
    #     buck.download_file(filename, to_save)
    #     return to_save
        
    def get_file_path(self):
        return self.saved_path
     
    def download_url(self, name, url, unzip = False):
        to_download = url
        final_name = name
        cur_folder = self.__class__.__name__
        # skip when file exists
        if name == "":
            if os.path.exists(f"{self.cache_dir}/{cur_folder}"):
                return f"{self.cache_dir}/{cur_folder}"
        else:
            if os.path.exists(f"{self.cache_dir}/{name}"):
                return f"{self.cache_dir}/{name}"
            if os.path.exists(f"{self.cache_dir}/{cur_folder}/{name}"):
                return f"{self.cache_dir}/{cur_folder}/{name}"
        # not exists, start to download
        if unzip:
            name = "download_tmp"
        to_save = f"{self.cache_dir}/{name}"
        # check if miniconda exsists
        if not os.path.exists(to_save):
            with requests.get(to_download, stream=True) as r:
                # check header to get content length, in bytes
                total_length = int(r.headers.get("Content-Length"))
                
                # implement progress bar via tqdm
                with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw: 
                    # save the output to a file
                    with open(f"{to_save}", 'wb')as output:
                        shutil.copyfileobj(raw, output)   

        def unzip_tgz(src, dst):
            print("try unzip using tarfile")
            if not os.path.exists(dst):
                os.mkdir(dst)
            try:
                with tarfile.open(src) as tf:
                    tf.extractall(dst)
            except Exception as e:
                shutil.rmtree(dst)
                raise e
            
        def unzip_zip(src, dst):
            print("try unzip using zipfile")
            if not os.path.exists(dst):
                os.mkdir(dst)
            try:
                with zipfile.ZipFile(src, "r") as tf:
                    tf.extractall(dst)
            except Exception as e:
                shutil.rmtree(dst)
                raise e
        
        def unzip_gz(src, dst):
            print("try unzip using gzip")
            with gzip.open(src, 'rb') as f_in:
                with open(dst, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
        if unzip:
            name = final_name
            unzip_success = True
            print(f"Start to unzip {to_save}")
            try:
                unzip_tgz(to_save, f"{self.cache_dir}/{cur_folder}")
                to_delete = to_save
                to_save = f"{self.cache_dir}/{cur_folder}/{name}"
            except:
                try:
                    unzip_gz(to_save, f"{self.cache_dir}/{name}")
                    to_delete = to_save
                    to_save = f"{self.cache_dir}/{name}"
                except:
                    try:
                        unzip_zip(to_save, f"{self.cache_dir}/{cur_folder}")
                        to_delete = to_save
                        to_save = f"{self.cache_dir}/{cur_folder}/{name}"
                    except:
                        print(f"downloaded, but unable to unzip {url}, please unzip manully, data saved at {to_save}")
                        unzip_success = False
            if unzip_success and os.path.exists(to_delete):
                os.remove(to_delete)
        return to_save
