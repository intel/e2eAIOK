import re
import os
import yaml
import hashlib
from pathlib import Path
from datetime import datetime

def update_list(orig, diff):
    dict_diff = {}
    for item in diff:
        dict_diff[item['name']] = item
    for i in range(len(orig)):
        if orig[i]['name'] in dict_diff:
            orig[i] = dict_diff[orig[i]['name']]
    return orig

def check_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def mkdir(dest_path):
    new_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = os.path.join(dest_path, new_name)
    path = Path(new_path)
    path.mkdir(parents=True)
    return new_path

def mkdir_or_backup_then_mkdir(dest_path):
    path = Path(dest_path)
    if path.exists():
        new_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = os.path.join(path.parent.absolute(), new_name)
        os.rename(dest_path, new_path)
    path.mkdir(parents=True)

def get_hash_string(in_str):
    return hashlib.sha256(in_str.encode()).hexdigest()

def timeout_input(printout, default, timeout = None, interactive = True):
    if not interactive:
        return default
    import sys, select
    print(printout)
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if (i):
        msg = sys.stdin.readline().strip()
        return default if len(msg) == 0 else msg
    else:
        return default

def parse_config(conf_path):
    settings = {}
    if not os.path.exists(conf_path):
        return settings
    try:
        with open(conf_path) as f:
            in_settings = yaml.load(f, Loader=yaml.FullLoader)
            settings.update(in_settings)
    except:
        pass
    return settings

def get_file(path):
    # return directory if path contains multi files
    file_or_dir = os.listdir(path)
    if len(file_or_dir) != 1:
        return path
    return os.path.join(path, file_or_dir[0])

def list_dir(path):
    source_path_dict = {'meta': "", 'train': "", 'valid': ""}
    dirs = os.listdir(path)
    for file_name in dirs:
        if (file_name.endswith('yaml')):
            source_path_dict['meta'] = os.path.join(path, file_name)
        if (file_name == "train"):
            source_path_dict['train'] = get_file(os.path.join(path, file_name))
        if (file_name == "valid"):
            source_path_dict['valid'] = get_file(os.path.join(path, file_name))
    if len(source_path_dict['meta']) == 0 or len(source_path_dict['train']) == 0 or len(source_path_dict['valid']) == 0:
        raise ValueError(f"This folder layout is not as expected, should have one train folder, one validate folder and a yaml file of metadata, current we found {source_path_dict}")
    return source_path_dict


def parse_size(size):
    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    size = size.upper()
    number, unit = re.findall('(\d+)(\w*)', size)[0]
    return int(float(number)*units[unit])


def get_estimate_size_of_dtype(dtype_name):
    units = {'byte': 1, 'short': 2, 'int': 4, 'long': 8, 'float': 4, 'double': 8, 'string': 10}
    return units[dtype_name] if dtype_name in units else 4

## Update multi-level dict, merge dict2 into dict1
def update_dict(dict1, dict2):
    keys = set(dict1.keys()) | set(dict2.keys())
    dict3 = {}
    for key in keys:
        if key in dict1.keys() and key in dict2.keys():
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                dict3[key] = update_dict(dict1[key],dict2[key])
            elif (isinstance(dict1[key], dict) and not isinstance(dict2[key], dict)) or \
                    (not isinstance(dict1[key], dict) and isinstance(dict2[key], dict)):
                raise ValueError(f"{key} in two dicts have different types, one is dict, another is not!")
            else:
                dict3[key] = dict2[key]
        elif key in dict1.keys():
            dict3[key] = dict1[key]
        else:
            dict3[key] = dict2[key]
    return dict3
