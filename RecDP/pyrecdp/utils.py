from .init_spark import *
import re

def convert_to_spark_dict(orig_dict, schema=['dict_col', 'dict_col_id']):
    ret = []
    for row_k, row_v in orig_dict.items():
        ret.append({schema[0]: row_k, schema[1]: row_v})
    return ret


def list_dir(path):
    source_path_dict = {}
    dirs = os.listdir(path)
    for files in dirs:
        try:
            sub_dirs = os.listdir(path + "/" + files)
            for file_name in sub_dirs:
                if (file_name.endswith('parquet') or file_name.endswith('csv')):
                    source_path_dict[files] = os.path.join(
                        path, files, file_name)
        except:
            source_path_dict[files] = os.path.join(path, files)
    return source_path_dict


def parse_size(size):
    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    size = size.upper()
    number, unit = re.findall('(\d+)(\w*)', size)[0]
    return int(float(number)*units[unit])


def get_estimate_size_of_dtype(dtype_name):
    units = {'byte': 1, 'short': 2, 'int': 4, 'long': 8, 'float': 4, 'double': 8, 'string': 10}
    return units[dtype_name] if dtype_name in units else 4
