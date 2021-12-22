from .init_spark import *

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
