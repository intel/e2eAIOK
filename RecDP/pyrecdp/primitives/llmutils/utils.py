import os

def get_data_files(data_dir):
    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))
    files = [os.path.join(data_dir, i) for i in files]
    return files
