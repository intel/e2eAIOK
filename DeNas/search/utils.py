import os
import timeit
import yaml

class Timer:
    level = 0

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        print(f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]

def parse_config(conf_file):
    settings = {}
    if not os.path.exists(conf_file):
        return settings
    with open(conf_file) as f:
        settings.update(yaml.load(f, Loader=yaml.FullLoader))
    return settings