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

def parse_config(conf_file):
    settings = {}
    if not os.path.exists(conf_file):
        return settings
    with open(conf_file) as f:
        settings.update(yaml.safe_load(f))
    return settings