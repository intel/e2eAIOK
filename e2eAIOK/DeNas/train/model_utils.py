import os
import yaml

def load_config(conf_file):
    settings = {}
    if not os.path.exists(conf_file):
        return settings
    with open(conf_file) as f:
        settings.update(yaml.safe_load(f))
    return settings