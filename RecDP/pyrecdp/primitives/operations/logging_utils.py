import logging
import sys
from loguru import logger

level = "INFO"
logging.root.setLevel(level)

# configure loguru
default_log_path = "/tmp/recdp/log/llmutils.log"
logger.configure(handlers=[
    {"sink": sys.stdout},
    {"sink": default_log_path, "rotation": "10 MB"},
])
