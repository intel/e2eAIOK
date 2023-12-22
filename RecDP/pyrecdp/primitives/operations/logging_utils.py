"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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
