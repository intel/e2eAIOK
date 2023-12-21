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

import os

# Default cache location
DEFAULT_CACHE_HOME = '~/.cache'
CACHE_HOME = os.getenv('CACHE_HOME', DEFAULT_CACHE_HOME)

# Default RECDP cache location
DEFAULT_RECDP_CACHE_HOME = os.path.join(CACHE_HOME, 'RECDP')
RECDP_CACHE_HOME = os.path.expanduser(
    os.getenv('RECDP_CACHE_HOME', DEFAULT_RECDP_CACHE_HOME))

# Default assets cache location
DEFAULT_RECDP_ASSETS_CACHE = os.path.join(RECDP_CACHE_HOME,
                                                'assets')
RECDP_ASSETS_CACHE = os.getenv('RECDP_ASSETS_CACHE',
                                     DEFAULT_RECDP_ASSETS_CACHE)
# Default models cache location
DEFAULT_RECDP_MODELS_CACHE = os.path.join(RECDP_CACHE_HOME,
                                                'models')
RECDP_MODELS_CACHE = os.getenv('RECDP_MODELS_CACHE',
                                     DEFAULT_RECDP_MODELS_CACHE)

