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

