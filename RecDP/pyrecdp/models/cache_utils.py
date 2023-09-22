import os

# Default cache location
DEFAULT_CACHE_HOME = '~/.cache'
CACHE_HOME = os.getenv('CACHE_HOME', DEFAULT_CACHE_HOME)

# Default recdp cache location
DEFAULT_REC_DP_CACHE_HOME = os.path.join(CACHE_HOME, 'recdp')
REC_DP_CACHE_HOME = os.path.expanduser(
    os.getenv('REC_DP_CACHE_HOME', DEFAULT_REC_DP_CACHE_HOME))

# Default assets cache location
DEFAULT_REC_DP_ASSETS_CACHE = os.path.join(REC_DP_CACHE_HOME,
                                                'assets')
REC_DP_ASSETS_CACHE = os.getenv('REC_DP_ASSETS_CACHE',
                                     DEFAULT_REC_DP_ASSETS_CACHE)
# Default models cache location
DEFAULT_REC_DP_MODELS_CACHE = os.path.join(REC_DP_CACHE_HOME,
                                                'models')
REC_DP_MODELS_CACHE = os.getenv('REC_DP_MODELS_CACHE',
                                     DEFAULT_REC_DP_MODELS_CACHE)

CACHE_COMPRESS = None
