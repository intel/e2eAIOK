from .astype import AsTypeFeatureGenerator
from .binned import BinnedFeatureGenerator
from .category import CategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .drop_duplicates import DropDuplicatesFeatureGenerator
from .drop_unique import DropUniqueFeatureGenerator
from .fillna import FillNaFeatureGenerator
from .identity import IdentityFeatureGenerator
from .encoder import LabelEncoderFeatureGenerator, OneHotEncoderFeatureGenerator
from .nlp import TextNgramFeatureGenerator, TextSpecialFeatureGenerator


cls_list = {
            'AsTypeFeatureGenerator': AsTypeFeatureGenerator,
            'DatetimeFeatureGenerator': DatetimeFeatureGenerator,
            'FillNaFeatureGenerator': FillNaFeatureGenerator,
            
            'BinnedFeatureGenerator': BinnedFeatureGenerator,
            'IdentityFeatureGenerator': IdentityFeatureGenerator,
            'CategoryFeatureGenerator': CategoryFeatureGenerator,

            'LabelEncoderFeatureGenerator': LabelEncoderFeatureGenerator,
            'OneHotEncoderFeatureGenerator': OneHotEncoderFeatureGenerator,
            
            'DropDuplicatesFeatureGenerator': DropDuplicatesFeatureGenerator,
            'DropUniqueFeatureGenerator': DropUniqueFeatureGenerator,
        }