from .dataframe import DataframeConvertFeatureGenerator, DataframeTransformFeatureGenerator
from .binned import BinnedFeatureGenerator
from .category import CategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .drop import DropUselessFeatureGenerator
from .name import RenameFeatureGenerator
from .fillna import FillNaFeatureGenerator
from .statics import StatisticsFeatureGenerator
from .type import TypeInferFeatureGenerator, TypeCheckFeatureGenerator
from .encoder import LabelEncoderFeatureGenerator, OneHotEncoderFeatureGenerator
from .nlp import DecodedTextFeatureGenerator, TextFeatureGenerator
from .geograph import GeoFeatureGenerator, CoordinatesInferFeatureGenerator
from .relation import RelationBuilder

feature_infer_list = [
    TypeInferFeatureGenerator,
    CoordinatesInferFeatureGenerator,
]

relation_builder_list = [
    RelationBuilder
]

pre_feature_generator_list = [
    FillNaFeatureGenerator,
    TypeInferFeatureGenerator,
    CoordinatesInferFeatureGenerator,
    DecodedTextFeatureGenerator,
]

transformation_generator_list = [
    DatetimeFeatureGenerator,
    GeoFeatureGenerator,
    TextFeatureGenerator,
]

index_generator_list = [
    BinnedFeatureGenerator,
    CategoryFeatureGenerator,
]

encode_generator_list = [
    LabelEncoderFeatureGenerator,
    OneHotEncoderFeatureGenerator
]

post_feature_generator_list = [
    DropUselessFeatureGenerator,
    RenameFeatureGenerator
]

final_generator_list = [
    TypeCheckFeatureGenerator,
    DropUselessFeatureGenerator
]