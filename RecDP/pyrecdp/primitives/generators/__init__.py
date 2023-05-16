from .binned import BinnedFeatureGenerator
from .category import CategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .drop import DropUselessFeatureGenerator
from .name import RenameFeatureGenerator
from .fillna import FillNaFeatureGenerator
from .type import TypeCheckFeatureGenerator,TypeConvertFeatureGenerator
from .nlp import DecodedTextFeatureGenerator, TextFeatureGenerator
from .geograph import GeoFeatureGenerator, CoordinatesInferFeatureGenerator
from .relation import RelationalFeatureGenerator
from .encode import OneHotFeatureGenerator, ListOneHotFeatureGenerator, TargetEncodeFeatureGenerator, LabelEncodeFeatureGenerator
from .feature_transform import ConvertToNumberFeatureGenerator

relation_builder_list = [
    RelationalFeatureGenerator
]

profiler_feature_generator_list = [
    ConvertToNumberFeatureGenerator,
    TypeConvertFeatureGenerator,
]

label_feature_generator_list = [
    RenameFeatureGenerator,
    TypeConvertFeatureGenerator,
    FillNaFeatureGenerator,
    LabelEncodeFeatureGenerator,
]

pre_feature_generator_list = [
    CoordinatesInferFeatureGenerator,
    ConvertToNumberFeatureGenerator,
    TypeConvertFeatureGenerator,
    FillNaFeatureGenerator,
    RenameFeatureGenerator,
]

transformation_generator_list = [
    DecodedTextFeatureGenerator,
    DatetimeFeatureGenerator,
    GeoFeatureGenerator,
    TextFeatureGenerator,
]

local_encode_generator_list = [
    OneHotFeatureGenerator,
    ListOneHotFeatureGenerator,
]

pre_enocode_feature_generator_list = [
    DropUselessFeatureGenerator,
]

global_dict_index_generator_list = [
    BinnedFeatureGenerator,
    CategoryFeatureGenerator,
    #TargetEncodeFeatureGenerator
]

post_feature_generator_list = [
    RenameFeatureGenerator
]

final_generator_list = [
    TypeCheckFeatureGenerator,
    DropUselessFeatureGenerator,
]