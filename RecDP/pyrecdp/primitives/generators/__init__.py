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

from .binned import BinnedFeatureGenerator
from .category import CategoryFeatureGenerator
from .group_category import GroupCategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .drop import DropUselessFeatureGenerator
from .name import RenameFeatureGenerator
from .fillna import FillNaFeatureGenerator
from .type import TypeCheckFeatureGenerator,TypeConvertFeatureGenerator
from .nlp import DecodedTextFeatureGenerator, TextFeatureGenerator
from .geograph import GeoFeatureGenerator, CoordinatesInferFeatureGenerator
from .relation import RelationalFeatureGenerator
from .encode import OneHotFeatureGenerator, ListOneHotFeatureGenerator, TargetEncodeFeatureGenerator, LabelEncodeFeatureGenerator, CountEncodeFeatureGenerator
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
    #DropUselessFeatureGenerator,
]

global_dict_index_generator_list = [
    LabelEncodeFeatureGenerator,
    BinnedFeatureGenerator,
    GroupCategoryFeatureGenerator,
    CategoryFeatureGenerator,
    TargetEncodeFeatureGenerator,
    CountEncodeFeatureGenerator
]

post_feature_generator_list = [
    RenameFeatureGenerator
]

final_generator_list = [
    TypeCheckFeatureGenerator,
    DropUselessFeatureGenerator,
]