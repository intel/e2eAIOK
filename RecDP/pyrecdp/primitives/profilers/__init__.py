from .type_infer import TypeInferFeatureGenerator
from .distribution_infer import DistributionInferFeatureProfiler
from .statics import StatisticsFeatureGenerator
from .time_series_infer import TimeSeriesInferFeatureProfiler

feature_infer_list = [
    TypeInferFeatureGenerator,
    TimeSeriesInferFeatureProfiler,
]