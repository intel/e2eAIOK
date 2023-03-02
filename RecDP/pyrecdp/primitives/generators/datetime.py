from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from featuretools.primitives import (
    Day,
    Month,
    Weekday,
    Year,
    Hour,
    PartOfDay
)

class DatetimeFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_list = [
            Day,
            Month,
            Weekday,
            Year,
            Hour,
            #PartOfDay()
        ]
        self.op_name = 'datetime_feature'            

    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if pa_field.is_datetime:
                self.feature_in.append(pa_field.name)
        return super().fit_prepare(pipeline, children, max_idx)
        