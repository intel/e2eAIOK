from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.core import SeriesSchema
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
            Day(),
            Month(),
            Weekday(),
            Year(),
            Hour(),
            PartOfDay()
        ]
    
    def is_useful(self, pa_schema):
        found = False
        for pa_field in pa_schema:
            if pa_field.is_datetime:
                self.feature_in.append(pa_field.name)
                found = True
        return found