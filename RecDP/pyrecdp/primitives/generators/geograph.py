from .base import BaseFeatureGenerator as super_class
from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.primitives.utils import SeriesSchema, DataFrameSchema

class GeoFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_list = []

    def is_useful(self, pa_schema):
        found = False
        for pa_field in pa_schema:
            if pa_field.is_coordinates:
                self.feature_in.append(pa_field.name)
                found = True
        return found
    
    def fit_prepare(self, pa_schema):
        if len(self.feature_in) == 2:
            # we assume we can calculate distance between them
            from featuretools.primitives import Haversine
            out_feat_name = f"haversine_{'_'.join(self.feature_in)}"
            op = Haversine()
            out_feat_type = op.return_type
            self.feature_in_out_map[str(self.feature_in)] = (SeriesSchema(out_feat_name, out_feat_type), op)

        return pa_schema

    def get_function_pd(self):
        def generate_ft_feature(df):
            for inputs_str, op in self.feature_in_out_map.items():
                inputs = eval(inputs_str)
                df[op[0].name] = op[1](df[inputs[0]], df[inputs[1]])
            return df
        return generate_ft_feature

class Point:
    def __init__(self, point = None, prefix = None, longitude = None, latitude = None):
        self.point = point
        self.feature_name_prefix = prefix
        self.longitude = longitude
        self.latitude = latitude
        
    def __eq__(self, other):
        return self.feature_name_prefix == other.feature_name_prefix
    
    def update(self, other):
        assert(self == other)
        self.point = other.point if other.point else self.point
        self.longitude = other.longitude if other.longitude else self.longitude
        self.latitude = other.latitude if other.latitude else self.latitude

    def get_feature_name(self):
        if self.point:
            return self.feature_name_prefix
        else:
            return f"{self.feature_name_prefix}_coordinates"
        
    def get_feature_type(self):
        from woodwork.column_schema import ColumnSchema
        from woodwork.logical_types import LatLong
        return ColumnSchema(logical_type=LatLong)
        
    def get_function(self):
        if self.point:
            def process(df):
                # convert string to latlong
                sch = SeriesSchema(df[self.point])
                if sch.is_coordinates:
                    return df
                elif sch.is_string:
                    def convert_to_point(x):
                        import re
                        ret = re.findall(r'\d+', x)
                        if not ret or len(ret) < 2:
                            return (-1, -1)
                        else:
                            return (ret[0], ret[1])
                    df[self.get_feature_name()] = df[self.point].apply(convert_to_point)
                return df
        else:
            def process(df):
                df[self.get_feature_name()] = df[[self.latitude, self.longitude]].apply(tuple, axis=1)
                return df
        return process

class CoordinatesInferFeatureGenerator(super_class):        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []
   
    def is_useful(self, pa_schema):
        for field in pa_schema:
            if self.is_coor_related_name_and_set(field.name):
                return True
        return False
    
    def fit_prepare(self, pa_schema):
        for field in pa_schema:
            self.is_coor_related_name_and_set(field.name)
        for p in self.points:
            pa_schema.append(SeriesSchema(p.get_feature_name(), p.get_feature_type()))
        return pa_schema

    def get_function_pd(self):
        def type_infer(df):            
            for p in self.points:
                df = p.get_function()(df)
            return df
        return type_infer

    def fit_transform(self, df):
        if len(self.points) == 0:
            self.fit_prepare(DataFrameSchema(df))
        return self.get_function_pd()(df)
    
    def is_coor_related_name_and_set(self, f_name):
            coor_related_names = ["longitude", "latitude", "coordinates", "point", "latlong", "longlat"]
            def get_prefix(in_name, remove_fix):
                return in_name.lower().replace(remove_fix, "").replace("_", "")
                
            for to_detect in coor_related_names:
                if to_detect in f_name.lower():
                    if to_detect == "coordinates" or to_detect == "point" or to_detect == "latlong" or to_detect == "longlat":
                        point = Point(point = f_name, prefix = f_name)
                    elif to_detect == "longitude":
                        point = Point(longitude = f_name, prefix = get_prefix(f_name, to_detect))
                    elif to_detect == "latitude":
                        point = Point(latitude = f_name, prefix = get_prefix(f_name, to_detect))
                    
                    update_inline = False
                    for exist_point in self.points:
                        if exist_point == point:
                            exist_point.update(point)
                            update_inline = True
                    if not update_inline:
                        self.points.append(point)
                    return True
                    
            return False