from .base import BaseFeatureGenerator as super_class
from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
from featuretools.primitives import Haversine

class GeoFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_list = [Haversine]
        self.op_name = 'haversine'

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if pa_field.is_coordinates:
                self.feature_in.append(pa_field.name)
                is_useful = True
        if len(self.feature_in) == 2:
            # we assume we can calculate distance between them
            out_feat_name = f"haversine_{'_'.join(self.feature_in)}"
            op = self.op_list[0]()
            out_feat_type = op.return_type
            out_schema = SeriesSchema(out_feat_name, out_feat_type)
            self.feature_in_out_map[str(self.feature_in)] = (out_schema.name, self.op_list[0])
            pa_schema.append(out_schema)

        if is_useful:
            cur_idx = max_idx + 1
            config = self.feature_in_out_map
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'haversine', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx


class Point:
    def __init__(self, prefix = None, longitude = None, latitude = None):
        self.feature_name_prefix = prefix
        self.longitude = longitude
        self.latitude = latitude
        
    def __eq__(self, other):
        return self.feature_name_prefix == other.feature_name_prefix
    
    def update(self, other):
        assert(self == other)
        self.longitude = other.longitude if other.longitude else self.longitude
        self.latitude = other.latitude if other.latitude else self.latitude

    def get_feature_name(self):
        return f"{self.feature_name_prefix}_coordinates"
        
    def get_feature_type(self):
        from woodwork.column_schema import ColumnSchema
        from woodwork.logical_types import LatLong
        return ColumnSchema(logical_type=LatLong)
    
    def get_config(self):
        return {'src':[self.latitude, self.longitude], 'dst': self.get_feature_name()}
 

class CoordinatesInferFeatureGenerator(super_class):        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = []

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        for field in pa_schema:
            if self.is_coor_related_name_and_set(field.name):
                is_useful = True
        for p in self.points:
            out_schema = SeriesSchema(p.get_feature_name(), p.get_feature_type()) 
            pa_schema.append(out_schema)
        if is_useful:
            cur_idx = max_idx
            child = children[0]
            for p in self.points:
                cur_idx = cur_idx + 1
                config = p.get_config()
                pipeline[cur_idx] = Operation(cur_idx, [child], pa_schema, op = 'tuple', config = config)
                child = cur_idx
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx
    
    def is_coor_related_name_and_set(self, f_name):
            coor_related_names = ["longitude", "latitude", "coordinates", "point", "latlong", "longlat"]
            def get_prefix(in_name, remove_fix):
                return in_name.lower().replace(remove_fix, "").replace("_", "")
                
            for to_detect in coor_related_names:
                if to_detect in f_name.lower():
                    if to_detect == "longitude":
                        point = Point(longitude = f_name, prefix = get_prefix(f_name, to_detect))
                    elif to_detect == "latitude":
                        point = Point(latitude = f_name, prefix = get_prefix(f_name, to_detect))
                    else:
                        continue
                    
                    update_inline = False
                    for exist_point in self.points:
                        if exist_point == point:
                            exist_point.update(point)
                            update_inline = True
                    if not update_inline:
                        self.points.append(point)
                    return True
                    
            return False