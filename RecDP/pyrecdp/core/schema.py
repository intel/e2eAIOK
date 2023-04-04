from woodwork.column_schema import ColumnSchema
from pandas import StringDtype
from pyrecdp.core.utils import is_text_series, is_tuple, is_encoded, is_integer_convertable
class TextDtype(StringDtype):
    pass

class SeriesSchema:
    def __init__(self, *args):
        if len(args) == 1:
            s = args[0]
            self.name = s.name
            in_type = s.dtype
            self.config = {}
            self.config['is_text'] = is_text_series(s)
            if self.config['is_text']:
                self.config['is_encoded'] = is_encoded(s)
            self.config['is_tuple'] = is_tuple(s)
            self.config['is_integer'] = is_integer_convertable(s)
        elif len(args) >= 2:
            # s_dtype is possible to be pandas.dtype or woodwork.dtype       
            self.name = args[0]
            # TODO: convert featuretools return_type to recdp return type
            in_type = args[1]
            if len(args) > 2:
                self.config = args[2]
            else:
                self.config = {}
            self.config['is_integer'] = False
        else:
            raise ValueError(f"SeriesSchema unsupport input as {args}")

        if in_type:
            # check all types
            from pandas.api.types import is_bool_dtype
            from pandas.api.types import is_string_dtype
            from pandas.api.types import is_numeric_dtype
            from pandas.api.types import is_float_dtype
            from pandas.api.types import is_integer_dtype
            from pandas.api.types import is_datetime64_any_dtype
            from pandas.api.types import is_categorical_dtype
            from pandas.api.types import is_object_dtype, is_list_like
            self.config['is_boolean'] = is_bool_dtype(in_type)
            self.config['is_string'] = is_string_dtype(in_type)
            self.config['is_numeric'] = is_numeric_dtype(in_type)
            self.config['is_integer'] = is_integer_dtype(in_type) or self.config['is_integer']
            self.config['is_float'] = is_float_dtype(in_type) if not self.config['is_integer'] else False
            self.config['is_datetime'] = is_datetime64_any_dtype(in_type)
            self.config['is_categorical'] = is_categorical_dtype(in_type)
            self.config['is_list'] = is_object_dtype(in_type) and is_list_like(in_type)

            if isinstance(in_type, ColumnSchema):
                self.config['is_categorical'] = in_type.is_categorical
                self.config['is_numeric'] = in_type.is_numeric or 'numeric' in str(in_type)
                self.config['is_categorical_string'] = in_type.is_categorical and not in_type.is_ordinal
                self.config['is_latlong'] = in_type.is_latlong

        self.post_fix()

    def post_fix(self):
        #post fix
        if 'is_datetime' in self.config and self.config['is_datetime']:
            self.config['is_text'] = False

    def copy_config_to(self, config):
        for k, v in self.config.items():
            if v:
                config[k] = v
        return config
    
    def copy_config_from(self, config):
        for k, v in config.items():
            if v is not False:
                self.config[k] = v
        self.post_fix()

    def mydump(self):
        return (self.name, list(k for k, v in self.config.items() if v is not False))

    def __repr__(self):
        return f"{self.mydump()}"
   
    @property
    def dtype_str(self):
        return str(dict((k, v) for k, v in self.config.items() if v))
         
    @property
    def is_boolean(self):
        return 'is_boolean' in self.config and self.config['is_boolean']

    @property
    def is_string(self):
        return 'is_string' in self.config and self.config['is_string']
    
    @property
    def is_numeric(self):
        return 'is_numeric' in self.config and self.config['is_numeric']
    
    @property
    def is_re_numeric(self):
        return 'is_re_numeric' in self.config and self.config['is_re_numeric']

    @property
    def is_float(self):
        return 'is_float' in self.config and self.config['is_float']

    @property
    def is_integer(self):
        return 'is_integer' in self.config and self.config['is_integer']

    @property
    def is_datetime(self):
        return 'is_datetime' in self.config and self.config['is_datetime']
    
    @property
    def is_categorical(self):
        return 'is_categorical' in self.config and self.config['is_categorical']
    
    @property
    def is_onehot(self):
        return 'is_onehot' in self.config and self.config['is_onehot'] is not False
    
    @property
    def is_list(self):
        return 'is_list' in self.config and self.config['is_list']
    
    @property
    def is_list_string(self):
        return 'is_list_string' in self.config and self.config['is_list_string'] is not False

    @property
    def is_categorical_and_string(self):
        if 'is_categorical_string' in self.config and self.config['is_categorical_string']:
            return True
        if 'is_categorical' in self.config and 'is_string' in self.config:
            return self.config['is_string'] and self.config['is_categorical']
        return False
    
    @property
    def is_coordinates(self):
        return 'is_latlong' in self.config and self.config['is_latlong']
    
    @property
    def is_encoded(self):
        return 'is_encoded' in self.config and self.config['is_encoded']

    @property
    def is_text(self):
        return 'is_text' in self.config and self.config["is_text"]
    

class DataFrameSchema(list):
    def __init__(self, df):
        for s_name in df.columns:
            s = df[s_name]
            super().append(SeriesSchema(s))

        
    