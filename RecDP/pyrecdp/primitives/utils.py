from woodwork.column_schema import ColumnSchema
from pandas import StringDtype
class TextDtype(StringDtype):
    pass

def is_text_series(s):
    from pandas.api import types as pdt
    import pandas as pd
    def try_text(s):
            if not pdt.is_string_dtype(s):
                return False
            if len(s) > 1000:
                # Sample to speed-up type inference
                result = s.sample(n=1000, random_state=0)
            try:
                avg_words = pd.Series(result).str.split().str.len().mean()
                if avg_words > 2:
                    return True
            except:
                return False
            return False
    
    type_change = False
    if not type_change:
        type_change = try_text(s)
        
    return type_change

class SeriesSchema:
    def __init__(self, *args):
        if len(args) == 1:
            s = args[0]
            self.name = s.name
            self.type = s.dtype
            self.actual_type = type(s.loc[s.first_valid_index()]) if s.first_valid_index() >= 0 else None
            self.is_text_flag = is_text_series(s)
        elif len(args) == 2:
            s_name = args[0]
            s_dtype = args[1]
            # s_dtype is possible to be pandas.dtype or woodwork.dtype       
            self.name = s_name
            self.type = s_dtype
            self.actual_type = None
            self.is_text_flag = isinstance(self.type, TextDtype)
        else:
            raise ValueError("SeriesSchema unsupport input arguments more than 2")
   
    @property
    def dtype_str(self):
        if isinstance(self.type, ColumnSchema):
            return self.type
        if self.is_string:
            return "string"
        if self.is_categorical_and_string:
            return "categorical string"
        return self.type.name
         
    @property
    def is_boolean(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_bool_dtype as check_func
        return check_func(self.type)

    @property
    def is_string(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_string_dtype as check_func
        return check_func(self.type)
    
    @property
    def is_numeric(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_numeric_dtype as check_func
        return check_func(self.type)

    @property
    def is_float(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_float_dtype as check_func
        return check_func(self.type)

    @property
    def is_int64(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_int64_dtype as check_func
        return check_func(self.type)

    @property
    def is_integer(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_integer_dtype as check_func
        return check_func(self.type)

    @property
    def is_datetime(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_datetime64_any_dtype as check_func
        return check_func(self.type)
    
    @property
    def is_categorical(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_categorical_dtype as check_func
        return check_func(self.type)
    
    @property
    def is_list(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_object_dtype, is_list_like
        return is_object_dtype(self.type) and is_list_like(self.type)
    
    @property
    def is_categorical_and_string(self):
        if isinstance(self.type, ColumnSchema):
            return False
        from pandas.api.types import is_categorical_dtype, is_string_dtype
        if not is_categorical_dtype(self.type):
            return False
        return is_string_dtype(self.type.categories)
    
    @property
    def is_coordinates(self):
        if self.actual_type is tuple:
            return True
        if not isinstance(self.type, ColumnSchema):
            return False
        return self.type.is_latlong
    
    @property
    def is_text(self):
        return self.is_text_flag
    

class DataFrameSchema(list):
    def __init__(self, df):
        for s_name in df.columns:
            s = df[s_name]
            super().append(SeriesSchema(s))

        
    