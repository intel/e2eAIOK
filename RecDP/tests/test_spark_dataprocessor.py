import unittest
import sys
import pandas as pd
from pathlib import Path
pathlib = str(Path(__file__).parent.parent.resolve())
print(pathlib)
try:
    import pyrecdp
except:
    print("Not detect system installed pyrecdp, using local one")
    sys.path.append(pathlib)

from pyrecdp.core import SparkDataProcessor
from pyrecdp.data_processor import Categorify
from pathlib import Path

class TestSparkDataProcessor(unittest.TestCase):
    def setUp(self):
        path_prefix = "file://"
        self.path = path_prefix + f"{pathlib}/tests/data/part-00008-3f7afb26-5b6e-4f4a-8a44-f439ddc4319f-c000.snappy.parquet"
    
    def test_local(self):
        self.rdp = SparkDataProcessor()
        self.test_categorify()
        del self.rdp
        
    def test_ray(self):
        self.rdp = SparkDataProcessor(spark_mode='ray')
        self.test_categorify()
        del self.rdp

    def test_categorify(self):
        proc = self.rdp
        path = self.path
        
        df = proc.spark.read.parquet(f"{path}")
        df = df.select("language")
        proc.reset_ops([Categorify(['language'])])
        ret_df = proc.apply(df).toPandas()
        self.assertEqual(ret_df.shape, (2193, 1))
        expected_desp = {'count': 2193.0, 'mean': 3.3461012311901506, 'std': 5.223049926213455, 'min': 0.0, '25%': 0.0, '50%': 1.0, '75%': 4.0, 'max': 39.0}
        actual_desp = ret_df['language'].describe().to_dict()
        #print(actual_desp)
        self.assertEqual(actual_desp, expected_desp)
        print("test_categorify ran successfully")