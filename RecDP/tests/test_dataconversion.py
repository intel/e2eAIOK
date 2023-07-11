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
from IPython.display import display
from pyrecdp.primitives.operations import Operation
from pyrecdp.core import SparkDataProcessor, utils


class TestConversion(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_parquet(f"{pathlib}/tests/data/test_frdtct.parquet")
        #self.df = pd.read_parquet(f"{pathlib}/../../applications/fraud_detect/raw_data/card_transaction.v1.parquet")
        self.rdp = SparkDataProcessor(spark_mode='local', spark_master="local[*]")

    def test_dataframe_to_RDD(self):
        from pyrecdp.primitives.operations.dataframe import DataFrameToRDDConverter
        _convert = DataFrameToRDDConverter().get_function(self.rdp)
        with utils.Timer("convert from DataFrame to spark RDD"):
            ret = _convert(self.df)
        display(ret)

    def test_dataframe_to_sparkDF(self):
        from pyrecdp.primitives.operations.dataframe import DataFrameToSparkDataFrameConverter
        _convert = DataFrameToSparkDataFrameConverter().get_function(self.rdp)
        with utils.Timer("convert from Dataframe to SparkDF"):
            ret = _convert(self.df)
            count_rt = ret.count()
        display(ret)
        print(count_rt)

    def test_RDD_to_sparkDF(self):
        from pyrecdp.primitives.operations.dataframe import DataFrameToRDDConverter
        from pyrecdp.primitives.operations.dataframe import RDDToSparkDataFrameConverter

        _convert = DataFrameToRDDConverter().get_function(self.rdp)
        with utils.Timer("convert from DataFrame to spark RDD"):
            ret = _convert(self.df)

        _convert = RDDToSparkDataFrameConverter().get_function(self.rdp)
        with utils.Timer("convert from SparkRDD to SparkDF"):
            ret1 = _convert(ret)
            count_rt = ret1.count()
        display(ret1)
        print(count_rt)