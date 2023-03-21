import os, sys
from pathlib import Path
try:
    import pyspark
except:
    libpath = str(Path(__file__).parent.resolve())
    spark_home_dir = f"{libpath}/spark-3.2.1-bin-hadoop3.2"
    py4j_path = f"{spark_home_dir}/python/lib/py4j-0.10.9.3-src.zip"
    os.environ["SPARK_HOME"]=spark_home_dir
    # Add the PySpark classes to the Python path:
    os.environ["PYTHONPATH"]=os.environ["SPARK_HOME"]+"/python/:"+os.environ["PYTHONPATH"]+":"+py4j_path

os.environ["JAVA_HOME"]="/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["PYSPARK_PYTHON"]=sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"]=sys.executable
os.environ["PYSPARK_WORKER_PYTHON"]=sys.executable

import pyrecdp.primitives.spark_data_processor.utils as utils
sys.modules['pyrecdp.utils'] = utils
import pyrecdp.primitives.spark_data_processor.encoder as encoder
sys.modules['pyrecdp.encoder'] = encoder
import pyrecdp.primitives.spark_data_processor.data_processor as data_processor
sys.modules['pyrecdp.data_processor'] = data_processor
