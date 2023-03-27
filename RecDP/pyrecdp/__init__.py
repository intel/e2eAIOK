import os, sys
from pathlib import Path
try:
    import pyspark
except:
    raise NotImplementedError("pyrecdp required pyspark pre-installed. Please do pip install pyspark==3.3.1")

os.environ["JAVA_HOME"]="/usr/lib/jvm/java-8-openjdk-amd64/"
os.environ["PYSPARK_PYTHON"]=sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"]=sys.executable
os.environ["PYSPARK_WORKER_PYTHON"]=sys.executable

__all__ = ["data_processor", "encoder", "init_spark", "utils"]
