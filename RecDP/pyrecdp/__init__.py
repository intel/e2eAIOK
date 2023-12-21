"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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

if not os.environ.get('JAVA_HOME', None): 
    os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-8-openjdk-amd64/"
    print(f"JAVA_HOME is not set, use default value of /usr/lib/jvm/java-8-openjdk-amd64/")
os.environ["PYSPARK_PYTHON"]=sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"]=sys.executable
os.environ["PYSPARK_WORKER_PYTHON"]=sys.executable

import pyrecdp.primitives.spark_data_processor.utils as utils
sys.modules['pyrecdp.utils'] = utils
import pyrecdp.primitives.spark_data_processor.encoder as encoder
sys.modules['pyrecdp.encoder'] = encoder
import pyrecdp.primitives.spark_data_processor.data_processor as data_processor
sys.modules['pyrecdp.data_processor'] = data_processor
