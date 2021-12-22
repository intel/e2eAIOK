import os
import sys
import pathlib
import_path = str(pathlib.Path(__file__).parent.absolute())
#print(import_path)
sys.path.append(import_path)
import findspark
findspark.init()
