import findspark
import os
import sys
import pathlib
import_path = str(pathlib.Path(
    __file__).parent.parent.parent.parent.absolute())
# print(import_path)
sys.path.append(import_path)

findspark.init()
