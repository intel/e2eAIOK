import os
import sys
import pathlib
import_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
sys.path.append(import_path)
print(import_path)
