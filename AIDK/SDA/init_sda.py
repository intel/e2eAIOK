import os
import sys
import pathlib
current_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.remove(current_path)
import_path = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(import_path)
print(import_path)
