import os
import sys
import pathlib
import_path = "/home/vmagent/app/recdp"
model_import_path = str(pathlib.Path(__file__).parent.absolute())
print(import_path, model_import_path)
sys.path.append(import_path)
sys.path.append(model_import_path)
