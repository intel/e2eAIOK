import os
import sqlite3
from e2eAIOK.utils.hydromodel import *

def test_pipeline_model_happypath_exist():
    con = sqlite3.connect(f"{os.environ['CUSTOM_RESULT_PATH']}/e2eaiok.db")
    for row in con.execute("SELECT hydro_model FROM models"):
        hydro_model = HydroModel(None, row)
        model_saved_path = hydro_model.model
        try:
            os.chdir(model_saved_path)
        except Exception as exc:
            assert False, f"raised an exception {exc}"
