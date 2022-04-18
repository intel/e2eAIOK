import os
import sqlite3
from AIDK.hydroai.hydromodel import *

def test_pipeline_model_happypath_exist():
    con = sqlite3.connect('/home/vmagent/app/hydro.ai/hydroai.db')
    for row in con.execute("SELECT hydro_model FROM models"):
        hydro_model = HydroModel(None, row)
        model_saved_path = hydro_model.model
        try:
            os.chdir(model_saved_path)
        except Exception as exc:
            assert False, f"raised an exception {exc}"
