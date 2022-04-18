import pickle
import os
import pytest
import sqlite3
from AIDK.hydroai.hydromodel import *
from example.sklearn_train import *

def test_pipeline_sklearn_train_accuracy():
    con = sqlite3.connect('/home/vmagent/app/hydro.ai/hydroai.db')
    for row in con.execute("SELECT hydro_model FROM models"):
            hydro_model = HydroModel(None, row)
            model_saved_path = hydro_model.model
            hydro_metrics = hydro_model.metrics
            accuracy_metric = hydro_metrics[0]
    model_path = os.path.join(model_saved_path, "saved_dictionary.pkl")
    with open(model_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    mean_accuracy = evaluate_xgboost_model(**loaded_dict)
    assert mean_accuracy == pytest.approx(accuracy_metric["value"])