import init_hydro_for_test
from hydroai.hydroDB import *
from hydroai.hydromodel import *
from hydroai.hydroconfig import *

settings = init_settings()
hydro_model = HydroModel(settings)
db = HydroDB()

out_hydro_model = db.get_model_by_id('dummy_id')
if out_hydro_model:
    print("This test has been ran before, to run full test, please delete hydroai.db")

db.insert_model_by_id('dummy_id', hydro_model)
out_hydro_model = db.get_model_by_id('dummy_id')
assert(out_hydro_model != None)
assert(out_hydro_model.to_json() == hydro_model.to_json())