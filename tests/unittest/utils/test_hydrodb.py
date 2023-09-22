from e2eAIOK.utils.hydroDB import *
from e2eAIOK.utils.hydromodel import *
from e2eAIOK.utils.hydroconfig import *

class TestHydroDB:
    settings = init_settings()
    hydro_model = HydroModel(settings)
    db = HydroDB(hydrodb_path='')

    def test_hydrodb(self):
        self.db.insert_model_by_id('dummy_id', self.hydro_model)
        out_hydro_model = self.db.get_model_by_id('dummy_id')
        assert(out_hydro_model != None)
        assert(out_hydro_model.to_json() == self.hydro_model.to_json())