import atheris
import sys
from e2eAIOK.SDA.SDA import SDA
from e2eAIOK.utils.hydromodel import HydroModel

@atheris.instrument_func
def fuzz_func(data):
  sda = SDA(model=data, hydro_model=HydroModel())

if __name__ == "__main__":
  atheris.Setup(sys.argv, fuzz_func)
  atheris.Fuzz()