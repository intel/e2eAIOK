import atheris
import sys
from easydict import EasyDict as edict
from e2eAIOK.DeNas.search.SearchEngineFactory import SearchEngineFactory

@atheris.instrument_func
def fuzz_func(data):
  params = {}
  params["domain"] = "unknown"
  params["search_engine"] = data
  params = edict(params)
  try:
    searcher = SearchEngineFactory.create_search_engine(params, None, None)
  except Exception as e:
    print(e)

if __name__ == "__main__":
  atheris.Setup(sys.argv, fuzz_func)
  atheris.Fuzz()