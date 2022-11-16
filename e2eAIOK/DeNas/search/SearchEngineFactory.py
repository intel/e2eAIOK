from search.RandomSearchEngine import RandomSearchEngine
from search.EvolutionarySearchEngine import EvolutionarySearchEngine
from search.SigoptSearchEngine import SigoptSearchEngine
from search.MOSigoptSearchEngine import MOSigoptSearchEngine

SEARCHER_TYPES = {
        "RandomSearchEngine": RandomSearchEngine,
        "EvolutionarySearchEngine": EvolutionarySearchEngine,
        "SigoptSearchEngine": SigoptSearchEngine,
        "MOSigoptSearchEngine": MOSigoptSearchEngine
}

class SearchEngineFactory(object):

    @staticmethod
    def create_search_engine(params, super_net, search_space):
        try:
            if params.search_engine in SEARCHER_TYPES:
                return SEARCHER_TYPES[params.search_engine](params, super_net, search_space)
            else:
                raise RuntimeError(f"Search Engine {params.search_engine} is not supported")
        except Exception as e:
            raise RuntimeError(f"Initialize {params.search_engine} for {params.domain} failed. Error Msg: {e}")