from search.RandomSearchEngine import RandomSearchEngine
from search.EvolutionarySearchEngine import EvolutionarySearchEngine
from search.SigoptSearchEngine import SigoptSearchEngine

SEARCHER_TYPES = {
        "RandomSearchEngine": RandomSearchEngine,
        "EvolutionarySearchEngine": EvolutionarySearchEngine,
        "SigoptSearchEngine": SigoptSearchEngine
}

class SearchEngineFactory(object):

    @staticmethod
    def create_search_engine(params, super_net, search_space):
        try:
            return SEARCHER_TYPES[params.search_engine](params, super_net, search_space)
        except Exception:
            return None