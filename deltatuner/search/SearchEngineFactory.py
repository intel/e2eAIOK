from .RandomSearchEngine import RandomSearchEngine
from .EvolutionarySearchEngine import EvolutionarySearchEngine
from .SigoptSearchEngine import SigoptSearchEngine

SEARCHER_TYPES = {
        "RandomSearchEngine": RandomSearchEngine,
        "EvolutionarySearchEngine": EvolutionarySearchEngine,
        "SigoptSearchEngine": SigoptSearchEngine
}

class SearchEngineFactory(object):

    @staticmethod
    def create_search_engine(params, super_net, search_space,peft_type):
        try:
            if params.search_engine in SEARCHER_TYPES:
                return SEARCHER_TYPES[params.search_engine](params, super_net, search_space,peft_type)
            else:
                raise RuntimeError(f"Search Engine {params.search_engine} is not supported")
        except Exception as e:
            raise RuntimeError(f"Initialize {params.search_engine} for {params.domain} failed. Error Msg: {e}")