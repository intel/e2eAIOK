from .EvolutionarySearchEngine import EvolutionarySearchEngine

SEARCHER_TYPES = {
        "EvolutionarySearchEngine": EvolutionarySearchEngine
}

class SearchEngineFactory(object):

    @staticmethod
    def create_search_engine(params, super_net, search_space,peft_type):
        if params.search_engine in SEARCHER_TYPES:
            return SEARCHER_TYPES[params.search_engine](params, super_net, search_space,peft_type)
        else:
            raise RuntimeError(f"Search Engine {params.search_engine} is not supported")