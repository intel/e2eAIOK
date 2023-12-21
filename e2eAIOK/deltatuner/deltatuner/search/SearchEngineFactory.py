"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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