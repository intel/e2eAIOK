from abc import ABC, abstractmethod
 
class BaseSearchEngine(ABC):

    def __init__(self, params=None, super_net=None, search_space=None):
        super().__init__()
        self.super_net = super_net
        self.search_space = search_space
        self.params = params
    
    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def get_best_structures(self):
        pass