from abc import ABC, abstractmethod
 
class BaseModelBuilder(ABC):
    """
    The basic model builder class for all models

    Note:
        You should implement specfic model builder class under model folder like vit_model_builder
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    '''
    Init the model from supernet config
    '''
    @abstractmethod
    def init_model(self):
        pass

    '''
    Apply the best model structure to the already inited model and warp the model with torch DDP if need
    '''
    @abstractmethod
    def create_model(self):
        pass