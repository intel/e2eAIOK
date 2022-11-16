from abc import ABC, abstractmethod
 
class BaseTrainer(ABC):
    """
    The basic trainer class for all models

    Note:
        You should implement specfic model trainer under model folder like vit_trainer
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    '''
    one epoch training function
    '''
    @abstractmethod
    def train_one_epoch(self):
        pass

    '''
    evluate the validation dataset during the training
    '''
    @abstractmethod
    def evaluate(self):
        pass

    '''
    training all epochs interface
    '''
    @abstractmethod
    def fit(self):
        pass