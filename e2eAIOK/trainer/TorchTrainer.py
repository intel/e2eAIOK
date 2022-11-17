from abc import ABC, abstractmethod
import time
 
class BaseTrainer(ABC):
    """
    The basic trainer class for all models

    Note:
        You should implement specfic model trainer under model folder like vit_trainer
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    @abstractmethod
    def create_model(self):
        pass
    
    @abstractmethod
    def create_dataloader(self):
        pass    

    @abstractmethod
    def preparation(self):
        pass
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
    
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def post_preprocess(self):
        pass

    '''
    training all epochs interface
    '''
    @abstractmethod
    def fit(self, train_args):
        start_time = time.time()
        self.create_dataloader()
        self.create_model()
        self.preparation()
        if train_args.mode == "train":
            for i in range(train_args.train_epochs):
                train_start = time.time()
                self.train_one_epoch()
                if i % train_args.eval_epochs == 0:
                    eval_start = time.time()
                    self.evaluate()
                    print(F"Evaluate time:{time.time() - eval_start}")
                self.save_model()
                print(F"This epoch training time:{time.time() - train_start}")
            self.post_preprocess()
        else:
            eval_start = time.time()
            self.evaluate()
            print(F"Evaluate time:{time.time() - eval_start}")
        
                
        print(F"Total time:{time.time() - start_time}")