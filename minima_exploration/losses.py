from abc import ABCMeta, abstractmethod

class LossClass(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def loss_fn(self):
        pass

    @abstractmethod
    def loss_params(self, x, y, y_pred):
        pass


