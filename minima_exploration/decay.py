from abc import abstractmethod, ABCMeta

class LRDecay(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def decay(self, optimizer, epoch):
        pass
