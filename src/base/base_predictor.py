from abc import abstractmethod

class BasePredictor:

    @abstractmethod
    def predict(self):
        raise NotImplementedError