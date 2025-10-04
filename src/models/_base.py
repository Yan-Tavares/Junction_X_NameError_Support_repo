"""
This module contains the base model class. Please use this as parent class when implementing new models and implement all the missing functions.
"""

class BaseModel:
    """
    Work in progress. 
    """

    def __init__(self, dt=0.1, **kwargs):
        self.dt = dt
        ...

    def predict(self, inp):
        """This method should obtain the model's predictions"""
        raise NotImplementedError()

    def preprocess(self):
        """This method should convert the standardized inputs to the input format the model requires"""
        raise NotImplementedError()
    
    def postprocess(self):
        """This method should convert the model's outputs to a list of probabilities per interval self.dt"""
        raise NotImplementedError()
