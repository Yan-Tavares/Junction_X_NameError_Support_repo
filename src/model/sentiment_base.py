"""
This module contains the base model class. Please use this as parent class when implementing new models and implement all the missing functions.
"""

class BaseSentimentModel:
    """
    Work in progress. 
    """

    def __init__(self, **kwargs):
        """
        This is a base class for the sentiment models.
        Each model should be wrapped in a class like this so it can be used in the ensemble.  
        Make sure to implement a version of 
        - __init__
        - predict
        """
        self.input_type = "text"
        raise NotImplementedError()


    def predict(self, split_text):
        """
        This method should obtain the model's predictions by calling the appropriate class methods.
        Input is a list of text samples.
        Output is an (n x m) array probabilities for "extremist", "non-extremist" or "potentially extremist".
        n is len(split_text)
        m is len(len(labels))
        """
        raise NotImplementedError()
    
