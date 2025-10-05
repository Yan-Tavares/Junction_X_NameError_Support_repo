"""
This module contains the base model class. Please use this as parent class when implementing new models and implement all the missing functions.
"""
import numpy as np

# CRITICAL: Label order must match ensemble expectations
# Ensemble expects: [safe, uncertain, extremist]
LABELS = ["non_extremist", "potentially_extremist", "extremist"]

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
        self.labels = LABELS
        self.input_type = "text"


    def predict(self, split_text):
        """
        This method should obtain the model's predictions by calling the appropriate class methods.
        Input is a list of text samples.
        Output is an (n x m) array probabilities for "extremist", "non-extremist" or "potentially extremist".
        n is len(split_text)
        m is len(len(labels))
        """
        raise NotImplementedError()
    

def to_prob3(x):
    """Ensure output is (n,3) probs."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1: x = x[None, :]
    # softmax if not already probabilities
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    p = ex / ex.sum(axis=1, keepdims=True)
    # If shape != 3 you must map externally.
    return p