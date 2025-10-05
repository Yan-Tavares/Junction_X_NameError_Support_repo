import os, sys
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.model.ensemble import Ensemble
from src.model.dummy import DummyModel
from src.model.sentiment_base import BaseSentimentModel
from src.model.vibechecker import VibeCheckerModel
from src.model.sentiment_model import SentimentModel

model_options = [
    "DummyModel",
    "BaseSentimentModel", 
    "VibeCheckerModel",
    "SentimentModel"
]

def load_config(config_path="ensemble_config.yaml"):
    """
    Load YAML configuration file into a Python dictionary.
    
    Args:
        config_path (str, optional): Path to the config file. 
                                     If None, uses ensemble_config.yaml from project root.
    
    Returns:
        dict: Configuration dictionary loaded from YAML file.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_DIR, "ensemble_config.yaml")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def create_ensemble(config):

    model_list = []
    
    for model_key, model_config in config["models"].items():
        # Extract the class name from the config
        model_class = model_config.get("class")
        
        if model_class and model_class in model_options:
            model_params = model_config.get("args", {})
            # Unpack the dictionary as keyword arguments
            model_instance = eval(model_class)(**model_params)
            model_list.append(model_instance)
    
    ensemble = Ensemble(model_list)

    return ensemble
