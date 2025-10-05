import os, sys
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.model.ensemble import Ensemble
from src.model.dummy import DummyModel
from src.model.text_models import HateXplainModel, ToxicityModel, ZeroShotExtremismNLI, HeuristicLexiconModel
from src.model.vibechecker import VibeCheckerModel

model_options = {
    "DummyModel": DummyModel,
    "VibeCheckerModel": VibeCheckerModel,
    "HateXplainModel": HateXplainModel,
    "ToxicityModel": ToxicityModel,
    "ZeroShotExtremismNLI": ZeroShotExtremismNLI,
    "HeuristicLexiconModel": HeuristicLexiconModel
}

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

    print("✅ Config loaded successfully")
    
    return config


def create_ensemble(config):

    model_list = []
    
    for model_key, model_config in config["models"].items():
        # Extract the class name from the config
        model_class = model_config.get("class")

        print(f"Loading model: {model_class}")
        
        if model_class and model_class in model_options:
            model_params = model_config.get("args", {})

            try:
                model_instance = model_options[model_class](**model_params)
            except Exception as e:
                print(f"❌ Failed to load {model_class}: {e}")
                continue

            model_list.append(model_instance)

            print(f"✅ {model_class} added to ensemble")
    
    ensemble = Ensemble(model_list)

    print("✅ Ensemble created successfully")

    return ensemble
