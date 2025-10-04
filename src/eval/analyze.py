import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dummy import DummyModel
from models._ensemble import ModelEnsemble
from preprocessing.preproc_ensemble import preprocess_for_ensemble, postprocess_predictions


def analyze_file(ensemble, filepath, method='average'):
    """Analyze an audio file using an ensemble of models.
    
    This function:
    1. Loads the audio file through preprocessing
    2. Extracts metadata and audio data
    3. Runs the data through the ensemble using the specified method
    4. Postprocesses the predictions
    5. Returns the final predictions
    
    Args:
        ensemble: ModelEnsemble object containing the models to use
        filepath: Path to the audio file to analyze
        method: Ensembling method - 'average' or 'voting' (default: 'average')
        
    Returns:
        List of timestamps (in seconds) where events are detected
    """
    # Step 1: Load file and extract metadata and audio data
    metadata, audio_data = preprocess_for_ensemble(filepath)
    
    # Step 2: Run through ensemble with specified method
    if method == 'voting':
        predictions = ensemble.predict_with_voting(audio_data)
    else:
        predictions = ensemble.predict(audio_data)
    
    # Step 3: Postprocess predictions
    final_predictions = postprocess_predictions(predictions, metadata)
    
    return final_predictions


def create_default_ensemble():
    """Create a default ensemble of dummy models for testing.
    
    Returns:
        ModelEnsemble object with 2 dummy models
    """
    model1 = DummyModel(seed=42)
    model2 = DummyModel(seed=123)
    return ModelEnsemble([model1, model2])

