import numpy as np
import torch
from faster_whisper import WhisperModel


class Ensemble:
    """Ensemble multiple models and combine their predictions."""
    

    def __init__(self, models, whisper_size="medium"):
        """Initialize the ensemble with a list of models.
        
        Args:
            models: List of model instances that have a predict method
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcriber = WhisperModel(
            whisper_size,
            device=self.device,
            compute_type=("float16" if torch.cuda.is_available() else "int8"),
            download_root=None
        )
        self.models = models

        # initialize ensemble weights and bias
        self.weights = np.ones(len(self.models))
        self.bias = 0.0

    
    def get_audio_from_file(self, filepath):
        """
        This function should load the audio data from a file. 
        Currently, it is used for all models in the population.
        """
        raise NotImplementedError()


    def split_audio(self, audio):
        """"
        This function should split the loaded audio into chunks, which are the same for all models. 
        """
        raise NotImplementedError()
    

    def convert_to_text(self, split_audio):
        """
        This function should use Whisper to convert the audio into text. 
        If possible, the timestamps of the text and audio fragments should match.
        """
        raise NotImplementedError()
    

    def predict(self, path_to_audio):
        """Ensemble predictions from all models.
        
        Args:
            audio_sample: Audio data to process
            
        Returns:
            List of timestamps after ensembling
        """
        audio = get_audio_from_file(path_to_audio)

        split_audio = split_audio(audio)
        split_text = get_text_from_audio(split_audio)
        assert len(split_text) == len(split_audio)

        predictions = np.empty((len(self.models), len(split_audio), len(self.labels)))
        
        # Collect predictions from all models
        for i, model in enumerate(self.models):

            if model.input_type == "audio":
                pred = model.predict(split_audio)

            elif model.input_type == "text":
                pred = model.predict(split_text)
            
            predictions[i] = pred

        ensembled_preds = np.average(predictions, axis=0, weights=self.weights)
        ensembled_preds += self.bias

        return ensembled_preds 
