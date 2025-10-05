import numpy as np
import torch
from pathlib import Path

# Import pipeline modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.preprocessing import preprocess_audio
from pipeline.transcription import AudioTranscriber, extract_timestamps
from pipeline.postprocessing import ResultAssembler, ensemble_predictions, validate_predictions

class Ensemble:
    """Ensemble multiple models and combine their predictions."""
    

    def __init__(self, models, whisper_size="medium"):
        """Initialize the ensemble with a list of models.
        
        Args:
            models: List of model instances that have a predict method
            whisper_size: Size of Whisper model (default: "medium")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize transcriber using pipeline module
        self.transcriber = AudioTranscriber(model_size=whisper_size, device=self.device)
        
        # Initialize result assembler
        self.result_assembler = ResultAssembler()
        
        self.models = models

        # Initialize ensemble weights and bias
        self.weights = np.ones(len(self.models))
        self.bias = [0.0, 0.0, 0.0]

        # Label mapping (for backwards compatibility)
        self.label_map = self.result_assembler.label_map



    def predict(self, path_to_audio):
        """Ensemble predictions from all models.
        
        Args:
            path_to_audio: Path to audio file to process
            
        Returns:
            numpy.ndarray: Ensemble predictions (n_segments, n_labels)
        """
        self.audio_path = path_to_audio
        
        # Step 1: Preprocess audio using pipeline module
        audio_path, emphasized_audio = preprocess_audio(path_to_audio)
        
        # Step 2: Transcribe and get segments using pipeline module
        full_transcript, segments = self.transcriber.transcribe(audio_path)
        
        # Step 3: Extract timestamps and texts
        timestamps = extract_timestamps(segments)
        
        self.transcript = full_transcript
        self.text_samples = [seg["text"] for seg in segments]
        self.timestamps = timestamps
        self.segments = segments  # Store for audio models if needed

        if not self.text_samples:
            print("⚠️ No text segments found, returning empty predictions")
            return np.zeros((0, len(self.label_map)))

        # Step 4: Run each model
        model_predictions = np.empty((len(self.models), len(self.text_samples), len(self.label_map)))
        
        for i, model in enumerate(self.models):
            if model.input_type == "audio":
                # For audio models, pass segment info
                pred = model.predict(segments, audio_path)
            elif model.input_type == "text":
                # For text models, pass just the text
                pred = model.predict(self.text_samples)
            else:
                print(f"⚠️ Unknown input type for model {i}: {model.input_type}")
                pred = np.ones((len(self.text_samples), len(self.label_map))) / len(self.label_map)
            
            # Validate prediction shape using pipeline module
            expected_shape = (len(self.text_samples), len(self.label_map))
            try:
                validate_predictions(pred, expected_shape)
            except ValueError as e:
                print(f"⚠️ Model {i} returned shape {pred.shape}, expected {expected_shape}")
                print(f"   Model type: {model.__class__.__name__}, input_type: {model.input_type}")
                raise
            
            model_predictions[i] = pred

        # Step 5: Ensemble predictions using pipeline module
        ensembled_preds = ensemble_predictions(model_predictions, weights=self.weights, bias=self.bias)

        return ensembled_preds 


    def assemble_preds(self, preds):
        """
        Assembles an array of predictions (shape (n_segments, n_labels)) into a dictionary 
        compatible with the API format.
        
        Args:
            preds: numpy array of shape (n_segments, n_labels) where n_labels=3
                   [p_normal, p_offensive, p_extremist]
        
        Returns:
            dict: Dictionary with keys:
                - audio_path: str
                - transcript: str
                - hate_spans: list of dicts with start, end, text, label, confidence
                - emotion_analysis: None (placeholder for future)
        """
        # Use pipeline module for result assembly
        return self.result_assembler.assemble_predictions(
            predictions=preds,
            timestamps=self.timestamps,
            text_samples=self.text_samples,
            audio_path=self.audio_path
        )
