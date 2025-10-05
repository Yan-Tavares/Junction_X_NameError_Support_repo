"""
Postprocessing module for Vocal Firewall.
Handles result assembly and formatting for API responses.
"""

import numpy as np


class ResultAssembler:
    """Assembles model predictions into API-compatible format."""
    
    # Label mapping: from model output indices to API labels
    # Model outputs: [normal, offensive, extremist]
    # API expects: ["non-hate", "uncertain", "hate"]
    LABEL_MAP = {
        0: "non-hate",    # normal -> non-hate
        1: "uncertain",   # offensive -> uncertain
        2: "hate"         # extremist -> hate
    }
    
    def __init__(self, label_map=None):
        """
        Initialize the result assembler.
        
        Args:
            label_map: Optional custom label mapping (dict)
        """
        self.label_map = label_map or self.LABEL_MAP
    
    def assemble_predictions(self, predictions, timestamps, text_samples, audio_path):
        """
        Assemble an array of predictions into a dictionary compatible with the API format.
        
        Args:
            predictions: numpy array of shape (n_segments, n_labels) where n_labels=3
                        [p_normal, p_offensive, p_extremist]
            timestamps: List of timestamps (start of each segment + end of last)
            text_samples: List of text strings for each segment
            audio_path: Path to the original audio file
        
        Returns:
            dict: Dictionary with keys:
                - audio_path: str
                - transcript: str
                - hate_spans: list of dicts with start, end, text, label, confidence
                - emotion_analysis: None (placeholder for future)
        """
        hate_spans = []
        
        for i, pred in enumerate(predictions):
            # Get start and end times for this segment
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            
            # Get text for this segment
            text = text_samples[i] if i < len(text_samples) else ""
            
            # Determine label (highest probability class)
            label_idx = int(np.argmax(pred))
            label = self.label_map[label_idx]
            
            # Confidence is the probability of the predicted class
            confidence = float(pred[label_idx])
            
            hate_spans.append({
                "start": float(start_time),
                "end": float(end_time),
                "text": text,
                "label": label,
                "confidence": confidence
            })
        
        # Build full transcript
        transcript = " ".join(text_samples)
        
        return {
            "audio_path": audio_path,
            "transcript": transcript,
            "hate_spans": hate_spans,
            "emotion_analysis": None  # Placeholder for future emotion analysis
        }
    
    def format_for_api(self, results):
        """
        Format results for API response (optional additional formatting).
        
        Args:
            results: Dict from assemble_predictions
            
        Returns:
            dict: Formatted results
        """
        # Currently just passes through, but can add additional formatting here
        return results


def ensemble_predictions(model_predictions, weights=None, bias=0.0):
    """
    Combine predictions from multiple models using weighted averaging.
    
    Args:
        model_predictions: numpy array of shape (n_models, n_segments, n_labels)
        weights: Optional weights for each model (default: equal weights)
        bias: Optional bias to add to ensemble predictions (default: 0.0)
        
    Returns:
        numpy.ndarray: Ensemble predictions of shape (n_segments, n_labels)
    """
    if weights is None:
        weights = np.ones(len(model_predictions))
    
    # Weighted average
    ensembled_preds = np.average(model_predictions, axis=0, weights=weights)
    ensembled_preds += bias
    
    # Ensure probabilities are non-negative and sum to 1
    ensembled_preds = np.maximum(0, ensembled_preds)
    ensembled_preds = ensembled_preds / (ensembled_preds.sum(axis=1, keepdims=True) + 1e-9)
    
    return ensembled_preds


def validate_predictions(predictions, expected_shape):
    """
    Validate model predictions have the expected shape.
    
    Args:
        predictions: numpy array of predictions
        expected_shape: Tuple of expected dimensions
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If shape doesn't match
    """
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Prediction shape mismatch: got {predictions.shape}, expected {expected_shape}"
        )
    return True
