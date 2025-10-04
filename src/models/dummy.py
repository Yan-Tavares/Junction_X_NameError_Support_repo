import random
import numpy as np


class DummyModel:
    """Dummy model that generates random timestamps for audio samples."""
    
    def __init__(self, seed=None):
        """Initialize the dummy model with an optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def predict(self, audio_sample):
        """Generate random timestamps for the given audio sample.
        
        Args:
            audio_sample: Audio data (can be file path, array, etc.)
            
        Returns:
            List of random timestamps in seconds
        """
        # Generate a random number of timestamps (between 3 and 10)
        num_timestamps = random.randint(3, 10)
        
        # Assume audio is between 0 and 60 seconds
        # Generate random timestamps and sort them
        timestamps = sorted([random.uniform(0, 60) for _ in range(num_timestamps)])
        
        return timestamps
    
    def predict_with_confidence(self, audio_sample):
        """Generate random timestamps with confidence scores.
        
        Args:
            audio_sample: Audio data (can be file path, array, etc.)
            
        Returns:
            List of tuples (timestamp, confidence_score)
        """
        timestamps = self.predict(audio_sample)
        
        # Add random confidence scores between 0.5 and 1.0
        timestamps_with_confidence = [
            (ts, random.uniform(0.5, 1.0)) for ts in timestamps
        ]
        
        return timestamps_with_confidence
