# vibechecker.py
import numpy as np
import librosa

class VibeCheckerModel:
    def __init__(self):
        self.input_type = "audio"


    def predict(self, audio_chunks):
        """
        To be implemented later
        """
        return np.zeros((len(audio_chunks), 3), dtype=np.float32)
