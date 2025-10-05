
import numpy as np

class DummyModel:

    def __init__(self, seed):
        self.input_type = "audio"
        np.random.seed(seed)
        self.labels = ["non-extremist", "potentially extremist", "extremist"]

    def predict(self, audio_segments):
        
        prediction = np.random.random(size=(len(audio_segments), len(self.labels)))

        return prediction

