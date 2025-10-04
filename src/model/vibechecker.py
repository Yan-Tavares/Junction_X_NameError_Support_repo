# vibechecker.py
import numpy as np
import librosa

class VibeChecker:
    def __init__(self, sr=16000):
        self.input_type = "audio"
        self.sr = sr

    def _features(self, y):
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1: y = y.mean(axis=0)
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        try:
            f0, _, _ = librosa.pyin(y, fmin=80, fmax=500, sr=self.sr)
            pitch = np.nanmean(f0)
        except Exception:
            pitch = 0.0
        return rms, zcr, pitch

    def predict(self, audio_chunks):
        """
        audio_chunks: list of mono numpy arrays (already sliced per utterance) at self.sr
        Returns weak priors over 3 classes.
        """
        out = []
        for y in audio_chunks:
            rms, zcr, pitch = self._features(y)
            intensity = float(rms > 0.05) + float(zcr > 0.12) + float(pitch and pitch > 260)
            if intensity >= 2:
                p = [0.18, 0.54, 0.28]  # more 'potentially' when agitated
            elif intensity >= 1:
                p = [0.10, 0.32, 0.58]
            else:
                p = [0.06, 0.14, 0.80]
            out.append(p)
        return np.asarray(out, dtype=np.float32)
