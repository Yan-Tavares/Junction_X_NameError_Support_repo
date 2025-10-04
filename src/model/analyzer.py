# analyzer.py (add/imports as needed)
import numpy as np, re
import librosa, soundfile as sf
from faster_whisper import WhisperModel
from sentiment_base import LABELS
from models_text import HateXplainModel, ToxicityModel, ZeroShotExtremismNLI, HeuristicLexiconModel
from vibechecker import VibeChecker
from ensemble import Ensemble

PAUSE_SEC = 0.8
MAX_UTT_SEC = 25.0
MAX_UTT_CHARS = 350

class UnifiedAnalyzer:
    def __init__(self, whisper_model_size="medium", text_model_path="models/hatexplain-roberta-mini", fast_mode=False):
        self.whisper = WhisperModel(whisper_model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
        self.sr = 16000
        # Models
        m1 = HateXplainModel(text_model_path)
        m2 = ToxicityModel()
        m3 = ZeroShotExtremismNLI()
        m4 = HeuristicLexiconModel()
        m5 = VibeChecker(sr=self.sr)
        self.models_text = [m1, m2, m3, m4]
        self.model_audio = m5
        self.ensemble = Ensemble(models=[m5, m1, m2, m3, m4])  # audio first, then texts

    def _transcribe(self, wav_path):
        segments, _ = self.whisper.transcribe(str(wav_path), word_timestamps=True, vad_filter=True)
        segs = []
        for s in segments:
            text = s.text.strip()
            if not text: continue
            segs.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": text
            })
        return segs

    def _merge_segments(self, segs):
        """ASR-aware merging into utterances."""
        merged = []
        if not segs: return merged
        cur = {"start": segs[0]["start"], "end": segs[0]["end"], "text": segs[0]["text"]}
        for s in segs[1:]:
            gap = s["start"] - cur["end"]
            new_text = (cur["text"] + " " + s["text"]).strip()
            too_long = (s["end"] - cur["start"]) > MAX_UTT_SEC or len(new_text) > MAX_UTT_CHARS
            boundary_punct = bool(re.search(r"[\.!\?]$", cur["text"]))
            if gap > PAUSE_SEC or too_long or boundary_punct:
                merged.append(cur)
                cur = {"start": s["start"], "end": s["end"], "text": s["text"]}
            else:
                cur["end"] = s["end"]
                cur["text"] = new_text
        merged.append(cur)
        return merged

    def _slice_audio(self, wav_path, utterances):
        y, sr = librosa.load(str(wav_path), sr=self.sr, mono=True)
        chunks = []
        for u in utterances:
            a = int(u["start"] * self.sr)
            b = int(u["end"] * self.sr)
            chunks.append(y[a:b].copy())
        return chunks

    def analyze(self, wav_path):
        segs = self._transcribe(wav_path)
        utts = self._merge_segments(segs)
        audio_chunks = self._slice_audio(wav_path, utts)
        texts = [u["text"] for u in utts]

        # Per-model predictions
        preds = {}
        preds["vibe"] = self.model_audio.predict(audio_chunks) if audio_chunks else np.zeros((0,3))
        preds["hatexplain"] = self.models_text[0].predict(texts) if texts else np.zeros((0,3))
        preds["toxicity"]   = self.models_text[1].predict(texts) if texts else np.zeros((0,3))
        preds["nli"]        = self.models_text[2].predict(texts) if texts else np.zeros((0,3))
        preds["lexicon"]    = self.models_text[3].predict(texts) if texts else np.zeros((0,3))

        # Ensemble (weighted average)
        # Reuse your Ensemble class if you want; here we do a simple average for clarity:
        stacked = np.stack([preds[k] for k in ["vibe","hatexplain","toxicity","nli","lexicon"]], axis=0)  # (K, N, 3)
        weights = np.array([0.5, 2.5, 1.0, 1.2, 0.8], dtype=np.float32)[:,None,None]
        ens = (stacked * weights).sum(axis=0) / (weights.sum(axis=0) + 1e-9)  # (N,3)

        # decision per utterance + aggregate
        ulabels = [LABELS[int(np.argmax(p))] for p in ens]
        uconfs  = [float(np.max(p)) for p in ens]
        final_label, final_conf = self._aggregate(ens)

        return {
            "labels": LABELS,
            "utterances": [
                {
                    "start": float(u["start"]),
                    "end": float(u["end"]),
                    "text": u["text"],
                    "probs": ens[i].tolist(),
                    "label": ulabels[i],
                    "confidence": uconfs[i],
                    "per_model": {k: preds[k][i].tolist() for k in preds}
                } for i, u in enumerate(utts)
            ],
            "final": {
                "label": final_label,
                "confidence": final_conf
            }
        }

    def _aggregate(self, ens_probs):
        # ens_probs: (N,3)
        p_ext = ens_probs[:,0]
        p_pot = ens_probs[:,1]
        mean_ext = float(p_ext.mean())
        max_ext = float(p_ext.max())
        n_ext06 = int((p_ext >= 0.6).sum())

        if max_ext >= 0.85 and n_ext06 >= 2:
            return "extremist", max_ext
        if (0.4 <= mean_ext < 0.85) or (n_ext06 >= 2) or (float(p_pot.mean()) >= 0.45):
            conf = max(mean_ext, float(p_pot.mean()))
            return "potentially_extremist", conf
        return "non_extremist", 1.0 - mean_ext
