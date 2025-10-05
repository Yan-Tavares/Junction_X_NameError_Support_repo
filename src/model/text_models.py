import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model.sentiment_base import BaseSentimentModel, LABELS, to_prob3


class HateXplainModel(BaseSentimentModel):
    def __init__(self, model_path="models/hatexplain-roberta-mini", device=None):
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        self.device = 0 if (device in (0, "cuda") or torch.cuda.is_available()) else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.pipe = pipeline("text-classification", model=self.model,
                             tokenizer=self.tokenizer, device=self.device,
                             return_all_scores=True, truncation=True)

        # Ensure your fine-tuned head outputs 3 logits in LABELS order.
        self.idxs = [0,1,2]  # map model indices → LABELS order if needed

    def predict(self, texts):
        outs = self.pipe(texts)
        # outs: list of list[{'label':..., 'score':...}]
        P = []
        for o in outs:
            p = [0,0,0]
            for i, comp in enumerate(o):
                p[self.idxs[i]] = comp["score"]
            P.append(p)
        return np.asarray(P, dtype=np.float32)


class ToxicityModel(BaseSentimentModel):
    """
    Map toxicity → {extremist, potentially, non} with a conservative rule:
    high toxicity -> lift 'potentially'; extreme toxicity + extreme lexicon (from analyzer) -> 'extremist'.
    For pure text-only, we map by thresholds.
    """
    def __init__(self, model_name="unitary/toxic-bert", device=None):
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        self.device = 0 if (device in (0, "cuda") or torch.cuda.is_available()) else -1
        self.pipe = pipeline("text-classification", model=model_name, device=self.device, truncation=True)
    
    def predict(self, texts):
        # returns [{'label': 'toxic', 'score': p}] or multi-label depending on model; standardize:
        probs = []
        for t in texts:
            out = self.pipe(t)
            if isinstance(out, list): out = out[0]
            tox = float(out["score"]) if "score" in out else float(out.get("toxic", 0.0))
            # heuristic mapping:
            if tox >= 0.8:
                p = [0.45, 0.45, 0.10]
            elif tox >= 0.5:
                p = [0.20, 0.60, 0.20]
            elif tox >= 0.2:
                p = [0.07, 0.28, 0.65]
            else:
                p = [0.03, 0.12, 0.85]
            probs.append(p)
        return np.asarray(probs, dtype=np.float32)


class ZeroShotExtremismNLI(BaseSentimentModel):
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        self.device = 0 if (device in (0, "cuda") or torch.cuda.is_available()) else -1
        self.pipe = pipeline("zero-shot-classification", model=model_name, device=self.device)

        self.candidate_labels = [
            "the text supports or praises extremist ideology or organizations",
            "the text may suggest potential extremist leanings or borderline propaganda",
            "the text clearly does not support extremist ideology"
        ]
        # map to LABELS order:
        self.map_idx = [0,1,2]

    def predict(self, texts):
        res = self.pipe(texts, candidate_labels=self.candidate_labels, multi_label=False)
        if not isinstance(res, list): res = [res]
        P = []
        for r in res:
            # r['scores'] aligned to r['labels']
            order = [r["labels"].index(lbl) for lbl in self.candidate_labels]
            scores = np.asarray([r["scores"][i] for i in order], dtype=np.float32)
            scores /= scores.sum() + 1e-9
            P.append(scores)
        return np.asarray(P, dtype=np.float32)


class HeuristicLexiconModel(BaseSentimentModel):
    def __init__(self, lexicon=None):
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        # {pattern: weight}
        self.lex = lexicon or {
            r"\b(islamic state|isis|daesh|al[- ]qaeda|kkk|nazi|white power)\b": 1.0,
            r"\b(heil hitler|14/88|14 words|blood and soil)\b": 1.2,
            r"\b(caliphate|jihadist)\b": 0.6,
            r"\b(gas the [^\s]+|ethnic cleansing|race war)\b": 1.3,
        }

    def predict(self, texts):
        P = []
        for t in texts:
            t_low = t.lower()
            score = 0.0
            for pat, w in self.lex.items():
                if __import__("re").search(pat, t_low):
                    score += w
            if score >= 2.0:
                p = [0.65, 0.30, 0.05]
            elif score >= 1.0:
                p = [0.30, 0.55, 0.15]
            elif score >= 0.4:
                p = [0.10, 0.35, 0.55]
            else:
                p = [0.05, 0.10, 0.85]
            P.append(p)
        return np.asarray(P, dtype=np.float32)
