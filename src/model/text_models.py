import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os, sys
import yaml
import json

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
    def __init__(self, model_path="unitary/toxic-bert", device=None):
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        self.device = 0 if (device in (0, "cuda") or torch.cuda.is_available()) else -1
        self.pipe = pipeline("text-classification", model=model_path, device=self.device, truncation=True)
    
    def predict(self, texts):
        # returns [{'label': 'toxic', 'score': p}] or multi-label depending on model; standardize:
        probs = []
        for t in texts:
            out = self.pipe(t)
            if isinstance(out, list): out = out[0]
            tox = float(out["score"]) if "score" in out else float(out.get("toxic", 0.0))
            # heuristic mapping to [non_extremist, potentially_extremist, extremist]:
            if tox >= 0.8:
                p = [0.10, 0.45, 0.45]  # High toxicity -> extremist
            elif tox >= 0.5:
                p = [0.20, 0.60, 0.20]  # Moderate toxicity -> potentially
            elif tox >= 0.2:
                p = [0.65, 0.28, 0.07]  # Slight toxicity -> mostly non
            else:
                p = [0.85, 0.12, 0.03]  # Non-toxic -> non_extremist
            probs.append(p)
        return np.asarray(probs, dtype=np.float32)


class ZeroShotExtremismNLI(BaseSentimentModel):
    def __init__(self, model_path="facebook/bart-large-mnli", device=None):
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        self.device = 0 if (device in (0, "cuda") or torch.cuda.is_available()) else -1
        self.pipe = pipeline("zero-shot-classification", model=model_path, device=self.device)

        # Candidate labels must match LABELS order: [non_extremist, potentially_extremist, extremist]
        self.candidate_labels = [
            "the text clearly does not support extremist ideology",
            "the text may suggest potential extremist leanings or borderline propaganda",
            "the text supports or praises extremist ideology or organizations"
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
    def __init__(self, lexicon_file="definition/lexicon.yaml"):
        """
        Initialize the HeuristicLexiconModel with either a lexicon file or direct lexicon dict.
        
        Args:
            lexicon_file (str): Path to YAML file containing patterns and thresholds.
                               Relative paths are resolved from PROJECT_ROOT.
            lexicon (dict): Direct dictionary of {pattern: weight} for backwards compatibility.
                           Ignored if lexicon_file is provided.
        """
        super().__init__()
        self.input_type = "text"
        self.labels = LABELS
        
        # Load lexicon from file or use provided dictionary
        self._load_from_file(lexicon_file)
    
    def _load_from_file(self, lexicon_file):
        """Load lexicon patterns from YAML or JSON file."""
        if not os.path.isabs(lexicon_file):
            lexicon_file = os.path.join(PROJECT_ROOT, lexicon_file)
        
        # Determine file format and load
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Build pattern dict
        self.lex = {}
        for item in config.get('patterns', []):
            self.lex[item['pattern']] = item['weight']

    
    def calculate_probs(self, score):
        """Map lexicon score to probabilities using a continuous mapping.
        
        Args:
            score: The accumulated lexicon score
            
        Returns:
            Array of [non_extremist, potentially_extremist, extremist] probabilities
            
        The mapping works as follows:
        - non_extremist (safe): scales inversely with score (1 / (1 + score))
        - potentially_extremist: fixed at 1.0
        - extremist: scales directly with score
        All values are then normalized to sum to 1.0
        """
        # Element 0 (safe): inverse scaling with score
        # Using 1/(1+score) to get smooth decay from 1.0 to near 0
        safe = 1.0 / (score + 0.5)
        
        # Element 1 (potentially): fixed at 1.0
        potentially = 1.0
        
        # Element 2 (extremist): direct scaling with score
        extremist = -1.0 / (score - 2.5)
        
        # Normalize so they sum to 1.0
        total = safe + potentially + extremist
        if total > 0:
            return np.array([safe / total, potentially / total, extremist / total], dtype=np.float32)
        else:
            # Fallback if score is exactly 0
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)


    def predict(self, texts):
        """
        Returns probability distribution [non_extremist, potentially_extremist, extremist]
        for each input text based on lexicon pattern matching.
        """
        import re
        P = []
        for t in texts:
            t_low = t.lower()
            score = 0.0
            for pat, w in self.lex.items():
                if re.search(pat, t_low):
                    score += w
            
            # Map lexicon score to probabilities using continuous function
            p = self.calculate_probs(score)
            P.append(p)
        return np.asarray(P, dtype=np.float32)
