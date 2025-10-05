# debug_analyzer.py - Fast minimal analyzer for debugging

import json
import random
from pathlib import Path
from typing import Dict, List
import re

class DebugAnalyzer:
    """Fast minimal analyzer that generates mock output for debugging"""
    
    def __init__(self, **kwargs):
        print("🚀 Debug Analyzer initialized (no models loaded)")
        self.labels = ["extremist", "potentially_extremist", "non_extremist"]
    
    def analyze(self, audio_path: str) -> Dict:
        """Generate mock analysis results quickly"""
        print(f"📋 Mock analyzing: {audio_path}")
        
        # Generate some fake transcript segments
        fake_segments = [
            {"start": 0.0, "end": 5.2, "text": "Hello everyone, welcome to today's discussion"},
            {"start": 5.5, "end": 12.3, "text": "We're going to talk about various political topics"},
            {"start": 12.8, "end": 18.9, "text": "It's important to have respectful dialogue"},
            {"start": 19.2, "end": 25.1, "text": "Different viewpoints help us understand issues better"},
            {"start": 25.5, "end": 32.0, "text": "Thank you for watching and please subscribe"}
        ]
        
        # Generate utterances with mock predictions
        utterances = []
        for i, seg in enumerate(fake_segments):
            # Mostly non-extremist with some variation
            probs = self._mock_prediction()
            label = self.labels[probs.index(max(probs))]
            
            utterances.append({
                "start": seg["start"],
                "end": seg["end"], 
                "text": seg["text"],
                "probs": probs,
                "label": label,
                "confidence": max(probs),
                "per_model": {
                    "vibe": [0.1, 0.2, 0.7],
                    "hatexplain": probs,
                    "toxicity": [0.05, 0.15, 0.8],
                    "nli": [0.08, 0.12, 0.8],
                    "lexicon": [0.02, 0.08, 0.9]
                }
            })
        
        # Determine final label
        avg_extremist = sum(u["probs"][0] for u in utterances) / len(utterances)
        if avg_extremist > 0.6:
            final_label = "extremist"
        elif avg_extremist > 0.3:
            final_label = "potentially_extremist"
        else:
            final_label = "non_extremist"
            
        return {
            "labels": self.labels,
            "utterances": utterances,
            "final": {
                "label": final_label,
                "confidence": 1.0 - avg_extremist if final_label == "non_extremist" else avg_extremist
            }
        }
    
    def _mock_prediction(self) -> List[float]:
        """Generate mock probability distribution"""
        # Bias towards non-extremist for realistic output
        base_probs = [0.05, 0.15, 0.80]  # [extremist, potentially, non]
        
        # Add some random variation
        noise = [random.uniform(-0.02, 0.02) for _ in range(3)]
        probs = [max(0.01, p + n) for p, n in zip(base_probs, noise)]
        
        # Normalize
        total = sum(probs)
        return [p/total for p in probs]

# Alias for compatibility
UnifiedAnalyzer = DebugAnalyzer