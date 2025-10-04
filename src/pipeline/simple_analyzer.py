# Simple fallback analyzer for testing
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any

class SimpleAnalyzer:
    """Simple fallback analyzer that just returns mock data"""
    
    def __init__(self, **kwargs):
        print("⚠️ Using simplified analyzer - complex models not available")
    
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Simple analysis that returns mock data for testing
        """
        return {
            "labels": ["extremist", "potentially_extremist", "non_extremist"],
            "utterances": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": "This is a test utterance from the audio file",
                    "probs": [0.1, 0.2, 0.7],
                    "label": "non_extremist",
                    "confidence": 0.7,
                    "per_model": {
                        "vibe": [0.1, 0.3, 0.6],
                        "hatexplain": [0.05, 0.15, 0.8],
                        "toxicity": [0.08, 0.22, 0.7],
                        "nli": [0.12, 0.18, 0.7],
                        "lexicon": [0.05, 0.1, 0.85]
                    }
                }
            ],
            "final": {
                "label": "non_extremist",
                "confidence": 0.7
            }
        }

# Make it available as UnifiedAnalyzer
UnifiedAnalyzer = SimpleAnalyzer