import sys
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from faster_whisper import WhisperModel
from GPT_api import analyze_extremism
import json
import os

class VibeCheckerModel:
    """
    LLM-based extremism classifier that uses audio prosodic features.
    Compatible with ensemble.py architecture.
    """
    
    def __init__(self, device=None):
        """
        Initialize the Vibe Checker model with emotion detection.
        
        Args:
            device: Device to run the model on (cuda/cpu/mps). If None, auto-detects.
        """
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Vibe_Checker_model using device: {self.device}")
        
        # This model processes audio segments
        self.input_type = "audio"
        
        # Load emotion model for audio analysis
        MODEL_DIR = "./fine_tuned_emotion_model"
        self.emotion_model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR).to(self.device)
        self.emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
        
        print("✓ Vibe_Checker_model loaded successfully")
    
    def extract_prosodic_features(self, signal, sr=16000):
        """Extract prosodic features from audio signal."""
        features = {}
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=signal, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = round(np.mean(pitch_values), 2)
            features['pitch_std'] = round(np.std(pitch_values), 2)
            features['pitch_variation'] = round(features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0, 3)
        else:
            features['pitch_variation'] = 0
        
        # Energy features
        rms = librosa.feature.rms(y=signal)[0]
        features['energy_mean'] = round(np.mean(rms), 4)
        features['energy_std'] = round(np.std(rms), 4)
        features['energy_variation'] = round(features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0, 3)
        
        # Speaking rate
        zcr = librosa.feature.zero_crossing_rate(signal)[0]
        features['speaking_rate'] = round(np.mean(zcr), 3)
        
        return features
    
    def categorize_prosodic_features(self, features):
        """Convert numerical prosodic features to categorical labels."""
        categories = {}
        
        # Pitch expressiveness
        if features['pitch_variation'] > 0.3:
            categories['pitch'] = 'high'
        elif features['pitch_variation'] > 0.15:
            categories['pitch'] = 'medium'
        else:
            categories['pitch'] = 'low'
        
        # Emphasis
        if features['energy_variation'] > 0.5:
            categories['emphasis'] = 'high'
        elif features['energy_variation'] > 0.3:
            categories['emphasis'] = 'medium'
        else:
            categories['emphasis'] = 'low'
        
        # Pace
        if features['speaking_rate'] > 0.15:
            categories['pace'] = 'fast'
        elif features['speaking_rate'] > 0.08:
            categories['pace'] = 'moderate'
        else:
            categories['pace'] = 'slow'
        
        return categories
    
    def classify_emotion(self, signal, sr=16000):
        """Classify emotion from audio signal."""
        inputs = self.emotion_feature_extractor(
            signal,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000*5
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.emotion_model(**inputs).logits
            pred_id = torch.argmax(logits, dim=-1).item()
            intensity = torch.softmax(logits, dim=-1).max().item()
        
        emotion = self.emotion_model.config.id2label[pred_id]
        return emotion, intensity
    
    def predict(self, segments, audio_path):
        """
        Predict extremism scores for audio segments using LLM with prosodic context.
        
        Args:
            segments: List of segment dictionaries with 'start', 'end', 'text' keys
                     (from faster-whisper or similar transcription)
            audio_path: Path to the audio file (if not in segment dict)
        
        Returns:
            numpy.ndarray: Predictions of shape (n_segments, 3) 
                          [p_normal, p_offensive, p_extremist]
        """
        segments = list(segments)
        n_segments = len(segments)
        predictions = np.zeros((n_segments, 3))  # 3 labels: normal, offensive, extremist

        
        print(f"🎤 Processing {n_segments} segments from: {audio_path}")
        
        for i, segment in enumerate(segments):
            try:
                # Handle both dict and object (Segment) formats
                if isinstance(segment, dict):
                    start = segment.get('start', 0)
                    end = segment.get('end', start + 5)
                    text = segment.get('text', '')
                else:
                    # Segment object with attributes
                    start = getattr(segment, 'start', 0)
                    end = getattr(segment, 'end', start + 5)
                    text = getattr(segment, 'text', '')
                
                if not text:
                    # No text, assume normal
                    predictions[i] = [1.0, 0.0, 0.0]
                    continue
                
                # Load audio segment
                duration = end - start
                audio, sr = librosa.load(audio_path, sr=16000, offset=start, duration=duration)
                
                # Get emotion and prosodic features
                emotion, intensity = self.classify_emotion(audio, sr=sr)
                prosodic_features = self.extract_prosodic_features(audio, sr=sr)
                prosodic_categories = self.categorize_prosodic_features(prosodic_features)
                
                # Create augmented text
                augmented_text = (
                    f"{text} "
                    f"[emotion={emotion}, emotion confidence={intensity:.2f}, "
                    f"pitch={prosodic_categories['pitch']}, "
                    f"emphasis={prosodic_categories['emphasis']}, "
                    f"pace={prosodic_categories['pace']}]"
                )
                
                # Analyze with LLM
                llm_result = analyze_extremism(augmented_text)
                
                # Use LLM's probability distribution directly
                if llm_result.get('validation_status') == 'PASSED':
                    # LLM returns [p_safe, p_uncertain, p_extremist]
                    # We need [p_normal, p_offensive, p_extremist]
                    # Map: safe→normal, uncertain→offensive, extremist→extremist
                    predictions[i] = [
                        llm_result.get('p_safe', 0.7),      # p_normal
                        llm_result.get('p_uncertain', 0.2),  # p_offensive
                        llm_result.get('p_extremist', 0.1)   # p_extremist
                    ]
                else:
                    # LLM failed, return neutral prediction
                    print(f"⚠️ LLM failed for segment {i}: {llm_result.get('error', 'Unknown error')}")
                    predictions[i] = [0.7, 0.2, 0.1]  # Slightly cautious default
                    
            except Exception as e:
                print(f"⚠️ Error processing segment {i}: {e}")
                predictions[i] = [0.7, 0.2, 0.1]  # Default to mostly normal
        
        # Normalize to ensure probabilities sum to 1
        row_sums = predictions.sum(axis=1, keepdims=True)
        predictions = predictions / np.maximum(row_sums, 1e-10)
        
        return predictions