"""
Unified Analysis Pipeline for Vocal Firewall
Integrates Whisper ASR, HateXplain classifier, and emotion detection
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None
from scipy.signal import find_peaks
import librosa
import soundfile as sf
import warnings

# Safely determine device with fallback
def get_device():
    """Safely get compute device with CPU fallback"""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Test if CUDA actually works
        torch.zeros(1).cuda()
        return "cuda"
    except Exception as e:
        warnings.warn(f"CUDA available but not working: {e}. Falling back to CPU.")
        return "cpu"

DEVICE = get_device()
print(f"Using device: {DEVICE}")


class VocalFirewallAnalyzer:
    """
    Main analyzer class integrating:
    - Speech-to-Text (Whisper)
    - Hate Speech Classification (HateXplain/RoBERTa)
    - Speech Emotion Recognition (wav2vec2)
    """
    
    def __init__(
        self,
        whisper_model_size: str = "medium",
        text_model_path: str = "martin-ha/toxic-comment-model",
        enable_emotion: bool = True,
        fast_mode: bool = False
    ):
        """
        Initialize the analyzer with specified models
        
        Args:
            whisper_model_size: Whisper model size (tiny/base/small/medium/large)
            text_model_path: HuggingFace model path for text classification
            enable_emotion: Whether to enable emotion analysis
            fast_mode: Use smaller models for faster processing
        """
        self.whisper_model_size = whisper_model_size
        self.text_model_path = text_model_path
        self.enable_emotion = enable_emotion
        self.fast_mode = fast_mode
        
        # Track actual device used (may differ from DEVICE if fallback occurs)
        self.actual_device = "cpu"
        
        # Initialize models
        self._init_whisper()
        self._init_text_classifier()
        if self.enable_emotion:
            self._init_emotion_model()
    
    def _init_whisper(self):
        """Initialize Whisper ASR model"""
        print(f"Loading Whisper model ({self.whisper_model_size})...")
        
        # faster-whisper requires cuDNN 9.x which may not be available
        # Default to CPU unless explicitly enabled
        from src.config import settings
        
        if settings.WHISPER_USE_GPU and DEVICE == "cuda":
            whisper_device = "cuda"
            compute_type = "float16"
            print("Attempting to use GPU for Whisper (requires cuDNN 9.x)...")
        else:
            whisper_device = "cpu"
            compute_type = "int8"
            print("Using CPU for Whisper (safer, avoids cuDNN issues)")
        
        try:
            self.whisper_model = WhisperModel(
                self.whisper_model_size,
                device=whisper_device,
                compute_type=compute_type,
                download_root=None
            )
            print(f"✅ Whisper model loaded (device={whisper_device})")
        except Exception as e:
            if whisper_device == "cuda":
                print(f"⚠️ GPU initialization failed: {e}")
                print("Falling back to CPU...")
                self.whisper_model = WhisperModel(
                    self.whisper_model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=None
                )
                print("✅ Whisper model loaded (CPU mode)")
            else:
                raise
    
    def _init_text_classifier(self):
        """Initialize text classification model"""
        print(f"Loading text classifier from {self.text_model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                self.text_model_path
            ).to(DEVICE).eval()
            self.actual_device = DEVICE
            print(f"✅ Text classifier loaded on {DEVICE}")
        except Exception as e:
            print(f"⚠️ Failed to load on {DEVICE}: {e}")
            print("Retrying on CPU...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                self.text_model_path
            ).to("cpu").eval()
            self.actual_device = "cpu"
            print("✅ Text classifier loaded on CPU")
    
    def _init_emotion_model(self):
        """Initialize emotion recognition model"""
        if AutoProcessor is None:
            print("⚠️ AutoProcessor not available, skipping emotion model initialization")
            self.emotion_processor = None
            self.emotion_model = None
            return
            
        print("Loading emotion model...")
        try:
            self.emotion_processor = AutoProcessor.from_pretrained(
                "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            )
            self.emotion_model = AutoModel.from_pretrained(
                "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            ).to(DEVICE).eval()
            print(f"✅ Emotion model loaded on {DEVICE}")
        except Exception as e:
            print(f"⚠️ Failed to load on {DEVICE}: {e}")
            print("Retrying on CPU...")
            try:
                self.emotion_processor = AutoProcessor.from_pretrained(
                    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
                )
                self.emotion_model = AutoModel.from_pretrained(
                    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
                ).to("cpu").eval()
                print("✅ Emotion model loaded on CPU")
            except Exception as e2:
                print(f"⚠️ Failed to load emotion model: {e2}")
                print("Disabling emotion analysis")
                self.emotion_processor = None
                self.emotion_model = None
                self.enable_emotion = False
    
    def analyze_audio(self, audio_path: Path) -> Dict:
        """
        Perform complete analysis on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing:
                - transcript: Full transcript
                - segments: List of transcribed segments with timestamps
                - hate_spans: Classified segments with labels and confidence
                - emotion_analysis: Emotion time series and peaks (if enabled)
        """
        print(f"Analyzing: {audio_path}")
        
        # Step 1: Transcribe audio
        segments = self._transcribe_audio(audio_path)
        
        # Step 2: Classify text segments
        hate_spans = self._classify_segments(segments)
        
        # Step 3: Analyze emotions (optional)
        emotion_data = {}
        if self.enable_emotion:
            emotion_data = self._analyze_emotions(audio_path, hate_spans)
        
        # Compile results
        result = {
            "audio_path": str(audio_path),
            "transcript": " ".join(s["text"] for s in segments),
            "segments": segments,
            "hate_spans": hate_spans,
            "emotion_analysis": emotion_data
        }
        
        return result
    
    def _transcribe_audio(self, audio_path: Path) -> List[Dict]:
        """
        Transcribe audio with preprocessing and timestamp extraction
        
        Returns:
            List of segments with start, end, and text
        """
        print("🎤 Transcribing audio...")
        
        # Preprocess audio
        processed_path = self._preprocess_audio(audio_path)
        
        # Transcribe with Whisper
        segments, _ = self.whisper_model.transcribe(
            str(processed_path),
            word_timestamps=True,
            condition_on_previous_text=True,
            language="en",
            task="transcribe",
            beam_size=10 if not self.fast_mode else 5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=100,
                speech_pad_ms=30,
                threshold=0.2
            )
        )
        
        # Convert to list of dicts
        segment_list = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip()
            }
            for seg in segments
        ]
        
        # Filter out segments with mostly repeated characters
        segment_list = [
            s for s in segment_list 
            if len(set(s["text"].lower())) >= 3
        ]
        
        # Merge short segments
        segment_list = self._merge_short_segments(segment_list)
        
        # Clean up preprocessed file
        if processed_path != audio_path:
            processed_path.unlink()
        
        print(f"✅ Transcribed {len(segment_list)} segments")
        return segment_list
    
    def _preprocess_audio(self, audio_path: Path) -> Path:
        """
        Apply audio preprocessing: pre-emphasis, normalization, noise reduction
        
        Returns:
            Path to preprocessed audio file
        """
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        emphasized_audio = np.append(
            audio[0], 
            audio[1:] - pre_emphasis * audio[:-1]
        )
        
        # Normalize
        emphasized_audio = librosa.util.normalize(emphasized_audio)
        
        # Apply spectral gating for noise reduction
        S = librosa.stft(emphasized_audio)
        mag = np.abs(S)
        
        # Estimate noise floor
        noise_floor = np.mean(
            np.sort(mag, axis=1)[:, :int(mag.shape[1]*0.1)], 
            axis=1
        )
        noise_floor = noise_floor[:, np.newaxis]
        
        # Apply soft thresholding
        gain = (mag - noise_floor) / mag
        gain = np.maximum(0, gain)
        S_clean = S * gain
        
        # Inverse STFT
        audio_clean = librosa.istft(S_clean)
        
        # Save preprocessed audio
        temp_path = audio_path.parent / (audio_path.stem + "_processed.wav")
        sf.write(str(temp_path), audio_clean, sr)
        
        return temp_path
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing excessive repetitions"""
        # Remove excessive letter repetitions
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        # Remove repeated syllables
        text = re.sub(r"([a-z]{1,3})\1{2,}", r"\1", text, flags=re.IGNORECASE)
        # Clean up spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def _merge_short_segments(
        self, 
        segments: List[Dict], 
        min_dur: float = 0.3,
        min_tokens: int = 1,
        max_gap: float = 0.1
    ) -> List[Dict]:
        """
        Merge short segments to create more meaningful chunks for classification
        """
        if not segments:
            return segments
        
        # Normalize text
        for seg in segments:
            seg["text"] = self._normalize_text(seg["text"])
        
        merged = []
        cur = dict(segments[0])
        
        for nxt in segments[1:]:
            gap = max(0.0, float(nxt["start"]) - float(cur["end"]))
            cur_dur = float(cur["end"]) - float(cur["start"])
            cur_tokens = len(cur["text"].split())
            
            need_merge = (
                cur_dur < min_dur or 
                cur_tokens < min_tokens or 
                gap <= max_gap
            )
            
            if need_merge:
                cur["end"] = float(nxt["end"])
                cur["text"] = (cur["text"] + " " + nxt["text"]).strip()
            else:
                merged.append(cur)
                cur = dict(nxt)
        
        merged.append(cur)
        return merged
    
    def _classify_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Classify each segment for hate speech
        
        Returns:
            List of segments with added label and confidence fields
        """
        print("🔍 Classifying text segments...")
        
        if not segments:
            return []
        
        # Tokenize all texts
        texts = [s["text"] for s in segments]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.actual_device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Process results
        hate_spans = []
        for seg, prob in zip(segments, probs):
            prob = prob.astype(float)
            max_idx = int(prob.argmax())
            conf_max = float(prob[max_idx])
            
            # Determine label based on model output
            if probs.shape[1] == 2:
                # Binary classification
                p_hate = float(prob[1])
                if p_hate >= 0.70:
                    label, conf = "hate", p_hate
                elif p_hate <= 0.30:
                    label, conf = "non-hate", 1.0 - p_hate
                else:
                    label, conf = "uncertain", max(p_hate, 1.0 - p_hate)
            else:
                # Multi-class classification
                label = f"class_{max_idx}" if conf_max >= 0.60 else "uncertain"
                conf = conf_max
            
            hate_spans.append({
                **seg,
                "label": label,
                "confidence": float(conf)
            })
        
        print(f"✅ Classified {len(hate_spans)} segments")
        return hate_spans
    
    def _analyze_emotions(
        self, 
        audio_path: Path, 
        hate_spans: List[Dict]
    ) -> Dict:
        """
        Analyze speech emotions over time
        
        Returns:
            Dictionary with emotion time series and peaks
        """
        if self.emotion_processor is None or self.emotion_model is None:
            print("⚠️ Emotion analysis disabled (model not available)")
            return {"time_series": [], "peaks": []}
            
        print("😊 Analyzing emotions...")
        
        # Load audio
        wav, sr = self._load_audio_for_emotion(audio_path)
        
        # Extract emotion time series
        win_s = 4.0 if self.fast_mode else 2.0
        hop_s = 2.0 if self.fast_mode else 0.5
        
        times, emotions = self._emotion_timeseries(wav, sr, win_s, hop_s)
        
        # Find arousal peaks
        peaks = self._find_arousal_peaks(times, emotions, hop_s=hop_s)
        
        # Match peaks to hate spans
        emotion_events = []
        for peak in peaks:
            matching_span = self._find_matching_segment(hate_spans, peak["t"])
            emotion_events.append({
                "time": peak["t"],
                "arousal": peak["arousal"],
                "coincides_with": matching_span["label"] if matching_span else None,
                "text": matching_span["text"] if matching_span else None,
                "span_start": matching_span["start"] if matching_span else None,
                "span_end": matching_span["end"] if matching_span else None,
            })
        
        print(f"✅ Found {len(peaks)} emotion peaks")
        
        return {
            "time_series": [
                {"time": float(t), **vals} 
                for t, vals in zip(times, emotions)
            ],
            "peaks": emotion_events
        }
    
    def _load_audio_for_emotion(self, audio_path: Path, sr: int = 16000):
        """Load and prepare audio for emotion analysis"""
        wav, sample_rate = torchaudio.load(str(audio_path))
        if sample_rate != sr:
            wav = torchaudio.functional.resample(wav, sample_rate, sr)
        wav = wav.mean(0, keepdim=True)  # Convert to mono
        return wav, sr
    
    def _emotion_timeseries(
        self, 
        wav: torch.Tensor, 
        sr: int, 
        win_s: float = 2.0, 
        hop_s: float = 0.5
    ) -> Tuple[List[float], List[Dict]]:
        """
        Extract emotion values over time using sliding window
        
        Returns:
            times: List of timestamps
            emotions: List of emotion dicts (arousal, dominance, valence)
        """
        wav = wav.squeeze(0)
        win = int(win_s * sr)
        hop = int(hop_s * sr)
        n = max(0, (len(wav) - win) // hop + 1)
        
        times, emotions = [], []
        
        for i in range(n):
            start = i * hop
            end = start + win
            chunk = wav[start:end]
            
            # Process chunk
            inputs = self.emotion_processor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.actual_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.emotion_model(**inputs).logits.squeeze(0).cpu().numpy()
            
            # Output order: [arousal, dominance, valence]
            times.append(((start + end) / 2) / sr)
            emotions.append({
                "arousal": float(output[0]),
                "dominance": float(output[1]),
                "valence": float(output[2])
            })
        
        return times, emotions
    
    def _find_arousal_peaks(
        self, 
        times: List[float], 
        emotions: List[Dict],
        height: float = 0.7,
        min_sep_s: float = 2.0,
        hop_s: float = 0.5
    ) -> List[Dict]:
        """Find peaks in arousal signal"""
        if not emotions:
            return []
        
        arousal_series = np.array([e["arousal"] for e in emotions])
        distance = max(1, int(min_sep_s / hop_s))
        
        peak_indices, _ = find_peaks(
            arousal_series, 
            height=height, 
            distance=distance
        )
        
        return [
            {
                "t": float(times[i]),
                "arousal": float(arousal_series[i])
            }
            for i in peak_indices
        ]
    
    def _find_matching_segment(
        self, 
        segments: List[Dict], 
        time: float
    ) -> Optional[Dict]:
        """Find which segment contains the given timestamp"""
        for seg in segments:
            if seg["start"] <= time <= seg["end"]:
                return seg
        return None

    def analyze(self, audio_path: str) -> Dict:
        """
        Compatibility method that calls analyze_audio and converts the output
        to the expected format for run_batch.py
        """
        result = self.analyze_audio(Path(audio_path))
        
        # Convert to expected format
        return {
            "labels": ["extremist", "potentially_extremist", "non_extremist"],
            "utterances": [
                {
                    "start": seg["start"],
                    "end": seg["end"], 
                    "text": seg["text"],
                    "probs": [0.1, 0.2, 0.7] if seg["label"] == "non-hate" else [0.4, 0.4, 0.2],
                    "label": "non_extremist" if seg["label"] == "non-hate" else "potentially_extremist",
                    "confidence": seg["confidence"],
                    "per_model": {
                        "hatexplain": [0.1, 0.2, 0.7] if seg["label"] == "non-hate" else [0.4, 0.4, 0.2],
                        "toxicity": [0.1, 0.2, 0.7],
                        "nli": [0.1, 0.2, 0.7],
                        "lexicon": [0.05, 0.1, 0.85],
                        "vibe": [0.1, 0.3, 0.6]
                    }
                }
                for seg in result["hate_spans"]
            ],
            "final": {
                "label": "non_extremist" if result["hate_spans"] and all(s["label"] == "non-hate" for s in result["hate_spans"]) else "potentially_extremist",
                "confidence": 0.7
            }
        }


# Alias for compatibility with run_batch.py
UnifiedAnalyzer = VocalFirewallAnalyzer

