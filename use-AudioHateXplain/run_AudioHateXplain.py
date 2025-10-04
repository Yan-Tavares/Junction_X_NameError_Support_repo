"""
AudioHateXplain Runner Script
Processes audio files through ASR and hate speech detection pipeline
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

import torch
import torchaudio
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)
from safetensors.torch import load_file
import numpy as np

# Try to set torchaudio backend for Windows compatibility
try:
    torchaudio.set_audio_backend("soundfile")
except RuntimeError:
    pass  # Backend already set or not available

warnings.filterwarnings('ignore')


class AudioHateXplainPipeline:
    """
    Pipeline for processing audio through ASR and hate speech detection.
    """
    
    def __init__(
        self,
        asr_model_path: str = "models/AudioHateXplain/ASR",
        classifier_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the AudioHateXplain pipeline.
        
        Args:
            asr_model_path: Path to the ASR model weights
            classifier_model_path: Path to the hate speech classifier (optional)
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load ASR model
        print("Loading ASR model...")
        self.asr_processor, self.asr_model = self._load_asr_model(asr_model_path)
        
        # Load hate speech classifier
        print("Loading hate speech classifier...")
        self.tokenizer, self.classifier = self._load_classifier(classifier_model_path)
        
        print("✅ Models loaded successfully!")
    
    def _load_asr_model(self, model_path: str):
        """Load ASR model from safetensors or use a pretrained model."""
        model_path = Path(model_path)
        
        if model_path.exists() and (model_path / "model.safetensors").exists():
            try:
                # Try to load custom model from safetensors
                print(f"Loading custom ASR model from {model_path}")
                
                # Load config if available
                config_path = model_path / "config.json"
                if config_path.exists():
                    # Load the model architecture based on config
                    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
                    model = Wav2Vec2ForCTC.from_pretrained(str(model_path))
                else:
                    # Fallback: use a standard ASR model and load weights
                    print("No config found, using Wav2Vec2 base model")
                    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
                    
                    # Load custom weights from safetensors
                    state_dict = load_file(str(model_path / "model.safetensors"))
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ Loaded custom weights from safetensors")
                
                model = model.to(self.device)
                model.eval()
                return processor, model
                
            except Exception as e:
                print(f"⚠️  Could not load custom ASR model: {e}")
                print("Falling back to pretrained Whisper model...")
        
        # Fallback to a pretrained ASR model (Whisper is good for transcription with timestamps)
        print("Using pretrained Whisper model for ASR with timestamps...")
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en")
        model = model.to(self.device)
        model.eval()
        
        return processor, model
    
    def _load_classifier(self, model_path: Optional[str] = None):
        """Load hate speech classifier."""
        if model_path and Path(model_path).exists():
            print(f"Loading custom classifier from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # Use a pretrained hate speech detection model
            print("Using pretrained BERT-based hate speech classifier...")
            model_name = "bert-base-uncased"  # Base model, can be fine-tuned
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,  # normal, offensive, hate
                output_attentions=True,
            )
        
        model = model.to(self.device)
        model.eval()
        return tokenizer, model
    
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of segments with timestamps and text
        """
        print(f"Transcribing audio: {audio_path}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary (most models expect 16kHz)
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process with ASR model
        segments = []
        
        # Split audio into chunks (e.g., 30 seconds) for better timestamp granularity
        chunk_duration = 30  # seconds
        chunk_samples = chunk_duration * sample_rate
        total_samples = waveform.shape[1]
        
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[:, start_sample:end_sample]
            
            # Skip very short chunks
            if chunk.shape[1] < sample_rate * 0.5:
                continue
            
            # Process chunk
            inputs = self.asr_processor(
                chunk.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get transcription
                if hasattr(self.asr_model, 'generate'):
                    # Whisper-style model
                    outputs = self.asr_model.generate(**inputs)
                    transcription = self.asr_processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )[0]
                else:
                    # Wav2Vec2-style model
                    logits = self.asr_model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.asr_processor.batch_decode(predicted_ids)[0]
            
            # Create segment
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            if transcription.strip():  # Only add non-empty transcriptions
                segments.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "text": transcription.strip()
                })
        
        print(f"✅ Transcribed {len(segments)} segments")
        return segments
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text for hate speech.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with label, confidence, and scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()
        
        # Map to labels (adjust based on your model's label mapping)
        label_map = {
            0: "normal",
            1: "offensive", 
            2: "hate"
        }
        
        # Determine label with uncertainty threshold
        if confidence < 0.5:
            label = "uncertain"
        else:
            label = label_map.get(predicted_class, "uncertain")
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {
                "normal": round(probs[0, 0].item(), 4),
                "offensive": round(probs[0, 1].item(), 4) if probs.shape[1] > 1 else 0.0,
                "hate": round(probs[0, 2].item(), 4) if probs.shape[1] > 2 else 0.0,
            }
        }
    
    def process_audio_file(
        self,
        input_path: str,
        output_path: str,
        save_json: bool = True
    ) -> Dict[str, Any]:
        """
        Process an audio file through the full pipeline.
        
        Args:
            input_path: Path to input .wav file
            output_path: Directory path for output analysis.json
            save_json: Whether to save results to JSON file
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}\n")
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Transcribe audio
        segments = self.transcribe_audio(input_path)
        
        # Step 2: Classify each segment
        print("\nClassifying segments for hate speech...")
        results = []
        
        for segment in segments:
            classification = self.classify_text(segment["text"])
            
            result = {
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "text": segment["text"],
                "label": classification["label"],
                "confidence": classification["confidence"],
                "scores": classification["scores"]
            }
            results.append(result)
            
            # Print summary
            print(f"  [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] "
                  f"{classification['label'].upper()} ({classification['confidence']:.2%}) "
                  f"- \"{segment['text'][:50]}...\"")
        
        # Create final output
        output_data = {
            "audio_file": str(Path(input_path).name),
            "total_segments": len(results),
            "segments": results,
            "summary": self._create_summary(results)
        }
        
        # Save to JSON
        if save_json:
            output_file = output_dir / "analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Analysis saved to: {output_file}")
        
        return output_data
    
    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the analysis results."""
        if not results:
            return {
                "total_segments": 0,
                "hate_segments": 0,
                "offensive_segments": 0,
                "normal_segments": 0,
                "uncertain_segments": 0,
            }
        
        label_counts = {"hate": 0, "offensive": 0, "normal": 0, "uncertain": 0}
        for result in results:
            label = result["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "total_segments": len(results),
            "hate_segments": label_counts["hate"],
            "offensive_segments": label_counts["offensive"],
            "normal_segments": label_counts["normal"],
            "uncertain_segments": label_counts["uncertain"],
            "hate_percentage": round(label_counts["hate"] / len(results) * 100, 2),
        }


def run_analysis(input_path: str, output_path: str, asr_model_path: str = "models/AudioHateXplain/ASR"):
    """
    Main function to run audio hate speech analysis.
    
    Args:
        input_path: Path to input .wav file
        output_path: Directory path for output analysis.json
        asr_model_path: Path to ASR model weights
    """
    # Initialize pipeline
    pipeline = AudioHateXplainPipeline(
        asr_model_path=asr_model_path,
        classifier_model_path=None  # Will use default BERT model
    )
    
    # Process audio
    results = pipeline.process_audio_file(input_path, output_path)
    
    return results


# Example usage
if __name__ == "__main__":
    # Example paths (adjust as needed)
    input_audio = "data/recordings/test.wav"
    output_directory = "output/analysis"
    
    # Run analysis
    results = run_analysis(input_audio, output_directory)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total segments: {results['summary']['total_segments']}")
    print(f"Hate speech segments: {results['summary']['hate_segments']}")
    print(f"Offensive segments: {results['summary']['offensive_segments']}")
    print(f"Normal segments: {results['summary']['normal_segments']}")
    print(f"Uncertain segments: {results['summary']['uncertain_segments']}")
    print(f"Hate speech percentage: {results['summary']['hate_percentage']:.2f}%")
    print("="*60)
