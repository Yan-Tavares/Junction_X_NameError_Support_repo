import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(file_path, target_sr=16000):
    """Load and resample audio file."""
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

def load_emotion_model():
    """Load emotion recognition model."""
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return feature_extractor, model

def analyze_chunk(audio_chunk, sr, feature_extractor, model):
    """Analyze emotion in an audio chunk."""
    # Normalize and get audio features
    audio_chunk = librosa.util.normalize(audio_chunk)
    rms = librosa.feature.rms(y=audio_chunk)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(audio_chunk)[0].mean()
    
    inputs = feature_extractor(audio_chunk, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_idx].item()

    emotion = model.config.id2label[predicted_idx]
    
    # Override prediction for intense speech
    if rms > 0.1 and zcr > 0.15:
        if emotion in ['neutral', 'calm']:
            emotion = 'angry'
            confidence = max(confidence, 0.8)
    
    return emotion, confidence

def analyze_audio_file(file_path, chunk_duration=10.0, overlap=0.5):
    """Analyze emotions in audio file with overlapping chunks."""
    # Load model and audio
    feature_extractor, model = load_emotion_model()
    audio, sr = load_audio(file_path)
    
    # Set up chunking
    chunk_size = int(chunk_duration * sr)
    hop_size = int(chunk_size * (1 - overlap))
    
    # Analyze emotions in chunks
    results = []
    for start_idx in range(0, len(audio), hop_size):
        end_idx = min(start_idx + chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        
        if len(chunk) >= sr:  # Only analyze chunks of at least 1 second
            emotion, confidence = analyze_chunk(chunk, sr, feature_extractor, model)
            results.append({
                "start": start_idx / sr,
                "end": end_idx / sr,
                "emotion": emotion,
                "confidence": confidence
            })
    
    return results

def print_analysis(results):
    """Pretty print analysis results."""
    print("\nEmotion Analysis Results:")
    print("-" * 60)
    for r in results:
        print(f"Time: {r['start']:.1f}s - {r['end']:.1f}s")
        print(f"Emotion: {r['emotion']} ({r['confidence']:.1%} confident)")
        print("-" * 60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        results = analyze_audio_file(audio_file)
        print_analysis(results)
    else:
        print("Please provide an audio file path")