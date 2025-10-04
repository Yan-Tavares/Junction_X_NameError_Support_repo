import torch
import librosa
import soundfile as sf
from pydub import AudioSegment
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def to_wav(input_path, sr=16000):
    """
    Convert mp3/mp4/wav to clean 16kHz mono audio data.
    Returns tuple of (audio_data, sample_rate).
    """
    if input_path.endswith(".wav"):
        # Load directly
        y, _ = librosa.load(input_path, sr=sr, mono=True)
    elif input_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(input_path)
        sound = sound.set_frame_rate(sr).set_channels(1)
        # Convert to numpy array directly
        y = np.array(sound.get_array_of_samples(), dtype=np.float32) / 32768.0
    elif input_path.endswith(".mp4"):
        sound = AudioSegment.from_file(input_path, "mp4")
        sound = sound.set_frame_rate(sr).set_channels(1)
        # Convert to numpy array directly
        y = np.array(sound.get_array_of_samples(), dtype=np.float32) / 32768.0
    else:
        raise ValueError("Unsupported file type. Use .wav, .mp3, or .mp4")
    
    return y, sr

def chunk_audio(wav_path, chunk_size_s=10):
    """
    Split WAV into chunks of `chunk_size_s` seconds.
    
    Args:
        wav_path: Path to WAV file
        chunk_size_s: Size of each chunk in seconds
    
    Returns:
        list of tuples: (audio_chunk, start_time, end_time)
        where audio_chunk is the numpy array of audio data
    """
    y, sr = librosa.load(wav_path, sr=None)
    chunk_len = chunk_size_s * sr
    chunks = []
    
    for i in range(0, len(y), chunk_len):
        chunk = y[i:i+chunk_len]
        start, end = i/sr, min((i+chunk_len)/sr, len(y)/sr)
        chunks.append((chunk, start, end))
    
    return chunks

def load_tone_model():
    """Load wav2vec2 model for tone/emotion recognition"""
    model_name = "superb/wav2vec2-base-superb-er"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(DEVICE).eval()
    return feature_extractor, model

def analyze_tone(wav_chunk, sr, feature_extractor, model):
    """
    Analyze emotion in an audio chunk using wav2vec2.
    
    Args:
        wav_chunk: Audio waveform chunk
        sr: Sample rate
        feature_extractor: Wav2Vec2 feature extractor
        model: Loaded emotion recognition model
    
    Returns:
        str: Predicted emotion label
    """
    inputs = feature_extractor(wav_chunk, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = torch.argmax(logits, dim=1).item()
    
    emotions = ['neutral', 'happy', 'sad', 'angry']
    return emotions[predicted_label]

def tone_timeseries(wav, sr, feature_extractor, model, win_s=10.0, hop_s=10.0):
    """
    Analyze tone/emotion over time windows.
    
    Args:
        wav: Full audio waveform
        sr: Sample rate
        feature_extractor: Wav2Vec2 feature extractor
        model: Loaded emotion recognition model
        win_s: Window size in seconds
        hop_s: Hop size in seconds
    
    Returns:
        tuple: (times, tones) where times are timestamps and tones are emotion labels
    """
    wav = wav.squeeze(0)
    win = int(win_s*sr)
    hop = int(hop_s*sr)
    n = max(0, (len(wav)-win)//hop + 1)
    
    times, tones = [], []
    for i in range(n):
        s = i*hop
        e = s+win
        chunk = wav[s:e]
        tone = analyze_tone(chunk, sr, feature_extractor, model)
        times.append(((s+e)/2)/sr)
        tones.append(tone)
    
    return times, tones

def analyze_audio_file(wav_path, feature_extractor, model, chunk_size_s=10):
    """
    Analyze entire audio file in chunks.
    
    Args:
        wav_path: Path to WAV file
        feature_extractor: WAV2Vec2 feature extractor
        model: Loaded emotion recognition model
        chunk_size_s: Size of chunks in seconds
    
    Returns:
        list of tuples: (timestamp, emotion) for each chunk analyzed
    """
    audio_data, sr = to_wav(wav_path)
    chunk_len = chunk_size_s * sr
    results = []
    
    for i in range(0, len(audio_data), chunk_len):
        chunk = audio_data[i:i+chunk_len]
        start, end = i/sr, min((i+chunk_len)/sr, len(audio_data)/sr)
        emotion = analyze_tone(chunk, sr, feature_extractor, model)
        timestamp = (start + end) / 2
        results.append((timestamp, emotion))
    
    return results