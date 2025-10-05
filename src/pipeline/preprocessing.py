"""
Audio preprocessing module for Vocal Firewall.
Handles audio loading, noise reduction, and normalization.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def preprocess_audio(filepath, sr=16000, pre_emphasis=0.97, apply_noise_reduction=True):
    """
    Load and preprocess audio file with noise reduction and normalization.
    
    Args:
        filepath: Path to audio file
        sr: Target sample rate (default: 16000)
        pre_emphasis: Pre-emphasis filter coefficient (default: 0.97)
        apply_noise_reduction: Whether to apply spectral gating (default: True)
        
    Returns:
        Path: Path to preprocessed audio file
    """
    filepath = Path(filepath)
    
    # Load audio
    audio, sample_rate = librosa.load(str(filepath), sr=sr)
    
    # Apply pre-emphasis filter
    emphasized_audio = np.append(
        audio[0], 
        audio[1:] - pre_emphasis * audio[:-1]
    )
    
    # Normalize
    emphasized_audio = librosa.util.normalize(emphasized_audio)
    
    # Apply spectral gating for noise reduction
    if apply_noise_reduction:
        emphasized_audio = apply_spectral_gating(emphasized_audio)
    
    # Save preprocessed audio
    temp_path = filepath.parent / (filepath.stem + "_processed.wav")
    sf.write(str(temp_path), emphasized_audio, sr)
    
    return temp_path


def apply_spectral_gating(audio):
    """
    Apply spectral gating for noise reduction.
    
    Args:
        audio: Audio signal as numpy array
        
    Returns:
        numpy.ndarray: Cleaned audio signal
    """
    # Apply STFT
    S = librosa.stft(audio)
    mag = np.abs(S)
    
    # Estimate noise floor from quietest 10% of frames
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
    
    return audio_clean
