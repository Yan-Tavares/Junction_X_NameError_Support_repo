"""
Configuration management for Vocal Firewall
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Model Configuration
    WHISPER_MODEL_SIZE: str = "medium"  # tiny, base, small, medium, large
    TEXT_MODEL_PATH: str = "cardiffnlp/twitter-roberta-base-hate-latest"
    EMOTION_MODEL_PATH: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    
    # Feature Flags
    ENABLE_EMOTION_ANALYSIS: bool = True
    FAST_MODE: bool = False  # Use smaller models for faster processing
    WHISPER_USE_GPU: bool = False  # Set to True if you have cuDNN 9.1+ installed
    
    # Processing Configuration
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_AUDIO_LENGTH_SECONDS: int = 600  # 10 minutes
    
    # Audio Preprocessing
    SAMPLE_RATE: int = 16000
    PRE_EMPHASIS: float = 0.97
    NOISE_REDUCTION: bool = True
    
    # Segment Merging Parameters
    MIN_SEGMENT_DURATION: float = 0.3  # seconds
    MIN_SEGMENT_TOKENS: int = 1
    MAX_SEGMENT_GAP: float = 0.1  # seconds
    
    # Emotion Analysis Parameters
    EMOTION_WINDOW_SIZE: float = 2.0  # seconds
    EMOTION_HOP_SIZE: float = 0.5  # seconds
    AROUSAL_PEAK_HEIGHT: float = 0.7
    MIN_PEAK_SEPARATION: float = 2.0  # seconds
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    TEMP_DIR: Path = BASE_DIR / "temp"
    OUT_DIR: Path = BASE_DIR / "out"
    
    # Classification Labels
    HATE_LABELS: list = ["hate", "offensive", "normal"]
    EXTREMIST_CATEGORIES: list = [
        "derogatory",
        "exclusionary", 
        "dangerous"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Create necessary directories
settings.DATA_DIR.mkdir(exist_ok=True)
settings.TEMP_DIR.mkdir(exist_ok=True)
settings.OUT_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)

