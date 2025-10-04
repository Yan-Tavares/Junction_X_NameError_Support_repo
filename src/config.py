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
    CLASSIFIER_MODEL_PATH: Optional[str] = None
    
    # Processing Configuration
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_AUDIO_LENGTH_SECONDS: int = 600  # 10 minutes
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "src" / "models"
    TEMP_DIR: Path = BASE_DIR / "temp"
    
    # Audio Processing
    SAMPLE_RATE: int = 16000
    
    # Categories
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

