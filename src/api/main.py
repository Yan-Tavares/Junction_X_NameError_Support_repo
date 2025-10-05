"""
FastAPI Backend for Vocal Firewall
Provides REST API endpoints for audio analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipeline.initialize import create_ensemble, load_config
from src.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Vocal Firewall API",
    description="API for detecting extremist speech in audio files",
    version="0.1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class SegmentInfo(BaseModel):
    start: float
    end: float
    text: str
    label: str
    confidence: float

class EmotionPeak(BaseModel):
    time: float
    arousal: float
    coincides_with: Optional[str]
    text: Optional[str]
    span_start: Optional[float]
    span_end: Optional[float]

class EmotionAnalysis(BaseModel):
    time_series: List[Dict]
    peaks: List[EmotionPeak]

class AnalysisResult(BaseModel):
    audio_path: str
    transcript: str
    overall_label: str
    confidence: float
    segments: List[SegmentInfo]
    hate_spans: List[SegmentInfo]
    emotion_analysis: Optional[EmotionAnalysis] = None
    flagged_count: int
    total_segments: int

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool

# Global analyzer instance
analyzer = None

@app.on_event("startup")
async def load_models():
    """Load ML models on startup"""
    global analyzer
    
    try:
        print("🚀 Initializing Ensemble Model...")
        config = load_config()
        analyzer = create_ensemble(config)
        print("✅ All models loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")
        print("API will run in degraded mode")
        analyzer = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="ok", 
        version="1.0.0",
        models_loaded=analyzer is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if analyzer else "degraded",
        version="1.0.0",
        models_loaded=analyzer is not None
    )

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an audio file for extremist speech
    
    Args:
        file: Audio file (wav)
        
    Returns:
        AnalysisResult with transcript, labels, and timestamps
    """
    
    # Validate file type
    allowed_extensions = {'.wav'}  #{'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm'}
    
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    
    if not file_ext or file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext or 'unknown'}. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # TODO: Replace with actual analysis logic
        # This is a placeholder for demonstration
        result = analyze_audio_file(tmp_file_path)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def analyze_audio_file(audio_path: str) -> AnalysisResult:
    """
    Process audio file and return analysis results
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        AnalysisResult object
    """
    
    if not analyzer:
        raise Exception("Analyzer not initialized. Models may not be loaded.")
    
    try:
        # Run full analysis pipeline
        # Ensemble.predict returns predictions array, then assemble_preds converts to dict
        preds = analyzer.predict(audio_path)
        results = analyzer.assemble_preds(preds)
        
        # Count flagged segments (hate or uncertain)
        flagged_segments = [
            s for s in results["hate_spans"]
            if s["label"] in ["hate", "uncertain"] and s["confidence"] > settings.CONFIDENCE_THRESHOLD
        ]
        
        # Determine overall label
        hate_count = sum(1 for s in results["hate_spans"] if s["label"] == "hate")
        
        if hate_count > 0:
            overall_label = "hate_detected"
            # Average confidence of hate segments
            hate_confs = [s["confidence"] for s in results["hate_spans"] if s["label"] == "hate"]
            confidence = sum(hate_confs) / len(hate_confs) if hate_confs else 0.0
        elif len(flagged_segments) > 0:
            overall_label = "uncertain"
            confidence = sum(s["confidence"] for s in flagged_segments) / len(flagged_segments)
        else:
            overall_label = "safe"
            # Average confidence of non-hate segments
            safe_confs = [s["confidence"] for s in results["hate_spans"] if s["label"] == "non-hate"]
            confidence = sum(safe_confs) / len(safe_confs) if safe_confs else 1.0
        
        # Format segments for response
        segment_infos = [
            SegmentInfo(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                label=s["label"],
                confidence=s["confidence"]
            )
            for s in results["hate_spans"]
        ]
        
        # Format emotion analysis if available
        emotion_analysis = None
        if results.get("emotion_analysis"):
            emo = results["emotion_analysis"]
            emotion_analysis = EmotionAnalysis(
                time_series=emo.get("time_series", []),
                peaks=[
                    EmotionPeak(**peak)
                    for peak in emo.get("peaks", [])
                ]
            )
        
        return AnalysisResult(
            audio_path=results["audio_path"],
            transcript=results["transcript"],
            overall_label=overall_label,
            confidence=float(confidence),
            segments=segment_infos,
            hate_spans=segment_infos,
            emotion_analysis=emotion_analysis,
            flagged_count=len(flagged_segments),
            total_segments=len(results["hate_spans"])
        )
        
    except Exception as e:
        raise Exception(f"Error analyzing audio: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analyze multiple audio files in batch
    """
    results = []
    
    for file in files:
        try:
            result = await analyze_audio(file)
            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

