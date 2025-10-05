# 🛡️ Vocal Firewall - Extremist Speech Detection

**Team NameError | Junction X Hackathon**

An automated system for detecting extremist speech in audio files, providing timestamped analysis of potentially harmful content.

## 🚀 Quick Start
### 1. Install PyTorch
```bash
# For CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# For CPU only (fallback)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Use the quickstart.sh script to set up the project
```bash
chmod +x quickstart.sh
./quickstart.sh
```

### 3. Run the Application

**Terminal 1 - Run API Server:**
```bash
chmod +x run_api.sh
./run_api.sh
```
API docs: http://localhost:8000/docs

**Terminal 2 - Run Frontend:**
```bash
python run_frontend.py
```
Open browser to: http://localhost:8080

## 🏗️ Architecture

```
┌─────────────────┐
│   Web Frontend  │ ← HTML/JS UI (Port 8080)
└────────┬────────┘
         │ REST API
    ┌────▼─────┐
    │ FastAPI  │ ← Backend Server (Port 8000)
    │  Server  │
    └────┬─────┘
         │
    ┌────▼────────────┐
    │ ML Pipeline     │
    │ 1. Whisper STT  │ ← Speech-to-Text
    │ 2. Classifier   │ ← Content Classification
    │ 3. Timestamps   │ ← Segment Extraction
    └─────────────────┘
```

## 📂 Project Structure

```
.
├── index.html               # Web UI
├── app.js                   # Frontend logic
├── run_frontend.py          # Frontend server
├── run_api.sh               # Launch API
├── requirements.txt         # Dependencies
│
├── src/
│   ├── config.py            # Configuration
│   ├── api/
│   │   └── main.py          # FastAPI backend
│   └── models/
│       ├── classic_ML_clf.py
│       └── MLP_clf_custom_structure.py
│
├── data/
│   └── sample database/     # Test audio files
│
└── temp/                    # Temporary uploads
```

## 🌐 API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /analyze` - Analyze single audio file
- `POST /analyze/batch` - Batch process multiple files

API documentation: http://localhost:8000/docs

## 🧪 Testing

```bash
# Test API with curl
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/audio.wav"
```
