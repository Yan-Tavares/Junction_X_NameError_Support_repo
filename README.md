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

#### Option A: Streamlit UI Only (for quick demo)
```bash
# Make script executable
chmod +x run_streamlit.sh

# Run Streamlit
./run_streamlit.sh
# or directly:
streamlit run streamlit_app.py
```

Open browser to: http://localhost:8501

#### Option B: Full Stack (API + UI)

**Terminal 1 - Run API Server:**
```bash
chmod +x run_api.sh
./run_api.sh
# or directly:
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

**Terminal 2 - Run Streamlit:**
```bash
./run_streamlit.sh
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │ ← User Interface (Port 8501)
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
├── streamlit_app.py          # Streamlit UI
├── requirements.txt          # Dependencies
├── SETUP.md                  # Detailed setup guide
├── run_streamlit.sh         # Launch UI
├── run_api.sh               # Launch API
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
