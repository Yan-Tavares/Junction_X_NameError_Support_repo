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

**OR - Use CLI for Research:**
```bash
# Check API health
./vfw health

# Analyze single file
./vfw analyze audio.wav -v

# Screen training data directory
./vfw screen ./training_data/

# See full CLI documentation
cat CLI_README.md
```

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Web Frontend  │         │   CLI Tool      │
│  (Port 8080)    │         │  (Research)     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
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

## 🎯 Two Interfaces, Two Audiences

### 🌐 Web UI - For General Users
- **Purpose**: Screen materials children may be exposed to
- **Use cases**: Parents, educators, content moderators
- **Features**: Drag & drop, visual results, easy to use

### 🖥️ CLI - For Researchers
- **Purpose**: Screen training data for speech models
- **Use cases**: ML researchers, dataset curation, automation
- **Features**: Batch processing, CSV/JSON export, filtering, reports
- **Documentation**: See [CLI_README.md](CLI_README.md)

## 📂 Project Structure

```
.
├── index.html               # Web UI
├── app.js                   # Frontend logic
├── run_frontend.py          # Frontend server
├── run_api.sh               # Launch API
├── requirements.txt         # Dependencies
│
├── vfw_cli.py               # CLI tool (main script)
├── vfw                      # CLI wrapper script
├── CLI_README.md            # CLI documentation
│
├── src/
│   ├── config.py            # Configuration
│   ├── pipeline.py          # Analysis pipeline
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

**API documentation**: http://localhost:8000/docs

**Access via CLI**: `./vfw --help`

## 🧪 Testing

```bash
# Test API with curl
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/audio.wav"

# Test with CLI (recommended)
./vfw analyze path/to/audio.wav -v

# Test batch processing
./vfw batch -d ./data/sample\ database/
```
