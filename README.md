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
├── requirements.txt         # Dependencies
├── ensemble_config.yaml     # Model configuration
├── quickstart.sh            # Setup script (Linux/Mac)
├── run_api.sh               # Launch API
├── SETUP.md                 # Detailed setup guide
│
├── vfw_cli.py               # CLI tool (main script)
├── vfw                      # CLI wrapper script
├── CLI_README.md            # CLI documentation
├── CLI_EXAMPLES.md          # CLI usage examples
│
├── src/                     # Source code
│   ├── config.py            # Application settings
│   ├── __init__.py
│   │
│   ├── api/                 # REST API
│   │   ├── main.py          # FastAPI endpoints
│   │   └── __init__.py
│   │
│   ├── model/               # ML Models
│   │   ├── ensemble.py      # Ensemble orchestrator
│   │   ├── text_models.py   # Text classification models
│   │   ├── dummy.py         # Test model (dev only)
│   │   ├── sentiment_base.py # Base model class
│   │   └── vibechecker.py   # Audio prosody analysis
│   │
│   ├── pipeline/            # Processing Pipeline
│   │   ├── preprocessing.py # Audio preprocessing
│   │   ├── transcription.py # Speech-to-text
│   │   ├── postprocessing.py # Result assembly
│   │   ├── initialize.py    # Model initialization
│   │   └── __init__.py
│   │
│   └── testing/             # Test utilities
|
├── models/                  # Saved model weights
│
├── finetune_ensemble.py     # Model fine-tuning
├── segment_augmentation.py  # Data augmentation
│
└── temp/                    # Temporary uploads
```

## Acknowledgment of the use of LLMs
- LLMs for coding: Claude 4.5 Sonnet in Cursor and VS Code. Used for quick prototyping, productive quick edits with the tab feature, creating documentation, and refactors
- LLMs for speech: Eleven Labs for voiceover of pitch video
- LLMs for research: ChatGPT for suggestions of useful papers

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
