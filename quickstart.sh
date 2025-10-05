#!/bin/bash
# Quick start script for Vocal Firewall

echo "🛡️  Vocal Firewall - Quick Start"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000

# Model Configuration
WHISPER_MODEL_SIZE=medium
CONFIDENCE_THRESHOLD=0.5
MAX_AUDIO_LENGTH_SECONDS=600
EOF
    echo "✅ .env file created"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "  Terminal 1: ./run_api.sh"
echo "  Terminal 2: python run_frontend.py"
echo ""
echo "Then open: http://localhost:8080"
echo ""
