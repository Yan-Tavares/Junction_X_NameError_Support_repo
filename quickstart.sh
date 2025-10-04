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

echo "📥 Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo "✅ Dependencies installed"


# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
WHISPER_MODEL_SIZE=small
TEXT_MODEL_PATH=cardiffnlp/twitter-roberta-base-hate-latest
ENABLE_EMOTION_ANALYSIS=true

# Performance
FAST_MODE=false
CONFIDENCE_THRESHOLD=0.5

# Audio Processing
SAMPLE_RATE=16000
PRE_EMPHASIS=0.97
NOISE_REDUCTION=true
EOF
    echo "✅ .env file created"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "  Option 1 (UI only):  ./run_streamlit.sh"
echo "  Option 2 (Full):     ./run_api.sh  (in terminal 1)"
echo "                       ./run_streamlit.sh  (in terminal 2)"
echo ""
echo "📚 See SETUP.md for detailed documentation"
echo ""
