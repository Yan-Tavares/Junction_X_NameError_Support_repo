#!/bin/bash
# Script to run FastAPI backend

echo "🚀 Starting Vocal Firewall API Server..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set LD_LIBRARY_PATH to include cuDNN libraries from venv
CUDNN_LIB_PATH="$(pwd)/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB_PATH"
else
    export LD_LIBRARY_PATH="$CUDNN_LIB_PATH:$LD_LIBRARY_PATH"
fi
echo "✓ Set cuDNN library path"

# Run FastAPI with uvicorn
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

echo ""
echo "✅ API Server stopped"

