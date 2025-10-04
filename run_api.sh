#!/bin/bash
# Script to run FastAPI backend

echo "🚀 Starting Vocal Firewall API Server..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run FastAPI with uvicorn
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

echo ""
echo "✅ API Server stopped"

