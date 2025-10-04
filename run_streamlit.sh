#!/bin/bash
# Script to run Streamlit UI

echo "🚀 Starting Vocal Firewall Streamlit UI..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address localhost

echo ""
echo "✅ Streamlit UI stopped"

