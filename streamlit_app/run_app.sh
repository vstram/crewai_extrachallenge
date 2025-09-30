#!/bin/bash
# Script to run the CrewAI Fraud Detection Streamlit app

echo "🔍 Starting CrewAI Fraud Detection Streamlit App..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "📍 Running from: $SCRIPT_DIR"

# Check if we're in the streamlit_app directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Make sure you're running this script from the streamlit_app directory"
    exit 1
fi

# Run the Streamlit app
echo "🚀 Launching Streamlit app..."
uv run streamlit run app.py

echo "✅ App stopped"