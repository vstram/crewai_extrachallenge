#!/bin/bash
# Script to run the CrewAI Fraud Detection Streamlit app

echo "🔍 Starting CrewAI Fraud Detection Streamlit App..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "📍 Script located at: $SCRIPT_DIR"

# Change to the script directory (streamlit_app)
cd "$SCRIPT_DIR"

echo "📍 Changed to: $(pwd)"

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in directory: $(pwd)"
    echo "Directory contents:"
    ls -la
    exit 1
fi

# Run the Streamlit app using absolute path to avoid path issues
APP_PATH="$SCRIPT_DIR/app.py"
echo "🚀 Launching Streamlit app: $APP_PATH"
echo "📖 App will be available at: http://localhost:8501"
uv run streamlit run "$APP_PATH" --server.headless true

echo "✅ App stopped"