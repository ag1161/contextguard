#!/bin/bash
# ContextGuard — Launch Script (macOS)
# Double-click this file in Finder to start the Streamlit dashboard.

# Change to the directory where this script lives
cd "$(dirname "$0")"

echo "======================================"
echo "  ContextGuard Dashboard Launcher"
echo "======================================"
echo ""

# Check for Python 3
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Please install Python 3.10+ from https://python.org"
  read -p "Press Enter to exit..."
  exit 1
fi

echo "Python: $(python3 --version)"
echo ""

# Install / upgrade dependencies
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt --quiet

echo ""
echo "Starting Streamlit dashboard..."
echo "Visit http://localhost:8502 in your browser."
echo ""
echo "(Press Ctrl+C in this window to stop the server)"
echo ""

# Launch Streamlit on port 8502 (8501 may be in use by another app)
python3 -m streamlit run dashboard/streamlit_app.py \
  --server.port 8502 \
  --server.headless false \
  --browser.gatherUsageStats false
