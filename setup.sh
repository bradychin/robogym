#!/bin/bash
# Installation script for RoboGym on macOS (Apple Silicon)

set -e  # Exit on any error

echo "=================================================="
echo "RoboGym Installation Script"
echo "Platform: macOS (Apple Silicon)"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python 3.10+ required. Found: $python_version"
    echo "Please install Python 3.10, 3.11, or 3.12"
    echo "Recommended: brew install python@3.11"
    exit 1
fi

echo "✓ Python version: $python_version"
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
    echo ""
    echo "⚠️  Please activate the virtual environment and re-run this script:"
    echo "    source .venv/bin/activate"
    echo "    ./setup.sh"
    exit 0
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Install package in editable mode
echo "Installing robogym in editable mode..."
pip install -e .
echo "✓ Package installed"
echo ""

# Verify installation
echo "Verifying installation..."
python test_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Installation successful!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run training: python main.py"
    echo "  2. View logs: tensorboard --logdir=./runs"
    echo "  3. Check documentation: cat README.md"
    echo ""
else
    echo ""
    echo "❌ Installation verification failed"
    echo "Please check the error messages above"
    exit 1
fi
