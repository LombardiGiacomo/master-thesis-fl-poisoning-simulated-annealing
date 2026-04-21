#!/bin/bash
set -euo pipefail

cd ~/fl_project || { echo "Directory fl_project not found!"; exit 1; }

echo "=== Client Node configuration ==="

echo "Installing python3-venv..."
sudo apt update
sudo apt install -y python3-venv python3-pip

echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing packages from local wheelhouse..."
python3 -m pip install --no-index --find-links=wheelhouse \
  torch flwr pandas numpy scikit-learn hatchling editables \
  filelock typing-extensions sympy networkx jinja2 fsspec setuptools markupsafe mpmath

echo "Verifying installation..."
python3 - << 'EOF'
try:
    import torch, flwr, pandas, numpy, sklearn
    print("SUCCESS: All libraries imported correctly.")
except ImportError as e:
    print(f"ERROR: {e}")
    raise
EOF

echo "Installing project in editable mode..."
if [ -f "pyproject.toml" ]; then
    python3 -m pip install -e . --no-deps --no-build-isolation
else
    echo "ERROR: pyproject.toml not found!"
    exit 1
fi

echo "Verifying dataset presence..."
if [ -d "aotizhongxin" ] || [ -d "changping" ] || [ -d "dingling" ] || [ -d "dongsi" ] || [ -d "guanyuan" ]; then
    echo "OK: Datasets' directories found."
else
    echo "ATTENTION: No dataset directory found ('aotizhongxin' or 'changping' or 'dingling' or 'dongsi' or 'guanyuan')."
    echo "Training will fail."
fi

echo "=== Setup Client Node completed ==="
echo "You can now launch ClientApp."