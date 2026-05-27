#!/usr/bin/env bash
set -euo pipefail

ENVS_DIR="./envs"
mkdir -p "$ENVS_DIR"

# =========================
# ---- ENV: ptsed ----
# =========================

ENV_PATH="$ENVS_DIR/ptsed"
PYTHON_BIN="$ENV_PATH/bin/python"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating environment at $ENV_PATH..."
    conda create -y -p "$ENV_PATH" python=3.9 cython
else
    echo "ptsed environment already exists."
fi

"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install "setuptools<82" # classic problem
"$PYTHON_BIN" -m pip install numpy cython


echo "Installing PyTorch..."
"$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
"$PYTHON_BIN" -m pip install torch torchvision torchaudio

echo "Installing requirements..."
"$PYTHON_BIN" -m pip install --no-build-isolation -r pretrainedsed_requirements.txt

echo "Installing minimp3..."
CFLAGS='-O3 -march=native' "$PYTHON_BIN" -m pip install https://github.com/f0k/minimp3py/archive/master.zip

echo "ptsed set up"

# =========================
# ---- ENV: emotion ----
# =========================

ENV_PATH="$ENVS_DIR/emotion"
PYTHON_BIN="$ENV_PATH/bin/python"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating emotion environment at $ENV_PATH..."
    conda create -y -p "$ENV_PATH" python=3.9
else
    echo "emotion environment already exists."
fi

"$PYTHON_BIN" -m pip install --upgrade pip

echo "Installing funasr in emotion env..."
"$PYTHON_BIN" -m pip install -U funasr

echo "Installing PyTorch..."
"$PYTHON_BIN" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
"$PYTHON_BIN" -m pip install torch torchaudio

echo "emotion set up"

# =========================
# ---- ENV: mfa ----
# =========================
ENV_PATH="$ENVS_DIR/mfa"
PYTHON_BIN="$ENV_PATH/bin/python"


if [ ! -d "$ENV_PATH" ]; then
    echo "Creating MFA environment at $ENV_PATH..."
    conda create -y -p "$ENV_PATH" python=3.9
else
    echo "mfa environment already exists."
fi

echo "Installing Montreal Forced Aligner..."
conda install -y -p "$ENV_PATH" -c conda-forge montreal-forced-aligner
"$PYTHON_BIN" -m pip install pandas textgrid

echo "mfa set up"

echo "All environments set up"