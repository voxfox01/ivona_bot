#!/bin/bash
# Installs llama-cpp-python with CUDA support for Jetson (Ampere sm_87)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"

source "$VENV/bin/activate"

CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --no-cache-dir --verbose

echo "llama-cpp-python installed with CUDA support."
