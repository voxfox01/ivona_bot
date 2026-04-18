#!/bin/bash
# Builds CTranslate2 with CUDA support for Jetson (sm_87) and installs the
# Python wheel into the project venv. Run once after cloning the repo.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CT2_DIR="$ROOT/CTranslate2"
VENV="$ROOT/.venv"

if [ ! -d "$CT2_DIR" ]; then
  echo "Cloning CTranslate2..."
  git clone --recursive https://github.com/OpenNMT/CTranslate2.git --depth=1 "$CT2_DIR"
fi

echo "Building CTranslate2 with CUDA (sm_87)..."
cmake -S "$CT2_DIR" -B "$CT2_DIR/build" \
  -DWITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DWITH_CUDNN=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -DCUDNN_INCLUDE_DIR=/usr/include \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$CT2_DIR/build" -j6

echo "Building Python wheel..."
source "$VENV/bin/activate"
cd "$CT2_DIR/python"
pip install -r install_requirements.txt
CT2_CUDA_PATH=/usr/local/cuda-12.6 python setup.py bdist_wheel
pip install dist/ctranslate2*.whl --force-reinstall

echo ""
echo "Done. Update config/settings.yaml:"
echo "  model_size: medium"
echo "  device: cuda"
echo "  compute_type: float16"
