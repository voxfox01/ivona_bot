#!/bin/bash
# Downloads and installs the Piper TTS binary and the default English voice.
# Run once from the project root after cloning.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

PIPER_DIR="$ROOT/services/piper"
MODEL_DIR="$ROOT/models/TTS"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
VOICE_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"

mkdir -p "$PIPER_DIR" "$MODEL_DIR"

echo "Downloading Piper binary (aarch64)..."
curl -L "$PIPER_URL" | tar -xz -C "$PIPER_DIR"

echo "Downloading en_US-amy-medium voice..."
curl -L "$VOICE_BASE/en_US-amy-medium.onnx"      -o "$MODEL_DIR/en_US-amy-medium.onnx"
curl -L "$VOICE_BASE/en_US-amy-medium.onnx.json" -o "$MODEL_DIR/en_US-amy-medium.onnx.json"

echo "Done. Piper installed at $PIPER_DIR/piper/piper"
