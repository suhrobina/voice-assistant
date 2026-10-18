#!/usr/bin/env bash
# One-time setup for Ubuntu/Debian with an NVIDIA GPU.
set -e
cd "$(dirname "$0")"

echo "==> Checking NVIDIA driver"
nvidia-smi >/dev/null || { echo "NVIDIA driver not found. Install it first: sudo ubuntu-drivers install && reboot"; exit 1; }

echo "==> System packages"
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential portaudio19-dev \
  espeak-ng ffmpeg libsndfile1 curl

echo "==> Ollama"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
MODEL="${LLM_MODEL:-llama3.2:3b}"
ollama pull "$MODEL"

echo "==> Python environment"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "RealtimeSTT[faster-whisper]" "RealtimeTTS[kokoro]" requests onnxruntime \
  nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

chmod +x run.sh
echo
echo "Done. Start with:  ./run.sh"
