#!/usr/bin/env bash
cd "$(dirname "$0")"

# Auto-start service (install-service.sh) already running? Only one copy fits on the GPU.
if [ -z "$VA_SERVICE" ] && command -v systemctl >/dev/null \
    && systemctl is-active --quiet voice-assistant 2>/dev/null; then
  echo "The voice assistant service is running (phone mode, started at boot)."
  echo "Only one copy fits on the GPU. Stop it first:  sudo systemctl stop voice-assistant"
  echo "(it starts again at the next boot, or now with: sudo systemctl start voice-assistant)"
  exit 1
fi

source .venv/bin/activate
# Let faster-whisper find the pip-installed CUDA libraries
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# Machine-specific defaults (not in git), e.g. KOKORO_GPU=1.
# Settings given on the command line win over local.env.
if [ -f local.env ]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -z "${!key+x}" ] && export "$key=$value"
  done < local.env
fi

# Fully offline once the models are downloaded: no update checks on Hugging Face.
# To download a new voice or Whisper model, run once with HF_HUB_OFFLINE=0 ./run.sh
if [ -z "${HF_HUB_OFFLINE+x}" ] && [ -d "$HOME/.cache/huggingface/hub/models--hexgrad--Kokoro-82M" ]; then
  export HF_HUB_OFFLINE=1
fi

# GPU numbers (WHISPER_GPU, KOKORO_GPU) match `nvidia-smi -L`.
# Ollama picks its own GPU (see README "Two GPUs").
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ "$1" = "--web" ]; then
  exec python web.py       # phone / browser mode
fi
exec python assistant.py
