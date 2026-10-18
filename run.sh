#!/usr/bin/env bash
cd "$(dirname "$0")"
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

# GPU numbers (WHISPER_GPU, KOKORO_GPU) match `nvidia-smi -L`.
# Ollama picks its own GPU (see README "Two GPUs").
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [ "$1" = "--web" ]; then
  exec python web.py       # phone / browser mode
fi
exec python assistant.py
