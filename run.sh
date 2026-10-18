#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
# Let faster-whisper find the pip-installed CUDA libraries
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
if [ "$1" = "--web" ]; then
  exec python web.py       # phone / browser mode
fi
exec python assistant.py
