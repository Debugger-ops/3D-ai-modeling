#!/usr/bin/env bash
# Start Image to 3D Studio at http://127.0.0.1:8000
#   I3D_ENGINE=demo ./run.sh     -> test the UI without the AI model
#   I3D_DEVICE=mps  ./run.sh     -> try the Apple GPU (faster, less tested)
cd "$(dirname "$0")"
[ -d .venv ] && . .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
PORT="${PORT:-8000}"
( sleep 2; command -v open >/dev/null && open "http://127.0.0.1:$PORT" ) &
exec python -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
