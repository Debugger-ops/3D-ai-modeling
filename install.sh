#!/usr/bin/env bash
# One-time setup: creates .venv, installs PyTorch + TripoSR + the web app.
set -euo pipefail
# CMake 4 rejects the old pybind11 inside some packages; this lets them build.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cd "$(dirname "$0")"

# TripoSR's pinned dependencies want Python 3.10 or 3.11.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.11 python3.10 python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
fi
[ -n "$PY" ] || { echo "Python not found. Install Python 3.11 (brew install python@3.11)."; exit 1; }
VER=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
case "$VER" in 3.10|3.11) ;; *) echo "WARNING: Python $VER found; 3.10 or 3.11 is recommended (brew install python@3.11, then PYTHON=python3.11 ./install.sh)";; esac

command -v git   >/dev/null || { echo "git is required.";   exit 1; }
command -v cmake >/dev/null || { echo "cmake is required to build torchmcubes. macOS: brew install cmake"; exit 1; }

echo "==> Creating virtual environment (.venv) with $PY ($VER)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -U pip wheel setuptools

echo "==> Installing PyTorch"
pip install torch torchvision

echo "==> Installing app + TripoSR dependencies"
pip install -r requirements.txt

echo "==> Fetching TripoSR source"
mkdir -p vendor
[ -d vendor/TripoSR ] || git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR

echo "==> Building torchmcubes (marching cubes)"
pip install scikit-build-core pybind11 ninja cmake setuptools_scm
pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git

echo
echo "Done. Start the app with:  ./run.sh"
echo "The first generation downloads ~1.7 GB of model weights (cached afterwards)."
