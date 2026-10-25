#!/usr/bin/env bash
# Double-click to start Image to 3D Studio. Keep this window open while you use the app;
# close the window (or press Ctrl+C here) to stop it.
cd "$(dirname "$0")"
echo "Starting Image to 3D Studio at http://127.0.0.1:8000 ..."
./run.sh
echo; echo "Server stopped. You can close this window."
read -r -p "Press Enter to close"
