#!/usr/bin/env bash
# Dev runner: (re)starts label_server.py on every save of the file, so browser-refresh
# picks up edits. The whole front end is embedded as Python strings, so there is no
# hot-reload -- this does a fast full restart (~1-2s) whenever label_server.py changes.
#
#   ./dev.sh            # port 8000
#   ./dev.sh 8011       # custom port
#
# Ctrl-C stops the loop and the server.
set -u
cd "$(dirname "$0")"
PORT="${1:-8000}"
FILE=label_server.py

cleanup(){ [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; exit 0; }
trap cleanup INT TERM

echo "dev: serving $FILE on http://localhost:$PORT  (auto-restart on save, Ctrl-C to stop)"
while true; do
  python3 "$FILE" --port "$PORT" &
  PID=$!
  LAST=$(stat -c %Y "$FILE")
  while kill -0 "$PID" 2>/dev/null && [ "$(stat -c %Y "$FILE")" = "$LAST" ]; do
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "── $FILE changed, restarting ──"
    kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  else
    echo "── server exited, restarting ──"; sleep 1
  fi
done
