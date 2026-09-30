#!/bin/zsh
set -e
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/mjpython ]]; then
  if command -v uv >/dev/null; then
    uv sync
  else
    print 'Install uv and run uv sync in this folder first.'
    read '?Press Return to close.'
    exit 1
  fi
fi
exec .venv/bin/python view.py "$@"
