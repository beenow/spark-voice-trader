#!/usr/bin/env bash
# Run the voice server locally (http://127.0.0.1:8300), or the tests / a text-only turn.
#
#   ./run.sh setup        create .venv and install requirements
#   ./run.sh test         run the test suite (model-dependent tests skip when weights are absent)
#   ./run.sh ask "How's my NVDA position, and is it near support?" ["What about its RSI?" ...]
#   ./run.sh eval         score the synthetic sample with the default model
#   ./run.sh serve        voice server; set VOICE_PASSCODE first
#
# Weights: put them under $VOICE_MODELS_DIR (default ~/.leaps-voice-agent/models) and the adapter under
# ~/.leaps-voice-agent/adapters/llama-1b-loop, then copy models.example.json to ~/.leaps-voice-agent/models.json.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
export VOICE_HOME="${VOICE_HOME:-$HOME/.leaps-voice-agent}"
export VOICE_MODELS="${VOICE_MODELS:-$VOICE_HOME/models.json}"
cmd="${1:-serve}"; shift || true
case "$cmd" in
  setup) python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt ;;
  test)  "$PY" -m pytest -q "$@" ;;
  ask)   "$PY" -m agent.cli --backend named --name "${VOICE_DEFAULT_MODEL:-llama-1b-loop}" "$@" ;;
  eval)  "$PY" -m eval.run_eval --model "${VOICE_DEFAULT_MODEL:-llama-1b-loop}" "$@" ;;
  serve)
    : "${VOICE_PASSCODE:?set VOICE_PASSCODE (the web page asks for it)}"
    "$PY" -m voice.server --agent real --port "${VOICE_PORT:-8300}" "$@" ;;
  *) echo "usage: $0 setup|test|ask|eval|serve" >&2; exit 2 ;;
esac
