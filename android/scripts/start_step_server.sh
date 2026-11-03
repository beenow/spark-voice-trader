#!/bin/zsh
# Start the Mac step server (:8310) detached, bound to a LAN/Tailscale address (never a public tunnel).
#   scripts/start_step_server.sh <host-ip> [repo-dir] [python]
# Env: VOICE_HOME (model files + .passcode), STEP_MODEL_FILES (GGUFs offered, first = app default).
HOST=${1:?usage: start_step_server.sh <host-ip> [repo-dir] [python]}
REPO=${2:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${3:-python3}
LOG=${STEP_LOG:-${VOICE_HOME:-$HOME/.leaps-voice-agent}/logs/step_server.log}
mkdir -p "$(dirname "$LOG")"
pkill -f "voice.step_server" 2>/dev/null; sleep 1
cd "$REPO" && nohup "$PY" -m voice.step_server --host "$HOST" > "$LOG" 2>&1 &
echo "step server starting on http://$HOST:8310 (log: $LOG); check: curl http://$HOST:8310/health"
