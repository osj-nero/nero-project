#!/usr/bin/env bash
# stop_qwen3vl.sh — clean shutdown of a vLLM server started outside this shell
# (e.g. in another tmux pane). Sends SIGINT, waits, then verifies GPU memory
# was reclaimed. Never escalates to SIGKILL on its own — a SIGKILL mid-alloc
# leaks nvmap until reboot; if SIGINT won't take, that's a decision for you.
set -uo pipefail
WAIT="${SHUTDOWN_WAIT:-45}"

mapfile -t PIDS < <(pgrep -f 'vllm.entrypoints.openai.api_server')
if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "no vLLM api_server process found."
else
  echo "SIGINT → ${PIDS[*]}  (waiting up to ${WAIT}s)"
  kill -INT "${PIDS[@]}" 2>/dev/null
  for _ in $(seq 1 "$WAIT"); do
    pgrep -f 'vllm.entrypoints.openai.api_server' >/dev/null || break
    sleep 1
  done
  if pgrep -f 'vllm.entrypoints.openai.api_server' >/dev/null; then
    echo "!! still running after ${WAIT}s. Left alone."
    echo "!! escalate manually if you must:  kill -TERM ${PIDS[*]}   (then -KILL, which leaks nvmap)"
    exit 1
  fi
  echo "vLLM stopped."
fi

sleep 3
exec "$HOME/vllm-venv/memcheck.sh"
