#!/usr/bin/env bash
# vLLM OpenAI server — Qwen3-VL-8B-Instruct on Jetson Thor (JetPack 7 / L4T R38 / CUDA 13)
#
# Run this in a PERSISTENT session (tmux / systemd), NOT as a child of a short-lived shell.
#   tmux new -s vllm
#   ~/vllm-venv/serve_qwen3vl.sh
#
# Stop it with Ctrl-C (SIGINT) — this script forwards a clean shutdown to vLLM,
# waits for it, then verifies GPU memory was actually reclaimed.
#
# Perf tuning via env:
#   EXTRA="--enforce-eager"                       # skip torch.compile (dodge sm_110a triton)
#   EXTRA="--quantization fp8 --enforce-eager"    # FP8 (needs the local cutlass.py patch on Thor)
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct             # smaller model
#   GPU_MEM_UTIL=0.3                              # KV-cache reservation fraction (default 0.25)
#   PORT=8005
#   SHUTDOWN_WAIT=45                              # grace seconds before escalating past SIGINT
#   SKIP_MEMCHECK=1                               # bypass the pre-flight leak guard
set -uo pipefail
source "$HOME/vllm-venv/env.sh"

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PORT="${PORT:-8005}"
EXTRA="${EXTRA:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
SHUTDOWN_WAIT="${SHUTDOWN_WAIT:-45}"
MEMCHECK="$HOME/vllm-venv/memcheck.sh"

# ---------------------------------------------------------------------------
# Pre-flight: refuse to start on an already-leaked box. Starting anyway just
# stacks another ~10-25 GB of unreclaimable nvmap and turns the eventual
# "Free memory ... less than desired GPU memory utilization" startup failure
# into a 60 s mystery.
# ---------------------------------------------------------------------------
if [ -x "$MEMCHECK" ] && [ -z "${SKIP_MEMCHECK:-}" ]; then
  echo "── pre-flight memory check ─────────────────────────────────────────"
  if ! "$MEMCHECK"; then
    echo "───────────────────────────────────────────────────────────────────"
    echo "ABORT: GPU/nvmap memory is leaked. Reboot before starting vLLM."
    echo "  try first (needs sudo, often a no-op on Tegra): sudo nvidia-smi -r"
    echo "  then: sudo reboot"
    echo "  (bypass: SKIP_MEMCHECK=1 ~/vllm-venv/serve_qwen3vl.sh)"
    exit 3
  fi
  echo "───────────────────────────────────────────────────────────────────"
fi

# ---------------------------------------------------------------------------
# Launch vLLM in the background so we can trap signals and supervise shutdown.
# ---------------------------------------------------------------------------
"$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3-vl \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 2}' \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs 8 \
  --mm-processor-kwargs '{"max_pixels": 262144}' \
  $EXTRA &
VLLM_PID=$!
echo "vLLM api_server pid=$VLLM_PID  (gpu-memory-utilization=$GPU_MEM_UTIL)"

verify_reclaim() {
  echo "── vLLM gone. settling 3s, verifying memory reclaim ──"
  sleep 3
  if [ -x "$MEMCHECK" ] && ! SKIP_MEMCHECK= "$MEMCHECK"; then
    echo
    echo "!! GPU memory did NOT come back — this box needs a reboot before the next run."
    echo "!! (Tegra: a CUDA context poisoned by an earlier crash won't release nvmap"
    echo "!!  on exit, even when this shutdown itself was clean.)"
  fi
}

SHUTTING_DOWN=0
shutdown() {
  [ "$SHUTTING_DOWN" = 1 ] && return
  SHUTTING_DOWN=1
  echo
  echo "── graceful shutdown: SIGINT → pid $VLLM_PID, waiting up to ${SHUTDOWN_WAIT}s ──"
  kill -INT "$VLLM_PID" 2>/dev/null
  for _ in $(seq 1 "$SHUTDOWN_WAIT"); do
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! still alive after ${SHUTDOWN_WAIT}s — SIGTERM"
    kill -TERM "$VLLM_PID" 2>/dev/null; sleep 5
  fi
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! STILL alive — SIGKILL. WARNING: SIGKILL mid-allocation leaks nvmap on Tegra."
    kill -KILL "$VLLM_PID" 2>/dev/null
  fi
  wait "$VLLM_PID" 2>/dev/null
  verify_reclaim
  exit 0
}
trap shutdown INT TERM

wait "$VLLM_PID"
RC=$?
SHUTTING_DOWN=1
echo
echo "!! vLLM exited on its own (rc=$RC)"
verify_reclaim
exit "$RC"
