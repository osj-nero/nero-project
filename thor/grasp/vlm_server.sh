#!/usr/bin/env bash
# Qwen3-VL (vLLM OpenAI :8005) — 시맨틱 패스(sem_infer.py)용. 선택 서비스.
#   ~/grasp/thor_all.sh 는 이걸 안 띄운다 (무겁고 GPU ~4GB+). 시맨틱 경로 쓸 때만 up.
#   기동 ~1.5분. tmux 세션 'vllm' 으로.
set -uo pipefail
LOG="$HOME/grasp/logs/vllm.log"
PORT=8005
mkdir -p "$HOME/grasp/logs"

case "${1:-status}" in
  up)
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/v1/models"; then echo "[vlm] 이미 UP"; exit 0; fi
    "$HOME/vllm-venv/memcheck.sh" || { echo "[vlm] 메모리 누수 상태 — 재부팅 후 재시도"; exit 1; }
    tmux kill-session -t vllm 2>/dev/null || true
    tmux new-session -d -s vllm "EXTRA=--enforce-eager '$HOME/vllm-venv/serve_qwen3vl.sh' 2>&1 | tee '$LOG'"
    printf '[vlm] 로드 대기 (~90s)'
    for i in $(seq 1 150); do
      curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/v1/models" && { echo ' — OK'; break; }
      printf '.'; sleep 1
      [ "$i" -eq 150 ] && { echo ' — 타임아웃'; tail -n 20 "$LOG" | sed 's/^/  /'; exit 1; }
    done
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json;print('[vlm]',json.load(sys.stdin)['data'][0]['id'])" || true
    ;;
  down)
    tmux send-keys -t vllm C-c 2>/dev/null || true
    sleep 3
    tmux kill-session -t vllm 2>/dev/null || true
    pkill -f "serve_qwen3vl" 2>/dev/null || true
    echo "[vlm] 내림 (GPU 회수 확인: ~/vllm-venv/memcheck.sh)"
    ;;
  restart) "$0" down; sleep 2; "$0" up ;;
  status)
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null \
      && echo "[vlm] UP :$PORT" || echo "[vlm] DOWN" ;;
  log) exec tmux attach -t vllm ;;
  *) echo "사용: $0 {up|down|restart|status|log}"; exit 1 ;;
esac
