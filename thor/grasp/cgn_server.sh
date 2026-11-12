#!/usr/bin/env bash
# CGN 상주 추론 서버 (:8010). 모델 1회 로드 -> run_pipeline 이 매번 재로드 안 함 (~10s 절약).
# setsid + PGID (thor_stack.sh 와 동일 방식).
set -eo pipefail
PY="$HOME/grasp/cgn_venv/bin/python3"
LOG="$HOME/grasp/logs/cgn_server.log"
PGIDFILE="$HOME/grasp/.cgn_server.pgid"
PORT=8010
mkdir -p "$HOME/grasp/logs"

case "${1:-status}" in
  up)
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health"; then echo "[cgn] 이미 UP"; exit 0; fi
    setsid bash -lc "cd '$HOME/grasp' && exec '$PY' cgn_server.py" >"$LOG" 2>&1 &
    ch=$!; sleep 0.5
    ps -o pgid= -p "$ch" 2>/dev/null | tr -d ' ' >"$PGIDFILE" || echo "$ch" >"$PGIDFILE"
    printf '[cgn] 모델 로드 대기'
    for i in $(seq 1 60); do
      curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health" && { echo ' — OK'; break; }
      printf '.'; sleep 1
      [ "$i" -eq 60 ] && { echo ' — 타임아웃'; tail -n 15 "$LOG" | sed 's/^/  /'; exit 1; }
    done
    curl -s "http://127.0.0.1:$PORT/health"; echo
    ;;
  down)
    [ -f "$PGIDFILE" ] && { kill -TERM -- "-$(cat $PGIDFILE)" 2>/dev/null || true; sleep 1; kill -KILL -- "-$(cat $PGIDFILE)" 2>/dev/null || true; rm -f "$PGIDFILE"; }
    pkill -9 -f "cgn_server.py" 2>/dev/null || true
    echo "[cgn] 내림"
    ;;
  restart) "$0" down; sleep 1; "$0" up ;;
  status)
    curl -s -m 2 "http://127.0.0.1:$PORT/health" && echo || echo "[cgn] DOWN" ;;
  log) exec tail -n 40 -f "$LOG" ;;
  *) echo "사용: $0 {up|down|restart|status|log}"; exit 1 ;;
esac
