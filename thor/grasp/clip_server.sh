#!/usr/bin/env bash
# clip_server.sh -- THOR 로컬. CLIP 접지 서버(:8030, ViT-L/14): 이름 없는 물체 크롭에 자연어 점수를 붙인다(PC discover_object/grounding.py 가 호출).
# 사용: ~/grasp/clip_server.sh {up|down|status|log}
set -eo pipefail
PY="$HOME/grasp/cgn_venv/bin/python3"; LOG="$HOME/grasp/logs/clip_server.log"; PGIDFILE="$HOME/grasp/.clip_server.pgid"; PORT=8030
mkdir -p "$HOME/grasp/logs"
case "${1:-status}" in
  up)
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health" && { echo "[clip] 이미 :$PORT 응답 중"; exit 0; }
    setsid bash -lc "cd $HOME/grasp/yolo && exec $PY clip_ground_server.py --port $PORT" >"$LOG" 2>&1 &
    child=$!; sleep 0.5; ps -o pgid= -p "$child" | tr -d " " >"$PGIDFILE" || echo "$child" >"$PGIDFILE"
    printf "[clip] /health 대기"
    for i in $(seq 1 90); do curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health" && { echo " — OK"; break; }; printf "."; sleep 1; done ;;
  down) [ -f "$PGIDFILE" ] && { kill -TERM -- "-$(cat $PGIDFILE)" 2>/dev/null || true; rm -f "$PGIDFILE"; }; echo "[clip] 내림" ;;
  status) curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null && echo "[clip] UP :$PORT" || echo "[clip] DOWN" ;;
  log) exec tail -n 40 -f "$LOG" ;;
esac
