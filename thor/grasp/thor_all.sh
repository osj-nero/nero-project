#!/usr/bin/env bash
# THOR 로컬. 이 프로젝트에 필요한 Thor 서비스를 한 번에.
#   thor_stack (move_group+pick_ik+STOMP) · yolo_server(:8002 YOLO+SAM) · cgn_server(:8010) · clip_server(:8030 CLIP ViT-L/14)
# 사용:  ~/grasp/thor_all.sh {up|down|status}
# 2026-10-01: up 을 중복 기동에 안전하게 -- (1) flock 으로 동시 호출 직렬화, (2) 프로세스가 이미 있으면(로딩 중이라 /health 가 아직 안 나와도) 새로 띄우지 않고 응답을 기다린다.
set -uo pipefail
D="$HOME/grasp"
# 2026-10-01 YOLO(:8002 vlm_boxyolo) 는 옛 인지 경로(perception_node_sim, nero-robot MCP 의 검출 도구, discover_boxes.py)에만 쓰인다. 새 시스템(CLIP 접지)에는 불필요해 기본 OFF.
# 켜려면:  NERO_YOLO=1 ~/grasp/thor_all.sh up   (코드는 지우지 않았다)
YOLO="${NERO_YOLO:-0}"

_wait_health() {   # _wait_health <이름> <포트> <최대초>
    local t=0
    while [ "$t" -lt "$3" ]; do
        curl -s -o /dev/null -m 2 "http://127.0.0.1:$2/health" && { echo "[$1] 응답 확인"; return 0; }
        sleep 3; t=$((t+3))
    done
    echo "[$1] 응답 없음(${3}s)"; return 1
}

_ensure() {        # _ensure <이름> <프로세스 패턴> <포트> <up 스크립트> <최대대기초>
    if pgrep -f "$2" >/dev/null 2>&1; then
        echo "[$1] 프로세스 이미 있음 -- 새로 띄우지 않고 응답만 기다림"
        _wait_health "$1" "$3" "$5"
    else
        "$4" up
    fi
}

case "${1:-status}" in
  up)
    exec 9>/tmp/thor_all.lock; flock -w 600 9 || { echo "[thor_all] 다른 up 이 오래 실행 중 -- 중단"; exit 1; }
    "$D/thor_stack.sh" up
    if [ "$YOLO" = "1" ]; then
        _ensure yolo "vlm_boxyolo.py --host" 8002 "$D/yolo_server.sh" 150
    else
        echo "[yolo] OFF (옛 인지 경로 전용; 켜려면 NERO_YOLO=1)"
    fi
    _ensure cgn  "cgn_server.py"         8010 "$D/cgn_server.sh"  150
    _ensure clip "clip_ground_server.py --port" 8030 "$D/clip_server.sh" 150
    ;;
  down) "$D/clip_server.sh" down; "$D/cgn_server.sh" down; "$D/yolo_server.sh" down; "$D/thor_stack.sh" down ;;
  status)
    "$D/thor_stack.sh" status
    "$D/yolo_server.sh" status
    "$D/cgn_server.sh" status
    "$D/clip_server.sh" status
    ;;
  *) echo "사용: $0 {up|down|status}"; exit 1 ;;
esac
