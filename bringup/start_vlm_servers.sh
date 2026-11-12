#!/usr/bin/env bash
# YOLO 서버(8002) + VLM grasp 서버(8003)를 0.0.0.0으로 띄워서
# 연구실 PC 등 다른 네트워크의 클라이언트가 접속할 수 있게 한다.
#
# ⚠️ VLM grasp 서버(8003)는 이제 로컬에서 웨이트를 로드하지 않는다.
#    별도로 떠 있는 vLLM OpenAI 서버(기본 127.0.0.1:8005, Qwen3-VL-8B)를
#    HTTP로 호출하는 얇은 어댑터다. 먼저 tmux에서
#      ~/vllm-venv/serve_qwen3vl.sh
#    를 띄워야 /infer_grasp·/analyze_scene 등이 동작한다(안 떠 있으면
#    grasp는 라벨 휴리스틱 fallback, scene/placement/ground는 503).
#    포트/모델 오버라이드: 아래 VLLM_URL / VLLM_MODEL 환경변수.
#
# 사용법:
#   ./start_vlm_servers.sh          # 두 서버 백그라운드로 기동
#   ./start_vlm_servers.sh stop     # 두 서버 종료
#   ./start_vlm_servers.sh status   # 실행 상태 확인
#
# [2026-09-02] vlm_grasp_server(8003)에 open-vocab 검출 엔드포인트
# /detect_open_vocab (YOLO-World) 추가. 가중치는 YOLOWORLD_MODEL 환경변수
# (기본 yolov8l-worldv2.pt, 레포 루트에 있음). ultralytics 없으면 그
# 엔드포인트만 503, 나머지는 정상.
#   segmentation backend(estimate_object_geometry): 클라이언트(mcp_robot_
#   server) 쪽 SEG_BACKEND 환경변수 -- noop(기본)|depth_plane|sam. 이 스크립트
#   가 아니라 mcp_robot_server 를 띄우는 쪽에서 export.
#
# 로그: /tmp/nero_servers/*.log
# PID:  /tmp/nero_servers/*.pid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO_DIR="$SCRIPT_DIR/yolo"
PY="$HOME/miniconda3/envs/demo/bin/python"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8005/v1}"
VLLM_MODEL="${VLLM_MODEL:-qwen3-vl}"
RUN_DIR="/tmp/nero_servers"
mkdir -p "$RUN_DIR"

BOX_PID_FILE="$RUN_DIR/vlm_boxyolo.pid"
VLM_PID_FILE="$RUN_DIR/vlm_grasp_server.pid"
BOX_LOG="$RUN_DIR/vlm_boxyolo.log"
VLM_LOG="$RUN_DIR/vlm_grasp_server.log"

is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

start() {
    if is_running "$BOX_PID_FILE"; then
        echo "YOLO 서버(8002) 이미 실행 중 (PID $(cat "$BOX_PID_FILE"))"
    else
        cd "$YOLO_DIR"
        nohup "$PY" vlm_boxyolo.py --host 0.0.0.0 --port 8002 \
            --model best.pt --conf-box 0.75 \
            --model-coco yolov8n.pt --conf-coco 0.25 \
            > "$BOX_LOG" 2>&1 &
        echo $! > "$BOX_PID_FILE"
        echo "YOLO 서버(8002) 기동: PID $!"
    fi

    if is_running "$VLM_PID_FILE"; then
        echo "VLM 서버(8003) 이미 실행 중 (PID $(cat "$VLM_PID_FILE"))"
    else
        cd "$YOLO_DIR"
        nohup "$PY" vlm_grasp_server.py --host 0.0.0.0 --port 8003 \
            --vllm-url "$VLLM_URL" --vllm-model "$VLLM_MODEL" \
            > "$VLM_LOG" 2>&1 &
        echo $! > "$VLM_PID_FILE"
        echo "VLM 서버(8003) 기동: PID $! (vLLM 어댑터 → $VLLM_URL)"
    fi

    echo
    echo "로그: $BOX_LOG / $VLM_LOG"
    echo "연구실 PC에서 접속할 IP 후보:"
    hostname -I | tr ' ' '\n' | grep -v '^127\.' | grep -v '^$' | sed 's/^/  http:\/\//'
    echo
    echo "클라이언트 쪽 환경변수 예:"
    echo "  BOX_SERVER_URL=http://<위 IP>:8002"
    echo "  VLM_SERVER_URL=http://<위 IP>:8003"
}

stop() {
    for pid_file in "$BOX_PID_FILE" "$VLM_PID_FILE"; do
        if is_running "$pid_file"; then
            kill "$(cat "$pid_file")"
            echo "종료: PID $(cat "$pid_file") ($pid_file)"
        fi
        rm -f "$pid_file"
    done
}

status() {
    if is_running "$BOX_PID_FILE"; then
        echo "YOLO 서버(8002): 실행 중 (PID $(cat "$BOX_PID_FILE"))"
    else
        echo "YOLO 서버(8002): 중지됨"
    fi
    if is_running "$VLM_PID_FILE"; then
        echo "VLM 서버(8003): 실행 중 (PID $(cat "$VLM_PID_FILE"))"
    else
        echo "VLM 서버(8003): 중지됨"
    fi
}

case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *)      echo "사용법: $0 [start|stop|status]"; exit 1 ;;
esac
