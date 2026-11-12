#!/usr/bin/env bash
# ============================================================================
# nero_pc_control.sh — PC 자연어 로봇 제어 스택 (planning_node + nero-robot MCP)
#
# 이걸 띄우면 Claude 에게 자연어로 조인트/픽 명령을 줄 수 있다:
#   "옵저베이션 자세로 이동해"      -> go_home  (= 저장된 observation 자세)
#   "joint1 을 0.5 로"              -> move_joints
#   "빨간 컵 집어"                  -> pick_object
#
# 체인:
#   Claude Code ──http──▶ nero-robot MCP(:9000) ──/arm_command──▶ planning_node
#            ──▶ move_group(Humble) ──▶ ros2_control ──▶ Isaac Sim
#
# 선행 조건: start_nero_isaac_all.sh 로 아래가 이미 떠 있어야 함
#   Isaac Sim(▶재생) · robot_state_publisher · ros2_control_node
#   · spawn_controllers · move_group · perception_node_sim
#   (이 스크립트는 그 위에 planning_node 와 MCP HTTP 서버만 얹는다.)
#
# 이 스택은 Thor plan-only 스택(thor_planning.sh)과 독립 — 섞여도 무방.
#
# 사용법:
#   ./nero_pc_control.sh up        # planning_node + mcp(:9000) 를 tmux 로
#   ./nero_pc_control.sh status
#   ./nero_pc_control.sh down
#   ./nero_pc_control.sh log       # tmux attach (Ctrl-b d 로 빠짐)
# 그 다음 Claude Code 에서:  /mcp  (nero-robot 이 connected 인지 확인)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="nero_pc"
SRC='source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash'

_have_node() { bash -lc "$SRC && ros2 node list 2>/dev/null" | grep -qx "$1"; }

_check_prereq() {
  local bad=0
  for n in /robot_state_publisher /move_group; do
    if _have_node "$n"; then echo "  ✓ $n"; else echo "  ✗ $n  (start_nero_isaac_all.sh 먼저)"; bad=1; fi
  done
  return $bad
}

case "${1:-status}" in
  up)
    echo "[pc] 선행 노드 확인:"
    _check_prereq || { echo "[pc] 선행 스택이 없다. start_nero_isaac_all.sh 를 먼저 실행."; exit 1; }

    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[pc] tmux '$SESSION' 이미 실행 중 (재시작은 down 먼저)"; exit 0
    fi
    tmux new-session -d -s "$SESSION" -n planning \
      "bash -lc '$SRC && exec ros2 run sj_pickplace planning_node --ros-args -p use_moveit2:=true'"
    sleep 3
    tmux new-window -t "$SESSION" -n mcp \
      "bash -lc 'exec $HERE/run_mcp_robot_server_http.sh'"

    # 실시간 pick 루프 MCP(:9100, nero-rt): 자연어 -> 대상/제외 문구 -> 자율 탐색 pick. ROS 선행 노드와 무관(루프 자체가 rclpy 를 쓴다)
    # 2026-11-03: NERO_RT_MCP_PLUS=1 이면 개편판(~/nero/mcp/rt_plus.py: rt_mcp_server 를 감싸 호출 기록·응답 보완·설명 갱신, phase_c 무수정)을 띄운다. 기본 0 = 기존 그대로
    NERO_ROOT="${NERO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    if [ "${NERO_RT_MCP_PLUS:-0}" = "1" ]; then RT_CMD="cd $NERO_ROOT/mcp && exec python3 rt_plus.py"; else RT_CMD="cd $NERO_ROOT/phase_c_L1 && exec python3 rt_mcp_server.py"; fi
    tmux new-window -t "$SESSION" -n rt "bash -lc '$RT_CMD'"

    printf '[pc] planning_node 대기'
    for i in $(seq 1 20); do
      _have_node /planning_node && { echo ' — OK'; break; }
      printf '.'; sleep 1
    done
    printf '[pc] MCP :9000 대기'
    for i in $(seq 1 20); do
      if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9000/mcp | grep -qE '2..|4..|405|406'; then
        echo ' — OK (listening)'; break
      fi
      printf '.'; sleep 1
      [ "$i" -eq 20 ] && echo ' — 아직 안 뜸 (log 확인)'
    done
    echo "[pc] 완료. Claude Code 에서 /mcp 로 nero-robot connected 확인."
    ;;
  down)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "[pc] '$SESSION' 종료" || echo "[pc] '$SESSION' 없음"
    pkill -f 'sj_pickplace planning_node' 2>/dev/null || true
    pkill -f 'mcp_robot_server' 2>/dev/null || true
    pkill -f 'rt_mcp_server' 2>/dev/null || true
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "[pc] tmux '$SESSION': UP" || echo "[pc] tmux '$SESSION': DOWN"
    for n in /robot_state_publisher /move_group /planning_node /perception_node_sim; do
      _have_node "$n" && echo "  ✓ $n" || echo "  ✗ $n"
    done
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9000/mcp 2>/dev/null || echo 000)
    code2=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9100/mcp 2>/dev/null || echo 000)
    echo "  MCP :9100/mcp (nero-rt) -> HTTP $code2"
    echo "  MCP :9000/mcp  -> HTTP $code  ($([ "$code" = 000 ] && echo 'DOWN' || echo 'listening'))"
    ;;
  log)
    exec tmux attach -t "$SESSION"
    ;;
  *)
    echo "사용: $0 {up|down|status|log}"; exit 1
    ;;
esac
