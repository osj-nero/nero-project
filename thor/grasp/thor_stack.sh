#!/usr/bin/env bash
# ============================================================================
# thor_stack.sh  —  THOR 로컬. MoveIt2 plan-only 스택을 "명령 한 번"으로.
#
#   tmux 안 씀. setsid 로 자체 프로세스 그룹을 만들어 백그라운드로 띄우고,
#   PGID 를 파일에 남긴다 -> down 이 그 그룹 전체(launch + rsp + jsp +
#   static_tf + move_group)를 한 방에 정리. (tmux 시절 남던 orphan 없음)
#
#   띄우는 것: headless_plan_test.launch.py
#     = robot_state_publisher + joint_state_publisher
#       + static_transform_publisher(world->base_link)
#       + move_group  (OMPL + STOMP + Pilz, pick_ik)
#     ros2_control/로봇 없음 — CGN grasp 파이프라인의 "궤적 계산" 전용.
#
# 사용 (Thor ssh 터미널):
#   ~/grasp/thor_stack.sh up       # 띄우고 /move_group 뜰 때까지 기다린 뒤 종료 (터미널 안 잡음)
#   ~/grasp/thor_stack.sh status   # 떠 있나 확인
#   ~/grasp/thor_stack.sh log      # move_group 로그 tail -f (Ctrl-C 로 빠짐, 스택은 유지)
#   ~/grasp/thor_stack.sh down     # 내리기
#   ~/grasp/thor_stack.sh restart
#
# up 이 끝나면 파이프라인:
#   ~/grasp/run_pipeline.sh ~/grasp/<scene>.npz
# ============================================================================
set -eo pipefail        # -u 금지: ROS setup.bash 가 unbound var 를 참조함

GRASP="$HOME/grasp"
LOGDIR="$GRASP/logs"
PGIDFILE="$GRASP/.thor_stack.pgid"
LOG="$LOGDIR/move_group.log"
LAUNCH_PKG="nero_gripper_moveit_config"
LAUNCH_FILE="headless_plan_test.launch.py"
mkdir -p "$LOGDIR"

_src() {
  source /opt/ros/jazzy/setup.bash
  source "$HOME/ros2_ws/install/setup.bash"
}

_running_nodes() { _src; ros2 node list 2>/dev/null || true; }

_is_up() { _running_nodes | grep -qx /move_group; }

_up() {
  if _is_up; then
    echo "[thor] 이미 실행 중 (/move_group). 재시작하려면: $0 restart"
    return 0
  fi
  # 자체 프로세스 그룹으로 백그라운드 실행
  setsid bash -c "
    source /opt/ros/jazzy/setup.bash
    source '$HOME/ros2_ws/install/setup.bash'
    cd /tmp
    exec ros2 launch $LAUNCH_PKG $LAUNCH_FILE
  " >"$LOG" 2>&1 &
  local child=$!
  # setsid 자식의 PGID = 그 자식의 PID (새 세션 리더)
  sleep 0.5
  local pgid
  pgid=$(ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ' || echo "$child")
  echo "$pgid" >"$PGIDFILE"
  echo "[thor] 기동 (PGID $pgid), 로그: $LOG"

  printf '[thor] /move_group 대기'
  for i in $(seq 1 45); do
    if _is_up; then echo ' — OK'; break; fi
    printf '.'; sleep 1
    if [ "$i" -eq 45 ]; then
      echo ' — 타임아웃'; echo "  마지막 로그:"; tail -n 20 "$LOG" | sed 's/^/    /'; return 1
    fi
  done
  _running_nodes | grep -E 'move_group|robot_state_publisher|joint_state' | sort -u | sed 's/^/  /'
  echo "[thor] 준비 완료 → ~/grasp/run_pipeline.sh ~/grasp/<scene>.npz"
}

_down() {
  local killed=0
  if [ -f "$PGIDFILE" ]; then
    local pgid; pgid=$(cat "$PGIDFILE")
    if [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-$pgid" 2>/dev/null || true
      killed=1
    fi
    rm -f "$PGIDFILE"
  fi
  # 안전망: 패턴으로 잔여 정리
  pkill -9 -f "ros2 launch $LAUNCH_PKG $LAUNCH_FILE" 2>/dev/null || true
  pkill -9 -f "moveit_ros_move_group/move_group"     2>/dev/null || true
  pkill -9 -f "robot_state_publisher --ros-args"     2>/dev/null || true
  pkill -9 -f "joint_state_publisher --ros-args"     2>/dev/null || true
  pkill -9 -f "static_transform_publisher 0 0 0 0 0 0 world base_link" 2>/dev/null || true
  for i in 1 2 3 4 5; do
    sleep 1
    pgrep -f "moveit_ros_move_group/move_group" >/dev/null 2>&1 || { \
      echo "[thor] 내림 완료$([ $killed = 1 ] && echo ' (PGID)')"; return 0; }
  done
  echo "[thor] 경고: move_group 프로세스가 아직 남아있음 — 수동 확인 필요"
}

case "${1:-status}" in
  up)      _up ;;
  down)    _down ;;
  restart) _down; sleep 1; _up ;;
  status)
    if _is_up; then
      echo "[thor] UP"
      _running_nodes | grep -E 'move_group|robot_state_publisher|joint_state|static_transform' | sort -u | sed 's/^/  /'
      [ -f "$PGIDFILE" ] && echo "  PGID $(cat "$PGIDFILE")"
    else
      echo "[thor] DOWN"
    fi
    ;;
  log)
    [ -f "$LOG" ] || { echo "로그 없음: $LOG"; exit 1; }
    exec tail -n 40 -f "$LOG"
    ;;
  *)
    echo "사용: $0 {up|down|restart|status|log}"; exit 1
    ;;
esac
