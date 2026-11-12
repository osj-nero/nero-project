#!/usr/bin/env bash
# apply_paths.sh <ros2_ws 경로> -- config/*.yml 안의 __ROS2_WS__ 자리표시자를 실제 ROS 2 작업공간 경로로 바꾼다.
# 예: ./config/apply_paths.sh $HOME/ros2_ws      (agx_arm_sim 을 $HOME/ros2_ws/src 아래에 받은 경우)
set -euo pipefail
WS="${1:?사용법: apply_paths.sh <ros2_ws 경로>}"
cd "$(dirname "$0")"
[ -d "$WS/src/agx_arm_sim" ] || echo "경고: $WS/src/agx_arm_sim 이 없습니다. agx_arm_sim 을 먼저 받으세요." >&2
sed -i "s#__ROS2_WS__#${WS}#g" nero.yml nero7.yml nero7_open.yml
grep -n "asset_root_path\|urdf_path" nero7_open.yml
