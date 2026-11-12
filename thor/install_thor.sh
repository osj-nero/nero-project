#!/usr/bin/env bash
# install_thor.sh -- Thor(Jetson) 에서 실행. 이 저장소의 thor/ 와 config/ 파일을 서버들이 기대하는 위치로 복사한다.
#   ~/grasp/                         CGN 서버, 후보 선별(B 단계), CLIP 접지 서버, 기동 스크립트
#   ~/nero/tools/phase_c_goalpose/   cuRobo 집기+놓기 계획 서버(:8021)
#   ~/nero/curobo_nero/              로봇 설정 yml
#   ~/vllm-venv/                     LLM(vLLM) 기동 스크립트
# 이미 있는 파일은 덮어쓰지 않는다(-n). 덮어쓰려면 FORCE=1 ./install_thor.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"
CP="cp -n"; [ "${FORCE:-0}" = "1" ] && CP="cp"
mkdir -p ~/grasp/yolo ~/grasp/logs ~/nero/tools/phase_c_goalpose ~/nero/curobo_nero ~/vllm-venv
$CP "$HERE"/grasp/*.py "$HERE"/grasp/*.sh ~/grasp/
$CP "$HERE"/grasp/clip_ground_server.py ~/grasp/yolo/
$CP "$HERE"/curobo/*.py ~/nero/tools/phase_c_goalpose/
$CP "$REPO"/config/*.yml ~/nero/curobo_nero/
$CP "$HERE"/vllm/*.sh ~/vllm-venv/
chmod +x ~/grasp/*.sh ~/vllm-venv/*.sh
if grep -q "__ROS2_WS__" ~/nero/curobo_nero/nero7_open.yml; then
    sed -i "s#__ROS2_WS__#${ROS2_WS:-$HOME/ros2_ws}#g" ~/nero/curobo_nero/*.yml
fi
echo "복사 완료. 모델 파일(README 의 '모델 파일' 절)을 ~/grasp 에 넣은 뒤 서버를 띄우세요."
