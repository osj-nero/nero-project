#!/usr/bin/env bash
# start_rviz_rt.sh [--no-cloud] -- 실시간 루프 3D 시각화: 빠른 TF(rsp_fast) + 브리지 노드(rt_rviz_bridge.py) + RViz2(~/nero/rviz/nero_rt.rviz).
# 단독 실행도 되고, start_nero_isaac_all.sh 가 ENABLE_RVIZ=true 일 때 백그라운드 job 으로 같이 띄운다.
# 종료: Ctrl+C / TERM / RViz 창 닫기 -> 이 스크립트가 띄운 것(RViz, 브리지, rsp_fast)을 모두 정리한다(스택의 다른 프로세스는 건드리지 않음).
# rsp_fast: 스택의 robot_state_publisher 는 TF 를 15Hz 로만 내서 RViz 에서 로봇이 끊겨 보인다 -> 같은 URDF/joint_states 로 100Hz TF 를 추가로 낸다.
source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash 2>/dev/null
export ROS_LOCALHOST_ONLY=1 DISPLAY=${DISPLAY:-:1}
[ -f "$HOME/.config/fastdds/nero_profile.xml" ] && export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/.config/fastdds/nero_profile.xml"
cd "$(dirname "$(readlink -f "$0")")"; pids=()
cleanup() { trap - EXIT INT TERM; kill "${pids[@]}" 2>/dev/null; sleep 1; kill -9 "${pids[@]}" 2>/dev/null; echo "RViz 시각화 종료"; }
trap cleanup EXIT INT TERM
alive() { ps -eo args | grep -F "$1" | grep -v grep | grep -v "start_rviz_rt" >/dev/null; }

# 1) 빠른 TF: /robot_description 이 올라올 때까지 최대 90초 기다린다(스택의 rsp 가 먼저 떠야 함)
if ! alive "robot_state_publisher_fast"; then
  if timeout 90 python3 - <<'PY'
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import String
rclpy.init(); n=Node('rd_get'); box={}
n.create_subscription(String,'/robot_description',lambda m:box.__setitem__('d',m.data),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
while 'd' not in box: rclpy.spin_once(n,timeout_sec=0.2)
open('/tmp/nero_urdf.xml','w').write(box['d'])
PY
  then
    /opt/ros/humble/lib/robot_state_publisher/robot_state_publisher --ros-args -r __node:=robot_state_publisher_fast -p publish_frequency:=100.0 -p robot_description:="$(cat /tmp/nero_urdf.xml)" > /tmp/rsp_fast.log 2>&1 & pids+=($!)
  else echo "⚠️ /robot_description 을 못 받음 -- 빠른 TF 없이 계속(로봇이 끊겨 보일 수 있음)"; fi
fi
# 2) 브리지
if ! alive "rt_rviz_bridge.py"; then python3 rt_rviz_bridge.py "$@" > /tmp/rt_rviz_bridge.log 2>&1 & pids+=($!); echo "브리지 시작"; fi
# 2.5) RViz 의 Image 디스플레이는 로드 시점에 토픽이 보이지 않으면 구독을 못 만들고 계속 비어 있다(2026-10-07 실측) -> 브리지가 /nero_rt/panel 을 내보낼 때까지 기다린 뒤 RViz 를 켠다
for i in $(seq 1 60); do ros2 topic list 2>/dev/null | grep -qx /nero_rt/panel && break; sleep 0.5; done
# 3) RViz (백그라운드 + wait: 종료 신호를 즉시 처리하려고). 2026-10-08: RViz 가 떠 있는데 /nero_rt/panel 구독이 0 으로 남는 경우(HUD/로봇 시점이 사라짐)가 반복돼, 구독 수를 감시해 RViz 만 다시 띄운다.
RESTART=/tmp/rviz_rt_restart.$$.flag; PIDF=/tmp/rviz_rt.$$.pid; rm -f $RESTART
start_rviz() { rviz2 -d ./rviz/nero_rt.rviz > /tmp/rviz2.log 2>&1 & RV=$!; echo $RV > $PIDF; pids+=($RV); }
start_rviz
# 4) 브리지 감시: 죽으면 다시 띄운다(2026-10-08 경쟁 상태로 죽은 적이 있다).
( while kill -0 "$(cat $PIDF 2>/dev/null || echo 0)" 2>/dev/null; do sleep 3; if ! ps -eo args | grep -F "rt_rviz_bridge.py" | grep -v grep | grep -v start_rviz_rt >/dev/null; then echo "$(date +%T) 브리지 종료 감지 -> 재기동" >> /tmp/rt_rviz_bridge.restart.log; cp /tmp/rt_rviz_bridge.log /tmp/rt_rviz_bridge.prev.log 2>/dev/null; python3 rt_rviz_bridge.py "$@" > /tmp/rt_rviz_bridge.log 2>&1 & fi; done ) & WD=$!; pids+=($WD)
# 5) 구독 감시: 패널 토픽의 구독자 수를 rclpy 로 직접 센다(2026-10-08: `ros2 topic info` 는 낡은 데몬 정보로 구독 0 을 잘못 보고해 정상인 RViz 를 40초마다 재기동했다). 연속 6번(약 30초) 0 이면 RViz 만 다시 띄우고, 재기동 사이는 60초 이상 띄운다. NERO_RVIZ_SUBWD=0 으로 끈다
# 2026-11-02: 기본 OFF. 구독 수(count_subscribers)가 패널이 실제로 표시되는 중에도 0 으로 나와(발견 지연·구독 깜빡임) 잘 뜬 RViz 를 죽였다(17:38·17:40 오탐 + 종료 시 세그폴트). 켜려면 NERO_RVIZ_SUBWD=1
if [ "${NERO_RVIZ_SUBWD:-0}" = "1" ]; then
( z=0; nrs=0; MAXRS=${NERO_RVIZ_SUBWD_MAX:-2}; sleep 40; while true; do sleep 5; RVP=$(cat $PIDF 2>/dev/null || echo 0); kill -0 $RVP 2>/dev/null || break
  c=$(timeout 8 python3 - <<'PY' 2>/dev/null
import rclpy,time
from rclpy.node import Node
rclpy.init(); n=Node('sub_probe'); t=time.time()
while time.time()-t<2.0: rclpy.spin_once(n,timeout_sec=0.1)
print(n.count_subscribers('/nero_rt/panel'))
PY
)
  # 프로브 노드 자신은 발행만 보는 구독이 아니므로 센 값이 곧 RViz 등 다른 구독자 수다
  if [ "$c" = "0" ]; then z=$((z+1)); else z=0; fi
  # 2026-11-02: 재기동 최대 횟수(NERO_RVIZ_SUBWD_MAX, 기본 2, 0=무제한). 한도에 닿으면 로그만 남기고 감시를 끝낸다(RViz 는 그대로 둠). 이전: 구독이 안 잡히면 105초마다 무한 재기동했다
  if [ $z -ge 6 ] && [ "$MAXRS" -gt 0 ] && [ $nrs -ge "$MAXRS" ]; then echo "$(date +%T) 패널 구독 0 이 계속되나 재기동 한도($MAXRS회) 도달 -> 재기동 안 함, 감시 종료(로그만)" >> /tmp/start_rviz.log; break; fi
  if [ $z -ge 6 ]; then nrs=$((nrs+1)); echo "$(date +%T) 패널 구독 0(rclpy 직접 확인) -> RViz 재기동 ($nrs/$MAXRS)" >> /tmp/start_rviz.log; touch $RESTART; kill $RVP 2>/dev/null; z=0; sleep 60; fi
done ) & pids+=($!)
fi
while true; do
  wait $RV     # RViz 창을 닫으면 여기서 끝나고 cleanup 이 나머지를 정리한다
  if [ -f $RESTART ]; then rm -f $RESTART; sleep 2; start_rviz; continue; fi
  break
done
