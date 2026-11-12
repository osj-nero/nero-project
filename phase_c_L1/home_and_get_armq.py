#!/usr/bin/env python3
"""home_and_get_armq.py — go_home MCP 툴과 동일하게 /arm_command 에 {'action':'home'}
직접 발행(place_via_ros.py/discover_boxes.py 와 같은 패턴, MCP 왕복 없이 스크립트 안에서
체이닝하기 위함) 후, 정착된 arm_q 를 읽어서 stdout 에 한 줄로 찍는다.
다중물체 stack_discovered.sh 가 물체 하나 pick+place 끝날 때마다 이걸 불러서
다음 물체의 B단계(NERO_ARM_Q) 시드를 매번 새로 잡는다 -- 안 하면 이전 arm_q 로 다음
물체를 계획해서 STOMP transit 시작점이 실제 위치랑 어긋남.

2026-09-18 추가(사용자 지적 "verification 이후 pick 전 딜레이"): place_via_ros.py 가
부르는 planning_node._do_place 는 place 직후 검증을 위해 이미 observation 자세로
복귀한다(verify-observation). 그 직후 여기서 또 {"action":"home"} 을 보내면
planning_node._home_sequence() 가 (거리 0짜리 이동인데도) 매번 그리퍼 재오픈+
GRIPPER_DELAY(2.0s)+하드코딩 settle(1.5s) 를 다시 태워 최소 3.5초+ 순오버헤드가 남
-- 이미 B/C단계가 선계산돼 있어 이 재홈이 아무것도 새로 얻지 못하는데도 그렇다.
그래서 여기서 먼저 joint_states 를 훑어 이미 observation 근처(각 관절 오차
<OBS_TOL rad)면 home 액션 자체를 안 보내고 그 값을 그대로 반환한다 -- 그 값이 아니면
(직전 place 의 verify-observation 복귀가 실패했거나 애초에 다른 자세였던 경우)
안전하게 기존 전체 home 경로로 폴백한다.

사용: python3 home_and_get_armq.py   ->  "0.00080 -0.99870 ... " (7개 공백구분)
"""
import json, os, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

ARM_J = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
OBS_TOL = 0.05  # rad -- planning_node._home_sequence() 재홈 스킵 판정 허용오차
POSES_FILE = os.environ.get(
    "NERO_POSES_FILE",
    os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
                 "nero_robot", "saved_poses.json"))


def _load_observation_ref():
    """planning_node.py 의 _get_observation_pose_joints() 와 같은 파일을 읽는다 --
    하드코딩 상수로 안 두는 이유도 그쪽과 동일(save_pose로 재저장해도 자동 추종)."""
    try:
        with open(POSES_FILE, "r", encoding="utf-8") as f:
            joints = json.load(f).get("observation", {}).get("joints")
        if not joints:
            return None
        return [joints.get(j, 0.0) for j in ARM_J]
    except Exception:
        return None


def main():
    rclpy.init()
    n = Node("home_and_get_armq")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
    got = {}

    def on_js(m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM_J):
            got["arm_q"] = [float(d[j]) for j in ARM_J]

    result = {}

    def on_result(m):
        try:
            result["data"] = json.loads(m.data)
        except Exception:
            pass

    pub = n.create_publisher(String, "/arm_command", qos)
    n.create_subscription(String, "/pick_result", on_result, qos)
    n.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)
    # ⚠ 2026-09-18 버그이력(place_via_ros.py 와 동일): 발행 전 구독자 연결을 확인 안
    # 하면 ROS2 디스커버리가 안 끝난 채 publish 해서 메시지가 사라진다(planning_node
    # 콘솔에 [CMD] 수신 로그 자체가 안 찍힘, 실측 확인). 최대 5초까지 연결 대기.
    t_disc = time.time()
    while pub.get_subscription_count() == 0 and time.time() - t_disc < 5.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print("[home] 경고: 5초 내 /arm_command 구독자 없음", file=sys.stderr)

    # 재홈 스킵 판정: 명령을 보내기 전에 현재 joint_states 를 먼저 확인.
    # 2026-09-18: 2.0s -> 4.0s. 실측으로 이 스크립트를 새로 띄울 때마다 /joint_states
    # 구독이 DDS 디스커버리를 새로 거쳐야 해서(위 /arm_command 발행자 디스커버리와 별개
    # 엔티티), 2.0s로는 가끔 못 받아 실제로는 observation 근처였는데도 못 알아채고
    # 풀 home으로 폴백하는 게 관찰됨(3번 중 1번만 스킵 성공). 놓쳐도 기존 폴백 경로로
    # 안전하게 떨어지니 넉넉하게 늘림.
    obs_ref = _load_observation_ref()
    t_peek = time.time()
    while "arm_q" not in got and time.time() - t_peek < 4.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    if obs_ref and "arm_q" in got and max(abs(a - b) for a, b in zip(got["arm_q"], obs_ref)) < OBS_TOL:
        print("[home] 이미 observation 근처 -- home 액션 생략(joint_states 그대로 사용)", file=sys.stderr)
        print(" ".join("%.5f" % x for x in got["arm_q"]))
        rclpy.shutdown(); return

    # 2026-09-23(사용자 지시): {"action":"home"}은 실측으로 all-zero(천장을 보는 자세)로
    # 가는 경우가 확인돼 더 이상 안 쓴다 -- 절대 all-zero로 안 가도록 저장된 observation
    # 관절값을 move_joints로 직접 보낸다(discover_boxes.py Discoverer.home()과 동일 수정).
    obs_joints = {}
    try:
        obs_joints = json.load(open(POSES_FILE))["observation"]["joints"]
    except Exception:
        pass
    if not obs_joints:
        print("[home] 경고: observation 저장 자세 없음 -- home 액션으로 폴백(all-zero 위험)",
              file=sys.stderr)
        msg = String(); msg.data = json.dumps({"action": "home"})
    else:
        msg = String(); msg.data = json.dumps({"action": "move_joints", "joints": obs_joints})
    pub.publish(msg)
    print(f"[home] /arm_command 발행: {msg.data}", file=sys.stderr)
    t0 = time.time()
    while time.time() - t0 < 30 and "data" not in result:
        rclpy.spin_once(n, timeout_sec=0.2)
    for _ in range(int(os.environ.get('HOME_SETTLE_SPINS', '20'))):
        rclpy.spin_once(n, timeout_sec=0.1)  # joint_states 정착 (기본 2.0s; arm_q 를 안 쓰는 백그라운드 호출은 HOME_SETTLE_SPINS=3 으로 줄임)

    if obs_joints and result.get("data", {}).get("status") != "success" and "arm_q" in got \
            and max(abs(got["arm_q"][k] - obs_joints.get(j, 0.0)) for k, j in enumerate(ARM_J)) > 0.08:
        print("[home] move_joints 실패 -- force_home.py(충돌검사 없는 직접 궤적)로 폴백", file=sys.stderr)
        import subprocess
        try:
            subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_home.py")],
                           stdout=sys.stderr, check=False, timeout=45)
        except subprocess.TimeoutExpired:
            print("[home] force_home.py 45초 타임아웃", file=sys.stderr)
        got.pop("arm_q", None)
        for _ in range(30):
            rclpy.spin_once(n, timeout_sec=0.1)

    if "arm_q" not in got:
        print("[home] arm_q 못 얻음", file=sys.stderr)
        rclpy.shutdown(); sys.exit(1)
    print(" ".join("%.5f" % x for x in got["arm_q"]))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
