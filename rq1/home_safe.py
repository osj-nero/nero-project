"""2026-11-09(사용자 보고: 놓은 뒤 홈 복귀가 충돌 검사 없이 움직여 손/카메라가 작업대에 걸려 로봇이 멈춤).
충돌을 피하는 홈 복귀: cuRobo /jmove(8021 사본)로 관측 자세까지 작업대(+놓인 물체)를 피하는 관절 경로를 계획해 실행한다.
계획이 안 되면 실행하지 않고 'FAIL' 을 출력(충돌 무시 복귀로 넘어가지 않음 -- 호출 쪽이 기록하고 사람이 확인).
사용: python3 home_safe.py '<obstacles json>'   (obstacles = [{id, xyz, raw, yaw}])"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import json, os, sys, time, urllib.request
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
sys.path.insert(0, "" + NERO_ROOT + "/phase_c_L1")
from discover_boxes import Discoverer
ARM_J = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
POSES_FILE = os.path.join(os.path.expanduser("~/.local/share"), "nero_robot", "saved_poses.json")
URL = os.environ.get("NERO_RT_CUROBO", "http://127.0.0.1:8021")


def main():
    obs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    goal = [json.load(open(POSES_FILE))["observation"]["joints"][j] for j in ARM_J]
    rclpy.init(); n = Node("home_safe"); got = {}
    n.create_subscription(JointState, "/joint_states", lambda m: got.update(q=[dict(zip(m.name, m.position))[j] for j in ARM_J]) if all(j in m.name for j in ARM_J) else None, qos_profile_sensor_data)
    t0 = time.time()
    while "q" not in got and time.time() - t0 < 5: rclpy.spin_once(n, timeout_sec=0.1)
    if "q" not in got: print("FAIL joint_states 없음"); return
    q = got["q"]
    if max(abs(a - b) for a, b in zip(q, goal)) < 0.05: print("OK 이미 관측 자세"); return
    mg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02      # 2026-11-09 v4: 여유를 인자로(놓은 직후 손이 물체/받침대에 붙어 있어 2cm 면 시작이 충돌 판정)
    req = json.dumps(dict(start_q=q, goal_q=goal, obstacles=obs, margin=mg)).encode()
    try: r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/jmove", req, {"Content-Type": "application/json"}), timeout=30).read())
    except Exception as e: print(f"FAIL 계획 요청 {e!r}"); return
    if not r.get("ok"):
        diag = ""
        try:      # 2026-11-09 진단: 시작 자세가 장애물(여유 포함)과 겹치는지 -- 링크별 최소 거리(음수=침범)
            bx = [dict(id=o["id"], xyz=o["xyz"], dims=o["raw"], yaw=o.get("yaw", 0.0)) for o in obs]
            d = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/sphere_dist", json.dumps(dict(q=[q, goal], boxes=bx)).encode(), {"Content-Type": "application/json"}), timeout=30).read())
            diag = " 시작/끝 최소거리 " + json.dumps({k: [v["min"], v["link"]] for k, v in d["per_box"].items()}, ensure_ascii=False)
        except Exception as e: diag = f" 진단 실패 {e!r}"
        print(f"FAIL 계획 실패 {r.get('why')}{diag} q={[round(x, 3) for x in q]}"); return
    tr, tt, vv, T = Discoverer.retime_minjerk(np.asarray(r["traj"], float), vmax=1.0, amax=3.0)
    ac = ActionClient(n, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
    if not ac.wait_for_server(timeout_sec=5): print("FAIL 액션 서버 없음"); return
    g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = ARM_J
    for i, w in enumerate(tr):
        p = JointTrajectoryPoint(); p.positions = [float(x) for x in w]; p.velocities = [float(x) for x in vv[i]]; t = tt[i] + 0.05
        p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9); g.trajectory.points.append(p)
    fut = ac.send_goal_async(g); t0 = time.time()
    while not fut.done() and time.time() - t0 < 5: rclpy.spin_once(n, timeout_sec=0.05)
    if not fut.done() or not fut.result().accepted: print("FAIL 궤적 거부"); return
    rf = fut.result().get_result_async(); t0 = time.time()
    while not rf.done() and time.time() - t0 < T + 8: rclpy.spin_once(n, timeout_sec=0.05)
    print(f"OK 충돌 회피 복귀 {T:.1f}s code={rf.result().result.error_code if rf.done() else None}")


if __name__ == "__main__":
    try: main()
    finally:
        try: rclpy.shutdown()
        except Exception: pass
