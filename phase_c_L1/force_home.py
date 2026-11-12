"""force_home.py — MoveIt이 start-state 자기충돌로 move_joints를 거부할 때(B단계는
avoid_collisions=False라 극단 손목 자세로 끝날 수 있음) 충돌검사 없이 저장된 observation
자세로 관절보간 궤적을 컨트롤러에 직접 보낸다. 모든 대기에 타임아웃 -- 실패 시 exit 1."""
import json, os, sys, time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

J = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
T = 10.0


def fail(n, msg):
    print(f"[force_home] 실패: {msg}", file=sys.stderr)
    n.destroy_node(); rclpy.shutdown(); sys.exit(1)


def spin_until(n, fut, timeout):
    t0 = time.time()
    while not fut.done() and time.time() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
    return fut.done()


rclpy.init(); n = Node("force_home")
cur = {}
def cb(m):
    for k, v in zip(m.name, m.position): cur[k] = v
n.create_subscription(JointState, "/joint_states", cb, qos_profile_sensor_data)
t0 = time.time()
while not all(j in cur for j in J) and time.time() - t0 < 5.0:
    rclpy.spin_once(n, timeout_sec=0.1)
if not all(j in cur for j in J):
    fail(n, "5초 내 /joint_states 미수신")

POSES = os.environ.get("NERO_POSES_FILE", os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "nero_robot", "saved_poses.json"))
obs = json.load(open(POSES))["observation"]["joints"]
home = [obs.get(j, 0.0) for j in J]
start = [cur[j] for j in J]

ac = ActionClient(n, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
if not ac.wait_for_server(timeout_sec=10.0):
    fail(n, "arm_controller 액션서버 없음")
g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = J
N = 40
for i in range(1, N + 1):
    a = i / N; p = JointTrajectoryPoint(); p.positions = [s + (h - s) * a for s, h in zip(start, home)]
    t = T * a; p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9)
    g.trajectory.points.append(p)
f = ac.send_goal_async(g)
if not spin_until(n, f, 5.0) or not f.result().accepted:
    fail(n, "goal 전송/수락 실패")
r = f.result().get_result_async()
if not spin_until(n, r, T + 10.0):
    fail(n, f"{T + 10:.0f}초 내 궤적 결과 없음")
code = r.result().result.error_code
print(f"[force_home] result error_code={code}", file=sys.stderr)
n.destroy_node(); rclpy.shutdown()
sys.exit(0 if code == 0 else 1)
