"""release_here.py -- 그리퍼를 그 자리에서 연다 (pick 전용 모드: 파지 판정 후 물체를 원래 자리 근처에 떨군다, 2026-09-29).
사용: python3 release_here.py   출력: opened / not_opened (종료코드는 항상 0)"""
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

GRIP = ["gripper_joint1", "gripper_joint2"]
GOPEN = [1.0, -1.0]
rclpy.init(); n = Node("release_here"); st = {}
n.create_subscription(JointState, "/joint_states", lambda m: [st.__setitem__(a, b) for a, b in zip(m.name, m.position)], 10)
gc = ActionClient(n, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
gc.wait_for_server(timeout_sec=10)
for _ in range(10):
    rclpy.spin_once(n, timeout_sec=0.1)
ok = False
for k in range(3):
    jt = JointTrajectory(); jt.joint_names = list(GRIP)
    p = JointTrajectoryPoint(); p.positions = [float(x) for x in GOPEN]
    p.time_from_start.sec = 0; p.time_from_start.nanosec = 400_000_000
    jt.points = [p]
    g = FollowJointTrajectory.Goal(); g.trajectory = jt
    fut = gc.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    t0 = time.time()
    while time.time() - t0 < 1.5:
        rclpy.spin_once(n, timeout_sec=0.05)
        if st.get("gripper_joint1", 0) > 0.040:
            ok = True; break
    if ok:
        break
print("opened" if ok else "not_opened")
rclpy.shutdown()
