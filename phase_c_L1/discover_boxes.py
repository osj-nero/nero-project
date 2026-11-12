#!/usr/bin/env python3
"""discover_boxes.py — PC. scan_for_boxes(하드코딩된 룰베이스 스윕) 대신, 우리 자체
joint1 스윕으로 시야 밖 물체를 찾는다. MCP 안 거치고 /arm_command 직접 발행(place_via_ros.py
와 동일 패턴)으로 한 rclpy 노드에서 전부 처리 -- 속도를 룰베이스와 맞추기 위한 핵심 설계.

2026-09-18 개정 (사용자 지시, coarse-to-fine 2단계):
  1단계(러프 와이드 스윕): joint1 가동범위 전체를 큰 스텝으로 빠르게 훑으며 "대략 이
     방향에 뭔가 있다"는 근사 위치만 잡는다. CGN 안 부름(느려짐 방지) -- 위치 추정용
     세그멘테이션/포인트클라우드만.
  2단계(정밀 재방문): 1단계에서 잡은 대략 위치마다 joint1을 그 방향으로 직접 겨냥
     (atan2 로 방위각 계산 -> j1 로 변환)해서 카메라를 그 물체 쪽으로 재조준한 뒤,
     그 자리에서 다시 정밀 캡처 + CGN 추론(구체 위치 + grasp 후보 확정). 재방문 중
     원래 안 찾던 물체가 같이 보이면 그것도 덤으로 확정한다.
  기존(1단계 겨우 5개 각도만 훑고 그 자리에서 바로 CGN)는 커버리지가 좁고, 러프
  탐색과 정밀 파지추정이 뒤섞여 있었음 -- 이제 "대략 어디 있는지 넓게 훑기"와 "거기로
  가서 자세히 보기"를 분리.

scan_for_boxes 와의 차이(이게 "이득"): scan_for_boxes 는 위치만 큐에 저장하고 grasp
방향은 나중에(pick_object 호출 시점) 따로 판단한다. 이 스크립트는 2단계(정밀 재방문)에서
위치 확정과 동시에 CGN grasp 후보까지 그 자리에서 확보 -- 다시 보러 갈 필요가 없다.

사용: python3 discover_boxes.py --labels box:6 \
        --wide-sweep=-2.3,-1.53,-0.77,0.0,0.77,1.53,2.3 \
        --dedupe-dist 0.08 --out ~/grasp/discovered.json
출력: ~/grasp/discovered.json {"boxes": {"box_1": {"world":[x,y,z], ...}, ...}, "arm_q":[...]}
      + Thor 에 {id}_grasps.json (CGN 후보, 기존 스키마와 동일 -- B/C단계 바로 사용 가능)
"""
import os as _ng_os; GRASP_DIR = _ng_os.environ.get("NERO_GRASP_DIR") or _ng_os.path.expanduser("~/grasp")  # shared work dir on PC and Thor (override: NERO_GRASP_DIR)
import argparse, colorsys, glob, json, math, os, re, shlex, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as _cf_wait, FIRST_COMPLETED
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_multi_npz import multi_detect, quat_to_R, BOTTLE_LABELS, _overlap_ratio  # 재사용
from agn_detect import agnostic_masks   # 2026-09-30 이름 없는 물체 탐지(NERO_AGNOSTIC=1 일 때만 사용)

CGN_URL_LOCAL = "http://127.0.0.1:8010/infer_multi"   # ssh thor 안에서 부르므로 localhost
NBR_FILTER = False  # 씬 점군(SAM 마스크) 기반 이웃 grasp 필터 -- 2026-09-29 효과 미미(0~7%)로 기본 OFF, --nbr-filter 로 켬
J1_LIMIT = 2.60  # 실제 한계 2.70526 보다 살짝 안쪽 (여유)

# [2026-09-24 추가/수정, 사용자 지적] 1단계 러프 스윕이 joint1만 돌려서, 프레임
# 위/아래로 잘린 박스(카메라 화각 밖 상/하단)를 놓치는 경우가 실측 확인됨. 매 joint1
# 스텝마다 joint7도 같이 스윕해서 세로 커버리지를 넓힌다.
#
# 최초엔 observation 기준 +/-0.15 델타로 시도했으나, joint7 상한(1.5708)에 거의
# 붙어있는 observation 값(1.5491) 때문에 + 방향은 거의 즉시 한계에 클램프돼 효과가
# 없었고 실측(스크린샷 비교)으로 확인함. 대신 **joint4를 1.399->1.8로 올려서 기본
# 시야 자체를 근처 워크스페이스로 낮추고(사용자 지시, saved_poses.json observation
# 갱신)**, joint7은 절대값 기준 [J7_SWEEP_LOWER, observation 값(상한)] 사이를
# J7_SWEEP_STEPS개로 나눠 스윕한다(사용자 지시: "지금을 상한, 1.1을 하한으로 3번").
# planning_node.py의 baseline scan_box(_scan_box_sequence)에도 동일 범위를 반영
# (SCAN_JOINT7_SWEEP_LOWER/STEPS, 같은 값 유지할 것 -- 두 조건의 씬 파악 능력을
# 동일하게 맞춰야 비교실험이 "탐색 방식" 차이만 측정하지, 우연한 시야 차이로 왜곡되지
# 않는다).
J7_SWEEP_LOWER = 1.1
J7_SWEEP_STEPS = int(os.environ.get("NERO_J7_STEPS", "3"))   # 2026-09-30 j7 스윕 개수(기본 3). 스윕 줄이기 실험(sweep_coverage_test.py): 7x2 가 13x3 과 같은 재현율(48/48)에 시간 절반


def bearing_to_j1(x, y):
    """base_link (x,y) 를 향하도록 joint1 각도 추정.
    관측 자세(j1=0)에서 카메라는 base_link -X 방향(bearing=pi)을 보고, joint1 을 돌리면
    그만큼 시야 방향도 같이 회전한다는 가정(1단계 러프 스윕 실측과 대체로 일치, 완전
    정밀할 필요 없음 -- 어차피 "더 잘 보이는 쪽으로 재조준"이 목적이라 물체가 그냥
    프레임 안에만 들어오면 됨)."""
    bearing = math.atan2(y, x)
    j1 = bearing - math.pi
    while j1 > math.pi:
        j1 -= 2 * math.pi
    while j1 < -math.pi:
        j1 += 2 * math.pi
    return float(np.clip(j1, -J1_LIMIT, J1_LIMIT))


class Discoverer(Node):
    def __init__(self):
        super().__init__("discover_boxes")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.buf = Buffer(); TransformListener(self.buf, self)
        self.got = {}
        self.pub_cmd = self.create_publisher(String, "/arm_command", qos)
        self.create_subscription(String, "/pick_result", self._on_result, qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/color/image_raw", self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self._result = {}

    def _on_js(self, m):
        ARM_J = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM_J):
            self.got["arm_q"] = [float(d[j]) for j in ARM_J]; self._js_n = getattr(self, "_js_n", 0) + 1

    def _on_info(self, m):
        self.got["K"] = np.array(m.k).reshape(3, 3)

    def _on_depth(self, m):
        d = np.frombuffer(m.data, dtype=np.float32 if m.encoding == "32FC1" else np.uint16)
        d = d.reshape(m.height, m.width).astype(np.float32)
        if m.encoding == "16UC1":
            d = d / 1000.0
        self.got["depth"] = d; self.got["depth_stamp"] = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec      # 2026-10-28: 프레임 헤더 시각 기록(TF 시각 정렬 점검용, 동작 불변)

    def _on_rgb(self, m):
        arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
        self.got["rgb"] = arr[..., ::-1].copy() if m.encoding.startswith("bgr") else arr.copy()
        self.got["rgb_stamp"] = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec

    def _on_result(self, m):
        try:
            self._result["data"] = json.loads(m.data)
        except Exception:
            pass

    # 2026-09-23(사용자 지시): "home"(전 관절 0, 천장을 봄)과 "observation"(저장된
    # 관측 자세)은 이름부터 명확히 구분한다 -- 이 둘을 같은 걸로 착각해서 스윕이
    # 천장을 보고 도는 조용한 실패가 반복 재현됐다. _FORBIDDEN_HOME_TOL 이내로 전
    # 관절이 0에 가까우면 "금지된 home 포즈"로 간주하고 구조적으로 거부한다.
    _FORBIDDEN_HOME_TOL = 0.05  # rad

    @staticmethod
    def _looks_like_forbidden_home(arm_q):
        """전 관절이 0 근처(=금지된 home 포즈)인지 판정. observation은 joint2/4/7이
        크게 벗어나 있어(-0.999/1.399/1.549) 정상 상황에서 이 함수가 True를 낼 수 없다."""
        return arm_q is not None and all(abs(v) < Discoverer._FORBIDDEN_HOME_TOL for v in arm_q)

    def _observation_pose(self):
        return json.load(open(os.path.expanduser(
            "~/.local/share/nero_robot/saved_poses.json")))["observation"]["joints"]

    def home(self, timeout=30.0, retries=3):
        """observation 저장 자세로 직접 move_joints (2026-09-23 변경, 사용자 지시).
        예전엔 {"action":"home"}을 썼는데, 이게 항상 observation으로 가는 게 아니라
        실측으로 all-zero(카메라가 천장을 보는 자세)로 가는 경우가 확인됨. 게다가
        move_joints 응답이 "success"라고 와도 실제로는 안 움직인 채(all-zero 그대로)
        인 사례가 실측 확인돼(2026-09-23), **응답 status를 안 믿고 실제 /joint_states로
        검증**한다 -- observation 근처에 실제로 도달할 때까지 최대 retries회 재시도,
        그래도 실패하면 절대 조용히 넘어가지 않고 RuntimeError로 멈춘다(스윕이 천장을
        보고 도는 것보다 여기서 확실히 죽는 게 낫다)."""
        pose = self._observation_pose()
        arm_j = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        target = [pose.get(j, 0.0) for j in arm_j]
        last = None
        for attempt in range(1, retries + 1):
            self._result.clear()
            t_disc = time.time()
            while self.pub_cmd.get_subscription_count() == 0 and time.time() - t_disc < 5.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            msg = String(); msg.data = json.dumps({"action": "move_joints", "joints": pose})
            self.pub_cmd.publish(msg)
            t0 = time.time()
            while time.time() - t0 < timeout and "data" not in self._result:
                rclpy.spin_once(self, timeout_sec=0.2)
            last = self._result.get("data", {"status": "timeout"})
            # 응답과 무관하게 실제 joint_states로 검증(위 이유).
            self.got.pop("arm_q", None)
            t1 = time.time()
            while "arm_q" not in self.got and time.time() - t1 < 3.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            arm_q = self.got.get("arm_q")
            ok = arm_q is not None and max(abs(a - b) for a, b in zip(arm_q, target)) < 0.08
            if ok and not self._looks_like_forbidden_home(arm_q):
                return last
            print(f"[discover] home() 시도 {attempt}/{retries}: 응답={last.get('status')} "
                  f"실제도달={ok} arm_q={arm_q} -- observation 근처 아님, 재시도", file=sys.stderr)
            if attempt >= 2:
                print("[discover] force_home.py(충돌검사 없는 직접 궤적)로 폴백", file=sys.stderr)
                try:
                    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_home.py")],
                                   stdout=sys.stderr, check=False, timeout=45)
                except subprocess.TimeoutExpired:
                    print("[discover] force_home.py 45초 타임아웃", file=sys.stderr)
        raise RuntimeError(
            f"observation 자세 도달 실패({retries}회 재시도) -- 마지막 arm_q={self.got.get('arm_q')}. "
            "all-zero(home)로 스윕 시작하는 걸 막기 위해 여기서 중단합니다.")

    def cur_q(self, n_fresh=3, timeout=1.5):
        """2026-10-16: 지금 실제 관절값. 이 프로세스는 exec_pick/exec_place 하위 프로세스를 기다리는 동안 스핀하지 않아 got["arm_q"] 가 놓기 직전 자세에 멈춰 있다. 그 낡은 값으로 jmove/swing 궤적을 만들면 첫 점이 실제 자세와 달라 팔이 순간 이동(0.25초에 0.4~1.3rad)하며 용기를 쳤다. 새 JointState 를 n_fresh 개 받은 뒤의 값을 돌려준다."""
        n0 = getattr(self, "_js_n", 0); t0 = time.time()
        while getattr(self, "_js_n", 0) < n0 + n_fresh and time.time() - t0 < timeout: rclpy.spin_once(self, timeout_sec=0.02)
        return self.got.get("arm_q")

    def follow_traj(self, traj, times, extra=8.0, vels=None):
        """관절 궤적(점 목록 + 누적 시간)을 FollowJointTrajectory 로 실행. 성공이면 True."""
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        if not hasattr(self, "_jm_ac"): self._jm_ac = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        if not self._jm_ac.wait_for_server(timeout_sec=5.0): return False
        _q = self.cur_q(); _e = max(abs(float(a_) - float(b_)) for a_, b_ in zip(traj[0], _q)) if _q is not None else 0.0
        if _e > 0.05: print(f"[traj] 첫 점이 실제 자세와 {_e:.2f}rad 다름 -- 실행 거부(순간 이동 방지)", file=sys.stderr); return False
        g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        for i_, (w, t) in enumerate(zip(traj, times)):
            p = JointTrajectoryPoint(); p.positions = [float(x) for x in w]; t = float(t) + 0.05
            if vels is not None: p.velocities = [float(x) for x in vels[i_]]
            p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9); g.trajectory.points.append(p)
        fut = self._jm_ac.send_goal_async(g); t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0: rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done() or not fut.result().accepted: return False
        rf = fut.result().get_result_async(); t0 = time.time(); T = float(times[-1])
        while not rf.done() and time.time() - t0 < T + extra: rclpy.spin_once(self, timeout_sec=0.05)
        return rf.done() and rf.result().result.error_code == 0

    def grip_set(self, pos, sec=1.0):
        """그리퍼 관절 목표(닫기: [0,0], 열기: [1,-1]) 를 보내고 끝날 때까지 기다린다."""
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        if not hasattr(self, "_gr_ac"): self._gr_ac = ActionClient(self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
        if not self._gr_ac.wait_for_server(timeout_sec=5.0): return False
        g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = ["gripper_joint1", "gripper_joint2"]
        p = JointTrajectoryPoint(); p.positions = [float(x) for x in pos]; p.time_from_start.sec = int(sec); p.time_from_start.nanosec = int((sec % 1) * 1e9); g.trajectory.points.append(p)
        fut = self._gr_ac.send_goal_async(g); t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0: rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done() or not fut.result().accepted: return False
        rf = fut.result().get_result_async(); t0 = time.time()
        while not rf.done() and time.time() - t0 < sec + 4.0: rclpy.spin_once(self, timeout_sec=0.05)
        return rf.done()

    def push_object(self, center, direction, radius, dist, obstacles, url=None):
        """닫은 그리퍼로 물체를 direction(수평 단위벡터) 쪽으로 dist 만큼 민다. 성공이면 True."""
        import urllib.request
        q = self.cur_q()
        if q is None: return False
        self.grip_set([0.0, 0.0], 1.0)
        req = json.dumps(dict(start_q=[float(x) for x in q], center=[float(center[0]), float(center[1])], dir=[float(direction[0]), float(direction[1])], radius=float(radius), dist=float(dist), obstacles=obstacles)).encode()
        try: open(f"/tmp/push_req_{int(time.time()) % 100000}.json", "wb").write(req)      # 디버그: 마지막 요청 보존
        except Exception: pass
        try: r = json.loads(urllib.request.urlopen(urllib.request.Request((url or getattr(self, "jmove_url", "http://localhost:8020")) + "/push", req, {"Content-Type": "application/json"}), timeout=30).read())
        except Exception as e:
            print(f"[push] 계획 요청 실패 {e!r}", file=sys.stderr); return False
        if not r.get("ok"):
            print(f"[push] 계획 실패: {r.get('why')}", file=sys.stderr); return False
        ok = self.follow_traj(r["traj"], r["times"])
        print(f"[push] 밀기 {dist * 100:.0f}cm 방향 ({direction[0]:+.2f},{direction[1]:+.2f}) 실행 {r['T']:.1f}s (계획 {r['sec']}s) ok={ok}", file=sys.stderr)
        return ok

    @staticmethod
    def retime_minjerk(W, vmax=1.2, amax=4.0, t_min=1.2, dt=0.05):
        """경유점 W(n,7)를 3차 스플라인으로 잇고, 최소 저크 시간 척도(시작/끝 속도·가속도 0)로 재시간화한다. 최고 속도 <= vmax rad/s, 최고 가속도 <= amax rad/s^2 를 만족하는 가장 짧은 T.
        -> (positions, times, velocities, T). 2026-10-16: cuRobo 는 최소 1.5s 만 보장해 큰 이동이 초속 2~3rad 이 됐다."""
        from scipy.interpolate import CubicSpline
        W = np.asarray(W, float); n = len(W)
        if n < 3: W = np.vstack([W[0], 0.5 * (W[0] + W[-1]), W[-1]]); n = 3
        u = np.linspace(0, 1, n); cs = CubicSpline(u, W, bc_type=((1, np.zeros(W.shape[1])), (1, np.zeros(W.shape[1]))))
        ug = np.linspace(0, 1, 400); d1 = np.abs(cs(ug, 1)).max(); d2 = np.abs(cs(ug, 2)).max()
        T = max(t_min, 1.875 * d1 / vmax, float(np.sqrt((5.77 * d1 + 3.52 * d2) / amax)))
        m = max(8, int(np.ceil(T / dt))); tt = np.linspace(0, T, m + 1); x = tt / T
        sm = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5; dsm = (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T
        return cs(sm).tolist(), tt.tolist(), (cs(sm, 1) * dsm[:, None]).tolist(), float(T)

    def jmove(self, goal7, timeout=40.0, vmax=None, wander_max=None):
        """2026-10-15: cuRobo(/jmove)로 용기/장애물을 피하는 관절 이동을 계획해 FollowJointTrajectory 로 직접 실행한다. 장애물 공급자(self.jmove_obstacles)가 없으면 False."""
        fn = getattr(self, "jmove_obstacles", None); q = self.cur_q()
        if fn is None or q is None or os.environ.get("NERO_JMOVE", "1") != "1": return False
        import urllib.request
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        try:
            req = json.dumps(dict(start_q=[float(x) for x in q], goal_q=[float(x) for x in goal7], obstacles=fn(), margin=float(os.environ.get("NERO_JMOVE_MARGIN", "0.03")))).encode()
            r = json.loads(urllib.request.urlopen(urllib.request.Request(getattr(self, "jmove_url", "http://localhost:8020") + "/jmove", req, {"Content-Type": "application/json"}), timeout=20).read())
        except Exception as e:
            print(f"[jmove] 계획 요청 실패 {e!r}", file=sys.stderr); return False
        if not r.get("ok"):
            print(f"[jmove] 계획 실패: {r.get('why')}", file=sys.stderr); return False
        if not hasattr(self, "_jm_ac"): self._jm_ac = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        if not self._jm_ac.wait_for_server(timeout_sec=5.0): return False
        g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        try:      # 2026-10-15: 용기를 친 사건 분석용으로 /jmove 요청과 계획을 최근 8개 남긴다(FK 로 링크-용기 간격 재계산)
            self._jm_n = getattr(self, "_jm_n", 0) + 1; _d = os.environ.get("NERO_MULTI_LOG", "/tmp/multi_exec_logs")
            json.dump(dict(t=time.time(), start=[float(x) for x in q], goal=[float(x) for x in goal7], obstacles=fn(), margin=float(os.environ.get("NERO_JMOVE_MARGIN", "0.03")), traj=r["traj"], T=r["T"]), open(f"{_d}/jmove_{self._jm_n % 8}.json", "w"), default=float)
        except Exception as e_: print(f"[jmove] 덤프 실패 {e_!r}", file=sys.stderr)
        tr = r["traj"]
        try:      # 2026-10-16: 경유점을 최소 저크로 재시간화(속도 vmax, 가속도 amax 제한). 시작 자세는 cur_q 라 첫 점 = 실제 자세
            _tr = np.asarray(tr, float); _tv = np.abs(np.diff(_tr, axis=0)).sum(0); _net = np.abs(_tr[-1] - _tr[0]); _wd = float((_tv - _net).max())
            if wander_max is not None and _wd > wander_max: print(f"[jmove] 경로가 돌아감(관절 낭비 {_wd:.2f}rad > {wander_max}) -- 거부", file=sys.stderr); return False
            _vm = float(vmax or os.environ.get("NERO_JMOVE_VMAX", "1.2")); tr, _tt, _vv, T = self.retime_minjerk(_tr, vmax=_vm, amax=float(os.environ.get("NERO_JMOVE_AMAX", "4.0")))
        except Exception as e_:
            print(f"[jmove] 재시간화 실패 {e_!r} -- 균일 시간", file=sys.stderr); T = float(r["T"]); _tt = [T * i / max(1, len(tr) - 1) for i in range(len(tr))]; _vv = None
        for i, w in enumerate(tr):
            p = JointTrajectoryPoint(); p.positions = [float(x) for x in w]; t = _tt[i] + 0.05
            if _vv is not None: p.velocities = [float(x) for x in _vv[i]]
            p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9); g.trajectory.points.append(p)
        fut = self._jm_ac.send_goal_async(g); t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0: rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done() or not fut.result().accepted: return False
        rf = fut.result().get_result_async(); t0 = time.time()
        while not rf.done() and time.time() - t0 < T + 8.0: rclpy.spin_once(self, timeout_sec=0.05)
        print(f"[jmove] cuRobo 충돌 회피 이동 {T:.1f}s (계획 {r.get('sec')}s, 점 {len(tr)})", file=sys.stderr)
        return rf.done() and rf.result().result.error_code == 0

    def rotate_j1_direct(self, angle, vmax=None):
        """2026-10-15: j1 만 직선 보간으로 돌린다(나머지 관절은 지금 값 그대로, 최소 저크 시간 척도, 최고 속도 vmax rad/s). 플래너 없이 접힌 관측 자세 그대로 돌리므로 매 정리 시작의 전체 스윕과 같은 경로이고, 계획이 이상한 호를 그릴 일이 없다."""
        q = self.cur_q()
        if q is None: return False
        vmax = float(vmax or os.environ.get("NERO_SWING_VMAX", "1.0")); q0 = [float(x) for x in q]; d = float(angle) - q0[0]
        if abs(d) < 1e-3: return True
        T = max(1.2, 1.875 * abs(d) / vmax); n = max(12, int(T / 0.1)); traj = []; times = []
        for i in range(n + 1):
            u = i / n; sm = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5      # 최소 저크: 시작/끝 속도·가속도 0
            w = list(q0); w[0] = q0[0] + d * sm; traj.append(w); times.append(T * u)
        print(f"[swing] j1 직접 회전 {q0[0]:+.2f} -> {float(angle):+.2f} ({abs(d):.2f}rad, {T:.1f}s, 최고 {vmax}rad/s)", file=sys.stderr)
        return self.follow_traj(traj, times)

    def move_j1(self, angle, timeout=20.0, j7_abs=None):
        """조인트1만 스윕하되, 나머지 관절(2~7)도 매번 observation 값으로 명시 지정한다
        (2026-09-23 수정). 예전엔 {"j1": angle}만 보내서 "다른 관절은 이미 observation에
        있겠지"라고 가정했는데, 그 가정이 깨지면(예: 직전 home()이 응답은 success였지만
        실제로 all-zero에 있었던 경우) 스윕 내내 카메라가 천장을 보게 된다 -- 매 스텝
        7개 관절 전부를 명시해서 이 가정 자체를 없앤다.

        j7_abs(2026-09-24 추가, 델타에서 절대값으로 변경): 지정하면 joint7을 이
        절대값으로 덮어써서 보낸다 -- 프레임 위/아래로 잘리는 물체를 잡기 위한 세로
        방향 스윕(J7_SWEEP_LOWER/STEPS 참고). None(기본값)이면 observation의 joint7을
        그대로 써서 기존 동작과 완전히 동일(하위호환)."""
        pose = dict(self._observation_pose())
        # 2026-10-15: 놓은 직후처럼 팔이 낮게 뻗은 자세에서 j1 을 크게 돌리면 관절 보간이 낮은 높이로 휘둘러 용기/홀더를 쳐서 날렸다(C18: 홀더 0.5m, bin 0.6m 이동).
        # 먼저 j1 은 그대로 두고 나머지 관절만 관측 자세로 올린 뒤(제자리에서 들어올림) j1 을 돌린다.
        _q = self.cur_q(); _jm_done = False
        if _q is not None and abs(float(_q[0]) - float(angle)) > 0.5 and os.environ.get("NERO_RETURN_SINGLE", "1") == "1":      # 2026-10-16: 놓은 뒤 복귀를 한 번의 부드러운 경로로(용기를 아는 cuRobo 가 관측 자세+목표 j1 까지 한 번에 계획, 최소 저크 재시간화). 올림-정지-회전의 끊김이 없다. 안 되면 아래 단계별 방식으로
            _obs1 = [pose[j] for j in ("joint2", "joint3", "joint4", "joint5", "joint6")]
            if self.jmove([float(angle)] + _obs1 + [float(j7_abs if j7_abs is not None else pose["joint7"])], vmax=float(os.environ.get("NERO_RETURN_VMAX", "1.6")), wander_max=float(os.environ.get("NERO_RETURN_WANDER", "0.9"))):
                _jm_done = True; _q = self.cur_q()
        if _q is not None and abs(float(_q[0]) - float(angle)) > 0.5 and os.environ.get("NERO_LIFT_FIRST", "1") == "1":      # 2026-10-15(E70): 놓은 뒤 /jmove 로 바로 크게 돌리자 bin 이 초속 1m 넘게 날아갔다(t=503.6s, 안에 든 물체 5개 포함 1.9m 이동). 용기 위에 낮게 뻗은 자세에서 출발하므로 제자리에서 먼저 관측 자세로 올린 뒤 돌린다.
            _obs0 = [pose[j] for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")]
            if max(abs(float(_q[i]) - _obs0[i]) for i in range(1, 6)) > 0.25:
                _lg = [float(_q[0])] + _obs0[1:6] + [float(j7_abs if j7_abs is not None else pose["joint7"])]
                if not self.jmove(_lg):      # 용기를 아는 cuRobo 로 먼저(MoveIt 은 용기를 몰라 쳐서 날렸던 계획기). 안 풀리면 MoveIt 제자리 올림으로 폴백, 그것도 안 돼도 멈추지 않고 진행
                    print("[discover] 제자리 올림: cuRobo 계획 실패 -> MoveIt 폴백", file=sys.stderr)
                    self.move_j1(float(_q[0]), timeout=min(timeout, 8.0), j7_abs=j7_abs)
                pose = dict(self._observation_pose()); _q = self.cur_q()
        if _q is not None and abs(float(_q[0]) - float(angle)) > 0.5 and os.environ.get("NERO_DIRECT_SWING", "1") == "1":      # 접힌 관측 자세로 올린 뒤(위) j1 만 직접 돌린다
            _jm_done = self.rotate_j1_direct(float(angle)); _q = self.cur_q()
        if _q is not None and not _jm_done and abs(float(_q[0]) - float(angle)) > 0.5:
            _g = [float(angle)] + [pose[j] for j in ("joint2", "joint3", "joint4", "joint5", "joint6")] + [float(j7_abs if j7_abs is not None else pose["joint7"])]
            _jm_done = self.jmove(_g)
        if _q is not None and not _jm_done and abs(float(_q[0]) - float(angle)) > 0.5 and os.environ.get("NERO_LIFT_BEFORE_SWING", "1") == "1":
            _obs = [pose[j] for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")]
            if max(abs(float(_q[i]) - _obs[i]) for i in range(1, 6)) > 0.25:
                self.move_j1(float(_q[0]), timeout=timeout, j7_abs=j7_abs)
                pose = dict(self._observation_pose())
        pose["joint1"] = float(angle)
        if j7_abs is not None:
            pose["joint7"] = float(j7_abs)
        self._result.clear()
        # ⚠ 2026-09-18 버그이력: 구독자 연결 확인 없이 publish 하면 ROS2 디스커버리가
        # 안 끝난 채 메시지가 사라질 수 있다(place_via_ros.py 에서 실측). 연결되면 매번
        # 즉시 통과하므로(count>0) 비용은 사실상 없음, 첫 호출만 최대 5초 대기.
        t_disc = time.time()
        while self.pub_cmd.get_subscription_count() == 0 and time.time() - t_disc < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        msg = String(); msg.data = json.dumps({"action": "move_joints", "joints": pose})
        self.pub_cmd.publish(msg)
        t0 = time.time()
        while time.time() - t0 < timeout and "data" not in self._result:
            rclpy.spin_once(self, timeout_sec=0.2)
        result = self._result.get("data", {"status": "timeout"})
        # 2026-09-25 추가(실측 확인): /pick_result 응답은 "명령을 접수/처리했다"는
        # 뜻일 뿐 조인트가 목표각도에 물리적으로 도달했다는 보장이 아니다. 큰 각도
        # 재조준(예: j1이 2.60->-2.34rad처럼 크게 이동) 직후 바로 fresh_frame을
        # 부르면, 응답은 왔지만 로봇이 아직 이전 각도 근처에 있어서 스테일 프레임을
        # 읽는 사례가 실측 확인됨(재조준 목표와 무관한 좌표의 물체가 검출됨). home()과
        # 동일한 패턴으로 arm_q(실시간 조인트 상태)를 폴링해서 실제 도달을 확인한다.
        target = [pose[j] for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")]
        t1 = time.time()
        while time.time() - t1 < 3.0:
            arm_q = self.got.get("arm_q")
            if arm_q is not None and max(abs(a - b) for a, b in zip(arm_q, target)) < 0.08:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        else:
            print(f"[discover] move_j1: 3.0s 내 실제 도달 확인 실패 -- "
                  f"arm_q={self.got.get('arm_q')} target={np.round(target, 3).tolist()}", file=sys.stderr)
        return result

    def fresh_frame(self, settle=0.3, timeout=5.0, _nosleep=False):
        """이동 직후 새 프레임 도착까지 대기(스테일 프레임 방지)."""
        self.got.pop("depth", None); self.got.pop("rgb", None)
        if not _nosleep: time.sleep(float(os.environ.get("NERO_FRAME_SETTLE", settle)))      # 2026-10-28: 이동 후 대기 시간을 환경변수로(기본 settle 인자 = 기존 0.3 s); _nosleep 은 NERO_FRAME_MULTI 의 추가 프레임용(2026-10-29, 기본 False = 기존)
        if os.environ.get("NERO_FRAME_QUIESCE", "0") == "1" and not _nosleep:      # 2026-10-29(기본 꺼짐): 고정 대기 뒤 관절 정지 확인 -- 최근 WIN 초 동안 7관절 위치 변동폭(최대-최소) <= EPS rad 일 때까지(상한 MAX 초; 정지 판정은 물리량(관절 변동)만 쓰고 인식 결과는 쓰지 않는다. 기본 EPS=1e-4 rad = 0.7 m 거리 횡이동 0.07 mm)
            _win = float(os.environ.get("NERO_QUIESCE_WIN", "0.3")); _eps = float(os.environ.get("NERO_QUIESCE_EPS", "1e-4")); _tmax = float(os.environ.get("NERO_QUIESCE_MAX", "8.0")); _h = []; _tq = time.time(); _ok = False; _rng = None
            while time.time() - _tq < _tmax:
                rclpy.spin_once(self, timeout_sec=0.05); _q = self.got.get("arm_q"); _now = time.time()
                if _q is None: continue
                _h.append((_now, list(_q))); _h = [x for x in _h if _now - x[0] <= _win]
                if _now - _tq >= _win and _h[0][0] <= _now - 0.9 * _win:
                    _rng = max(max(x[1][i] for x in _h) - min(x[1][i] for x in _h) for i in range(7))
                    if _rng <= _eps: _ok = True; break
            print(f"[quiesce] 정지 확인 {'성공' if _ok else '시간초과'} 대기 {time.time() - _tq:.2f}s 최근 변동 {_rng}", file=sys.stderr)
            if os.environ.get("NERO_FLAG_LOG"):      # 시험 기록용(기본 없음): 정지 대기 결과를 파일에 덧붙임
                try: open(os.environ["NERO_FLAG_LOG"], "a").write(f"quiesce ok={int(_ok)} wait={time.time() - _tq:.3f} range={_rng}\n")
                except Exception: pass
            self.got.pop("depth", None); self.got.pop("rgb", None)
        t0 = time.time()
        while time.time() - t0 < timeout and not ("depth" in self.got and "rgb" in self.got):
            rclpy.spin_once(self, timeout_sec=0.1)
        if not ("depth" in self.got and "rgb" in self.got):
            print("[discover] 새 프레임 미수신 -- 이 캡처 스킵", file=sys.stderr)
            return False
        t_tf = time.time(); last_e = None
        while time.time() - t_tf < 3.0:       # 2026-10-03: 새 프로세스 직후 첫 촬영은 TF 버퍼가 아직 안 찼을 수 있다(15도 파일럿: 정면 뷰가 통째로 버려져 막대를 못 찾음). 최대 3초 재시도
            try:
                tf = self.buf.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time())
                if os.environ.get("NERO_TF_AT_STAMP", "0") == "1" and self.got.get("depth_stamp") is not None:      # 2026-10-28(기본 꺼짐): 최신 TF 가 아니라 깊이 영상 헤더 시각의 TF 로 조회(영상-자세 시각 불일치 보정), 실패하면 최신 TF
                    try:
                        _ds = self.got["depth_stamp"]; tf = self.buf.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time(seconds=int(_ds), nanoseconds=int(round((_ds - int(_ds)) * 1e9))))
                    except Exception as _e2: print(f"[discover] 깊이 시각 TF 실패, 최신 TF 사용: {_e2}", file=sys.stderr)
                self.got["tf"] = tf; self.got["tf_stamp"] = tf.header.stamp.sec + 1e-9 * tf.header.stamp.nanosec; self.got["wall_capture"] = time.time()
                return True
            except Exception as e:
                last_e = e; rclpy.spin_once(self, timeout_sec=0.1)
        print(f"[discover] TF 못 얻음(3초 재시도 후): {last_e}", file=sys.stderr)
        return False

    def capture_geom(self):
        """현재 프레임 기준 K/포즈/포인트클라우드 계산 (러프/정밀 공용)."""
        K3 = self.got["K"]; fx, fy, cx, cy = K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]
        depth = self.got["depth"]; H, W = depth.shape
        tr = self.got["tf"].transform
        R = quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
        base_T_opt = np.eye(4); base_T_opt[:3, :3] = R
        base_T_opt[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
        W_T_cam = base_T_opt @ np.diag([1.0, -1.0, -1.0, 1.0])
        zf = depth.copy(); zf[~np.isfinite(zf)] = 0.0
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        Xc = (uu - cx) / fx * zf; Yc = (vv - cy) / fy * zf
        pts_opt = np.stack([Xc, Yc, zf, np.ones_like(zf)], -1)
        pts_base = pts_opt @ base_T_opt.T
        return dict(fx=fx, fy=fy, cx=cx, cy=cy, H=H, W=W, zf=zf, pts_base=pts_base, W_T_cam=W_T_cam)


def _nearest(w, known):
    """known: {gid: pos} -- w 와 가장 가까운 gid, 거리. known 비어있으면 (None, inf)."""
    if not known:
        return None, float("inf")
    gids = list(known.keys())
    dists = [np.linalg.norm(w - np.array(known[g])) for g in gids]
    i = int(np.argmin(dists))
    return gids[i], dists[i]



def _log_nbr(gid, b):
    for ln in (b.stdout or "").splitlines():
        if "이웃 충돌 필터" in ln:
            print(f"[discover] {gid}: {ln.strip()}", file=sys.stderr)


def _run(args, timeout, **kw):
    """Thor 원격 호출이 멈춰도 파이프라인이 무한대기하지 않도록 상한을 둔다(초과 시 rc=124)."""
    try:
        return subprocess.run(args, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as e:
        print(f"[discover] !! {timeout}s 타임아웃: {' '.join(map(str, args))[:120]}", file=sys.stderr)
        return subprocess.CompletedProcess(args, 124, stdout="" if kw.get("text") else b"", stderr="" if kw.get("text") else b"")

# 2026-09-29 재조준 재검출 실패 폴백: 텍스트 프롬프트("mustard bottle")가 안 맞아 물체가 화면 중앙에
# 멀쩡히 보여도 0건이 나오는 사례(시드 100 머스타드, "yellow bottle" 로 물으면 0.56). 1단계에서 잡힌
# 마스크의 주 색상으로 "<색> <물체 종류>" 대체 프롬프트를 만들어 한 번 더 검출한다(라벨 하드코딩 없음).
def _color_name(rgb, mask):
    try:
        px = np.asarray(rgb)[mask]
        if len(px) < 30:
            return None
        r, g, b = [float(np.median(px[:, k])) / 255.0 for k in range(3)]
        h, sat, v = colorsys.rgb_to_hsv(r, g, b)
        if sat < 0.18:
            return "white" if v > 0.72 else ("black" if v < 0.28 else "gray")
        deg = h * 360.0
        if deg < 15 or deg >= 345:
            return "red"
        if deg < 45:
            return "brown" if v < 0.5 else "orange"
        if deg < 70:
            return "yellow"
        if deg < 165:
            return "green"
        if deg < 200:
            return "cyan"
        if deg < 260:
            return "blue"
        if deg < 300:
            return "purple"
        return "pink"
    except Exception:
        return None


def _plane_bbox(P, tol=0.015, q=1.0):
    """place target(테이블) 마스크 3D 점 -> (윗면 중심xy, half_extent xy, 윗면 z).
    가장 점이 많은 높이(=테이블 윗면) 근처 점만 남겨 bbox 를 구한다. 시드 100 실측(시점 5곳): 예전 median 중심은
    카메라에 가까운 쪽 점이 촘촘해서 2.4~9.2cm 쏠렸고(place 슬롯 첫 자리가 실제 테이블 밖으로 나가 물체가
    떨어짐), 이 방식은 평균 1.1cm/최대 1.8cm. 마스크가 바닥으로 번져도 높이로 걸러진다."""
    z = P[:, 2]
    h, e = np.histogram(z, bins=np.arange(float(z.min()), float(z.max()) + 0.005, 0.005))
    zt = 0.5 * (e[int(h.argmax())] + e[int(h.argmax()) + 1])
    Q = P[np.abs(z - zt) < tol][:, :2]
    if len(Q) < 50:
        Q = P[:, :2]
    lo, hi = np.percentile(Q, q, axis=0), np.percentile(Q, 100 - q, axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0, float(zt), len(Q)


def _plane_rect(P, tol=0.015):
    """윗면 점들의 최소면적 회전 사각형 -> {"center":[x,y], "half":[a,b], "yaw": rad}. 테이블이 z축으로 회전돼 있으면
    축 정렬 bbox 는 모서리 밖(=테이블 밖) 슬롯을 만든다. 실패하면 None (호출부는 축 정렬 값으로 폴백)."""
    try:
        import cv2
        z = P[:, 2]
        h, e = np.histogram(z, bins=np.arange(float(z.min()), float(z.max()) + 0.005, 0.005))
        zt = 0.5 * (e[int(h.argmax())] + e[int(h.argmax()) + 1])
        Q = P[np.abs(z - zt) < tol][:, :2].astype(np.float32)
        if len(Q) < 50:
            return None
        lo, hi = np.percentile(Q, 1.0, axis=0), np.percentile(Q, 99.0, axis=0)
        Q = Q[np.all((Q >= lo) & (Q <= hi), axis=1)]
        (cx, cy), (w, hh), ang = cv2.minAreaRect(Q)
        return {"center": [float(cx), float(cy)], "half": [float(w) / 2.0, float(hh) / 2.0], "yaw": float(np.deg2rad(ang))}
    except Exception:
        return None


def _alias_prompt(label, color):
    if not color or not label:
        return None
    return f"{color} {label.split()[-1]}"


def _parse_b(out):
    ok = "reachable" in out and "no reachable" not in out and "no candidate" not in out
    m = re.search(r"(\d+) reachable", out)
    return ok, (int(m.group(1)) if m else None)


# 파이프라인 상태 JSON (시각화 오른쪽 위 패널이 읽음). 갱신은 스레드 안전.
_STATUS_PATH = os.path.expanduser(os.environ.get("NERO_STATUS_PATH", "~/grasp/pipeline_status.json"))
_STATUS = {"stage": "", "tag": "", "objects": {}}
_STATUS_LOCK = threading.Lock()


def _status_flush():
    _STATUS["updated"] = time.time()
    tmp = _STATUS_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(_STATUS_PATH), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(_STATUS, f, ensure_ascii=False)
        os.replace(tmp, _STATUS_PATH)
    except Exception:
        pass


def _status_set(gid, **kw):
    with _STATUS_LOCK:
        o = _STATUS["objects"].setdefault(gid, {})
        for k, v in kw.items():
            o[k] = float(v) if isinstance(v, np.floating) else v
        _status_flush()


def _status_del(gid):
    with _STATUS_LOCK:
        _STATUS["objects"].pop(gid, None)
        _status_flush()


def _status_stage(stage):
    with _STATUS_LOCK:
        _STATUS["stage"] = stage
        _status_flush()


def _status_reset(tag):
    with _STATUS_LOCK:
        _STATUS.update(stage="", tag=tag, objects={})
        _status_flush()


def _run_cgn(n, geom, seg, step_labels, step_ids, step_worlds, tag):
    npz_path = f"/tmp/discover_{tag}.npz"
    np.savez(npz_path, depth_m=geom["zf"].astype(np.float32), seg=seg,
              K=np.array([geom["fx"], geom["fy"], geom["cx"], geom["cy"]], dtype=np.float64),
              W_T_cam=geom["W_T_cam"].astype(np.float64),
              target_labels=np.array(step_labels), target_ids=np.array(step_ids, dtype=np.int32),
              target_worlds=np.array(step_worlds, dtype=np.float64),
              target_is_bottle=np.array([lb.rsplit("_", 1)[0] in BOTTLE_LABELS for lb in step_labels], dtype=bool))
    remote = f"{GRASP_DIR}/{os.path.basename(npz_path)}"
    _run(["rsync", "-q", npz_path, f"thor:grasp/{os.path.basename(npz_path)}"], timeout=60)
    # ⚠ ssh 는 인자를 로컬이 아니라 "원격 셸이 한 문자열로 재파싱"하므로 JSON 을 여러
    # argv 로 넘기면 따옴표가 깨진다(실측). 원격 명령 전체를 문자열 하나로, JSON 은
    # 작은따옴표로 감싼다(경로 문자열엔 작은따옴표가 안 들어가므로 안전).
    body = json.dumps({"npz_path": remote, "out_dir": "" + GRASP_DIR + "", "nbr_filter": NBR_FILTER})
    remote_cmd = f"curl -s -X POST {CGN_URL_LOCAL} -H 'Content-Type: application/json' -d '{body}'"
    r = _run(["ssh", "thor", remote_cmd], capture_output=True, text=True, timeout=180)
    print(f"[discover] {tag} CGN: {r.stdout.strip()}", file=sys.stderr)
    try:
        for item in json.loads(r.stdout).get("results", []):
            for k, v in item.items():
                mm = re.match(r"(\d+) grasps", str(v))
                if mm:
                    _status_set(k, cgn=int(mm.group(1)))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="box:6", help="capture_multi_npz.py 와 동일 문법(라벨:스텝당최대개수)")
    ap.add_argument("--place-label", default=None,
                     help="놓을 곳(바구니 등) 자연어 라벨. 지정하면 같은 360도 스윕에서 "
                          "박스와 함께 찾는다 -- vlm_boxyolo.py가 'box'/COCO 고정클래스가 "
                          "아닌 라벨은 자동으로 YOLO-World open-vocab로 라우팅하므로(실측"
                          "확인, 2026-09-18) 별도 ground_object 왕복 없이 그냥 라벨 문자열만 "
                          "넘기면 된다. CGN(파지후보)은 안 돌리고 위치만 확정해 출력의 "
                          "'place_target'에 넣는다(집을 대상이 아니므로 'boxes'엔 안 들어감).")
    # 2026-09-18 (사용자 지시 "360도 끝까지 꼼꼼히"): 7 -> 13스텝으로 밀도 2배(스텝
    # 간격 ~50° -> ~25°). 참고: joint1 실제 가동범위는 -J1_LIMIT~+J1_LIMIT(약 ±149°,
    # 총 ~298°)로 물리적 한계상 진짜 360도 전체회전은 불가능 -- 이 가동범위 안에서
    # 최대한 촘촘히 훑는다는 뜻으로 해석.
    ap.add_argument("--stop-after", type=int, default=0,
                     help="반응형 모드(2026-10-01): 1단계 스윕 중 후보가 이만큼 모이면 스윕을 멈추고 바로 2단계로 간다(0=끔, 기존처럼 끝까지 스윕). "
                          "NERO_SWEEP_FROM=<j1 rad> 이면 그 각도부터 한 방향으로 훑고 나머지 쪽을 훑는다.")
    ap.add_argument("--wide-sweep", default=",".join(str(round(x, 2)) for x in
                     np.linspace(-J1_LIMIT, J1_LIMIT, 13)),
                     help="1단계 러프 스윕 joint1 각도(rad, 콤마구분) -- 기본 가동범위 전체를 13등분")
    ap.add_argument("--dedupe-dist", type=float, default=0.08, help="같은 물체로 볼 base_link 거리(m)")
    ap.add_argument("--cross-xy-dist", type=float, default=0.04, help="서로 다른 라벨을 한 물체로 볼 xy 거리(m), 더 높은 쪽 유지")
    ap.add_argument("--out", default=os.path.expanduser("~/grasp/discovered.json"))
    ap.add_argument("--floor-z", type=float, default=0.012)
    ap.add_argument("--box-z-max", type=float, default=0.09,
                     help="'box' 후보의 정상 중심 높이 상한(m) -- box_height_m=0.05 고정값 "
                          "기준 정상범위(~0.05)보다 확실히 높으면(예: 가구를 부분크롭으로 "
                          "오검출) 후보에서 제외. 실측: 테이블 부분크롭이 z=0.117로 box "
                          "오검출된 사고(2026-09-18) 방지용.")
    ap.add_argument("--box-xy-extent-max", type=float, default=0.15,
                     help="'box' 후보의 xy 퍼짐(마스크 포인트 최대 가로/세로 폭) 상한(m) -- "
                          "박스는 0.05m 정육면체라 정상 범위가 좁다. 테이블을 낮춰서(z 필터가 "
                          "안 걸러지는 z대) 부분크롭으로 box 오검출되는 사고(2026-09-24) 방지용, "
                          "z 필터와 별개의 독립 신호(테이블은 프레임 가장자리에 걸려도 xy 퍼짐이 "
                          "여전히 훨씬 큼).")
    ap.add_argument("--no-precompute-bc", action="store_false", dest="precompute_bc",
                     help="B/C단계(IK랭킹+STOMP) 선계산을 끄고 pick 직전(stack_discovered.sh)에 "
                          "매번 새로 계산. 기본은 켜짐(선계산) -- 2026-09-18 실측으로 SEED 신선도는 "
                          "원인이 아님이 확인돼 되돌림(pick_ik 자체의 비결정성이 원인, kinematics.yaml "
                          "참고). A/B 비교용으로만 끌 것.")
    ap.add_argument("--task", default=None,
                     help="논문 candidate2(태스크-적응형 cost 가중치, 2026-09-23). 자연어 태스크 "
                          "지시를 주면 Thor에서 task_weight.py를 1회(물체당 아니라 이 실행 전체에 "
                          "1번) 호출해 box cost의 d_com 계수 배율(low=0.7/normal=1.0/high=1.5, "
                          "VLM은 등급만 고르고 배율 숫자는 서버가 고정 매핑 -- CoRAL 논문이 보고한 "
                          "가중치 극단값 실패모드를 설계로 차단)을 정하고, 이후 모든 gid의 "
                          "two_pick_plan_fast.py 호출에 NERO_TASK_WEIGHT=1로 적용한다. 생략하면 "
                          "기존 동작과 100%% 동일(배율 1.0).")
    ap.add_argument("--pipeline-exec", action="store_true",
                     help="2026-09-25 추가(사용자 제안): gid별 B/C단계(IK랭킹+STOMP, 팔 불필요한 "
                          "순수 원격계산)가 끝나는 즉시 그 물체의 pick+place(팔 필요)를 바로 실행하고, "
                          "그 동안 나머지 물체들의 B/C 계산은 배경 스레드에서 계속 진행한다 -- 지금까지는 "
                          "discover가 3개 다 끝난 뒤에야 pick+place를 시작해서, 물체가 물리적으로 옮겨지는 "
                          "동안(보통 B/C보다 훨씬 오래 걸림) 나머지 물체 계산을 겹칠 기회를 날리고 있었다. "
                          "place_on_target_one.sh(물체 1개 처리)를 gid별로 직접 호출 -- place_on_target.sh는 "
                          "더 이상 안 부른다(호출자가 직접 부르면 pick+place가 중복 실행됨, run_trial.sh 참고). "
                          "--place-label 필수(놓을 곳 없이는 이 모드 의미 없음), --precompute-bc(기본 켜짐)와 "
                          "함께 써야 함.")
    ap.add_argument("--serial-bc", action="store_true",
                     help="B/C단계 선계산을 gid별 순차(B->C, 물체 하나씩)로 실행. 기본은 gid 동시 "
                          "실행(병렬). 직렬/병렬 ablation용 -- 순차 로직은 커밋 9f423c6과 동일.")
    ap.add_argument("--nbr-filter", action="store_true", dest="nbr_filter",
                     help="이웃 물체 grasp 필터(SAM 마스크 씬 점군 기반) 켬 -- 기본 OFF.")
    ap.add_argument("--no-place", action="store_true", dest="no_place",
                    help="pick 전용 모드(2026-09-29): place 계획/슬롯/실행을 쓰지 않고 pick 만 실행한다(--place-label 을 주면 테이블은 장애물로만 사용). "
                         "HELD 면 그 자리에서 그리퍼를 열어 물체를 놓는다(pick_one.sh). place 개발은 별도로 병렬 진행.")
    ap.set_defaults(precompute_bc=True, nbr_filter=False)
    a = ap.parse_args()
    try:
        os.remove('/tmp/place_slots_' + os.path.basename(a.out).rsplit('.json', 1)[0] + '.json')
    except OSError:
        pass
    global NBR_FILTER
    NBR_FILTER = a.nbr_filter
    _status_reset(os.path.basename(a.out).rsplit('.json', 1)[0]); _status_stage('sweep')

    label_counts = {}
    for tok in a.labels.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            lb, cnt = tok.split(":", 1); label_counts[lb.strip()] = int(cnt)
        else:
            label_counts[tok] = 1
    place_label = a.place_label.strip().lower() if a.place_label else None    # --no-place 여도 주면 테이블을 찾아 C단계 장애물로만 쓴다(place 는 안 함)
    if place_label and place_label not in label_counts:
        label_counts[place_label] = 1
    wide_angles = [float(x) for x in a.wide_sweep.split(",")]
    if os.environ.get("NERO_SWEEP_FROM", ""):
        _a0 = float(os.environ["NERO_SWEEP_FROM"])
        _up = sorted(x for x in wide_angles if x >= _a0 - 0.30); _dn = sorted((x for x in wide_angles if x < _a0 - 0.30), reverse=True)
        wide_angles = _up + _dn          # 직전 위치 근처부터 한 방향으로 훑고 반대쪽으로(왕복 최소)
        print(f"[discover] 스윕 시작 각도 {_a0:.2f}rad 부터: {[round(x, 2) for x in wide_angles]}", file=sys.stderr)

    # 2026-09-18 (버그 발견 중 확인): gid("box_1" 등)는 실행마다 재사용되는 이름이라,
    # 이번 실행에서 어떤 gid가 (z 필터 등으로 걸러지거나 재검출 실패로) CGN을 안 태워도
    # Thor에 "이전 실행"이 같은 이름으로 남겨둔 {gid}_grasps.json 이 있으면 B/C단계가
    # 그 stale 파일을 그대로 읽어 "OK"를 내버린다(전혀 다른/존재 안 하는 물체 계획을
    # 유효한 것처럼 보고하는 조용한 실패 -- 실측으로 의심됨, box_4/5가 실제론 테이블
    # 부분크롭 오검출이었는데 B/C단계가 OK로 나온 사례). 매 실행 시작 시 이번에 쓸 라벨의
    # gid 패턴에 해당하는 이전 산출물을 전부 지워서, 이번에 실제로 확정 안 된 gid는
    # 절대 "우연히 성공"으로 안 보이게 한다.
    for _lb in label_counts:
        # gid = f"{lb}_{idx}" 그대로라 라벨에 공백 있으면 파일명에도 공백 -- 원격 셸이
        # glob(*)은 펼치되 공백은 하나의 인자로 보게 공백만 백슬래시 이스케이프.
        _pat = _lb.replace(" ", "_") + "_*"
        _run(["ssh", "thor",
                        f"cd ~/grasp && rm -f {_pat}_grasps.json {_pat}_pick.json "
                        f"{_pat}_step6_pick.json {_pat}_obstacles.npy {_pat}_target.npy 2>/dev/null; true"], timeout=30)

    rclpy.init()
    n = Discoverer()

    # 2026-09-29 조기 B단계(사용자 지시 "방향 1,2 동시"): 파이프라인 모드에선 물체의 CGN 이 끝나는
    # 즉시 그 물체의 B단계(IK 랭킹, Thor 원격)를 배경에서 시작한다 -- 팔은 그동안 다른 물체를
    # 재조준. 예전엔 스윕이 다 끝난 뒤에야 B단계를 시작해 팔이 ~16s 서 있었다. B단계는 이웃
    # 물체 정보가 필요 없어 조기 실행해도 결과가 같다. C단계(이웃 장애물 필요)는 모든 물체가
    # 확정된 뒤에 돈다. --task 는 task_weight 가 스윕 뒤에 계산되므로 조기 B 를 쓰지 않는다.
    _early_b = {}
    _early_pool = ThreadPoolExecutor(max_workers=4) \
        if (a.pipeline_exec and a.precompute_bc and not a.task) else None

    def _table_env():
        """place 를 고려한 파지 선택용(NERO_PLACE_AWARE_PICK=1 일 때만): 알고 있는 테이블 사각형을 B단계 env 로 넘긴다.
        기본 OFF -- 학습 모델이 아직 파일럿으로 검증되지 않았다(2026-09-29)."""
        # 2026-09-30 NERO_PHOLD=1 이면 학습 파지 점수(p_hold)를 B단계로 넘긴다(Thor 쪽 two_pick_plan_fast.py 가 읽음). 아래 place-aware 와 독립.
        _ph = ("NERO_PHOLD=1 NERO_PHOLD_W=%s " % os.environ.get("NERO_PHOLD_W", "3.0")) if os.environ.get("NERO_PHOLD", "0") == "1" else ""
        # 2026-10-01 B단계 IK 를 cuRobo 서버(PC)로: Thor 에서 PC 주소로 부른다. NERO_CUROBO_IK_URL 이 있을 때만 B단계로 넘긴다.
        if os.environ.get("NERO_CUROBO_IK_URL", ""):
            _ph += "NERO_CUROBO_IK_URL=%s NERO_CUROBO_IK_CHUNK=%s " % (os.environ["NERO_CUROBO_IK_URL"], os.environ.get("NERO_CUROBO_IK_CHUNK", "32"))
        if os.environ.get("NERO_PLACE_AWARE_PICK", "0") != "1":
            return _ph
        try:
            rc = None
            for _g, _d in list(confirmed.items()) + [(None, place_target_box[0])] if place_target_box[0] else list(confirmed.items()):
                if _d and _d.get("label") == place_label and _d.get("rect"):
                    rc = dict(_d["rect"]); rc["top_z"] = _d["world"][2]
                    break
            if rc is None:
                for _g, _w in rough.items():
                    if gid_lb.get(_g) == place_label and _g in rough_half_extent:
                        hx_, hy_ = rough_half_extent[_g]
                        rc = {"center": [_w[0], _w[1]], "half": [hx_, hy_], "yaw": 0.0, "top_z": _w[2]}
                        break
            if rc is None:
                return _ph
            return _ph + "NERO_TABLE_RECT=" + shlex.quote(json.dumps(rc)) + " "
        except Exception:
            return _ph

    def _submit_early_b(gids):
        if _early_pool is None:
            return
        _obsp = n._observation_pose()
        _aq = " ".join("%.5f" % float(_obsp.get(j, 0.0)) for j in
                       ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"])
        for g in gids:
            prev = _early_b.get(g)

            def _job(g=g, prev=prev):
                if prev is not None:
                    prev.result()      # 같은 gid 가 재-CGN 되면 이전 B 가 끝난 뒤 다시(파일 경합 방지)
                _t0 = time.time()
                _r = _run(["ssh", "thor",
                           f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                           f"cd ~/grasp && export NERO_ARM_Q='{_aq}' {_table_env()}&& python3 two_pick_plan_fast.py {shlex.quote(g)}"],
                          capture_output=True, text=True, timeout=180)
                _ok, _nr = _parse_b(_r.stdout or "")
                _status_set(g, b=_ok, b_n=_nr, b_sec=round(time.time() - _t0, 1), b_note=("no CGN grasps" if "no grasps file" in (_r.stdout or "") else None))
                return _r
            _early_b[g] = _early_pool.submit(_job)
            print(f"[discover] {g}: CGN 직후 B단계 조기 시작(배경)", file=sys.stderr)
    for _ in range(20):
        rclpy.spin_once(n, timeout_sec=0.1)  # pub/sub discovery
    # ⚠ 2026-09-18 버그이력: 위 2초 웜업 안에 CameraInfo(/camera_info, depth/rgb 보다
    # 드물게 발행됨)가 항상 도착하는 게 아니었다 -- capture_geom()에서 self.got["K"]
    # KeyError 로 실측. K 는 스윕 내내 안 바뀌는 상수라 딱 한 번만 확실히 받아두면 됨.
    t_k = time.time()
    while "K" not in n.got and time.time() - t_k < 5.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    if "K" not in n.got:
        print("[discover] CameraInfo 5초 내 못 받음 -- 중단", file=sys.stderr)
        rclpy.shutdown(); sys.exit(1)

    # 2026-09-18 추가(사용자 지적): 이전 호출(스태킹 등)이 팔을 observation 이 아닌
    # 자세에 남겨둔 채로 바로 스윕을 시작하면, 카메라 프레이밍이 캘리브레이션된
    # 스윕 각도와 안 맞아 방 안 장식용 박스류를 오검출하는 문제가 실측됨(2026-09-18,
    # x≈1.27 위치에 실재하지 않는 "박스" 2개 검출). 스윕 시작 전에 항상 observation
    # 자세로 복귀해서 매번 같은 기준에서 스윕하도록 한다.
    print("[discover] 스윕 시작 전 observation 자세로 복귀", file=sys.stderr)
    n.home()
    for _ in range(20):
        rclpy.spin_once(n, timeout_sec=0.1)

    # 2026-10-01 실시간 pick 모드(NERO_FAST_VIEW=1): 1단계 스윕 프레임 중 후보가 가장 완전히 보인 프레임(화면 경계 여유 최대)을 기억해 뒀다가,
    # 그 여유가 NERO_REVISIT_MARGIN(기본 20px) 이상이면 2단계 재방문(이동+촬영+시야 맞추기, 약 5~11초)을 생략하고 그 프레임으로 바로 CGN 한다.
    # 여유가 모자라면(잘려 보이면) 기존 재방문으로 간다 -- 잘린 물체를 충돌객체로 등록하던 문제(2026-09-30)를 되풀이하지 않기 위해.
    FAST_VIEW = os.environ.get("NERO_FAST_VIEW", "0") == "1"
    best_view = {}   # gid -> (geom, by_label, rgb, margin_px)
    rough = {}       # {gid: [x,y,z]} -- 1단계 대략 위치
    rough_j7 = {}    # {gid: j7v} -- 1단계에서 이 후보가 실제로 잡힌 joint7 값(2026-09-25 추가)
    place_target_box = [None]
    rough_half_extent = {}  # {gid: [hx,hy]} -- place_label의 1단계 half_extent(2026-09-25 추가, 아래 설명)
    confirmed = {}   # {gid: {"world":[..], "label":.., "conf":..}} -- 2단계 정밀 확정
    next_idx = {lb: 1 for lb in label_counts}
    gid_lb = {}  # gid -> 원본 라벨. gid는 공백 제거(ROS 노드명/셸 인자 안전), 라벨은 YOLO용 원문 유지
    rough_conf = {}  # gid -> 1단계 검출 conf (교차라벨 중복 판정용)
    rough_color = {}  # gid -> 1단계 마스크의 주 색상 이름 (재검출 실패 시 대체 프롬프트용)

    # ── 1단계: 러프 와이드 스윕 (CGN 없음, 위치만) ─────────────────────────────
    # [2026-09-24 추가/수정] joint1 스텝마다 joint7도 같이 스윕해서 프레임 위/아래로
    # 잘리는 물체까지 잡는다(사용자 실측 지적). observation의 joint7 값을 상한,
    # J7_SWEEP_LOWER를 하한으로 J7_SWEEP_STEPS개 절대값을 스윕(사용자 지시). baseline의
    # scan_box(_scan_box_sequence, planning_node.py)에도 동일 범위를 반영해 두 조건의
    # 씬 커버리지를 맞춤.
    j7_upper = float(n._observation_pose()["joint7"])
    j7_sweep_values = list(np.linspace(J7_SWEEP_LOWER, j7_upper, J7_SWEEP_STEPS))
    if os.environ.get("NERO_J7_SWEEP_VALUES", ""):      # 2026-10-01 j7 를 직접 지정(예: 1.325 하나): 7x1 스윕이 7x2 와 같은 탐지에 18s(7x2 는 30s), 물체가 화면 중앙에 놓인다
        j7_sweep_values = [float(x) for x in os.environ["NERO_J7_SWEEP_VALUES"].split(",")]
    print(f"[discover] === 1단계: 러프 와이드 스윕 {len(wide_angles)}스텝 x "
          f"{len(j7_sweep_values)}(joint7={[round(v,3) for v in j7_sweep_values]}) "
          f"(범위 {min(wide_angles):.2f}~{max(wide_angles):.2f}rad) ===",
          file=sys.stderr)
    # 2026-09-30 NERO_AGNOSTIC=1: 이름 없는 탐지. 이름 있는 대상(place 라벨 = 테이블 등)은 기존 검출기, 나머지 물체는 "object" 라벨로
    # agn_detect.agnostic_masks(깊이에서 바닥 위 덩어리 -> 월드 top-down 재분할). 반환 형식은 multi_detect 와 동일해서 뒤쪽 코드는 그대로다.
    AGN = os.environ.get("NERO_AGNOSTIC", "0") == "1"
    if AGN:
        label_counts.setdefault("object", 8)
        print(f"[discover] 이름 없는 탐지 모드(NERO_AGNOSTIC=1): 'object' 라벨 최대 {label_counts['object']}개/프레임, 이름 있는 라벨은 기존 검출기: "
              f"{[k for k in label_counts if k != 'object']}", file=sys.stderr)

    def _det(rgb_, geom_):
        if not AGN:
            return multi_detect(rgb_, label_counts, geom_["H"], geom_["W"])
        named = {k: v for k, v in label_counts.items() if k != "object"}
        by_ = multi_detect(rgb_, named, geom_["H"], geom_["W"]) if named else {}
        tab_ = [sm for lb_, insts_ in by_.items() if lb_ == place_label for sm, _, _ in insts_]
        ag_ = agnostic_masks(geom_, floor_z=a.floor_z, exclude_masks=tab_, max_count=label_counts.get("object", 8))
        if ag_:
            by_["object"] = ag_
        return by_

    def _view_margin(geom_, by_, w_):
        """정답 위치 w_ 근처(xy 10cm)의 검출 마스크 합집합이 화면 경계(이름 없는 탐지는 그리퍼 영역 경계 포함)에서 몇 픽셀 떨어져 있는가. 검출이 없으면 None."""
        H_, W_ = geom_["H"], geom_["W"]; u = np.zeros((H_, W_), bool)
        for lb_, insts_ in by_.items():
            if lb_ == place_label:
                continue
            for sm_, _sb, _cf in insts_:
                mm_ = sm_ & (geom_["zf"] > 0.1) & (geom_["zf"] < 2.0)
                if int(mm_.sum()) < 50:
                    continue
                c_ = np.median(geom_["pts_base"][mm_][:, :2], axis=0)
                if float(np.hypot(c_[0] - w_[0], c_[1] - w_[1])) < 0.10:
                    u |= sm_
        if not u.any():
            return None
        ys_, xs_ = np.where(u); bl_ = int(0.72 * H_) if AGN else H_ - 1
        _sides = (int(xs_.min()), int(ys_.min()), int(W_ - 1 - xs_.max()), int(bl_ - ys_.max()))
        _view_margin.last = dict(zip(("좌", "상", "우", "하(제외선)"), _sides)); _view_margin.last["bbox"] = (int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max()), int(bl_), int(H_), int(W_))
        return int(min(_sides))

    _sat_at = None
    for si, ang in enumerate(wide_angles):
        # [2026-09-24 추가, 속도 최적화 -- **미검증, Isaac Sim 재기동 후 라이브로
        # 반드시 먼저 확인할 것, 이 세션은 Isaac Sim 없이 작성됨**] 로봇 이동/캡처는
        # 물리적으로 순차적이어야 하지만(팔이 동시에 두 자세에 있을 수 없음), YOLO
        # 추론 네트워크 요청(Thor로 HTTP POST)은 서로 독립적이라 병렬로 쏠 수 있다.
        # 같은 j1 각도의 j7 서브스텝 전부를 먼저 캡처만 해두고, multi_detect() 호출
        # 전부를 ThreadPoolExecutor로 동시에 쏴서 왕복시간을 (스텝당 j7 개수)배에서
        # 1배로 줄인다. 필터링/dedup 로직 자체는 내용·순서 전부 기존과 동일 -- 언제
        # 실행되느냐만 바뀜(캡처 직후 하나씩 -> 이 j1의 캡처 전부 끝난 뒤 한꺼번에).
        frames = []  # [(ti, j7v, geom, rgb), ...] -- fresh_frame 성공한 것만
        for ti, j7v in enumerate(j7_sweep_values):
            print(f"[discover] 1단계 step {si}.{ti}: j1={ang:.2f}rad j7={j7v:.3f}rad (캡처)",
                  file=sys.stderr)
            n.move_j1(ang, j7_abs=j7v)
            if not n.fresh_frame():
                continue
            geom = n.capture_geom()
            frames.append((ti, j7v, geom, n.got["rgb"]))

        if not frames:
            continue
        with ThreadPoolExecutor(max_workers=len(frames)) as ex:
            detect_results = list(ex.map(
                lambda fr: _det(fr[3], fr[2]), frames))

        for (ti, j7v, geom, _rgb), by_label in zip(frames, detect_results):
            for lb, insts in by_label.items():
                for smask, sbbox, conf in insts:
                    m = smask & (geom["zf"] > 0.1) & (geom["zf"] < 2.0) & (geom["pts_base"][..., 2] > a.floor_z)
                    if int(m.sum()) < 50:
                        continue
                    w = np.median(geom["pts_base"][m][:, :3], axis=0)
                    if lb == "object":      # 2026-09-30 같은 물체도 시점마다 보이는 부분이 달라 중앙값 높이가 크게 달라진다(손전등 0.05 vs 0.23) -> 중복 판정은 xy 만 본다
                        w = np.array([w[0], w[1], 0.05])
                    # 2026-09-18 (사용자 실측 지적): 테이블이 프레임 가장자리에 부분적으로만
                    # 걸리면 순간적으로 "box"로 오검출되는 사고 확인(z=0.117로 테이블 윗면과
                    # 일치, 정상 박스는 z≈0.05). box_height_m=0.05 고정값(CLAUDE.md)이라 박스
                    # 중심 z는 뻔한 좁은 범위에 있어야 한다 -- 그 범위를 크게 벗어나면 다른
                    # 물체(가구 등)를 부분적으로 잘라서 본 것일 가능성이 높으니 후보에서 제외.
                    if lb == "box" and w[2] > a.box_z_max:
                        print(f"[discover] 1단계: box 후보 z={w[2]:.3f} > {a.box_z_max} -- "
                              f"박스 정상높이 벗어남, 다른 물체(가구 부분크롭 등)로 추정, 버림", file=sys.stderr)
                        continue
                    # [2026-09-24 추가, 사용자 지적] 테이블 높이를 0.1145->0.0625m로
                    # 낮춘 뒤(충돌방지), 테이블 표면 z가 box_z_max(0.09) 범위 안으로
                    # 들어와서 위 z 필터를 더 이상 못 걸렀다 -- z 대신 물리적 크기
                    # (박스는 0.05m 정육면체, 테이블은 훨씬 큼)로 걸러낸다. 테이블이
                    # 프레임 가장자리에 일부만 걸려도 xy 퍼짐은 여전히 박스보다 훨씬
                    # 크므로 z보다 강건한 신호.
                    if lb == "box":
                        pts_xy = geom["pts_base"][m][:, :2]
                        xy_extent = float(max(pts_xy[:, 0].max() - pts_xy[:, 0].min(),
                                               pts_xy[:, 1].max() - pts_xy[:, 1].min()))
                        if xy_extent > a.box_xy_extent_max:
                            print(f"[discover] 1단계: box 후보 xy_extent={xy_extent:.3f}m > "
                                  f"{a.box_xy_extent_max} -- 박스치고 너무 큼(테이블 등으로 추정), 버림",
                                  file=sys.stderr)
                            continue
                    # 교차 라벨 xy 중복(2026-09-29, 일반규칙): 서로 다른 라벨이 xy상 거의 같은
                    # 자리(<cross_xy_dist)면 한 물체다. 부분크롭/하단 오탐(예: 머스타드 하단을
                    # box로)은 무게중심 z가 낮아 3D 거리(_nearest)로는 dedupe_dist 밖으로 빠져
                    # 별개 후보가 됐다. 무게중심이 더 높은(=위쪽까지 다 본) 쪽을 남긴다.
                    # place_label(테이블 등)은 위에 물체가 정상적으로 놓이므로 제외.
                    if lb != place_label:
                        _xy_hits = [(g, float(np.linalg.norm(np.array(p_[:2]) - w[:2]))) for g, p_ in rough.items()
                                    if gid_lb.get(g) not in (None, place_label)]
                        _xy_hits = [(g, dd) for g, dd in _xy_hits if dd < a.cross_xy_dist]
                        # 같은 라벨은 |dz|>4cm(부분크롭: 사람 피규어가 상/하체 2개로 잡힘 -- nbr21)일 때만 같은 물체로 본다
                        _xy_hits = [(g, dd) for g, dd in _xy_hits
                                    if gid_lb.get(g) != lb or abs(w[2] - rough[g][2]) > 0.04]
                        if _xy_hits:
                            og, odd = min(_xy_hits, key=lambda t: t[1])
                            # |dz|>4cm: 부분크롭(하단만 본 것) -> 더 높은 쪽이 온전한 물체.
                            # |dz|<=4cm: 같은 물체의 라벨 혼동(bottle<->mustard bottle) -> conf 높은 쪽.
                            _dz = w[2] - rough[og][2]
                            _new_wins = (_dz > 0.04) if abs(_dz) > 0.04 else (conf > rough_conf.get(og, 0.0))
                            if not _new_wins:
                                print(f"[discover] 1단계: {lb} 후보 xy={np.round(w[:2],3).tolist()}는 {og}와 xy {odd*100:.1f}cm "
                                      f"(다른 라벨, dz={_dz:+.3f}, conf {conf:.2f} vs {rough_conf.get(og,0):.2f}) -- 같은 물체로 보고 버림", file=sys.stderr)
                                continue
                            print(f"[discover] 1단계: 기존 {og}는 {lb} 후보와 xy {odd*100:.1f}cm "
                                  f"(dz={_dz:+.3f}, conf {rough_conf.get(og,0):.2f} vs {conf:.2f}) -- 같은 물체로 보고 기존 제거", file=sys.stderr)
                            rough.pop(og); rough_j7.pop(og, None); rough_half_extent.pop(og, None); gid_lb.pop(og, None); rough_conf.pop(og, None)
                            rough_color.pop(og, None); _status_del(og)
                    gid, d = _nearest(w, rough)
                    if d < a.dedupe_dist:
                        continue  # 이미 러프하게 알고 있음
                    gid = f"{lb.replace(' ', '_')}_{next_idx[lb]}"; next_idx[lb] += 1; gid_lb[gid] = lb
                    rough[gid] = w.tolist()
                    rough_j7[gid] = j7v
                    rough_conf[gid] = float(conf)
                    rough_color[gid] = _color_name(_rgb, m)
                    _status_set(gid, label=lb, conf=float(conf), kind=('target' if lb == place_label else 'obj'))
                    if lb == place_label:
                        # 2026-09-25 실측 확인: place_label(테이블 등 큰 물체)은 2단계에서
                        # bearing_to_j1로 정면 재조준하면 오히려 카메라가 물체에 너무 가깝게
                        # 밀착돼(진단 스크린샷으로 확인 -- 테이블이 화면을 꽉 채우고 가장자리가
                        # 잘림) YOLO-World가 "table" 형태를 인식 못 하는 역효과가 있었다
                        # (bearing_to_j1 각도 계산 자체는 정확했음 -- arm_q로 실제 조준 방향
                        # 확인, RGB에 테이블이 프레임 중앙에 크게 보임에도 raw 검출 0건).
                        # 1단계 와이드 스윕 프레임은 이미 정상 검출에 성공한 구도이므로, 그
                        # 시점의 마스크로 half_extent를 미리 계산해 둔다 -- 2단계 재조준이
                        # 실패해도 이 값을 폴백으로 쓸 수 있게(아래 fallback 분기 참고).
                        try:
                            _pc, _ph, _pz, _pk = _plane_bbox(geom["pts_base"][m][:, :3])
                            rough_half_extent[gid] = [float(_ph[0]), float(_ph[1])]
                            rough[gid] = [float(_pc[0]), float(_pc[1]), float(_pz)]   # 재조준 목표/폴백 위치도 윗면 중심으로
                        except Exception:
                            pts_xy_pt = geom["pts_base"][m][:, :2]
                            rough_half_extent[gid] = [
                                float((pts_xy_pt[:, 0].max() - pts_xy_pt[:, 0].min()) / 2.0),
                                float((pts_xy_pt[:, 1].max() - pts_xy_pt[:, 1].min()) / 2.0),
                            ]
                    print(f"[discover] 1단계 새 후보: {gid} 대략위치={np.round(w,3).tolist()} conf={conf:.2f} j7={j7v:.3f}", file=sys.stderr)

        # 2026-09-29 속도: 요청한 물체(라벨별 개수)와 place target 을 다 찾았으면 남은 각도 스윕을 접는다(스윕이 첫 pick 전 95s 중 대부분).
        # 오검출로 조기 종료하지 않도록 conf>=SWEEP_EARLY_CONF 인 후보만 세고, 충족 후에도 SWEEP_EARLY_EXTRA 스텝은 더 본다.
        # NERO_SWEEP_EARLY=0 이면 끔(기존 전체 스윕). 미검증(Isaac 없이 작성) -- 첫 파일럿에서 "조기 종료" 로그와 놓친 물체 여부를 확인할 것.
        if FAST_VIEW:
            for (_ti_f, _j7_f, _geom_f, _rgb_f), _bl_f in zip(frames, detect_results):
                for _g_f, _w_f in rough.items():
                    if gid_lb.get(_g_f) == place_label:
                        continue
                    try:
                        _mg_f = _view_margin(_geom_f, _bl_f, _w_f)
                    except Exception:
                        _mg_f = None
                    if _mg_f is not None and (_g_f not in best_view or _mg_f > best_view[_g_f][3]):
                        best_view[_g_f] = (_geom_f, _bl_f, _rgb_f, _mg_f)
        if a.stop_after and len([g for g in rough if rough_conf.get(g, 1.0) >= float(os.environ.get("NERO_SWEEP_EARLY_CONF", "0.40"))]) >= a.stop_after:
            print(f"[discover] 1단계 반응형 중단: 후보 {len(rough)}개 >= {a.stop_after} (step {si}/{len(wide_angles) - 1}) -- 바로 정밀 재방문/pick", file=sys.stderr)
            break
        if os.environ.get("NERO_SWEEP_EARLY", "1") == "1":
            _need = {}
            for _g, _lb in gid_lb.items():
                if rough_conf.get(_g, 0.0) >= float(os.environ.get("NERO_SWEEP_EARLY_CONF", "0.40")):
                    _need[_lb] = _need.get(_lb, 0) + 1
            _tbl_ok = (not place_label) or any(gid_lb.get(_g) == place_label and _g in rough_half_extent for _g in rough)
            _sat = _tbl_ok and all(_need.get(_lb, 0) >= _c for _lb, _c in label_counts.items())
            if not _sat:
                _sat_at = None
            else:
                if _sat_at is None:
                    _sat_at = si
                if si - _sat_at >= int(os.environ.get("NERO_SWEEP_EARLY_EXTRA", "1")):
                    print(f"[discover] 1단계 조기 종료: 요청 물체 {dict(label_counts)} 전부 발견(step {si}/{len(wide_angles) - 1}) -- 남은 {len(wide_angles) - 1 - si}스텝 생략",
                          file=sys.stderr)
                    break

    _status_stage("refine")
    if AGN and place_label:
        # 2026-09-30 이름 없는 탐지는 테이블 윗면 조각도 "솟은 덩어리"로 잡는다(기존 테이블 검출 신뢰도가 낮아 제외 마스크가 불완전). 알려진 테이블 후보의
        # 사각형(+10cm) 안이고 높이가 테이블 윗면(+-3cm)인 object 후보는 테이블 조각으로 보고 뺀다. (테이블 위에 놓인 진짜 물체는 중심 높이가 더 높아 남는다)
        _trs = [(rough[g], rough_half_extent[g]) for g in list(rough) if gid_lb.get(g) == place_label and rough_half_extent.get(g)]
        for g in [g for g in list(rough) if gid_lb.get(g) == "object"]:
            for _c, _he in _trs:
                if abs(rough[g][0] - _c[0]) <= _he[0] + 0.10 and abs(rough[g][1] - _c[1]) <= _he[1] + 0.10 and abs(rough[g][2] - _c[2]) <= 0.03:
                    print(f"[discover] 1단계: {g} 는 테이블 윗면 조각으로 판단(테이블 {np.round(_c[:2], 3).tolist()} 안, 높이 {rough[g][2]:.3f}) -- 제외", file=sys.stderr)
                    for _d in (rough, rough_j7, rough_half_extent, gid_lb, rough_conf, rough_color):
                        _d.pop(g, None)
                    _status_del(g)
                    break
    print(f"[discover] === 1단계 완료: {len(rough)}개 후보 -> 2단계 정밀 재방문 ===", file=sys.stderr)

    # ── 2단계: 후보마다 joint1 재조준 + 정밀 캡처 + CGN ────────────────────────
    visited = set()
    # 2026-10-01 반응형 "정렬 후 바로 pick"(--stop-after K): 스윕 중 물체가 걸리면 그 물체를 화면 중앙으로 정렬(j1 재조준 + 중간 j7 1회 촬영) -> CGN -> B -> C -> pick.
    # 이번 에피소드에서는 정렬하는 후보를 K개(현재 팔 방향에서 가장 가까운 것부터)로 제한하고, 나머지 후보는 장애물 전용(fallback)으로만 등록한다
    # (CGN/B 없음, 이웃 충돌 회피용 보수적 박스). 그 물체들은 다음 에피소드에서 다시 발견해 처리한다.
    _rx_done = 0
    _rough_items = list(rough.items())
    if a.stop_after:
        try:
            _cur_j1 = float(ang)
        except Exception:
            _cur_j1 = 0.0
        _rough_items.sort(key=lambda kv: abs(bearing_to_j1(kv[1][0], kv[1][1]) - _cur_j1))
    for gid, w in _rough_items:
        if gid in visited:
            continue  # 다른 후보 재방문 중에 덤으로 이미 확정됨
        if a.stop_after and _rx_done >= a.stop_after and gid_lb.get(gid) != place_label:
            visited.add(gid)
            confirmed[gid] = {"world": [float(x) for x in w], "label": gid_lb.get(gid, "object"), "conf": None, "fallback": True}
            _status_set(gid, label=gid_lb.get(gid, "object"), conf=None, kind="obj", fallback=True)
            print(f"[discover] 2단계: {gid} 이번 에피소드는 정렬 생략(반응형 K={a.stop_after}) -- 장애물 전용으로 등록, 다음 에피소드에서 처리", file=sys.stderr)
            continue
        # 2026-09-29 속도: place target(테이블) 후보가 여러 개(table_1..3, 한 테이블이 시점별로 따로 잡힘)여도 place_target 은 conf 최고
        # 하나만 쓴다. 이미 하나가 half_extent 까지 확정됐으면 나머지 재방문(후보당 약 8s)을 생략. NERO_ONE_TABLE_VISIT=0 이면 끔.
        if (place_label and gid_lb.get(gid) == place_label and os.environ.get("NERO_ONE_TABLE_VISIT", "1") == "1"
                and any(d.get("label") == place_label and d.get("half_extent") for d in confirmed.values())):
            print(f"[discover] 2단계: {gid} 생략 -- place target 이 이미 확정됨(테이블 중복 후보)", file=sys.stderr)
            continue
        target_j1 = bearing_to_j1(w[0], w[1])
        # [2026-09-25 추가, 실측 확인] 2단계 재조준이 항상 observation 기본 joint7
        # (상한값)만 써서 재캡처했는데, 이게 1단계에서 그 후보가 실제로 잡혔던
        # joint7 각도와 다르면 재검출이 실패하는 경우가 실측으로 반복 확인됨
        # (테이블처럼 confidence가 애초에 낮은 라벨에서 특히 두드러짐 -- N=10 이후
        # 재검증 파일럿 2회 연속 "재조준했는데 재검출 안 됨"으로 half_extent를
        # 못 얻었음). 1단계에서 잡혔던 j7을 그대로 재사용해서 재검출 성공률을 높임.
        j7_reuse = rough_j7.get(gid)
        _rx_done += 1
        print(f"[discover] 2단계: {gid} 재조준 j1={target_j1:.2f}rad j7={j7_reuse} "
              f"(대략위치 {np.round(w,3).tolist()})", file=sys.stderr)
        # 2026-09-30 재방문 시야 검사(NERO_REVISIT_FIT=1 기본): 예전엔 j7 한 개로 한 번만 찍어서 물체가 화면 가장자리에 걸려 잘린 채(긴 손전등 72%, 줄자 40%만 잡힘)
        # 충돌 객체로 등록됐다. 정답 위치 근처 마스크가 화면 경계에서 NERO_REVISIT_MARGIN(기본 20)픽셀 이상 떨어져 있으면 "완전히 보임". 아니면 스윕과 같은 다른 j7 로
        # 다시 찍고, 그래도 안 되면 j1 을 +-6도 돌려 중앙으로 맞춘다. 가장 여유가 큰 설정을 최종 프레임으로 쓴다.
        _mg_ok = int(os.environ.get("NERO_REVISIT_MARGIN", "20"))
        _fast = best_view.get(gid) if FAST_VIEW else None
        if _fast is not None and _fast[3] >= _mg_ok:
            geom, by_label = _fast[0], _fast[1]
            print(f"[discover] 2단계: {gid} 재방문 생략 -- 1단계 프레임에서 완전히 보임(화면 경계 여유 {_fast[3]}px >= {_mg_ok}px)", file=sys.stderr)
        else:
            if FAST_VIEW:
                print(f"[discover] 2단계: {gid} 1단계 프레임이 잘려 보임(여유 {None if _fast is None else _fast[3]}px) -- 기존 재방문", file=sys.stderr)
            _fast = None
            _fit_on = os.environ.get("NERO_REVISIT_FIT", "1") == "1" and gid_lb.get(gid) != place_label
            _mg_ok = int(os.environ.get("NERO_REVISIT_MARGIN", "20"))

            def _grab_eval(j1c, j7c):
                n.move_j1(j1c, j7_abs=j7c)
                if not n.fresh_frame():
                    return None
                g_ = n.capture_geom(); bl_ = _det(n.got["rgb"], g_)
                return g_, bl_, (_view_margin(g_, bl_, w) if _fit_on else None)

            _cands = [(target_j1, j7_reuse)]
            if _fit_on:
                # 2026-10-01 진단: j7=1.1 에선 물체가 화면 아래(그리퍼 제외선)에, j7=1.549 에선 위에 걸려 어느 쪽도 완전히 안 보였다(시야 확보 14%).
                # 7x2 스윕으로 줄이며 사라진 중간 j7 이 물체를 화면 중앙에 놓는다 -> 스윕 j7 사이 중간값을 재방문 후보에 추가(스윕 자체는 그대로).
                _sv = sorted(float(v) for v in j7_sweep_values)
                _mid = [(a + b) / 2 for a, b in zip(_sv, _sv[1:])]
                _mid.sort(key=lambda v: abs(v - (j7_reuse if j7_reuse is not None else _sv[0])))
                _cands += [(target_j1, v) for v in _mid + _sv if (j7_reuse is None or abs(v - j7_reuse) > 1e-3)]
            _best = None; _last = None
            for _ci, (_j1c, _j7c) in enumerate(_cands):
                _r = _grab_eval(_j1c, _j7c)
                if _r is None:
                    continue
                _last = (_j1c, _j7c)
                if _fit_on:
                    print(f"[discover] 2단계 시야 진단 {gid}: {getattr(_view_margin, 'last', None)}", file=sys.stderr)
                    print(f"[discover] 2단계 시야 검사 {gid}: j1={_j1c:.2f} j7={_j7c} -> 화면 경계 여유 {_r[2]}px "
                          f"({'완전히 보임' if (_r[2] is not None and _r[2] >= _mg_ok) else '잘림 또는 미검출'})", file=sys.stderr)
                if _best is None or (_r[2] is not None and (_best[2] is None or _r[2] > _best[2])):
                    _best = (_r[0], _r[1], _r[2], _j1c, _j7c)
                if not _fit_on or (_r[2] is not None and _r[2] >= _mg_ok):
                    break
            _lim = getattr(_view_margin, "last", None) or {}
            _ud_limited = bool(_lim) and min(("좌", "상", "우", "하(제외선)"), key=lambda k: _lim.get(k, 9999)) in ("상", "하(제외선)")
            if _fit_on and _best is not None and (_best[2] is None or _best[2] < _mg_ok) and not _ud_limited:    # 위/아래가 한계면 j1(좌우)을 돌려도 소용없다
                for _dj in (0.105, -0.105):                       # j1 을 6도 돌려 화면 중앙 쪽으로 맞춰 본다(여유가 더 좋아지는 쪽 채택)
                    _r = _grab_eval(_best[3] + _dj, _best[4])
                    if _r is not None:
                        _last = (_best[3] + _dj, _best[4])
                        print(f"[discover] 2단계 시야 보정 {gid}: j1 {_dj:+.3f}rad -> 여유 {_r[2]}px", file=sys.stderr)
                        if _r[2] is not None and (_best[2] is None or _r[2] > _best[2]):
                            _best = (_r[0], _r[1], _r[2], _best[3] + _dj, _best[4])
                            if _r[2] >= _mg_ok:
                                break
            if _best is None:
                print(f"[discover] {gid}: 재방문 프레임 실패 -- 스킵", file=sys.stderr)
                continue
            if _fit_on and _last != (_best[3], _best[4]):
                _r = _grab_eval(_best[3], _best[4])               # 최종 프레임이 최선이 아니면 그 설정으로 다시 찍는다
                if _r is not None:
                    geom, by_label = _r[0], _r[1]
                else:
                    geom, by_label = _best[0], _best[1]
            else:
                geom, by_label = _best[0], _best[1]
        _blb = gid_lb.get(gid)
        if _blb and _blb != place_label and _blb != "object" and not by_label.get(_blb) and rough_color.get(gid):
            _alias = _alias_prompt(_blb, rough_color[gid])
            if _alias and _alias != _blb:
                _byA = multi_detect((_fast[2] if _fast is not None else n.got["rgb"]), {_alias: 1}, geom["H"], geom["W"])
                _keep = [(_sm, _sb, _cf) for _sm, _sb, _cf in _byA.get(_alias, [])
                         if not any(_overlap_ratio(_sb, _ob[1]) > 0.6 for _v in by_label.values() for _ob in _v)]
                if _keep:
                    by_label[_blb] = _keep
                    print(f"[discover] {gid}: '{_blb}' 재검출 0건 -> 대체 프롬프트 '{_alias}' 로 검출 성공 "
                          f"(conf {_keep[0][2]:.2f})", file=sys.stderr)
                else:
                    print(f"[discover] {gid}: 대체 프롬프트 '{_alias}' 로도 검출 못 함", file=sys.stderr)

        seg = np.zeros((geom["H"], geom["W"]), dtype=np.int32)
        step_labels, step_ids, step_worlds = [], [], []
        sid = 1
        _step_sid = {}     # 이 프레임에서 확정한 gid -> seg id (같은 물체의 조각을 한 seg 로 합치기 위함)
        any_match = False  # 2026-09-25 버그수정: place_label 매칭은 step_labels에
        # 안 들어가므로(CGN 후보 아님) step_labels만 보고 "재검출 실패"를 판단하면
        # 테이블이 정상 재검출+half_extent 계산까지 성공한 경우도 아래 else 폴백이
        # 실행돼 confirmed[gid]를 덮어써서 half_extent를 파괴하는 버그가 있었다
        # (실측 확인: half_extent가 모든 파일럿에서 계속 null이었던 원인).
        base_lb = gid_lb[gid]
        for lb, insts in by_label.items():
            for smask, sbbox, conf in insts:
                m = smask & (geom["zf"] > 0.1) & (geom["zf"] < 2.0) & (geom["pts_base"][..., 2] > a.floor_z)
                if int(m.sum()) < 50:
                    continue
                w2 = np.median(geom["pts_base"][m][:, :3], axis=0)
                if lb == "box" and w2[2] > a.box_z_max:
                    print(f"[discover] 2단계: box 재검출 z={w2[2]:.3f} > {a.box_z_max} -- "
                          f"박스 정상높이 벗어남, 버림", file=sys.stderr)
                    continue
                if lb == "box":  # 2026-09-24: 1단계와 동일 xy 퍼짐 필터(위 주석 참고)
                    pts_xy2 = geom["pts_base"][m][:, :2]
                    xy_extent2 = float(max(pts_xy2[:, 0].max() - pts_xy2[:, 0].min(),
                                            pts_xy2[:, 1].max() - pts_xy2[:, 1].min()))
                    if xy_extent2 > a.box_xy_extent_max:
                        print(f"[discover] 2단계: box 재검출 xy_extent={xy_extent2:.3f}m > "
                              f"{a.box_xy_extent_max} -- 박스치고 너무 큼, 버림", file=sys.stderr)
                        continue
                # 이 라벨의 아직 안 찾아간 러프 후보 중 가장 가까운 것에 매칭.
                cand_gids = {g: p for g, p in rough.items() if gid_lb.get(g) == lb and g not in visited}
                _zfix = (lb == "object")
                w2m = np.array([w2[0], w2[1], 0.05]) if _zfix else w2
                match_gid, d = _nearest(w2m, cand_gids)
                if match_gid is None or d >= a.dedupe_dist * 1.5:
                    # rough(미방문) 후보 중엔 없음 -- 그래도 "완전히 새 물체"로 단정하기 전에
                    # 이미 확정된(confirmed) 같은 라벨 물체와도 겹치는지 확인한다. 다른 gid를
                    # 재조준하다가(예: place_label 재조준 중) 이미 확정된 물체가 카메라에 다시
                    # 걸리는 경우가 실측 확인됨(2026-09-21, table 재조준 중 box_1이 box_4로
                    # 중복 확정) -- rough만 보면 이미 visited라 놓치므로 confirmed까지 봐야 함.
                    confirmed_same_lb = {g: ([v["world"][0], v["world"][1], 0.05] if _zfix else v["world"]) for g, v in confirmed.items()
                                         if v.get("label") == lb and g != gid}
                    dup_gid, dup_d = _nearest(w2m, confirmed_same_lb)
                    if dup_gid is not None and dup_d < a.dedupe_dist * 1.5:
                        print(f"[discover] 2단계 재검출은 이미 확정된 {dup_gid}와 동일 물체로 판단"
                              f"(d={dup_d:.3f}m) -- 중복 생성 안 함", file=sys.stderr)
                        continue
                    # 그래도 안 겹침 -- 재방문 중 덤으로 걸린 완전히 새 물체.
                    match_gid = f"{lb.replace(' ', '_')}_{next_idx[lb]}"; next_idx[lb] += 1; gid_lb[match_gid] = lb
                    print(f"[discover] 2단계 덤 발견: {match_gid} @ {np.round(w2,3).tolist()}", file=sys.stderr)
                # 2026-09-30 중복/조각 병합(NERO_MERGE_DUP=1 기본): 미방문 후보와 짝이 지어진 경우에도 이미 확정된 같은 라벨 물체와 xy 6cm 안이면 같은 물체로 본다.
                # (이전엔 위의 확정 물체 검사가 "짝 없음"일 때만 실행돼서, 한 물체가 0.2cm 간격의 중복(줄자) 또는 4cm 간격의 조각(테이프 고리 양쪽 호) gid 둘로 확정되어
                #  서로를 장애물로 보며 둘 다 C단계 실패했다 -- vis2 육안 확인에서 발견). 같은 프레임에서 확정된 것이면 마스크를 합쳐 CGN 입력으로 쓴다.
                if os.environ.get("NERO_MERGE_DUP", "1") == "1" and lb != place_label:
                    _dg, _dd = None, 9.0
                    for _g, _v in confirmed.items():
                        if _g == match_gid or _v.get("label") != lb:
                            continue
                        _d = math.hypot(_v["world"][0] - w2[0], _v["world"][1] - w2[1])
                        if _d < 0.06 and _d < _dd:
                            _dg, _dd = _g, _d
                    if _dg is not None:
                        print(f"[discover] 2단계: {match_gid} 는 이미 확정된 {_dg} 와 xy {_dd*100:.1f}cm -- 같은 물체의 중복/조각으로 보고 병합(새 물체 만들지 않음)", file=sys.stderr)
                        if _dg in _step_sid:
                            seg[m] = _step_sid[_dg]
                        visited.add(match_gid)
                        _status_del(match_gid)
                        any_match = True        # 병합했으니 "재검출 실패 -> 러프 위치 폴백"이 같은 gid 를 다시 확정하지 않게 한다
                        continue
                visited.add(match_gid)
                confirmed[match_gid] = {"world": [float(x) for x in w2], "label": lb, "conf": conf}
                _status_set(match_gid, label=lb, conf=float(conf), kind=("target" if lb == place_label else "obj"))
                any_match = True
                if lb == place_label:
                    # [2026-09-24 추가, 사용자 아이디어] place-target(테이블 등)의 발판
                    # 크기(월드 x/y 축 기준 half-extent)도 같이 계산해서 place_on_target.sh가
                    # 중심점 하나만 보고 단순 x축 선형 스프레드(SPREAD_DX)로 박스를 밀어내다
                    # 우연히 top-down IK 도달불가 지점에 떨어뜨리는 문제(실측 확인, N=10의
                    # discovery 성공 시행 중 11%)를 줄인다 -- place_on_target.sh가 이 테두리
                    # 안에서 IK 가능한 후보를 여러 개 시도할 수 있게. 회전각(PCA 기반)은
                    # CLAUDE.md에 이미 "estimate_object_geometry의 major_axis_yaw_deg 신뢰
                    # 불가"로 명시된 known-unreliable 방법이라 의도적으로 안 씀 -- 대신 월드
                    # 축 기준 axis-aligned half-extent만 써서(테이블이 randomize_trial.py로
                    # 항상 축정렬 배치되는 이 실험 세팅에 안전) 회전 추정 없이도 안전한 사각형을
                    # 얻는다. 실제 회전된 테이블에도 안전하게 하려면(과소평가 방향이라 완전히
                    # 틀리진 않음, 다만 더 보수적인 사각형이 됨) 추후 견고한 회전 추정 방법이
                    # 필요.
                    try:
                        _pc, _ph, _pz, _pk = _plane_bbox(geom["pts_base"][m][:, :3])
                        hx, hy = float(_ph[0]), float(_ph[1])
                        confirmed[match_gid]["world"] = [float(_pc[0]), float(_pc[1]), float(_pz)]
                    except Exception:
                        pts_xy_pt = geom["pts_base"][m][:, :2]
                        hx = float((pts_xy_pt[:, 0].max() - pts_xy_pt[:, 0].min()) / 2.0)
                        hy = float((pts_xy_pt[:, 1].max() - pts_xy_pt[:, 1].min()) / 2.0)
                    confirmed[match_gid]["half_extent"] = [hx, hy]
                    _rc = _plane_rect(geom["pts_base"][m][:, :3])
                    if _rc:
                        confirmed[match_gid]["rect"] = _rc
                        print(f"[discover] {match_gid}: 윗면 회전 사각형 yaw={np.rad2deg(_rc['yaw']):.0f}도 half={np.round(_rc['half'], 3).tolist()} "
                              f"(축정렬 half={[round(hx, 3), round(hy, 3)]})", file=sys.stderr)
                    # 2026-09-18: 놓을 곳(바구니 등)은 집을 대상이 아니므로 CGN 파지후보를
                    # 안 만든다 -- seg/step_*에 안 넣어서 이 아래 _run_cgn 배치에서 제외.
                    continue
                seg[m] = sid
                _step_sid[match_gid] = sid
                step_labels.append(match_gid); step_ids.append(sid); step_worlds.append(w2)
                sid += 1

        if step_labels:
            _run_cgn(n, geom, seg, step_labels, step_ids, step_worlds, tag=gid)
            _submit_early_b(step_labels)
        elif not any_match:
            print(f"[discover] {gid}: 재조준했는데 재검출 안 됨 -- 러프위치로 폴백", file=sys.stderr)
            visited.add(gid)
            confirmed[gid] = {"world": w, "label": base_lb, "conf": None, "fallback": True}
            _status_set(gid, label=base_lb, conf=None, kind=("target" if base_lb == place_label else "obj"), fallback=True)
            if base_lb == place_label and gid in rough_half_extent:
                # 2026-09-25: 2단계 재조준 재검출 실패 -- 1단계에서 미리 계산해둔
                # half_extent로 폴백(위 rough_half_extent 계산부 설명 참고).
                confirmed[gid]["half_extent"] = rough_half_extent[gid]

    # 2026-09-18: 놓을 곳(place_label) 항목을 confirmed(="boxes", 집을 대상 큐)에서
    # 빼내 별도 place_target으로 분리 -- 안 그러면 stack_discovered.sh 류가 바구니를
    # 집으려 시도한다. 여러 개 잡혔으면(오검출 포함) confidence 가장 높은 것만 채택.
    place_target = None
    place_target_box[0] = None
    if place_label:
        cands = [(gid, d) for gid, d in confirmed.items() if d.get("label") == place_label]
        for gid, d in cands:
            del confirmed[gid]
        if cands:
            best_gid, best = max(cands, key=lambda kv: (kv[1].get("conf") or 0.0))
            _status_set(place_label, label=place_label, conf=best.get("conf"), kind="target")
            place_target = {"label": place_label, "world": best["world"], "conf": best.get("conf"),
                             "half_extent": best.get("half_extent"), "rect": best.get("rect")}  # None이면 place_on_target.sh가 기존 선형 스프레드로 폴백
            # 2026-09-30 테이블 장애물 크기: 한 시점에서 본 부분 점군의 half_extent 는 실제보다 작다(검출 0.148x0.095 vs 실제 축정렬 0.195x0.17 -> 팔이 테이블 모서리를 침).
            # 1단계에서 여러 시점에 잡힌 테이블 후보들(table_1..N)의 범위 합집합으로 장애물 박스를 만든다(place 슬롯 계산용 half_extent 는 그대로).
            try:
                if os.environ.get("NERO_TABLE_UNION", "1") == "1" and place_target.get("half_extent"):
                    _c0 = place_target["world"]; _h0 = place_target["half_extent"]
                    _xs = [_c0[0] - _h0[0], _c0[0] + _h0[0]]; _ys = [_c0[1] - _h0[1], _c0[1] + _h0[1]]
                    for _g, _w in rough.items():
                        if gid_lb.get(_g) == place_label and _g in rough_half_extent:
                            _hx, _hy = rough_half_extent[_g]
                            _xs += [_w[0] - _hx, _w[0] + _hx]; _ys += [_w[1] - _hy, _w[1] + _hy]
                    _uc = [(min(_xs) + max(_xs)) / 2.0, (min(_ys) + max(_ys)) / 2.0]; _uh = [(max(_xs) - min(_xs)) / 2.0, (max(_ys) - min(_ys)) / 2.0]
                    if _uh[0] > _h0[0] + 0.005 or _uh[1] > _h0[1] + 0.005:
                        place_target["obstacle"] = {"world": [_uc[0], _uc[1], _c0[2]], "half_extent": [_uh[0], _uh[1]]}
                        print(f"[discover] 테이블 장애물 확장: half {np.round(_h0, 3).tolist()} -> {np.round(_uh, 3).tolist()}, 중심 {np.round(_uc, 3).tolist()} (시점별 후보 합집합)", file=sys.stderr)
            except Exception as _eu:
                print(f"[discover] 테이블 합집합 실패({_eu}) -- 기존 크기", file=sys.stderr)
            place_target_box[0] = dict(place_target, rect=place_target.get("rect"))
            print(f"[discover] place_target({place_label}) 확정: {np.round(best['world'],3).tolist()} "
                  f"half_extent={best.get('half_extent')}"
                  + (f" (후보 {len(cands)}개 중 최고 conf 채택)" if len(cands) > 1 else ""), file=sys.stderr)
        else:
            print(f"[discover] place_target({place_label}) 스윕 중 못 찾음", file=sys.stderr)

    # 스윕 끝나면 관측 자세로 복귀 -- 이후 B/C단계가 쓸 NERO_ARM_Q 시드 확보.
    for _g in list(_STATUS["objects"]):
        if _g not in confirmed and _g != place_label:
            _status_del(_g)
    _status_stage("plan")
    print("[discover] 전체 종료 -- 관측 자세로 복귀", file=sys.stderr)
    n.home()
    for _ in range(20):
        rclpy.spin_once(n, timeout_sec=0.1)
    arm_q = n.got.get("arm_q")

    # 2026-09-18: B/C단계를 여기서 미리 계산해두는 걸 한 번 껐다가(SEED 신선도가
    # 원인이라는 가설) 다시 켬 -- 같은 SEED로 two_pick_plan_fast.py 를 반복 실행해
    # 직접 반증함(box_1 min_margin 0.17->0.13, box_3 0.21->0.39, 매번 다름). 원인은
    # SEED 가 아니라 IK 서비스(pick_ik, kinematics.yaml `mode: global`, memetic/진화
    # 알고리즘)가 동일 입력에도 매 호출 랜덤 결과를 낸다는 것 -- `minimal_displacement_weight:
    # 0.0`이라 현재 관절각에 가까운 해를 선호하지도 않는다. 즉 precompute 시점을
    # 언제로 잡든(discover 직후든 pick 직전이든) 이 비결정성은 그대로다 -- fresh-seed로
    # 미루는 건 실제 원인을 못 고치면서 물체당 B/C단계(~6~12s) 시간만 날린다.
    # 진짜 대응은 kinematics.yaml 의 minimal_displacement_weight 를 올리는 것(A/B 검증
    # 중, docs 참고). 여기서는 다시 기본 precompute(discover 직후 1회 계산 후 재사용)로
    # 되돌림 -- --no-precompute-bc 로 끌 수 있다(A/B 비교용).
    # candidate2(태스크-적응형 cost 가중치): --task 있으면 Thor에서 task_weight.py를
    # 이 실행 전체에 1번만 호출(물체당 아님) -- ~/grasp/_taskw.json에 캐싱, 이후 모든
    # gid의 B단계 호출에 NERO_TASK_WEIGHT=1을 실어 배율을 적용한다. --task 없으면
    # taskw_env=""(기존 동작과 100% 동일, two_pick_plan_fast.py 쪽도 플래그 없으면
    # 배율 1.0 폴백이라 이중 안전).
    taskw_env = ""
    if a.task:
        print(f"[discover] candidate2: task_weight.py 1회 호출 -- \"{a.task}\"", file=sys.stderr)
        tw = _run(["ssh", "thor",
                             f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                             f"cd ~/grasp && python3 task_weight.py {shlex.quote(a.task)} ~/grasp/_taskw.json"],
                            capture_output=True, text=True, timeout=90)
        print(f"  {tw.stderr.strip().splitlines()[-1] if tw.stderr.strip() else '(출력 없음)'}", file=sys.stderr)
        taskw_env = "NERO_TASK_WEIGHT=1 "

    # C단계 안전망의 장애물 폴백: CGN 이 실패한(box_fit 없는) 물체도 장애물로 쓰도록 탐색 위치를 넘긴다.
    _run(["ssh", "thor", "cat > ~/grasp/scene_objs.json"],
         input=json.dumps({**{g: {"world": v["world"], "label": v.get("label")} for g, v in confirmed.items()},
                           **({"__table__": {"world": place_target.get("obstacle", place_target)["world"], "half_extent": place_target.get("obstacle", place_target).get("half_extent")}}
                              if place_target else {})}),
         text=True, timeout=30)

    if a.pipeline_exec and (not a.precompute_bc or not arm_q or (place_target is None and not a.no_place)):
        print("[discover] --pipeline-exec 은 --precompute-bc(기본 켜짐) + arm_q 확보 + "
              "--place-label로 place_target 확정을 전제로 한다 -- 조건 불충족, 파이프라인 "
              "모드 비활성화하고 기존 방식으로 진행", file=sys.stderr)
        a.pipeline_exec = False

    if a.pipeline_exec:
        # 2026-09-25 구현(사용자 제안): "1번 물체 계산 끝나면 바로 1번 pick+place 시작,
        # 그동안 2-3번 계산은 배경에서 계속" -- 지금까지는 discover_boxes.py가 3개
        # 물체의 B/C단계(팔 불필요한 순수 Thor 원격계산)를 전부 끝내고 나서야 리턴하고,
        # 그제서야 place_on_target.sh가 pick+place(팔 필요)를 시작했다. 이 사이 pick+place
        # 실행 시간(보통 60~100s/물체)에 나머지 물체 계산(몇~십몇 초)이 겹칠 기회를
        # 날리고 있었다. gid별 B/C를 백그라운드 스레드로 던져두고, as_completed로
        # 먼저 끝나는 gid부터 즉시 place_on_target_one.sh(팔 사용, 메인스레드에서 블로킹
        # 호출)를 실행 -- 그 사이 다른 스레드들의 SSH 호출(I/O 대기라 GIL 안 막음)은
        # 계속 진행된다.
        arm_q_str = " ".join("%.5f" % x for x in arm_q)
        precomputed = {gid: False for gid in confirmed}

        def _write_out():
            _out = {"boxes": confirmed, "arm_q": arm_q, "precomputed": precomputed, "place_target": place_target}
            json.dump(_out, open(os.path.expanduser(a.out), "w"), indent=1)

        _write_out()  # place_on_target_one.sh가 읽을 place_target을 먼저 기록
        for gid in confirmed:
            _rm = [gid + "_step6_pick.json"] + ([] if gid in _early_b else [gid + "_pick.json"])
            _run(["ssh", "thor", "rm -f " + " ".join("~/grasp/" + shlex.quote(f) for f in _rm)
                  + " ~/grasp/step6_pick.json"], timeout=30)

        WIDE_HS = int(os.environ.get("NERO_WIDE_HIT_SPAN", "120"))   # 0 = 끔
        SWEEP_TOL_DEADLOCK = float(os.environ.get("NERO_SWEEP_TOL_DEADLOCK", "0.035"))   # 교착(전부 막힘)일 때만 쓰는 2단계 최대 침투(m)
        SWEEP_TOL_MAX = float(os.environ.get("NERO_SWEEP_TOL_MAX", "0.02"))   # 막힘 해소 불가 시 허용할 최대 침투(m)
        # 2026-09-30 허용오차 버그 수정(NERO_TOL_FIX=1 일 때만, 기본 OFF -- 야간 큐 조건을 안 흔들려고): step6_base 의 판정은 `간격 < 여유(NERO_SWEEP_MARGIN, 1cm) - 허용오차` 인데
        # 완화 로직은 허용오차를 `-간격 + 0.3cm` 로 계산해 여유 1cm 를 빼먹었다 -> 허용오차가 항상 1cm 모자라 완화 재계산 19번이 전부 실패(간격 +0.0cm 후보도 탈락).
        # 새 식: 허용오차 = max(0, 여유 - 간격) + 0.3cm, 상한 = 허용할 최대 실제 침투(cap) + 여유.
        TOL_FIX = os.environ.get("NERO_TOL_FIX", "0") == "1"
        SWEEP_MARGIN = float(os.environ.get("NERO_SWEEP_MARGIN", "0.01"))

        def _tol_for(gap, cap):
            if TOL_FIX:
                return min(cap + SWEEP_MARGIN, max(0.0, SWEEP_MARGIN - gap) + 0.003)
            return min(cap, max(0.0, -gap) + 0.003)
        c_bestgap = {}    # gid -> 마지막 C단계에서 가장 덜 막힌 후보의 간격(m, 음수=침투)
        # 2026-09-30 (a) 여유 = 순위 점수: C단계가 후보마다 계산한 최소 간격(clearance, m)을 물체별로 모아 pick 순서에 쓴다.
        # 막혀서 실패한 물체는 stall 을 기다리지 않고 바로 스윕 여유를 조금 완화한 "soft" 계획을 만들어 대기열에 올리되 순서는 뒤로 미룬다.
        c_clear = {}      # gid -> 실행할 첫 plan 의 clearance (m)
        c_soft = {}       # gid -> soft 로 통과시킨 물체를 막던 blocker 집합

        def _widen_b(gid):
            """C단계가 이웃 충돌로 후보를 전부 버려 실패했을 때, 그 물체만 B(IK 랭킹)를 넓은 탐색폭으로
            다시 돌린다. 시드 100 실측: 기본 폭(HIT_SPAN=20)은 병 후보 2개가 전부 머스타드 쪽 접근이라
            C 실패, 폭 120 은 8개 -> 반대편 접근 1개 통과. 상시 확대는 B 시간이 2~3배라 실패 때만 쓴다."""
            _status_set(gid, b=None, b_note="wider IK search")
            _t = time.time()
            b2 = _run(["ssh", "thor",
                       f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                       f"cd ~/grasp && export NERO_ARM_Q='{arm_q_str}' {taskw_env}{_table_env()}NERO_HIT_SPAN={WIDE_HS} NERO_KEEP_TOPN=16 && "
                       f"python3 two_pick_plan_fast.py {shlex.quote(gid)}"],
                      capture_output=True, text=True, timeout=300)
            ok2, nr2 = _parse_b(b2.stdout or "")
            _status_set(gid, b=ok2, b_n=nr2, b_note=("wider IK search" if ok2 else "wider search: no reachable"),
                        b_sec=round(time.time() - _t, 1))
            print(f"[discover] {gid}: 넓은 B 재탐색 {'OK' if ok2 else 'FAIL'} "
                  f"(도달 후보 {nr2}, {time.time() - _t:.0f}s)", file=sys.stderr)
            return ok2

        def _run_bc(gid):
            _tb = time.time()
            if gid in _early_b:
                b = _early_b[gid].result()
                print(f"[discover] {gid}: 조기 B단계 결과 사용", file=sys.stderr)
            else:
                b = _run(["ssh", "thor",
                                   f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                                   f"cd ~/grasp && export NERO_ARM_Q='{arm_q_str}' {taskw_env}{_table_env()}&& python3 two_pick_plan_fast.py {shlex.quote(gid)}"],
                                  capture_output=True, text=True, timeout=180)
            ok_b = "reachable" in b.stdout and "no reachable" not in b.stdout and "no candidate" not in b.stdout
            _log_nbr(gid, b)
            _status_set(gid, b=ok_b, b_n=_parse_b(b.stdout or "")[1],
                        b_note=("no CGN grasps" if "no grasps file" in (b.stdout or "") else None),
                        **({} if gid in _early_b else {"b_sec": round(time.time() - _tb, 1)}))
            if not ok_b and "no grasps file" not in (b.stdout or "") and WIDE_HS > 0:
                print(f"[discover] {gid}: B 도달 후보 0 -- HIT_SPAN={WIDE_HS} 로 넓혀 재탐색", file=sys.stderr)
                ok_b = _widen_b(gid)
            if not ok_b:
                _status_set(gid, c="-")
                return gid, False, "B", set()
            _oth = [g for g in confirmed if g != gid]
            ok_c, blk = _run_c(gid, _oth, [])
            if not ok_c and blk and WIDE_HS > 0:
                print(f"[discover] {gid}: 이웃 충돌로 C 실패(지목 {sorted(blk)}) -- B를 HIT_SPAN={WIDE_HS} 로 "
                      f"넓혀 재탐색", file=sys.stderr)
                if _widen_b(gid):
                    ok_c, blk = _run_c(gid, _oth, [])
            if ORDER_C:
                c1_info[gid] = (frozenset(blk) if not ok_c else frozenset(), c_bestgap.get(gid, -9.0))
            if not ok_c and blk and ORDER_C:
                _asm = frozenset(blk)
                _o_s = [g for g in confirmed if g != gid and g not in _asm and g not in removed_set]
                print(f"[discover] {gid}: 순서 기반 C 선계산 시작 -- {sorted(_asm)} 가 치워진다고 가정", file=sys.stderr)
                spec_pre[gid] = (ex.submit(_run_c, gid, _o_s, [], 0.0, None, "_pre"), _asm)
                # 순환 막힘(서로 지목)이면 덜 막힌 쪽만 soft 로 먼저 나가고, 더 막힌 쪽은 soft 를 돌리지 않고 선계산이 채택되길 기다린다(즉시 반환).
                _tw = time.time()
                while time.time() - _tw < 8.0 and any(b not in c1_info and b not in removed_set for b in blk):
                    time.sleep(0.5)
                _mut = [b for b in blk if b in c1_info and gid in c1_info[b][0]]
                if _mut and any((c1_info[b][1], b) > (c1_info[gid][1], gid) for b in _mut):
                    print(f"[discover] {gid}: 순환 막힘 {sorted(_mut)} -- 더 막힌 쪽이라 soft 계산 생략, 상대가 먼저 나간 뒤 선계산 채택", file=sys.stderr)
                    return gid, False, "C", blk
            if not ok_c and blk and os.environ.get("NERO_SOFT_C", "1") == "1":
                _bg = c_bestgap.get(gid)
                if _bg is not None and _bg >= -SWEEP_TOL_MAX:
                    _tol = _tol_for(_bg, SWEEP_TOL_MAX)
                    print(f"[discover] {gid}: 막힘(지목 {sorted(blk)}) -- 가장 덜 막힌 후보 간격 {_bg * 100:+.1f}cm, stall 을 기다리지 않고 "
                          f"스윕 여유 {_tol * 100:.1f}cm 완화한 soft 계획 계산", file=sys.stderr)
                    _ok_s, _ = _run_c(gid, _oth, [], _tol)
                    if _ok_s:
                        c_soft[gid] = set(blk)
                        return gid, True, None, set()
            return gid, ok_c, (None if ok_c else "C"), blk


        CUROBO_URL = os.environ.get("NERO_CUROBO_URL", "")     # 2026-10-01 예: http://localhost:8020 -- 비우면 기존 STOMP(step6_base) 경로

        def _run_c_curobo(gid, others, placed, tol, start_q, tag):
            """C단계를 PC 의 cuRobo 서버로 계산한다. 장애물/후보는 Thor 의 step6_scene.py(step6_base 의 장애물 구성 로직)가 만들고, 결과 플랜은
            step6_base 와 같은 형식의 JSON({gid}_step6_pick{tag}.json)으로 Thor 에 써서 이후 exec 경로를 그대로 쓴다."""
            import urllib.request
            if not tag:
                _status_set(gid, c="run")
            _tc = time.time()
            _qs = start_q or arm_q_str
            _pz = float(place_target["world"][2]) if place_target else 0.065
            _pl = " ".join(f"{g}:{x:.4f}:{y:.4f}:{_pz:.4f}" for g, x, y in placed)
            _out = f"~/grasp/{shlex.quote(gid + '_step6_pick' + tag + '.json')}"
            _run(["ssh", "thor", f"rm -f {_out}"], timeout=30)
            blk = set(); ok = False; note = None
            try:
                r = _run(["ssh", "thor", f"cd ~/grasp && NERO_OTHER_OBJS={shlex.quote(' '.join([gid] + list(others)))} NERO_PLACED_OBJS={shlex.quote(_pl)} "
                          f"NERO_OBB={os.environ.get('NERO_OBB', '1')} python3 step6_scene.py {shlex.quote(gid)}"], capture_output=True, text=True, timeout=60)
                sc = json.loads(r.stdout)
                req = dict(start_q=[float(x) for x in _qs.split()], candidates=sc["candidates"], obstacles=sc["obstacles"], target=sc["target"], tol=float(tol),
                           margin=float(os.environ.get("NERO_CUROBO_MARGIN", "0.01")), max_cands=int(os.environ.get("NERO_CUROBO_MAXC", "4")))
                rq = urllib.request.Request(CUROBO_URL + "/plan", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
                resp = json.loads(urllib.request.urlopen(rq, timeout=90).read())
                for b in resp.get("blockers", []) or []:
                    blk.add(b.split("_", 1)[1] if b.startswith(("other_", "placed_")) else b)
                if resp.get("ok"):
                    pl = resp["plan"]; top = dict(pl); top["plans"] = [pl]; top["run_ts"] = time.time()
                    w = _run(["ssh", "thor", f"cat > {_out}"], input=json.dumps(top), text=True, timeout=30)
                    ok = (w.returncode == 0)
                    print(f"  [{gid}] cuRobo C OK ({resp.get('sec')}s, 후보 {pl.get('cand_index')}, 장애물 {len(sc['obstacles'])})", file=sys.stderr)
                else:
                    note = "cuRobo: " + "; ".join(resp.get("why", []))[:120]
                    print(f"  [{gid}] cuRobo C FAIL ({resp.get('sec')}s) {resp.get('why')} 막힘 {sorted(blk) or '없음'}", file=sys.stderr)
                    for b in blk:
                        print(f"  [{gid}] 팔 스윕 충돌 grip <-> other_{b} (cuRobo 진단: 이 물체를 빼면 계획 성공)", file=sys.stderr)
            except Exception as _e:
                note = f"cuRobo error {_e!r}"[:120]
                print(f"  [{gid}] cuRobo C 호출 실패: {_e!r}", file=sys.stderr)
            if not tag:
                c_bestgap.pop(gid, None)
                if ok:
                    c_clear[gid] = 0.05
                _status_set(gid, c=ok, c_note=note, c_sec=round(time.time() - _tc, 1))
            return ok, blk

        def _run_c(gid, others, placed, tol=0.0, start_q=None, tag=""):
            """C단계(STOMP). others = 원위치 장애물 gid, placed = [(gid, x, y)] 옮겨 놓은 물체.
            start_q/tag: chain(홈 복귀 없이 이어 붙이기)용 -- 앞 물체 lift 끝 자세를 시작점으로 별도 파일({gid}_step6_pick{tag}.json)에 계획."""
            if CUROBO_URL:
                _okc, _blkc = _run_c_curobo(gid, others, placed, tol, start_q, tag)
                if _okc or _blkc or os.environ.get("NERO_CUROBO_FALLBACK", "1") != "1":
                    return _okc, _blkc
                # cuRobo 가 이웃 탓이 아닌데(막힘 지목 없음) 실패한 경우만 STOMP 로 폴백한다(관절한계/손목 자세처럼 cuRobo IK 가 못 찾는 후보).
                print(f"  [{gid}] cuRobo 실패(막힘 없음) -- STOMP 로 폴백", file=sys.stderr)
            if not tag:
                _status_set(gid, c="run")
            _tc = time.time()
            _qs = start_q or arm_q_str
            _pz = float(place_target["world"][2]) if place_target else 0.065     # pick 전용 모드엔 place target 없음(놓인 물체도 없어 값 미사용)
            _pl = " ".join(f"{g}:{x:.4f}:{y:.4f}:{_pz:.4f}" for g, x, y in placed)
            _run(["ssh", "thor", f"rm -f ~/grasp/{shlex.quote(gid+'_step6_pick'+tag+'.json')}"], timeout=30)
            c = _run(["ssh", "thor",
                      f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                      f"cd ~/grasp && export NERO_ARM_Q='{_qs}' NERO_STEP6_TAG='{tag}' NERO_OBB='{os.environ.get('NERO_OBB','1')}' NERO_VEL_SCALE='{os.environ.get('NERO_VEL_SCALE','1.0')}' "
                      f"NERO_OTHER_OBJS={shlex.quote(' '.join([gid] + list(others)))} NERO_PLACED_OBJS={shlex.quote(_pl)} "
                      f"NERO_SWEEP_TOL={tol:.4f} && "
                      f"bash step6_single.sh {shlex.quote(gid)}"],
                     capture_output=True, text=True, timeout=300)
            try:      # 2026-09-30 진단: C단계 전체 출력을 파일로 (STOMP/충돌 실패 위치 분석용)
                with open(f"/tmp/C_{gid}.log", "a") as _cf:
                    _cf.write(f"\n===== {time.strftime('%H:%M:%S')} tol={tol} others={others} placed={placed}\n" + (c.stdout or "") + (c.stderr or ""))
            except Exception:
                pass
            blk = set(); gaps = []
            for ln in (c.stdout or "").splitlines():
                if "옮겨 놓은" in ln or "스윕 충돌" in ln or "OBB" in ln:
                    print(f"  [{gid}] {ln.strip()}", file=sys.stderr)
                m = re.search(r"<-> (?:other|placed)_(\S+)", ln)
                if m:
                    blk.add(m.group(1))
                    mg = re.search(r"최소간격 (-?\d+\.\d+)", ln) or re.search(r"간격 (-?\d+\.\d+)", ln)   # 전 부위 최소 간격 우선(첫 위반 부위 값은 낙관적)
                    if mg:
                        gaps.append(float(mg.group(1)))
            if not tag:
                if gaps:
                    c_bestgap[gid] = max(gaps)
                else:
                    c_bestgap.pop(gid, None)
            r2 = _run(["ssh", "thor", f"test -f ~/grasp/{shlex.quote(gid+'_step6_pick'+tag+'.json')}"], timeout=30)
            _txt = c.stdout or ""
            _note = None
            if r2.returncode != 0:
                _note = ("STOMP budget" if "예산" in _txt else "STOMP TIMED_OUT" if "TIMED_OUT" in _txt
                         else ("collision: " + ", ".join(sorted(blk)) if blk else ("no output (timeout?)" if not _txt.strip() else "plan failed")))
            if tag:
                return r2.returncode == 0, blk
            if r2.returncode == 0:
                try:
                    _rc = _run(["ssh", "thor", "python3 -c \"import json;d=json.load(open('" + GRASP_DIR + "/" + gid + "_step6_pick.json'));"
                                "print(d['plans'][0].get('clearance', 9.0))\""], capture_output=True, text=True, timeout=30)
                    c_clear[gid] = float((_rc.stdout or "9.0").strip())
                except Exception:
                    c_clear.pop(gid, None)
            _status_set(gid, c=(r2.returncode == 0), c_note=_note, c_sec=round(time.time() - _tc, 1))
            return r2.returncode == 0, blk

        print(f"[discover] === 파이프라인 실행 모드: gid별 B/C 완료되는 즉시 그 물체부터 "
              f"pick+place 시작(나머지는 배경 계산 계속) ===", file=sys.stderr)
        # 2026-09-29 (사용자 지시): 잡을 수 있는 물체부터 먼저 옮기고, C단계(이웃 장애물 의존)에서
        # 실패한 물체는 다른 물체가 옮겨질 때마다 바뀐 장애물 집합으로 C단계만 재계산한다(B단계는
        # 이웃과 무관해 재사용). 옮긴 물체 = 슬롯 레지스트리에 기록된 gid -> 원위치 장애물에서 빼고
        # 놓은 자리에 장애물로. 장애물 집합이 지난 시도와 같으면 재시도 안 함.
        # 2026-09-29 막힘 그래프 순서 계획(사용자 지시 "앞에 있는 걸 먼저 치운다"). C단계 실패가 지목한
        # 막던 물체(c_blockers)로 "X 는 Y 가 막고 있다" 관계를 만들고:
        #  1) 팔이 비었을 때 준비된 물체 중 "막힌 다른 물체를 가장 많이 풀어주는" 것을 먼저 집는다
        #     (막힌 물체의 C 재계산이 나머지 물체 pick 과 겹쳐 돌아 끝에서 기다리지 않게).
        #  2) 막던 물체가 옮겨지면(슬롯 레지스트리) 막혔던 물체의 C단계만 재계산한다(B 는 재사용).
        #  3) 더 진행할 수 없는 정지 상태(막던 물체가 못 집힘/순환)면 가장 덜 막힌 물체 하나를 골라
        #     스윕 허용오차를 조금(<= NERO_SWEEP_TOL_MAX) 완화해 재계산한다 -- 그래도 안 되면 포기.
        box_index = 0
        _slot_reg = "/tmp/place_slots_" + os.path.basename(a.out).rsplit(".json", 1)[0] + ".json"
        c_failed = {}          # gid -> 마지막 시도의 장애물 시그니처
        c_blockers = {}        # gid -> 마지막 실패에서 충돌로 지목된 물체 gid
        retry_futs = {}        # gid -> (future, sig)  -- C단계 재계산은 배경에서(팔은 그동안 다른 물체 작업)
        ready = []             # C까지 통과해 팔만 있으면 집을 수 있는 물체
        unpickable = {}        # gid -> 사유 (B 후보 없음 / pick 실패)
        tol_tried = set()
        # 2026-10-01 순서 기반 C 선계산 (NERO_ORDER_C=1): 초기 C 가 이웃 충돌(blk)로 실패하면 곧바로 "blk 가 치워졌다"고 가정한 C 를 배경에서 계산해 둔다.
        # blk 가 실제로 치워지면(_submit_retries) 이 결과를 그대로 쓴다. 겹침 조건에서 앞 물체를 집고 난 뒤에야 재계산(17~35s)을 시작하던 대기를 없앤다.
        ORDER_C = os.environ.get("NERO_ORDER_C", "0") == "1"
        spec_pre = {}          # gid -> (future, frozenset(가정한 제거 물체))
        c1_info = {}           # gid -> (초기 C 가 지목한 막던 물체 집합(성공이면 빈 집합), 가장 덜 막힌 후보 간격) -- 순환 막힘 tie-break 용

        def _pre_adopt(fut, gid):
            """선계산(_pre 태그, 별도 파일)이 끝나길 기다렸다가 성공이면 정식 계획 파일로 복사한다(정식 파일은 소프트 계획/pick 이 쓰고 있을 수 있어 채택 시점에만 덮는다)."""
            ok, blk = fut.result()
            if ok:
                _run(["ssh", "thor", f"cp -f ~/grasp/{shlex.quote(gid+'_step6_pick_pre.json')} ~/grasp/{shlex.quote(gid+'_step6_pick.json')}"], timeout=30)
            return ok, blk

        def _placed_now():
            try:
                return [(u[2], float(u[0]), float(u[1])) for u in json.load(open(_slot_reg)) if len(u) > 2]
            except Exception:
                return []

        # 2026-09-30 pick 전용 모드: 집은 물체는 치워진 것으로 본다 -> 이후 다른 물체 C단계의 장애물에서 빼고, 그 물체 때문에 C가 막혀 있던
        # 물체(c_failed)는 자동으로 다시 계산한다(_submit_retries). 아래 moved 집합에 합쳐서 기존 "옮겨진 물체" 처리 경로를 그대로 쓴다.
        removed_set = set()

        chain = {}      # nxt gid -> {"frm":앞 물체, "q":시작 자세 문자열, "ok":bool, "done":bool}  (홈 복귀 없이 이어 붙이기용 C 선계산)

        def _spec_chain(best):
            """2026-09-30 홈 복귀 없이 pick 이어 붙이기: best 를 집는 동안, best 의 lift 끝 자세를 시작점으로 다음 물체 nxt 의 C단계(transit)를 미리 계산한다.
            nxt 는 이 thread 가 고른다 -- 다른 물체의 B/C 는 best 를 집는 동안에야 끝나므로 최대 12s 기다리며 (대기열 + 방금 끝난 B/C) 중 같은 기준으로 최선을 고른다.
            best 는 성공해서 치워질 것으로 가정하고 장애물에서 뺀다(실패하면 discover 가 이 결과를 안 쓰고 홈 복귀 경로로 간다).
            성공하면 /tmp/chain_ready_<best> 플래그를 만들어 pick_one.sh 가 홈 복귀를 생략하게 한다(진행 중엔 /tmp/chain_pending_<best>). NERO_CHAIN=0 이면 끔."""
            try:
                _r = _run(["ssh", "thor", "python3 -c \"import json;p=json.load(open('" + GRASP_DIR + "/" + best + "_step6_pick.json'))['plans'][0];"
                           "L=p.get('lift');print(' '.join('%.5f'%x for x in L[-1]) if L else '')\""], capture_output=True, text=True, timeout=30)
                _q = (_r.stdout or "").strip()
                if len(_q.split()) != 7:
                    return
                nxt = None
                for _ in range(24):
                    _cs = list(ready)
                    for _f, _g in list(pending.items()):
                        if _f.done():
                            try:
                                if _f.result()[1]:
                                    _cs.append(_g)
                            except Exception:
                                pass
                    _cs = [g for g in _cs if g != best and g not in removed_set and g not in c_failed and g not in retry_futs]
                    if _cs:
                        nxt = max(_cs, key=lambda g: (_unblock_score(g), 0 if g in c_soft else 1, c_clear.get(g, 1.0)))
                        break
                    if not any(g != best and g not in removed_set and g not in unpickable for g in confirmed):
                        return
                    time.sleep(0.5)
                if nxt is None:
                    return
                rec = {"frm": best, "q": _q, "ok": False, "done": False}
                chain[nxt] = rec
                _oth = [g for g in confirmed if g != nxt and g != best and g not in removed_set]
                try:
                    rec["ok"], _ = _run_c(nxt, _oth, [], 0.0, start_q=_q, tag="_chain")
                except Exception as _e:
                    print(f"[discover] {nxt}: chain 선계산 실패({_e})", file=sys.stderr)
                rec["done"] = True
                print(f"[discover] {nxt}: {best} 의 lift 끝 자세에서 이어 붙일 chain 계획 " + ("준비됨(홈 복귀 생략 가능)" if rec["ok"] else "실패 -- 홈 복귀 경로로"),
                      file=sys.stderr)
                if rec["ok"]:
                    try:
                        open(f"/tmp/chain_ready_{best}", "w").write(nxt)
                    except Exception:
                        pass
            finally:
                try:
                    os.remove(f"/tmp/chain_pending_{best}")
                except OSError:
                    pass

        spec_c = {}     # gid -> {""sig":..., "ok":..., "bk":..., "done": bool}   (앞 물체 place 중 미리 돌린 C 재검증)

        def _sig(placed):
            return tuple(sorted((g, round(x, 2), round(y, 2)) for g, x, y in placed))

        def _spec_watch(best, nxt, t0):
            """2026-09-29 속도: best 를 놓는 동안(place_plan 이 정한 슬롯을 알게 되는 즉시) 다음 물체 nxt 의 C 재검증을 미리 돌린다.
            실제로 놓은 자리(슬롯 레지스트리)가 계획한 슬롯과 같으면 _do_pick 이 그 결과를 재사용한다(다르면 기존대로 다시 계산).
            NERO_SPEC_C=0 이면 끔."""
            plan_p = os.path.expanduser(f"~/grasp/{best}_place_plan.json")
            d = None
            for _ in range(480):                       # 최대 240s 대기
                try:
                    if os.path.getmtime(plan_p) > t0:
                        d = json.load(open(plan_p))
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if not d or not d.get("ok") or not d.get("slot"):
                return
            hyp = _placed_now() + [(best, float(d["slot"][0]), float(d["slot"][1]))]
            moved = {g for g, _, _ in hyp}
            oth = [g for g in confirmed if g != nxt and g not in moved]
            rec = {"sig": _sig(hyp), "done": False, "ok": False}
            spec_c[nxt] = rec
            print(f"[discover] {nxt}: {best} 놓는 동안 C단계 재검증 선계산 시작(슬롯 {np.round(d['slot'], 3).tolist()})", file=sys.stderr)
            try:
                rec["ok"], rec["bk"] = _run_c(nxt, oth, hyp)
            except Exception as _e:
                print(f"[discover] {nxt}: 선계산 실패({_e})", file=sys.stderr)
            rec["done"] = True

        def _do_pick(gid):
            nonlocal box_index
            # 2026-09-29 실행 직전 C단계 재검증: 미리 계산한 C 는 다른 물체가 원위치에 있다고 가정했다. 이미 놓인 물체는
            # 장애물로 안 보이고(놓인 컵을 치고 지나감) 옮겨진 물체의 옛 자리는 유령 장애물로 남는다. 놓인 물체가 있으면
            # 그 위치를 반영해 이 물체의 C 만 다시 계산한다(B 재사용, 보통 2~13초). 실패하면 pick 을 보류한다.
            if gid in c_soft and (c_soft[gid] & removed_set):
                # soft 로 통과시켰는데 그 물체를 막던 blocker 가 이미 치워졌다 -> 정상 여유로 다시 계산(실패하면 soft 로 재계산)
                _oth2 = [g for g in confirmed if g != gid and g not in removed_set]
                print(f"[discover] {gid}: soft 계획의 blocker {sorted(c_soft[gid] & removed_set)} 가 치워짐 -- 정상 여유로 재계산", file=sys.stderr)
                _sp2 = spec_pre.get(gid) if ORDER_C else None
                if _sp2 is not None and _sp2[1] <= removed_set:
                    spec_pre.pop(gid)
                    print(f"[discover] {gid}: 순서 기반 선계산 결과 사용(가정 {sorted(_sp2[1])} 충족) -- 정상 여유 재계산 생략/대기 단축", file=sys.stderr)
                    _ok2, _bk2 = _pre_adopt(_sp2[0], gid)
                else:
                    _ok2, _bk2 = _run_c(gid, _oth2, _placed_now())
                if _ok2:
                    c_soft.pop(gid, None)
                else:
                    _bg2 = c_bestgap.get(gid)
                    _tol2 = _tol_for(_bg2 if _bg2 is not None else 0.0, SWEEP_TOL_MAX)
                    _ok3, _ = _run_c(gid, _oth2, _placed_now(), _tol2)
                    if not _ok3:
                        print(f"[discover] {gid}: 재계산 실패 -- pick 보류", file=sys.stderr)
                        c_failed[gid] = (tuple(sorted(_oth2)), tuple(sorted(removed_set))); c_blockers[gid] = _bk2
                        c_soft.pop(gid, None); precomputed[gid] = False; _write_out()
                        return
            _pl0 = _placed_now()
            _home_th = None
            if _pl0 and os.environ.get("NERO_HOME_OVERLAP", "1") == "1":
                # 2026-09-29 속도: 앞 물체를 놓은 팔은 C 재검증(Thor ssh 2~13s) 동안 놀고 있다 -> 그 사이 홈 복귀를 미리 한다
                # (place_on_target_one.sh 의 홈 복귀는 이미 홈이라 거의 즉시 끝남). NERO_HOME_OVERLAP=0 이면 끔.
                def _bg_home():
                    try:
                        subprocess.run(["python3", "home_and_get_armq.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
                    except Exception:
                        pass
                _home_th = threading.Thread(target=_bg_home, daemon=True)
                _home_th.start()
            if _pl0:
                _mv = {g for g, _, _ in _pl0}
                _oth = [g for g in confirmed if g != gid and g not in _mv]
                print(f"[discover] {gid}: 옮겨 놓은 물체 {sorted(_mv)} 반영해 C단계 재검증(실행 직전)", file=sys.stderr)
                _spec = spec_c.pop(gid, None)
                _reuse = False
                if _spec is not None:
                    for _ in range(600):               # 선계산이 아직 돌고 있으면 끝날 때까지(최대 60s)
                        if _spec.get("done"):
                            break
                        time.sleep(0.1)
                    _reuse = bool(_spec.get("done") and _spec.get("ok") and _spec["sig"] == _sig(_pl0))
                    print(f"[discover] {gid}: C 재검증 선계산 결과 " + ("재사용(시간 절약)" if _reuse else "불일치/실패 -- 다시 계산"), file=sys.stderr)
                _ok, _bk = (True, set()) if _reuse else _run_c(gid, _oth, _pl0)
                if not _ok and _bk and WIDE_HS > 0:
                    print(f"[discover] {gid}: 재검증 충돌(지목 {sorted(_bk)}) -- B를 넓혀 재탐색", file=sys.stderr)
                    if _widen_b(gid):
                        _ok, _bk = _run_c(gid, _oth, _pl0)
                if not _ok:
                    print(f"[discover] {gid}: 재검증 C단계 FAIL(충돌 지목 {sorted(_bk) or '없음'}) -- 놓인 물체를 칠 수 있어 pick 보류",
                          file=sys.stderr)
                    c_failed[gid] = (tuple(sorted(_oth)), tuple(sorted(_mv))); c_blockers[gid] = _bk
                    precomputed[gid] = False; _write_out()
                    if _home_th is not None:
                        _home_th.join(timeout=90)
                    return
            if _home_th is not None:
                _home_th.join(timeout=90)      # 홈 복귀가 끝난 뒤 실행 (C 재검증이 더 오래 걸렸으면 이미 끝나 있음)
            _status_stage("exec"); _status_set(gid, pick="run", place=None)
            _pf = glob.glob(f"/tmp/{gid}_place_try*.json") + [f"/tmp/{gid}_place_side.json"]
            for _f in _pf + [f"/tmp/{gid}.exec_res.json"]:
                try:
                    os.remove(_f)
                except OSError:
                    pass
            _env = dict(os.environ)
            if a.no_place and os.environ.get("NERO_CHAIN", "1") == "1":
                _env["NERO_CHAIN_ENABLE"] = "1"
                try:
                    os.remove(f"/tmp/chain_ready_{gid}")
                except OSError:
                    pass
                _rec = chain.pop(gid, None)
                if _rec is not None:                      # 선계산이 앞 물체 pick_one 이 끝날 때(홈 복귀 생략 판단, 최대 8s 대기)까지 안 끝났으면 chain 을 안 쓴다
                    try:
                        _up = json.load(open(f"/tmp/{_rec['frm']}.exec_res.json")).get("plan_used", 0) or 0
                    except Exception:
                        _up = 9
                    if _rec.get("done") and _rec.get("ok") and _rec["frm"] in removed_set and _up == 0:
                        _env["NERO_CHAIN_PLAN_TAG"] = "_chain"; _env["NERO_CHAIN_START"] = _rec["q"]
                        print(f"[discover] {gid}: {_rec['frm']} 다음으로 홈 복귀 없이 chain pick", file=sys.stderr)
            subprocess.run(["./pick_one.sh" if a.no_place else "./place_on_target_one.sh", os.path.expanduser(a.out), gid, str(box_index)], env=_env)
            box_index += 1
            _v = None
            try:
                _v = json.load(open(f"/tmp/{gid}.exec_res.json")).get("verdict")
            except Exception:
                pass
            _pl = None
            if _v in ("HELD", "EXEC_OK") and a.no_place:
                _pl = None                     # pick 전용: place 안 함
                removed_set.add(gid)
                _blocked = [g for g, bl in c_blockers.items() if g in c_failed and gid in bl]
                print(f"[discover] {gid}: pick 성공 -> 치워진 것으로 처리(이후 장애물에서 제외)"
                      + (f", 막혀 있던 {sorted(_blocked)} 재계산 대상" if _blocked else ""), file=sys.stderr)
            elif _v in ("HELD", "EXEC_OK"):
                _pl = "FAIL"
                for _f in glob.glob(f"/tmp/{gid}_place_try*.json") + [f"/tmp/{gid}_place_side.json"]:
                    try:
                        if json.load(open(_f)).get("status") == "success":
                            _pl = "OK"; break
                    except Exception:
                        pass
            else:
                unpickable[gid] = f"pick {_v or 'FAIL'}"
            _status_set(gid, pick=(_v or "FAIL"), place=_pl)

        def _unblock_score(g):
            return sum(1 for x, bl in c_blockers.items() if x in c_failed and g in bl)

        def _pick_next():
            def _chain_ok(g):
                r_ = chain.get(g)
                return 1 if (r_ and r_.get("done") and r_.get("ok") and r_["frm"] in removed_set) else 0
            best = max(ready, key=lambda g: (_unblock_score(g), _chain_ok(g), 0 if g in c_soft else 1, c_clear.get(g, 1.0), -ready.index(g)))
            sc = _unblock_score(best)
            others_wait = [g for g in ready if g != best]
            if sc or others_wait:
                print(f"[discover] 순서 계획: {best} 먼저 -- 막힌 물체 {sc}개를 풀어줌"
                      f"{' (대기: ' + ', '.join(others_wait) + ')' if others_wait else ''}", file=sys.stderr)
            ready.remove(best)
            print(f"[discover] {best}: pick+place 실행", file=sys.stderr)
            if os.environ.get("NERO_SPEC_C", "1") == "1" and ready and not a.no_place:
                _nx = max(ready, key=lambda g: (_unblock_score(g), 0 if g in c_soft else 1, c_clear.get(g, 1.0), -ready.index(g)))
                if _nx not in c_failed and _nx not in retry_futs:
                    threading.Thread(target=_spec_watch, args=(best, _nx, time.time()), daemon=True).start()
            if a.no_place and os.environ.get("NERO_CHAIN", "1") == "1":
                for _f in (f"/tmp/chain_ready_{best}", f"/tmp/chain_pending_{best}"):
                    try:
                        os.remove(_f)
                    except OSError:
                        pass
                if any(g != best and g not in removed_set and g not in unpickable for g in confirmed):
                    try:
                        open(f"/tmp/chain_pending_{best}", "w").write("1")
                    except Exception:
                        pass
                    threading.Thread(target=_spec_chain, args=(best,), daemon=True).start()
            if os.environ.get("NERO_SPEC_RETRY", "1") == "1":
                _spec_retries(best)
                threading.Thread(target=_spec_pending, args=(best,), daemon=True).start()
            _do_pick(best)
            _submit_retries()

        spec_res = {}       # gid -> (future, 가정한 제거 물체 best): pick 중 미리 돌린 "초기 B/C 실패 물체"의 재계산(pick 성공을 가정)

        def _spec_pending(best):
            """pick 중 메인 스레드는 막혀 있어 다른 물체의 초기 B/C 결과(pending)를 못 읽는다 -> 이 감시 스레드가 대신 최대 25초 동안 읽어서,
            C 단계에서 best 등에 막혀 실패한 물체는 best 를 집는 동안(성공 가정) 곧바로 C 를 재계산한다. 결과는 _absorb 가 pick 뒤에 채택/폐기한다.
            NERO_SPEC_RETRY=0 이면 끔."""
            seen = set()
            t_end = time.time() + 25.0
            while time.time() < t_end:
                for _f, _g in list(pending.items()):
                    if _g in seen or not _f.done():
                        continue
                    seen.add(_g)
                    try:
                        _gid, _ok, _stage, _blk = _f.result()
                    except Exception:
                        continue
                    if _ok or _stage != "C" or not _blk:
                        continue
                    _moved = {g for g, _, _ in _placed_now()} | removed_set | {best}
                    if not (set(_blk) & _moved):
                        continue
                    _oth = [g for g in confirmed if g != _g and g not in _moved]
                    print(f"[discover] {_g}: {best} 를 집는 동안(성공 가정) C단계 선재계산(초기 결과의 막힘 지목 {sorted(_blk)})", file=sys.stderr)
                    spec_res[_g] = (ex.submit(_run_c, _g, _oth, _placed_now()), best)
                if len(seen) >= len([1 for _f, _g in list(pending.items())]):
                    break
                time.sleep(0.5)

        spec_assume = {}    # gid -> 이 재계산이 "제거됐을 것"으로 가정한 물체 (그 물체 pick 이 실패하면 결과 폐기)

        def _spec_retries(best):
            """2026-09-30 속도(겹침 병목): 이웃이 치워진 뒤에야 시작하던 C 재계산(30~50s)을 best 를 집는 동안 미리 돌린다.
            best 가 성공해서 치워진다고 가정하고 best 때문에 막혀 있던 물체(c_failed)의 C를 계산해 두고, best 가 실패하면 결과를 버린다.
            NERO_SPEC_RETRY=0 이면 끔."""
            placed = _placed_now(); moved = {g for g, _, _ in placed} | removed_set | {best}
            for gid in list(c_failed):
                if gid in retry_futs:
                    continue
                if not (c_blockers.get(gid, set()) & moved):
                    continue
                others = [g for g in confirmed if g != gid and g not in moved]
                sig = (tuple(sorted(others)), tuple(sorted(moved)))
                if sig == c_failed[gid]:
                    continue
                print(f"[discover] {gid}: {best} 를 집는 동안(성공 가정) C단계 선재계산", file=sys.stderr)
                retry_futs[gid] = (ex.submit(_run_c, gid, others, placed), sig)
                spec_assume[gid] = best

        def _submit_retries():
            placed = _placed_now(); moved = {g for g, _, _ in placed} | removed_set
            for gid in list(c_failed):
                if gid in retry_futs:
                    continue
                others = [g for g in confirmed if g != gid and g not in moved]
                sig = (tuple(sorted(others)), tuple(sorted(moved)))
                if sig == c_failed[gid]:
                    continue
                blk = c_blockers.get(gid, set())
                if blk and not (blk & moved):
                    continue     # 막던 물체가 아직 제자리(옮겨지지 않는 검출 포함) -> 재계산해도 같은 결과
                _sp = spec_pre.get(gid)
                if _sp is not None and _sp[1] <= moved:
                    spec_pre.pop(gid)
                    print(f"[discover] {gid}: 장애물 변경(옮긴 물체 {sorted(moved)}) -- 순서 기반 선계산 결과 사용(가정 {sorted(_sp[1])} 충족)", file=sys.stderr)
                    retry_futs[gid] = (ex.submit(_pre_adopt, _sp[0], gid), sig)
                    continue
                print(f"[discover] {gid}: 장애물 변경(옮긴 물체 {sorted(moved)}) -- C단계 재계산(배경)", file=sys.stderr)
                retry_futs[gid] = (ex.submit(_run_c, gid, others, placed), sig)

        def _collect_retries(block=False):
            for gid, (f, sig) in list(retry_futs.items()):
                if not (block or f.done()):
                    continue
                ok, blk = f.result()
                del retry_futs[gid]
                _asm = spec_assume.pop(gid, None)
                if _asm is not None and _asm not in removed_set:
                    print(f"[discover] {gid}: 선재계산이 가정한 {_asm} 이 치워지지 않음(pick 실패) -- 결과 폐기", file=sys.stderr)
                    continue
                if not ok:
                    c_failed[gid] = sig; c_blockers[gid] = blk
                    print(f"[discover] {gid}: 재계산 C단계 FAIL (충돌 지목: {sorted(blk) or '없음'})", file=sys.stderr)
                    continue
                del c_failed[gid]
                precomputed[gid] = True; _write_out()
                print(f"[discover] {gid}: 재계산 C단계 OK -- pick 대기열 추가", file=sys.stderr)
                ready.append(gid)

        def _resolve_deadlock():
            """정지 상태(준비/계산 중인 것 없음)인데 C 실패 물체가 남았을 때: 가장 덜 막힌 물체 하나만 골라
            허용오차를 완화해 재계산. 후보가 없으면 False."""
            placed = _placed_now(); moved = {g for g, _, _ in placed} | removed_set
            def _cands(cap):
                return [(c_bestgap[g], g) for g in c_failed
                        if g not in tol_tried and g not in retry_futs
                        and c_bestgap.get(g) is not None and c_bestgap[g] >= -cap]
            cands = _cands(SWEEP_TOL_MAX)
            cap_used = SWEEP_TOL_MAX
            # 2026-09-30 교착 2단계 완화: 모든 물체가 막혀 아무것도 못 하면(예: 겹친 3개, c3v9 seed 100/105/109 에서 재스윕까지 가며 로봇이 100~150s 정지)
            # 재스윕 대신 침투 상한을 SWEEP_TOL_DEADLOCK(기본 3.5cm)까지 올려 가장 덜 막힌 물체를 바로 시도한다. 캡슐 반경(팔 4.5cm/그리퍼 6cm)이 보수적이라
            # 침투 판정이 실제 접촉이 아닐 수 있으나, 실제로 이웃을 쳤는지는 SNAP 이웃 변위 비교로 검증한다. NERO_SWEEP_TOL_DEADLOCK=0 이면 끔.
            if not cands and SWEEP_TOL_DEADLOCK > SWEEP_TOL_MAX:
                cands = _cands(SWEEP_TOL_DEADLOCK); cap_used = SWEEP_TOL_DEADLOCK
            if not cands:
                return False
            gap, gid = max(cands)
            tol_tried.add(gid)
            tol = _tol_for(gap, cap_used)
            blk = sorted(c_blockers.get(gid, set()))
            why = [f"{b}({unpickable[b]})" if b in unpickable else b for b in blk]
            others = [g for g in confirmed if g != gid and g not in moved]
            print(f"[discover] 막힘 해소 불가(막던 물체 {why or '없음'}: 못 집히거나 순환) -- {gid}: 가장 덜 막힌 후보"
                  f"(간격 {gap*100:+.1f}cm)로 스윕 허용오차 {tol*100:.1f}cm 완화 재계산", file=sys.stderr)
            retry_futs[gid] = (ex.submit(_run_c, gid, others, placed, tol), c_failed[gid])
            return True

        with ThreadPoolExecutor(max_workers=max(3, len(confirmed) * (2 if ORDER_C else 1) + 2)) as ex:
            pending = {ex.submit(_run_bc, gid): gid for gid in confirmed}

            def _absorb(fut):
                gid, ok, stage, blk = fut.result()
                precomputed[gid] = ok
                _write_out()
                if ok:
                    print(f"[discover] {gid}: B/C단계 OK -- pick 대기열 추가", file=sys.stderr)
                    ready.append(gid)
                    return
                print(f"[discover] {gid}: {stage}단계 FAIL -- pick 보류", file=sys.stderr)
                if stage == "B":
                    unpickable[gid] = "B 도달 후보 없음"
                elif stage == "C":
                    c_failed[gid] = (tuple(sorted(g for g in confirmed if g != gid)), ())
                    c_blockers[gid] = blk
                    _sp = spec_res.pop(gid, None)
                    if _sp is not None:
                        _spf, _spb = _sp
                        if _spb in removed_set:
                            _ok_s, _blk_s = _spf.result()
                            if _ok_s:
                                del c_failed[gid]
                                precomputed[gid] = True; _write_out()
                                print(f"[discover] {gid}: 선재계산한 C단계 OK({_spb} 치워짐) -- pick 대기열 추가", file=sys.stderr)
                                ready.append(gid)
                                return
                            print(f"[discover] {gid}: 선재계산한 C단계도 FAIL (지목 {sorted(_blk_s) or '없음'})", file=sys.stderr)
                            c_blockers[gid] = _blk_s
                            c_failed[gid] = (tuple(sorted(g for g in confirmed if g != gid and g not in removed_set)), tuple(sorted(removed_set)))
                        else:
                            print(f"[discover] {gid}: 선재계산이 가정한 {_spb} 이 치워지지 않음 -- 결과 폐기", file=sys.stderr)
                    _submit_retries()

            while pending or retry_futs or ready:
                for f in [f for f in pending if f.done()]:
                    _absorb(f); del pending[f]
                _collect_retries()
                if ready:
                    _pick_next()
                    _collect_retries()
                    continue
                waitset = list(pending) + [f for f, _ in retry_futs.values()]
                if waitset:
                    _cf_wait(waitset, timeout=1.0, return_when=FIRST_COMPLETED)
                    continue
                if not _resolve_deadlock():
                    break
        _status_stage("done")
        if c_failed:
            print(f"[discover] 재계산 후에도 실패: {sorted(c_failed)} (충돌 지목: "
                  f"{ {g: sorted(b) for g, b in c_blockers.items()} }, 못 집히는 물체: {unpickable})", file=sys.stderr)

        ok_gids = [g for g, ok in precomputed.items() if ok]
        if ok_gids:
            names = " ".join(shlex.quote(g) for g in ok_gids)
            _run(["ssh", "thor",
                           f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                           f"cd ~/grasp && if [ -f multi.npz ]; then python3 viz_export_multi.py "
                           f"{GRASP_DIR}/multi.npz {names}; fi"],
                          capture_output=True, text=True, timeout=90)

    elif a.precompute_bc and arm_q and a.serial_bc:
        arm_q_str = " ".join("%.5f" % x for x in arm_q)
        precomputed = {}
        for gid in confirmed:
            print(f"[discover] {gid}: B/C단계 미리 계산", file=sys.stderr)
            _run(["ssh", "thor", f"rm -f ~/grasp/{shlex.quote(gid+'_pick.json')} ~/grasp/{shlex.quote(gid+'_step6_pick.json')} ~/grasp/step6_pick.json"], timeout=30)
            b = _run(["ssh", "thor",
                               f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                               f"cd ~/grasp && export NERO_ARM_Q='{arm_q_str}' {taskw_env}{_table_env()}&& python3 two_pick_plan_fast.py {shlex.quote(gid)}"],
                              capture_output=True, text=True, timeout=180)
            ok_b = "reachable" in b.stdout and "no reachable" not in b.stdout and "no candidate" not in b.stdout
            _log_nbr(gid, b)
            print(f"  B단계: {'OK' if ok_b else 'FAIL'} ({b.stdout.strip().splitlines()[-1] if b.stdout.strip() else b.stderr.strip()[:200]})", file=sys.stderr)
            if not ok_b:
                precomputed[gid] = False; continue
            c = _run(["ssh", "thor",
                               f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                               f"cd ~/grasp && export NERO_ARM_Q='{arm_q_str}' NERO_VEL_SCALE='{os.environ.get('NERO_VEL_SCALE','1.0')}' NERO_OTHER_OBJS={shlex.quote(' '.join(confirmed))} && "
                               f"bash step6_single.sh {shlex.quote(gid)}"],
                              capture_output=True, text=True, timeout=300)
            r2 = _run(["ssh", "thor", f"test -f ~/grasp/{shlex.quote(gid+'_step6_pick.json')}"], timeout=30)
            ok_c = r2.returncode == 0
            print(f"  C단계: {'OK' if ok_c else 'FAIL'}", file=sys.stderr)
            if "lift-z Cartesian 실패" in c.stdout:
                print(f"  [경고] {gid}: lift-z Cartesian 실패 -- exec 시 폴백 관절 lift 사용됨", file=sys.stderr)
            precomputed[gid] = ok_c
        ok_gids = [g for g, ok in precomputed.items() if ok]
        if ok_gids:
            names = " ".join(shlex.quote(g) for g in ok_gids)
            _run(["ssh", "thor",
                           f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                           f"cd ~/grasp && if [ -f multi.npz ]; then python3 viz_export_multi.py "
                           f"{GRASP_DIR}/multi.npz {names}; fi"],
                          capture_output=True, text=True, timeout=90)

    elif a.precompute_bc and arm_q:
        arm_q_str = " ".join("%.5f" % x for x in arm_q)
        precomputed = {}

        # 2026-09-25 최적화(사용자 제안: "구동 대기 없이 계산도 동시에"): B단계
        # (two_pick_plan_fast.py)는 gid별로 {gid}_pick.json에만 쓰고 REACH_LOG만
        # append 공유라 병렬 SSH가 안전함(확인됨) -- ThreadPoolExecutor로 gid 전체를
        # 동시에 쏴서 "물체 수 x B단계 시간"을 "B단계 시간 1번"으로 줄인다. 이 단계는
        # 로봇 팔이 전혀 필요 없는 순수 원격계산(Thor)이라 팔 동작과 무관하게 항상
        # 안전하게 겹칠 수 있다.
        # C단계도 병렬화(2026-09-25, 후속 조사): step6_multi.sh는 side_grasp.json/
        # step6_pick.json이라는 gid와 무관한 전역 공유파일을 거쳐서 순차 전용으로만
        # 쓰였지만, 그 내부에서 부르는 step6_base.py 자체는 이미 NERO_SIDE_GRASP_JSON/
        # NERO_STEP6_OUT_JSON 환경변수로 입출력 경로를 오버라이드하게 설계돼 있고
        # (확인됨, 다른 공유파일 의존 없음 -- {oname}_grasps.json/{oname}_sem.json은
        # 원래 gid별 격리) 이 두 env var만 gid별로 지정하면 병렬 SSH도 안전하다.
        # step6_multi.sh를 우회하는 병렬-안전 버전 `step6_single.sh`를 Thor에 신설.
        for gid in confirmed:
            _run(["ssh", "thor", f"rm -f ~/grasp/{shlex.quote(gid+'_pick.json')} ~/grasp/{shlex.quote(gid+'_step6_pick.json')} ~/grasp/step6_pick.json"], timeout=30)

        def _run_b(gid):
            b = _run(["ssh", "thor",
                               f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                               f"cd ~/grasp && export NERO_ARM_Q='{arm_q_str}' {taskw_env}{_table_env()}&& python3 two_pick_plan_fast.py {shlex.quote(gid)}"],
                              capture_output=True, text=True, timeout=180)
            ok_b = "reachable" in b.stdout and "no reachable" not in b.stdout and "no candidate" not in b.stdout
            return gid, ok_b, b

        print(f"[discover] B단계 {len(confirmed)}개 물체 동시 실행", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, len(confirmed))) as ex:
            b_results = list(ex.map(_run_b, list(confirmed.keys())))

        c_targets = []
        for gid, ok_b, b in b_results:
            _log_nbr(gid, b)
            print(f"[discover] {gid}: B단계 {'OK' if ok_b else 'FAIL'} "
                  f"({b.stdout.strip().splitlines()[-1] if b.stdout.strip() else b.stderr.strip()[:200]})", file=sys.stderr)
            if not ok_b:
                precomputed[gid] = False; continue
            c_targets.append(gid)

        def _run_c(gid):
            c = _run(["ssh", "thor",
                               f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                               f"cd ~/grasp && export NERO_ARM_Q='{arm_q_str}' NERO_VEL_SCALE='{os.environ.get('NERO_VEL_SCALE','1.0')}' NERO_OTHER_OBJS={shlex.quote(' '.join(confirmed))} && "
                               f"bash step6_single.sh {shlex.quote(gid)}"],
                              capture_output=True, text=True, timeout=300)
            return gid, c

        print(f"[discover] C단계 {len(c_targets)}개 물체 동시 실행", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=max(1, len(c_targets) or 1)) as ex:
            c_results = list(ex.map(_run_c, c_targets)) if c_targets else []

        for gid, c in c_results:
            r2 = _run(["ssh", "thor", f"test -f ~/grasp/{shlex.quote(gid+'_step6_pick.json')}"], timeout=30)
            ok_c = r2.returncode == 0
            print(f"[discover] {gid}: C단계 {'OK' if ok_c else 'FAIL'}", file=sys.stderr)
            # 2026-09-18: step6_multi.sh(=step6_base.py) stdout을 그동안 캡처만 하고
            # 버려서 "lift-z Cartesian 실패 -- exec 폴백 관절 lift 사용" 진단이 한 번도
            # 안 보였다(관절 lift 폴백 원인 조사 중 발견). 나오면 반드시 표시.
            if "lift-z Cartesian 실패" in c.stdout:
                print(f"  [경고] {gid}: lift-z Cartesian 실패 -- exec 시 폴백 관절 lift 사용됨", file=sys.stderr)
            precomputed[gid] = ok_c

        # step6_multi.sh가 순차 루프 끝에 한 번만 부르던 오버레이 갱신
        # (viz_export_multi.py)을 병렬화 후에도 동일하게 1회만 호출.
        if c_targets:
            names = " ".join(shlex.quote(g) for g in c_targets)
            _run(["ssh", "thor",
                           f"source /opt/ros/jazzy/setup.bash && source $HOME/ros2_ws/install/setup.bash && "
                           f"cd ~/grasp && if [ -f multi.npz ]; then python3 viz_export_multi.py "
                           f"{GRASP_DIR}/multi.npz {names}; fi"],
                          capture_output=True, text=True, timeout=90)
    else:
        precomputed = {gid: False for gid in confirmed}
        if not arm_q:
            print("[discover] arm_q 못 잡음 -- B/C단계 선계산 스킵", file=sys.stderr)
        else:
            print("[discover] B/C단계는 pick 직전(fresh seed)에 계산 -- 여기선 CGN candidate만 저장", file=sys.stderr)

    out = {"boxes": confirmed, "arm_q": arm_q, "precomputed": precomputed, "place_target": place_target}
    json.dump(out, open(os.path.expanduser(a.out), "w"), indent=1)
    print(json.dumps({"found": list(confirmed.keys()), "n": len(confirmed), "arm_q": arm_q,
                       "precomputed": precomputed, "place_target": place_target}, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
