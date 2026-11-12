#!/usr/bin/env python3
"""fast_scan.py -- 경로 커버 탐색을 정지-촬영 대신 한 번의 연속 스윕으로(2026-10-06, 연속 스윕 1단계). NERO_COVER_FAST=1 로 켠다(기본 꺼짐: 기존 정지-촬영 보존).
 - 관절 1 을 사다리꼴 속도 곡선(기본 1.5 rad/s, 가속 6 rad/s^2)으로 한 번에 훑고(다른 관절은 관측 자세 고정), 도착하는 프레임을 일정 각도 간격(0.30 rad)으로 기록한다.
 - 기록한 프레임은 이동 중에 **작업 스레드가 즉시** 처리한다(분할+CLIP, TF 는 도착 실제 시각 - 45 ms). 카메라 헤더는 시뮬 시간, TF 는 실제 시계라 헤더로 조회할 수 없다(fast_sweep2.py 실험: 오차 최소 Δ≈30~60 ms).
 - 카메라/관절/TF 는 별도 노드·스레드(메인 스레드 spin_once 는 콜백을 하나씩만 처리해 값이 밀린다).
사용: fs = fast_scan.get(n); fs.sweep(angles, j7_abs, grab_fn, on_dets) -> 처리한 프레임 수"""
import os, sys, time, math, threading, collections, types
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image, CameraInfo, JointState
from tf2_ros import Buffer, TransformListener
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
SPEED = float(os.environ.get("NERO_COVER_SPEED", "1.5")); AMAX = float(os.environ.get("NERO_COVER_AMAX", "6.0")); DELTA = float(os.environ.get("NERO_COVER_TF_DELTA", "0.045")); SPACING = float(os.environ.get("NERO_COVER_SPACING", "0.30"))
PAIR = os.environ.get("NERO_SWEEP_PAIR", "0") == "1"      # 2026-10-12: 컬러/깊이를 헤더 시각으로 짝지음(기본 꺼짐)
_INST = {}
def get(n):
    if "fs" not in _INST: _INST["fs"] = FastScanner(n)
    return _INST["fs"]
class FastScanner:
    def __init__(self, n):
        self.n = n; self.node = Node("fast_scan_sn"); self.got = {}; self.j1 = collections.deque(maxlen=4000); self.rec = None; self.last_rec_j1 = None; self.lock = threading.Lock(); self.dq = collections.deque(maxlen=12); self.pair_dt = None
        nd = self.node
        nd.create_subscription(CameraInfo, "/camera/camera_info", lambda m: self.got.__setitem__("K", np.array(m.k).reshape(3, 3)), qos_profile_sensor_data)
        nd.create_subscription(Image, "/camera/depth/image_raw", self._d, qos_profile_sensor_data); nd.create_subscription(Image, "/camera/color/image_raw", self._c, qos_profile_sensor_data)
        nd.create_subscription(JointState, "/joint_states", self._j, qos_profile_sensor_data)
        self.tbuf = Buffer(cache_time=Duration(seconds=60)); self.tfn = Node("fast_scan_tf"); TransformListener(self.tbuf, self.tfn, spin_thread=True)
        self.ex = SingleThreadedExecutor(); self.ex.add_node(nd); threading.Thread(target=self.ex.spin, daemon=True).start()
        self.ac = ActionClient(nd, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        if not self.ac.wait_for_server(timeout_sec=10): raise RuntimeError("arm_controller 액션 서버 없음")
        t0 = time.time()
        while time.time() - t0 < 5 and not (self.j1 and "K" in self.got and "rgb" in self.got and "depth" in self.got): time.sleep(0.05)
    def _j(self, m):
        d = dict(zip(m.name, m.position))
        if "joint1" in d: self.j1.append((time.time(), float(d["joint1"])))
    def _d(self, m):
        x = np.frombuffer(m.data, dtype=np.float32 if m.encoding == "32FC1" else np.uint16).reshape(m.height, m.width).astype(np.float32)
        self.got["depth"] = x / 1000.0 if m.encoding == "16UC1" else x
        if PAIR: self.dq.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, self.got["depth"]))      # 2026-10-12: 헤더 시각이 가장 가까운 깊이를 컬러와 짝짓기 위해 보관(기본 꺼짐)
    def _c(self, m):
        arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]; rgb = arr[..., ::-1].copy() if m.encoding.startswith("bgr") else arr.copy(); self.got["rgb"] = rgb
        if self.rec is not None and "depth" in self.got and self.j1:
            now = time.time(); j = self.j1[-1][1]
            if self.last_rec_j1 is None or abs(j - self.last_rec_j1) >= SPACING:
                dep = self.got["depth"]
                if PAIR and self.dq:      # 컬러 헤더 시각에 가장 가까운 깊이(스윕 중 두 영상의 시각 차로 마스크가 크롭에서 어긋나는 것을 줄임)
                    ts = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec; k = min(self.dq, key=lambda e: abs(e[0] - ts)); dep = k[1]; self.pair_dt = (k[0] - ts) / 1e9
                self.last_rec_j1 = j; self.rec.append(dict(wall=now, rgb=rgb, depth=dep.copy(), j1=self._j1_at(now - DELTA)))
    def _j1_at(self, t):
        h = list(self.j1)[-400:]
        if not h: return 0.0
        ts = [x[0] for x in h]; return float(np.interp(t, ts, [x[1] for x in h]))
    @staticmethod
    def _trap(q0, tgt, vmax, amax, dt=0.05):
        dist = tgt - q0; s = 1.0 if dist >= 0 else -1.0; Dd = abs(dist); ta = vmax / amax; da = 0.5 * amax * ta * ta
        if 2 * da > Dd: vp = math.sqrt(amax * Dd); ta = vp / amax; tc = 0.0
        else: vp = vmax; tc = (Dd - 2 * da) / vmax
        T = 2 * ta + tc; pts = []
        for t in list(np.arange(0.0, T, dt)) + [T]:
            if t < ta: x, v = 0.5 * amax * t * t, amax * t
            elif t < ta + tc: x, v = 0.5 * amax * ta * ta + vp * (t - ta), vp
            else: tt = T - t; x, v = Dd - 0.5 * amax * tt * tt, amax * tt
            pts.append((t, q0 + s * x, s * v))
        return pts, T
    def _send(self, pts, Q):
        jt = JointTrajectory(); jt.joint_names = ARM
        for t, x, v in pts:
            p = JointTrajectoryPoint(); p.positions = [x] + list(Q[1:]); p.velocities = [v] + [0.0] * 6; p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9); jt.points.append(p)
        g = FollowJointTrajectory.Goal(); g.trajectory = jt; f = self.ac.send_goal_async(g); t0 = time.time()
        while not f.done() and time.time() - t0 < 3: time.sleep(0.005)
        return f.result()
    def _settle(self, timeout=4.0):
        t0 = time.time(); prev = self.j1[-1][1]; st = 0
        while time.time() - t0 < timeout:
            time.sleep(0.08); c = self.j1[-1][1]; st = st + 1 if abs(c - prev) < 0.0015 else 0; prev = c
            if st >= 4: return True
        return False
    def sweep(self, angles, j7_abs, grab_fn, on_dets, log=print):
        """angles: 확인할 j1 각도들. 반환: 처리한 프레임 수."""
        pose = dict(self.n._observation_pose()); Q = [float(pose[f"joint{i}"]) for i in range(1, 8)]
        if j7_abs is not None: Q[6] = float(j7_abs)
        q0 = self.j1[-1][1]; lo, hi = min(angles), max(angles)
        legs = [hi] if lo >= q0 - 0.05 else ([lo] if hi <= q0 + 0.05 else sorted([lo, hi], key=lambda x: abs(x - q0)))      # 한쪽이면 가장 먼 각도까지, 양쪽이면 가까운 쪽 먼저
        view = types.SimpleNamespace(got=self.got); view.fresh_frame = lambda *a, **k: True
        import discover_boxes as DB
        view.capture_geom = lambda: DB.Discoverer.capture_geom(view)
        todo = collections.deque(); done = {"n": 0, "run": True}
        def worker():
            while done["run"] or todo:
                if not todo: time.sleep(0.01); continue
                fr = todo.popleft()
                try: tf = self.tbuf.lookup_transform("base_link", "camera_color_optical_frame", Time(seconds=fr["wall"] - DELTA))
                except Exception: continue
                self.got["rgb"], self.got["depth"], self.got["tf"] = fr["rgb"], fr["depth"], tf
                try: r = grab_fn(view)
                except Exception as e: log(f"[fast_scan] 프레임 처리 실패 {e!r}"); continue
                if r is not None: on_dets(r[1], fr["j1"]); done["n"] += 1
        th = threading.Thread(target=worker, daemon=True); th.start()
        cur = q0; t_all = time.time()
        for tgt in legs:
            pts, T = self._trap(cur, tgt, SPEED, AMAX); self.last_rec_j1 = None; self.rec = []
            self._send(pts, Q); t0 = time.time(); drained = 0
            while time.time() - t0 < T + 0.45:
                while drained < len(self.rec): todo.append(self.rec[drained]); drained += 1     # 이동 중 도착한 프레임은 바로 처리 큐로
                time.sleep(0.01)
            while drained < len(self.rec): todo.append(self.rec[drained]); drained += 1
            self.rec = None; cur = tgt
        if os.environ.get("NERO_COVER_RETURN", "1") == "1" and abs(cur - q0) > 0.1:     # 끝난 자리가 대상에서 멀면 재정렬 이동이 늘어난다 -> 시작 각도로 돌아오며 마저 기록/처리(정지 없이 이어서)
            pts, T = self._trap(cur, q0, SPEED, AMAX); self.last_rec_j1 = None; self.rec = []; self._send(pts, Q); t0 = time.time(); drained = 0
            while time.time() - t0 < T + 0.45:
                while drained < len(self.rec): todo.append(self.rec[drained]); drained += 1
                time.sleep(0.01)
            while drained < len(self.rec): todo.append(self.rec[drained]); drained += 1
            self.rec = None
        self._settle(); done["run"] = False; th.join(timeout=15)
        log(f"[fast_scan] 연속 스윕 {legs} 속도 {SPEED} rad/s, 프레임 {done['n']}개 처리, 총 {time.time() - t_all:.1f}s")
        return done["n"]

    def search(self, j7_abs, grab_fn, on_dets, found_fn, first_dir=-1.0, lim=2.6, log=print):
        """초기 탐색(2단계): 현재 각도에서 한쪽 끝(first_dir)으로, 못 찾으면 반대쪽 끝까지 연속 스윕하며 프레임이 오면 최신 것만 처리하고,
        found_fn() 이 참이 되면 즉시 감속 정지한다(명령된 궤적의 현재 상태에서 정지 궤적을 새 목표로 보낸다: 시작 지연 0.3 s·추종 지연 0.24 s 때문에 실제 위치는 명령보다 뒤).
        반환: (찾았나, 정지 j1, 처리 프레임 수, 소요 s)"""
        pose = dict(self.n._observation_pose()); Q = [float(pose[f"joint{i}"]) for i in range(1, 8)]
        if j7_abs is not None: Q[6] = float(j7_abs)
        import discover_boxes as DB
        view = types.SimpleNamespace(got=self.got); view.fresh_frame = lambda *a, **k: True; view.capture_geom = lambda: DB.Discoverer.capture_geom(view)
        todo = collections.deque(); st = {"n": 0, "run": True, "found": False}
        def worker():
            while st["run"]:
                if not todo: time.sleep(0.005); continue
                fr = todo.pop()                     # 최신 프레임만(밀리면 오래된 것은 버려 정지 반응을 늦추지 않는다)
                todo.clear()
                try: tf = self.tbuf.lookup_transform("base_link", "camera_color_optical_frame", Time(seconds=fr["wall"] - DELTA))
                except Exception: continue
                self.got["rgb"], self.got["depth"], self.got["tf"] = fr["rgb"], fr["depth"], tf
                try: r = grab_fn(view)
                except Exception as e: log(f"[fast_scan] 프레임 처리 실패 {e!r}"); continue
                if r is not None:
                    on_dets(r[1], fr["j1"]); st["n"] += 1
                    if found_fn(): st["found"] = True
        th = threading.Thread(target=worker, daemon=True); th.start(); t_all = time.time()
        cur = self.j1[-1][1]; far = [first_dir * lim, -first_dir * lim]; j_stop = cur
        for leg_i, tgt in enumerate(far):
            if abs(tgt - cur) < 0.05: continue
            pts, T = self._trap(cur, tgt, SPEED, AMAX); self.last_rec_j1 = None; self.rec = []; drained = 0
            self._send(pts, Q); t_send = time.time()
            while time.time() - t_send < T + 0.45 and not st["found"]:
                while drained < len(self.rec): todo.append(self.rec[drained]); drained += 1
                time.sleep(0.005)
            if st["found"]:
                te = max(0.0, min(time.time() - t_send, T)); k = min(range(len(pts)), key=lambda i: abs(pts[i][0] - te)); qc, vc = pts[k][1], pts[k][2]      # 명령된 현재 위치·속도
                vv = abs(vc); sgn = 1.0 if vc >= 0 else -1.0; Ts = vv / AMAX; sp = [(t, qc + sgn * (vv * t - 0.5 * AMAX * t * t), sgn * (vv - AMAX * t)) for t in list(np.arange(0.0, Ts, 0.05)) + [Ts]] if vv > 1e-3 else [(0.0, qc, 0.0), (0.1, qc, 0.0)]
                self.rec = None; self._send(sp, Q); self._settle(timeout=4.0); j_stop = self.j1[-1][1]; break
            self.rec = None; cur = tgt
        st["run"] = False; th.join(timeout=5)
        if not st["found"]: self._settle(timeout=4.0); j_stop = self.j1[-1][1]
        dur = time.time() - t_all; log(f"[fast_scan] 탐색 스윕 {'발견' if st['found'] else '못 찾음'} 정지 j1={j_stop:+.2f} 프레임 {st['n']}개 {dur:.1f}s")
        return st["found"], j_stop, st["n"], dur
