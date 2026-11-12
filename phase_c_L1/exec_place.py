"""exec_place.py -- Thor place_plan.py 가 만든 ~/grasp/{name}_place_plan.json 재생 (2026-09-29).

pick 때의 그리퍼 자세를 그대로 유지한 채 슬롯으로 옮겨 세워 놓는다:
  transit(현재 -> 슬롯 위 pre) -> descend(수직 하강) -> 그리퍼 open -> retreat(pick 접근의 역방향으로 후퇴).
결과는 /tmp/{name}_place_try0.json (place_on_target_one.sh/discover 가 읽는 기존 형식 {"status": ...}) 과
/tmp/{name}.place_same.json 에 쓴다. 실패해도 물체를 놓치지 않도록 각 단계 도달 여부를 확인하고,
아직 물체를 쥔 단계에서 실패하면 status=failed(stage 표시) 로 끝내 호출부가 기존 place 로 폴백하게 한다.
"""
import json, os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

name = sys.argv[1]
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
GRIP = ["gripper_joint1", "gripper_joint2"]
_GO = os.environ.get("NERO_EXEC_GO_FILE")     # 예열 모드(exec_pick.py 와 같음)
PLAN = None if _GO else json.load(open(os.path.expanduser(f"~/grasp/{name}_place_plan.json")))
NO_PHYS = os.environ.get("NERO_NO_PHYSICS", "0") == "1"
_H = (lambda real, fast: fast if NO_PHYS else real)
MOVE_SCALE = float(os.environ.get("NERO_PLACE_MOVE_SCALE", "0.5"))     # 물체를 든 채 움직이는 속도(작을수록 느리게)
_MOVE_DT = 1.0 / max(0.05, MOVE_SCALE)
GOPEN = [1.0, -1.0]
GRIP_J1_MAX = 0.05

rclpy.init(); n = Node("exec_place")
st = {}
n.create_subscription(JointState, "/joint_states",
                      lambda m: [st.__setitem__(a, b) for a, b in zip(m.name, m.position)], 10)
ex = ActionClient(n, ExecuteTrajectory, "/execute_trajectory")
gc = ActionClient(n, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
ex.wait_for_server(timeout_sec=10); gc.wait_for_server(timeout_sec=10)
for _ in range(20):
    rclpy.spin_once(n, timeout_sec=0.1)

# ── 정밀도 연구용 캡처(NERO_PLACE_CAPTURE=1, 기본 꺼짐): 놓기 직전 호버 자세에서 영상/깊이/카메라 내부·외부 파라미터와 시뮬 정답(전체 강체 자세)을 저장한다.
#    목적지와 쥔 물체가 같은 영상에 보이므로, 손목 카메라만으로 상대 오프셋을 재는 추정기를 오프라인으로 개발/검증할 데이터셋이 된다. ──
_CAP = os.environ.get("NERO_PLACE_CAPTURE", "0") == "1"
_cap = {}
if _CAP:
    import rclpy.time
    from sensor_msgs.msg import Image as _Img, CameraInfo as _CI
    from rclpy.qos import qos_profile_sensor_data as _qsd
    from tf2_ros import Buffer as _Buf, TransformListener as _TL
    _tfb = _Buf(); _TL(_tfb, n)
    n.create_subscription(_Img, "/camera/color/image_raw", lambda m: _cap.__setitem__("rgb", m), _qsd)
    n.create_subscription(_Img, "/camera/depth/image_raw", lambda m: _cap.__setitem__("depth", m), _qsd)
    n.create_subscription(_CI, "/camera/camera_info", lambda m: _cap.__setitem__("info", m), _qsd)
if _GO:
    _tw = time.time()
    while not os.path.exists(_GO) and time.time() - _tw < float(os.environ.get("NERO_EXEC_GO_TIMEOUT", "300")):
        rclpy.spin_once(n, timeout_sec=0.02)
    if not os.path.exists(_GO):
        print("[warm] 신호 없음 -- 종료"); rclpy.shutdown(); sys.exit(0)
    PLAN = json.load(open(os.path.expanduser(f"~/grasp/{name}_place_plan.json"))); print("[warm] 신호 수신 -- place 실행 시작")
_t0 = time.time()


PHASES = {}      # 2026-10-05: 단계별 시각(time.time()) -- peg_trace 의 막대 시계열/접촉 시각과 맞춰 "어느 단계에서 떨어졌나"를 가른다


def mark(label):
    PHASES[label] = round(time.time(), 3)


def lap(label):
    global _t0
    now = time.time(); print(f"[lap] {label}: +{now - _t0:.2f}s"); _t0 = now

SMOOTH = os.environ.get("NERO_SMOOTH", "1") == "1"     # 2026-10-01 궤적 스무딩: 균일 간격 웨이포인트(구간별 후방차분 속도 -> 점마다 속도 불연속)를 3차 스플라인으로 다시 표본화(50ms)하고 속도를 스플라인 미분으로 채운다


def smooth_points(pts, dt, step=0.05):
    """pts: [현재상태, 웨이포인트...] (간격 dt 균일). 양 끝 속도 0 인 3차 스플라인으로 step 간격 재표본화. -> [(t, pos, vel)]"""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return None
    from scipy.interpolate import CubicSpline
    T = np.arange(len(P)) * dt
    cs = CubicSpline(T, P, bc_type=((1, np.zeros(P.shape[1])), (1, np.zeros(P.shape[1]))))
    ts = np.arange(0.0, T[-1], step); ts = np.append(ts[ts < T[-1] - 1e-3], T[-1])      # 마지막 격자점이 T[-1]과 1e-3 안이면 뺀다(부동소수 차이로 같은 시각 점이 둘 생겨 컨트롤러가 'not strictly increasing' 으로 거부했음)
    return [(float(t), cs(t), cs(t, 1)) for t in ts]


# ── 진단 계측(NERO_PLACE_SNAPS=1): 놓는 단계 세 지점에서 물체 기울기(시뮬 정답)를 잰다. 쓰러짐이 손 안/놓는 순간/후퇴 중 어디서 생기는지 가르기 위함. 시행당 약 6s 추가라 진단 때만 켠다. ──
SNAPS = []
_SNAP_ON = os.environ.get("NERO_PLACE_SNAPS", "0") == "1"


def snap_tilt(tag):
    if not _SNAP_ON:
        return
    try:
        import subprocess as _sp
        pr = json.load(open(f"/tmp/{name}.place_prim.json"))          # discover 가 pick 전 스냅샷 기준으로 정해 둔다: {prim, q0}
        out = f"/tmp/_snap_{name}_{tag}.json"
        _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap_pose.py"), out], capture_output=True, timeout=40)
        d = json.load(open(out))[pr["prim"]]
        import place_ops as _po
        SNAPS.append(dict(tag=tag, tilt=round(_po.tilt_deg(pr["q0"], d["q"]), 1), z=round(d["p"][2], 3), p=d["p"][:2], p3=[round(float(v), 4) for v in d["p"]], q=[round(float(v), 5) for v in d["q"]], joints=[round(float(st.get(j, 0.0)), 4) for j in ARM]))
        print(f"[snap] {tag}: 기울기 {SNAPS[-1]['tilt']}도 z={SNAPS[-1]['z']}")
    except Exception as e:
        print("[snap] 실패", repr(e))


def reached(target, tol=0.15):
    e = max(abs(st.get(j, 0.0) - t) for j, t in zip(ARM, target))
    return e, e < tol


def spin_for(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(n, timeout_sec=0.05)


# 2026-10-08 (사용자 지시: 단계 사이 끊김 제거): NERO_FAST_EXEC=1(기본)이면 정지 확인을 즉시 판정하고, 이송+하강을 한 궤적으로 합치고, 놓기 전후 고정 대기를 줄인다. 0 이면 예전 동작 그대로.
FAST = os.environ.get("NERO_FAST_EXEC", "1") == "1"
MERGE = FAST and os.environ.get("NERO_MERGE_SEGS", "1") == "1"
POST_OPEN = float(os.environ.get("NERO_PLACE_POST_OPEN", "0.5" if FAST else "0.9"))     # 놓은 뒤 물체가 안정되길 기다리는 시간(후퇴 전). FAST 기본 0.5(예전 0.9)


def smooth_multi(pts, dts, step=0.05):
    """pts: [현재상태, 웨이포인트...], dts[i] = pts[i] -> pts[i+1] 구간 시간. 양 끝 속도 0, 중간 점은 속도를 이어 가는 3차 스플라인 -> [(t, pos, vel)]."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return None
    from scipy.interpolate import CubicSpline
    T = np.concatenate([[0.0], np.cumsum(np.maximum(np.asarray(dts, float), 1e-3))])
    cs = CubicSpline(T, P, bc_type=((1, np.zeros(P.shape[1])), (1, np.zeros(P.shape[1]))))
    ts = np.arange(0.0, T[-1], step); ts = np.append(ts[ts < T[-1] - 1e-3], T[-1])      # 마지막 격자점이 T[-1]과 1e-3 안이면 뺀다(부동소수 차이로 같은 시각 점이 둘 생겨 컨트롤러가 'not strictly increasing' 으로 거부했음)
    return [(float(t), cs(t), cs(t, 1)) for t in ts]


def dedupe_join(a, b, tol=0.01):
    """b 의 앞쪽 점 중 a 의 마지막 점과 거의 같은 것(정지-재출발이 생기는 원인)을 뺀다."""
    last = np.asarray(a[-1], float); i = 0
    while i < len(b) - 1 and np.max(np.abs(np.asarray(b[i], float) - last)) < tol:
        i += 1
    return list(b[i:])


def settle(timeout=2.5):
    """팔이 멈출 때까지 대기(직전 궤적 직후 미세 진동/처짐). MoveIt 은 궤적 첫 점이 현재 상태와 0.01rad 넘게
    다르면 실행 전에 ABORTED(-4) 로 거부한다 -- 정지 후에 첫 점을 잡는다."""
    if FAST:        # 이미 멈춰 있으면(대부분) 0.1초 안에 끝낸다. 움직이고 있으면 예전 루프로
        spin_for(0.05); a_ = [st.get(j, 0.0) for j in ARM]; spin_for(0.05); b_ = [st.get(j, 0.0) for j in ARM]
        if max(abs(x - y) for x, y in zip(a_, b_)) < 0.0008:
            return True
    t0 = time.time(); prev = [st.get(j, 0.0) for j in ARM]; stable = 0
    while time.time() - t0 < timeout:
        spin_for(0.12)
        cur = [st.get(j, 0.0) for j in ARM]
        stable = stable + 1 if max(abs(a - b) for a, b in zip(cur, prev)) < 0.0015 else 0
        prev = cur
        if stable >= 2:
            return True
    return False


def build_traj(wps, dt, dts=None):
    """현재 관절 상태를 첫 점으로 하는 JointTrajectory (매 시도마다 다시 만든다)."""
    jt = JointTrajectory(); jt.joint_names = list(ARM)
    pts = [[st.get(j, 0.0) for j in ARM]] + [list(w) for w in wps]
    sm = (smooth_multi(pts, dts) if dts is not None else smooth_points(pts, dt)) if SMOOTH else None
    if sm is not None:
        for ts_, pos_, vel_ in sm:
            p = JointTrajectoryPoint(); p.positions = [float(x) for x in pos_]; p.velocities = [float(x) for x in vel_]
            p.time_from_start.sec = int(ts_); p.time_from_start.nanosec = int((ts_ % 1) * 1e9)
            jt.points.append(p)
        return jt, sm[-1][0]
    prev = np.array(pts[0]); t = 0.0
    for i, w in enumerate(pts):
        w = np.array(w, float)
        if i > 0:
            t += dt
        p = JointTrajectoryPoint(); p.positions = [float(x) for x in w]
        v = (w - prev) / dt if i > 0 else np.zeros(7)
        if i == len(pts) - 1:
            v = np.zeros(7)
        p.velocities = [float(x) for x in v]
        p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t % 1) * 1e9)
        jt.points.append(p); prev = w
    return jt, t


def exec_traj(wps, dt, label, tries=6, dts=None):
    """exec_pick 와 달리 재시도마다 첫 점을 현재 상태로 다시 만든다(낡은 첫 점을 재전송하면 팔이 조금이라도
    움직인 뒤엔 영원히 시작 오차로 거부된다 -- nbr18 하강 5회 연속 ABORTED 의 원인)."""
    code = None
    for attempt in range(tries):
        settle()
        jt, t = build_traj(wps, dt, dts)
        g = ExecuteTrajectory.Goal(); g.trajectory = RobotTrajectory(joint_trajectory=jt)
        fut = ex.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
        gh = fut.result()
        if not gh or not gh.accepted:
            print(f"[{label}] REJECTED"); return None
        rf = gh.get_result_async()
        tend = time.time() + t + _H(0.8, 0.2)
        while time.time() < tend and not rf.done():
            rclpy.spin_once(n, timeout_sec=0.1)
        code = rf.result().result.error_code.val if rf.done() else None
        if code == -4 and attempt < tries - 1:
            print(f"[{label}] err=-4 -- 재정지 후 첫 점 재생성해 재시도 {attempt + 2}/{tries}")
            spin_for(0.3 * (attempt + 1)); continue
        print(f"[{label}] err={code}")
        return code
    return code


def grip_open(label="place-open"):
    """그리퍼 열기. GOPEN=1.0 은 관절 한계(0.05)로 잘리므로 도달 판정은 클램프한 값으로 한다."""
    OPEN_T = float(os.environ.get("NERO_PLACE_OPEN_T", "1.0"))      # 그리퍼를 여는 데 쓰는 시간(초): 길수록 손가락이 물체를 천천히 놓는다(놓는 순간 쓰러짐 실험)
    hold = _H(max(3.0, OPEN_T + 2.0), 0.6)
    for k in range(3):
        jt = JointTrajectory(); jt.joint_names = list(GRIP)
        p = JointTrajectoryPoint(); p.positions = [float(x) for x in GOPEN]
        tt = _H(OPEN_T, 0.15); p.time_from_start.sec = int(tt); p.time_from_start.nanosec = int((tt % 1) * 1e9)
        jt.points = [p]
        g = FollowJointTrajectory.Goal(); g.trajectory = jt
        fut = gc.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
        gh = fut.result()
        if gh is not None and gh.accepted:
            rf = gh.get_result_async(); t0 = time.time()
            while time.time() - t0 < hold:
                rclpy.spin_once(n, timeout_sec=0.1)
                if rf.done() or (abs(st.get("gripper_joint1", 0) - GRIP_J1_MAX) < 0.004 and time.time() - t0 > min(0.6, hold * 0.5)):
                    break
        if st.get("gripper_joint1", 0) > 0.040:
            print(f"[{label}] grip={st.get('gripper_joint1', 0):.3f}")
            return True
        time.sleep(0.3)
    print(f"[{label}] 안 열림 grip={st.get('gripper_joint1', 0):.3f}")
    return False


def capture_pre(tag="hover"):
    if not _CAP:
        return
    try:
        _cap.pop("rgb", None); _cap.pop("depth", None)
        t0_ = time.time()
        while time.time() - t0_ < 3.0 and not ("rgb" in _cap and "depth" in _cap and "info" in _cap):
            rclpy.spin_once(n, timeout_sec=0.05)
        m = _cap["rgb"]; d = _cap["depth"]; info = _cap["info"]
        rgb = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[..., :3]
        rgb = rgb[..., ::-1].copy() if m.encoding.startswith("bgr") else rgb.copy()
        depth = np.frombuffer(bytes(d.data), np.float32 if "32F" in d.encoding else np.uint16).reshape(d.height, d.width).astype(np.float32)
        if "16U" in d.encoding:
            depth = depth / 1000.0
        K = np.array(info.k).reshape(3, 3)
        tf = _tfb.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time()).transform
        from scipy.spatial.transform import Rotation as _R
        T = np.eye(4); T[:3, :3] = _R.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix(); T[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
        import subprocess as _sp
        gt = {}
        try:
            _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap_pose.py"), f"/tmp/_cap_gt_{name}.json"], capture_output=True, timeout=40)
            gt = json.load(open(f"/tmp/_cap_gt_{name}.json"))
        except Exception as e_:
            print("[cap] GT 스냅샷 실패", repr(e_))
        os.makedirs("/tmp/place_cap", exist_ok=True)
        np.savez_compressed(f"/tmp/place_cap/{name}_{tag}.npz", rgb=rgb, depth=depth, K=K, T=T, q=np.array([st.get(j, 0.0) for j in ARM]), slot=np.array(PLAN.get("slot") or [0, 0]),
                            tcp=np.array(PLAN.get("tcp") or [0, 0, 0]), flange=np.array(PLAN.get("flange") or [0, 0, 0]), gt=json.dumps(gt), t=time.time())
        print(f"[cap] {tag} 저장: /tmp/place_cap/{name}_{tag}.npz depth 유효 {np.isfinite(depth).mean():.2f}")
    except Exception as e:
        print("[cap] 실패", repr(e))


def finish(status, stage, **kw):
    kw.setdefault("phases", PHASES); kw.setdefault("snaps", SNAPS)       # 실패 경로에서도 단계 시각/스냅샷을 남긴다
    res = dict(status=status, stage=stage, slot=PLAN.get("slot"), yaw_deg=PLAN.get("yaw_deg"), **kw)
    json.dump(res, open(f"/tmp/{name}.place_same.json", "w"))
    _slot = PLAN.get("slot") or [0.0, 0.0]; _tcp = PLAN.get("tcp") or [0.0, 0.0, 0.0]
    json.dump(dict(status=status, reason=stage, actual_place_pos=dict(x=_slot[0], y=_slot[1], z=_tcp[2]), mode="same_pose"),
              open(f"/tmp/{name}_place_try0.json", "w"))
    print(f"[place_same] {status} ({stage})")
    rclpy.shutdown(); sys.exit(0)


if not PLAN.get("ok"):
    finish("failed", "plan 없음: " + str(PLAN.get("reason")))

# 1) transit: 물체를 든 채 슬롯 위 pre 자세로 + 2) 수직 하강
if MERGE and not _SNAP_ON and not _CAP:
    # 2026-10-08: 이송 끝에서 멈춰 pre 도달을 확인하고 하강을 다시 쏘던 것을 한 궤적으로 이어서 실행한다(구간별 속도는 그대로). 중간 정지/정지 확인/재출발이 없다.
    _desc = dedupe_join(PLAN["transit"], PLAN["descend"])
    _dts = [0.13 * _MOVE_DT] * len(PLAN["transit"]) + [0.20 * _MOVE_DT] * len(_desc)
    mark("transit_start"); _tm = time.time()
    exec_traj(list(PLAN["transit"]) + _desc, dt=0.13 * _MOVE_DT, label="place-transit+descend", dts=_dts); lap("place-transit+descend")
    mark("transit_end"); mark("descend_start"); mark("descend_end")
    settle(3.0)
    e, ok = reached(PLAN["place_cfg"], tol=0.15)
    if not ok:
        finish("failed", f"이송+하강 후 place 자세 미도달(오차 {e:.2f} rad)", held=True)
    time.sleep(0.15 if FAST else _H(0.4, 0.15))
else:
    mark("transit_start"); exec_traj(PLAN["transit"], dt=0.13 * _MOVE_DT, label="place-transit"); lap("place-transit")
    settle(4.0)       # 2026-10-01 cuRobo 의 큰 transit(관절 3rad/2s)은 컨트롤러가 성공을 보고한 뒤에도 아직 움직이는 중이라 정지 후에 도달을 확인한다
    e, ok = reached(PLAN["pre_cfg"], tol=0.20)
    if not ok:
        finish("failed", f"transit 후 pre 미도달(오차 {e:.2f} rad)", held=True)
    mark("transit_end")
    if os.environ.get("NERO_PEG_TRACE", "1") == "1": snap_tilt("pre_descend")      # 이송 끝(하강 전) 막대 상태: 이송 중에 떨어졌는지 가른다
    capture_pre("hover")   # 호버 자세(놓기 직전 위): 목적지와 쥔 물체가 같은 영상에 보인다
    # 2) 수직 하강 (pick 때와 같은 자세 그대로)
    mark("descend_start"); exec_traj(PLAN["descend"], dt=0.20 * _MOVE_DT, label="place-descend"); lap("place-descend"); mark("descend_end")
    settle(3.0)
    e, ok = reached(PLAN["place_cfg"], tol=0.15)
    if not ok:
        finish("failed", f"descend 후 place 자세 미도달(오차 {e:.2f} rad)", held=True)
    time.sleep(_H(0.4, 0.15))
snap_tilt("pre_open")        # 하강 끝, 놓기 직전(손 안에서 이미 기울었나)
print(f"[grip] pre_open 그리퍼 관절 {st.get('gripper_joint1', 0):.3f} (물체를 쥐고 있으면 물체 폭의 절반 근처, 0.01 미만이면 빈손으로 닫혀 있음)")      # 2026-10-14: 열기 실패 원인 가르기(쥔 채 끼었나 / 이미 놓쳤나)
# 3) 놓기
mark("open_start"); opened = grip_open(); lap("place-open"); mark("open_end")
if not opened and os.environ.get("NERO_PROBE_CONTACT", "0") == "1":      # 2026-10-14: 열기 실패 직후 손가락/베이스와 바구니 벽의 접촉을 시뮬에서 읽는다(진단, 기본 꺼짐)
    try:
        import subprocess as _sp; _o = _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_contact.py")], capture_output=True, text=True, timeout=30)
        print(_o.stdout.strip() or _o.stderr.strip()[-300:])
    except Exception as _e: print("[probe] 실패", repr(_e))
if not opened and os.environ.get("NERO_OPEN_RECOVER", "0") == "1":      # 2026-10-14(기본 꺼짐): 열기 실패 복구 -- 실측: 놓기는 바구니 안에서 끝났는데 그리퍼 관절이 0.005~0.03 에서 안 열려 EXEC_FAIL 로 세어졌다
    _g = float(st.get("gripper_joint1", 0.0)); print(f"[place-open-recover] 열기 실패 grip={_g:.3f}")
    if _g < 0.015:      # 손가락이 빈 채 닫혀 있다 = 물체는 이미 손을 떠났다(또는 얇은 가장자리만 물었다) -> 계획된 후퇴로 손을 뺀다. 물체는 그대로 두고, 판정은 시뮬 정답이 한다
        mark("retreat_start"); exec_traj(PLAN["retreat"], dt=0.20 * _MOVE_DT, label="place-retreat-closed"); lap("place-retreat-closed"); mark("retreat_end")
        snap_tilt("post_retreat"); finish("success", "place_complete_gripper_closed", plan_sec=PLAN.get("plan_sec"), snaps=SNAPS, phases=PHASES)
    time.sleep(1.5); opened = grip_open()      # 쥔 채 낀 경우: 물리가 안정된 뒤 한 번 더
if not opened:
    finish("failed", "그리퍼가 안 열림", held=True)
time.sleep(POST_OPEN if not NO_PHYS else 0.3)      # 2026-10-01 놓은 직후 물체가 면에 안정될 때까지 잠시 기다린 뒤 후퇴한다(바로 빼면 손가락에 끌려 쓰러지는 경우가 있다)
snap_tilt("post_open")       # 그리퍼를 연 뒤 안정 대기 끝(놓는 순간에 쓰러졌나)
# 4) 후퇴: pick 접근의 역방향(물체와 부딪히지 않는 방향)으로
_rw = PLAN["retreat"]; _slow = float(os.environ.get("NERO_RETREAT_SLOW", "1")); _k = max(2, (len(_rw) + 3) // 4)      # 2026-10-15: 놓은 직후 후퇴 앞 1/4 구간은 느리게 -- 사자가 손가락에 끼어 팔과 함께 빠르게 끌려 나가다 책상 밖으로 튕겼다
_rd = [0.20 * _MOVE_DT * (_slow if i < _k else 1.0) for i in range(len(_rw))]
mark("retreat_start"); exec_traj(_rw, dt=0.20 * _MOVE_DT, label="place-retreat", dts=(_rd if SMOOTH and len(_rw) >= 2 else None)); lap("place-retreat"); mark("retreat_end")
snap_tilt("post_retreat")    # 후퇴 뒤(손가락에 끌려 쓰러졌나)
finish("success", "place_complete", plan_sec=PLAN.get("plan_sec"), snaps=SNAPS, phases=PHASES)
