"""step6_pick.json 궤적을 MoveIt /execute_trajectory 액션으로 재생 (move_group 실행 경로).
Isaac 스크립트/Isaac MCP 안 씀. transit -> advance -> 그리퍼 close -> lift (후퇴 단계 없음,
2026-09-17 -- ros2_ws 사본과 동기화, retreat 있으면 lift 시작점과 안 맞아 도로 내려갔다
올라가는 버그)."""
import os as _ng_os; GRASP_DIR = _ng_os.environ.get("NERO_GRASP_DIR") or _ng_os.path.expanduser("~/grasp")  # shared work dir on PC and Thor (override: NERO_GRASP_DIR)
import json, time, os, subprocess, sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
GRIP = ["gripper_joint1", "gripper_joint2"]
_GO = os.environ.get("NERO_EXEC_GO_FILE")     # 예열 모드: 노드/액션 클라이언트까지 준비한 뒤 신호 파일이 생길 때까지 기다리고, 그때 계획을 읽는다(C 결과가 나오자마자 출발)
D = None if _GO else json.load(open("" + GRASP_DIR + "/step6_pick.json"))

# 2026-09-17 (사용자 지시): "lift 전에 딜레이가 있다"는 실측 지적 진단용 -- 단계별
# 소요시간을 찍어서 어디서 시간이 새는지 정확히 잡는다.
_t_lap0 = time.time()


def _lap(label):
    global _t_lap0
    now = time.time()
    print(f"[lap] {label}: +{now - _t_lap0:.2f}s")
    _t_lap0 = now

rclpy.init(); n = Node("exec_pick")
st = {}
GSER = []     # 2026-09-30 그리퍼 폭 시계열 [시각(s), joint1, joint2] (30Hz 로 솎음) -- 학습 판정기 입력/판정 분석용. exec_res.json 의 grip_series 로 저장
_gser_t0 = [time.time()]; _gser_last = [0.0]


def _on_js(m):
    for a, b in zip(m.name, m.position):
        st[a] = b
    t = time.time()
    if t - _gser_last[0] >= 0.033 and len(GSER) < 4000:
        _gser_last[0] = t
        GSER.append([round(t - _gser_t0[0], 3), round(st.get("gripper_joint1", 0.0), 4), round(st.get("gripper_joint2", 0.0), 4)])


n.create_subscription(JointState, "/joint_states", _on_js, 10)
ex = ActionClient(n, ExecuteTrajectory, "/execute_trajectory")
gc = ActionClient(n, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
ex.wait_for_server(timeout_sec=10); gc.wait_for_server(timeout_sec=10)
for _ in range(20): rclpy.spin_once(n, timeout_sec=0.1)
# 2026-09-18 (사용자 지시, 딜레이 분석): _t_lap0가 모듈 최상단(rclpy.init 이전)에서
# 시작해서, 지금까지의 노드/액션클라이언트 초기화+워밍업 비용이 전부 다음 _lap()
# 호출인 "grip-open"에 숨어서 찍히고 있었다(실측: grip-open이 매번 3.6~5.6s로 커
# 보였던 이유 중 하나 -- 실제 grip-open 자체 비용이 아니었다). 여기서 리셋해서
# 초기화 비용을 "node-init"으로 따로 분리해 보이게 한다.
_lap("node-init")
if _GO:
    _tw = time.time()
    while not os.path.exists(_GO) and time.time() - _tw < float(os.environ.get("NERO_EXEC_GO_TIMEOUT", "180")):
        rclpy.spin_once(n, timeout_sec=0.02)           # 관절 상태 구독은 계속 받는다
    if not os.path.exists(_GO):
        print("[warm] 신호 없음 -- 종료"); rclpy.shutdown(); sys.exit(0)
    D = json.load(open("" + GRASP_DIR + "/step6_pick.json")); _t_lap0 = time.time()
    print("[warm] 신호 수신 -- 실행 시작")


def jl(j): return ", ".join("%.2f" % st.get(x, 0) for x in ARM)


# 2026-09-18 (사용자 지시, NO_REACH 원인 규명 -- "grasp_cfg 도달 오차 -> 미도달"이
# 실제로는 IK/도달성 문제가 아니었다): 실측으로 err 코드를 moveit_msgs.MoveItErrorCodes 와
# 직접 대조 -> -4 는 NO_IK_SOLUTION(-31)이 아니라 CONTROL_FAILED(컨트롤러 레벨 거부).
# 서명이 항상 동일: 직전 exec_traj가 err=1(성공)로 끝난 직후 바로 이어붙인 다음
# exec_traj가 err=-4로 +0.01s만에 즉시 실패(시도도 거의 안 함). 처음엔 "완료 직후
# 쉬는 시간이 없어서"로 보고 0.15s 슬립을 넣었으나 재현 결과 그걸로도 안 고쳐짐 --
# 게다가 advance 가 -4로 실패한 직후 이어진 backoff 까지 똑같이 +0.01s 즉시 실패해서,
# "한 번 거부되면 잠시 동안 컨트롤러가 계속 거부 상태로 남는다"는 게 더 정확한 모델로
# 보임(그 뒤 한참 지난 p1-transit은 정상 성공 -- 영구 고장은 아님, 시간이 지나면 풀림).
# 그래서 고정 슬립 대신: CONTROL_FAILED 받으면 그 자리에서 점증 백오프하며 같은 goal을
# 재전송(최대 3회) -- 시간이 지나면 저절로 풀리는 상태라는 관찰과 정확히 맞아떨어짐.
def _send_and_wait(jt_msg, t, label, tries=5):
    for attempt in range(tries):
        g = ExecuteTrajectory.Goal(); g.trajectory = RobotTrajectory(joint_trajectory=jt_msg)
        fut = ex.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
        gh = fut.result()
        if not gh or not gh.accepted:
            print(f"[{label}] REJECTED"); return None
        rf = gh.get_result_async()
        tend = time.time() + t + (_H(0.8, 0.2) if "_H" in globals() else 0.8)  # settle
        while time.time() < tend and not rf.done():
            rclpy.spin_once(n, timeout_sec=0.1)
        code = rf.result().result.error_code.val if rf.done() else None
        if code == -4 and attempt < tries - 1:
            backoff = 0.3 * (attempt + 1)
            print(f"[{label}] err=-4(CONTROL_FAILED) -- {backoff:.1f}s 후 재전송 {attempt+2}/{tries}")
            time.sleep(backoff)
            continue
        print(f"[{label}] err={code}  j=({jl(None)})")
        return code
    return code

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


def _spin_for(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(n, timeout_sec=0.05)


# 2026-10-08 (사용자 지시: 단계 사이 끊김 제거): NERO_FAST_EXEC=1(기본)이면 (1) 그리퍼 대기를 고정 시간이 아니라 폭이 안정되면 끝내고, (2) 정지 확인을 즉시 판정하고,
# (3) 접근+하강처럼 이어지는 구간을 한 궤적으로 합친다. 0 이면 예전 동작 그대로.
FAST = os.environ.get("NERO_FAST_EXEC", "1") == "1"
MERGE = FAST and os.environ.get("NERO_MERGE_SEGS", "1") == "1"


def smooth_multi(pts, dts, step=0.05):
    """pts: [현재상태, 웨이포인트...], dts[i] = pts[i] -> pts[i+1] 구간 시간. 양 끝 속도 0, 중간 점은 속도를 이어 가는 3차 스플라인으로 step 간격 재표본화 -> [(t, pos, vel)]."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return None
    from scipy.interpolate import CubicSpline
    T = np.concatenate([[0.0], np.cumsum(np.maximum(np.asarray(dts, float), 1e-3))])
    cs = CubicSpline(T, P, bc_type=((1, np.zeros(P.shape[1])), (1, np.zeros(P.shape[1]))))
    ts = np.arange(0.0, T[-1], step); ts = np.append(ts[ts < T[-1] - 1e-3], T[-1])      # 마지막 격자점이 T[-1]과 1e-3 안이면 뺀다(부동소수 차이로 같은 시각 점이 둘 생겨 컨트롤러가 'not strictly increasing' 으로 거부했음)
    return [(float(t), cs(t), cs(t, 1)) for t in ts]


def _dedupe_join(a, b, tol=0.01):
    """b 의 앞쪽 점 중 a 의 마지막 점과 거의 같은 것(정지-재출발이 생기는 원인)을 뺀다."""
    last = np.asarray(a[-1], float); i = 0
    while i < len(b) - 1 and np.max(np.abs(np.asarray(b[i], float) - last)) < tol:
        i += 1
    return list(b[i:])


def _settle(timeout=2.5):
    """팔이 멈출 때까지 대기. MoveIt 은 궤적 첫 점이 현재 상태와 0.01rad 넘게 다르면 실행 전에 ABORTED(-4)로 거부한다."""
    if FAST:        # 이미 멈춰 있으면(대부분) 0.1초 안에 끝낸다. 움직이고 있으면 예전 루프로
        _spin_for(0.05); a_ = [st.get(j, 0.0) for j in ARM]; _spin_for(0.05); b_ = [st.get(j, 0.0) for j in ARM]
        if max(abs(x - y) for x, y in zip(a_, b_)) < 0.0008:
            return True
    t0 = time.time(); prev = [st.get(j, 0.0) for j in ARM]; stable = 0
    while time.time() - t0 < timeout:
        _spin_for(0.12)
        cur = [st.get(j, 0.0) for j in ARM]
        stable = stable + 1 if max(abs(a - b) for a, b in zip(cur, prev)) < 0.0015 else 0
        prev = cur
        if stable >= 2:
            return True
    return False


TRACE = []      # exec_traj 실행 기록 (종료 시 /tmp/exec_trace_last.json; 어떤 경로로 끝나도 atexit 로 저장)
import atexit


def _save_trace():
    try:
        json.dump(TRACE, open('/tmp/exec_trace_last.json', 'w'))
    except Exception:
        pass


atexit.register(_save_trace)


def exec_traj(wps, dt, label, grip=None, tries=6, dts=None):
    """2026-09-29: 재시도마다 첫 점을 현재 상태로 다시 만든다. 예전(_send_and_wait)은 처음 만든 낡은 첫 점을 그대로
    재전송해서, 팔이 조금이라도 움직인 뒤엔 시작 오차(0.01rad)로 5번 연속 -4 가 났다(nbr18 시드101: advance/transit
    -4 x5 -> NO_REACH 연쇄). 시도 전에 정지도 기다린다."""
    code = None
    for attempt in range(tries):
        _settle()
        jt = JointTrajectory(); jt.joint_names = list(ARM)
        pts = [list(st.get(j, 0) for j in ARM)] + [list(w) for w in wps]   # 현재 자세에서 시작
        sm = (smooth_multi(pts, dts) if dts is not None else smooth_points(pts, dt)) if SMOOTH else None
        if sm is not None:
            t = sm[-1][0]
            for ts_, pos_, vel_ in sm:
                pp = JointTrajectoryPoint(); pp.positions = [float(x) for x in pos_]; pp.velocities = [float(x) for x in vel_]
                pp.time_from_start.sec = int(ts_); pp.time_from_start.nanosec = int((ts_ % 1) * 1e9)
                jt.points.append(pp)
        else:
            prev = np.array(pts[0]); t = 0.0
            for i, w in enumerate(pts):
                w = np.array(w, float)
                if i > 0:
                    t += dt
                pp = JointTrajectoryPoint()
                pp.positions = [float(x) for x in w]
                v = (w - prev) / dt if i > 0 else np.zeros(7)
                if i == len(pts) - 1:
                    v = np.zeros(7)
                pp.velocities = [float(x) for x in v]
                pp.time_from_start.sec = int(t); pp.time_from_start.nanosec = int((t % 1) * 1e9)
                jt.points.append(pp); prev = w
        g = ExecuteTrajectory.Goal(); g.trajectory = RobotTrajectory(joint_trajectory=jt)
        fut = ex.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
        gh = fut.result()
        if not gh or not gh.accepted:
            print(f"[{label}] REJECTED"); return None
        rf = gh.get_result_async()
        tend = time.time() + t + (_H(0.8, 0.2) if "_H" in globals() else 0.8)
        _t_start = time.time(); _tr = []
        while time.time() < tend and not rf.done():
            rclpy.spin_once(n, timeout_sec=0.05)
            _tr.append([round(time.time() - _t_start, 3)] + [round(st.get(j, 0.0), 4) for j in ARM])
        # 2026-09-30 진단: 실제로 실행된 관절 궤적 + 계획 waypoint 를 저장 -> 계획 대비 추종 오차/막힌 지점 분석(chain 충돌 조사)
        try:
            TRACE.append(dict(label=label, dt=dt, code=(rf.result().result.error_code.val if rf.done() else None), plan=[list(map(float, w)) for w in wps], actual=_tr))
        except Exception:
            pass
        code = rf.result().result.error_code.val if rf.done() else None
        if code == -4 and attempt < tries - 1:
            print(f"[{label}] err=-4(CONTROL_FAILED) -- 정지 후 첫 점 재생성 재시도 {attempt + 2}/{tries}")
            _spin_for(0.3 * (attempt + 1)); continue
        print(f"[{label}] err={code}  j=({jl(None)})")
        return code
    return code


def reached(target, tol=0.12):
    """현재 관절이 target(7-list) 근처인가 — advance 가 실제로 grasp pose 에 갔나 검증."""
    e = max(abs(st.get(j, 0.0) - t) for j, t in zip(ARM, target))
    return e, e < tol


GRIP_J1_MAX = 0.05   # ros2_control 관절 한계 -- GOPEN=1.0 은 여기로 클램프됨


def _gclamp(v):
    """명령값을 실제 도달 가능한 관절값으로. GOPEN=1.0 을 그대로 비교하면 도달 판정이
    영원히 안 돼서 grip-open 이 매번 hold(3s) 를 다 채웠다(2026-09-29 실측 3.76s 고정)."""
    return min(max(float(v), 0.0), GRIP_J1_MAX)


def _grip_once(pos, hold, settle):
    jt = JointTrajectory(); jt.joint_names = list(GRIP)
    p = JointTrajectoryPoint(); p.positions = [float(x) for x in pos]
    # 2026-09-18 (사용자 지시, waste time 제거): 기존엔 매 그립 명령(open/close/squeeze/
    # recheck/hold, 픽당 최대 5회)마다 무조건 1.0초짜리 궤적을 보냈다 -- NO_PHYSICS라
    # 그리퍼가 저항 없이 명령대로 바로 움직이는데, 목표 도달에 필요도 없는 1초를 매번
    # 강제한 것. NO_PHYS면 0.15s, 실물(그리퍼 저항 있음)은 기존 1.0s 그대로 유지.
    _traj_sec = _H(1.0, 0.15) if "_H" in globals() else 1.0
    # 2026-09-30 사용자 지시: 그리퍼 slip 이 닫는 속도와 관련 있을 수 있다 -> 개폐 궤적 시간을 env 로 조정(NERO_GRIP_TRAJ_SEC, 초).
    # 기본(미설정)은 기존 그대로(물리 모드 1.0s). 0.15 = 가능한 가장 빠르게. 팔 이동 배율(NERO_MOVE_SCALE)은 따로 유지.
    if os.environ.get("NERO_GRIP_TRAJ_SEC"):
        _traj_sec = float(os.environ["NERO_GRIP_TRAJ_SEC"])
    p.time_from_start.sec = int(_traj_sec)
    p.time_from_start.nanosec = int((_traj_sec % 1) * 1e9)
    jt.points = [p]
    g = FollowJointTrajectory.Goal(); g.trajectory = jt
    fut = gc.send_goal_async(g); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    gh = fut.result()
    if gh is None or not gh.accepted:
        return False
    rf = gh.get_result_async()
    t0 = time.time()
    # 2026-09-18: settle 조기종료 문턱이 0.6으로 고정돼 있어서 hold<=0.6이면 절대
    # 발동 못 하던 죽은 코드였다(hold 전체를 항상 다 씀) -- hold의 절반으로 스케일.
    _settle_floor = min(0.6, hold * 0.5)
    _g0 = st.get("gripper_joint1", 0.0); _need = abs(_g0 - _gclamp(pos[0])) > 0.008; _hist = []
    while time.time() - t0 < hold:
        rclpy.spin_once(n, timeout_sec=0.05 if FAST else 0.1)
        if rf.done():
            break
        if settle is not None and abs(st.get("gripper_joint1", 0) - _gclamp(pos[0])) < settle and time.time() - t0 > _settle_floor:
            break
        if FAST:      # 2026-10-08 사건 기반 종료: 폭이 0.2초 동안 0.4mm 안에서 안정되면(물체 폭에 막혔거나 목표에 도달) 고정 hold 를 다 채우지 않고 끝낸다
            _now = time.time(); _hist.append((_now, st.get("gripper_joint1", 0.0))); _hist = [h for h in _hist if _now - h[0] <= 0.2]
            if len(_hist) >= 4 and _now - _hist[0][0] >= 0.15 and max(h[1] for h in _hist) - min(h[1] for h in _hist) < 0.0004:
                _moved = abs(_hist[-1][1] - _g0) > 0.004
                if (_now - t0 > 0.30 and (_moved or not _need)) or _now - t0 > 0.60:
                    break
    return True


def grip(pos, label, hold=2.0, settle=None, tries=3):
    """그리퍼 명령이 씹히는 경우가 있어(궤적 직후 컨트롤러 busy) 안 움직이면 재전송."""
    g0 = st.get("gripper_joint1", 0.05)
    for k in range(tries):
        _grip_once(pos, hold if k == 0 else max(1.0, hold * 0.6), settle)
        g1 = st.get("gripper_joint1", 0.05)
        # 목표쪽으로 유의미하게 움직였거나 (닫힘: 감소 / 열림: 증가) 목표 근처면 OK
        moved = abs(g1 - g0) > 0.006 or abs(g1 - _gclamp(pos[0])) < 0.006
        if moved:
            break
        if k < tries - 1:
            print(f"[{label}] 안 움직임 ({g1:.3f}) — 재전송 {k+2}/{tries}")
            time.sleep(0.3)
    print(f"[{label}] grip=({st.get('gripper_joint1',0):.3f}, {st.get('gripper_joint2',0):.3f})")


def close_gripper(label, hold=2.5, tries=4):
    """2026-09-17: "완전히 안 닫혀서 중간에 떨어뜨림" 사용자 실측 지적에 대한 대응.
    두 가지 서로 다른 원인이 섞여있었다:
      (1) 컨트롤러가 이따금 close 명령 자체를 거부/무시(관절이 완전개방 0.050 에 그대로
          박혀있는 경우, banana 케이스에서 실측) -- GCLOSE_TIGHT(정확히 0) 로 먼저 시도
          해서 안 움직이면 폴백값 GCLOSE(0.003)로 재시도하면 해결됨.
      (2) 그리퍼가 물체의 실제 폭에 막혀 중간(예 0.039)에서 멈추는 경우(box 케이스) --
          이건 목표값을 더 조여도 안 바뀐다(이미 물리적으로 막힌 지점). 이 경우는 트랜짓
          중 마찰만으로 붙어있어야 하는데 불안정할 수 있음 -- 소프트웨어로 완전 해결
          불가, Isaac Sim 쪽 attach/rigid-body 없이는 근본 해결 안 됨(사용자에게 별도 보고).
    반환: 최종 gripper_joint1 값(정보용, 어느 케이스였는지 호출부에서 판단 가능)."""
    g_before = st.get("gripper_joint1", 0.05)
    grip(GCLOSE_TIGHT, f"{label}(tight)", hold=hold, tries=tries)
    g1 = st.get("gripper_joint1", 0.05)
    if abs(g1 - g_before) < 0.006:   # 완전개방에서 전혀 안 움직였음 = 명령 거부 케이스
        print(f"[{label}] tight(0) 거부된 것으로 보임 -- 폴백 0.003 재시도")
        grip(GCLOSE, f"{label}(fallback)", hold=hold, tries=tries)
    return st.get("gripper_joint1", 0.05)


START_Q = [st.get(j, 0.0) for j in ARM]
res = {"start": {j: round(x, 3) for j, x in zip(ARM, START_Q)}}
# 그리퍼 하드코딩 (2026-09-10 사용자 지시): open=1, close=0.
# ros2_control 이 관절 limit (joint1: 0~0.05, joint2: -0.05~0) 으로 클램프.
# ★ 정확히 limit(0.0) 을 goal 로 주면 FollowJointTrajectory 컨트롤러가 goal 을 거부해
#   (관절값이 안 움직임) → limit 에서 3mm 안쪽으로. 사실상 완전 폐합 + 물체 최대 압력.
GOPEN  = [1.0, -1.0]
GCLOSE = [0.003, -0.003]           # 폴백값 (limit 회피, 기존)
GCLOSE_TIGHT = [0.0, 0.0]          # 2026-09-17: 최우선 시도 -- "완전히 닫혀야 안 빠짐"
                                    # (사용자 실측 지적: 그리퍼가 물체 폭 근처에서만 멈춰서
                                    # transit 중 떨어뜨림). limit 정확히 0 은 컨트롤러가
                                    # goal 거부할 수 있다는 게 기존 관찰이라, tight 시도 후
                                    # "안 움직임"이 감지되면(_grip_once 의 moved 판정) 폴백값
                                    # GCLOSE(0.003)으로 다시 시도한다 -- close_gripper() 참고.
_gc = 0.0
HELD_MIN = 0.006                                   # 손끝 사이 6mm 이상 = 파지됨 (close 0.003 기준)
print("[exec] gripper 하드코딩: open=1(→0.05) / close 시도 0(완전폐합)->실패시 0.003 폴백")
NO_PHYS = os.environ.get("NERO_NO_PHYSICS", "0") == "1"
if NO_PHYS:
    print("[exec] NERO_NO_PHYSICS=1 -- grip 위치 기반 파지판정 무시, 궤적 실행 완료=성공(EXEC_OK)")
PLANS = D.get("plans", [D])

# 2026-09-17: 아래 grip hold 시간들은 원래 "실제로 물체를 쥐었는지 정착시간을 확보"하려고
# 넉넉하게 잡은 값들이다. NERO_NO_PHYSICS=1(물체에 rigid body 없음)이면 그리퍼가 저항
# 없이 명령대로 움직이므로 이 대기가 실질 정보를 안 주면서 물체당 delay 만 키운다(다중물체
# 라운드에서 물체 수만큼 누적되는 게 눈에 띄게 느림 -- 사용자 지적). NO_PHYS 일 때만 짧게,
# 실물/물리 있는 실행 경로(기본값)는 절대 안 건드림.
_H = (lambda real, fast: fast if NO_PHYS else real)

# 2026-09-17 (사용자 지시): 이동 속도를 구간별로 분리 -- grasp 이후 lift 는 0.5,
# 그 외(transit/advance)는 0.8. dt(waypoint당 재생시간)를 스케일의 역수로
# 키우면 그 배수만큼 느리게 재생됨(1.0=원래 기준 dt). 이전엔 파지 후 구간을 통째로
# 0.4로 늦췄는데, attach/물리가 없어서 느려도 안 놓치는 게 아니라 시간만 더 썼음
# (사용자 실측) -- 그래도 lift 는 좀 더 안정적으로, 나머지는 좀 더 빠르게로 재조정.
# (retreat 단계는 별도 버그로 완전히 제거함 -- 아래 주석 참고.)
MOVE_SCALE = float(os.environ.get("NERO_MOVE_SCALE", "0.8"))   # transit/advance
_MOVE_DT = 1.0 / max(0.05, MOVE_SCALE)
# 2026-09-18 (사용자 지시): lift는 더 이상 NERO_LIFT_SCALE로 느려지지 않는다 -- dt 고정
# 0.30(아래 lift-z/lift-fallback 참고). 예전 LIFT_SCALE=0.5 하위호환 로그만 남김.
print(f"[exec] 이동속도 -- move(transit/advance)={MOVE_SCALE} lift=고정(0.30, NERO_LIFT_SCALE 미적용)")

grip(GOPEN, "grip-open", hold=_H(3.0, 0.6), settle=0.004)   # 완전 개방
if st.get("gripper_joint1", 0) < 0.040:
    print(f"[warn] 그리퍼가 덜 열림 ({st.get('gripper_joint1',0):.3f}) — 재시도")
    grip(GOPEN, "grip-open2", hold=_H(3.0, 0.6), settle=0.004)
_lap("grip-open")

# A-2 수정: plan 을 순서대로 시도. advance 가 실제로 grasp_cfg 에 도달해야 close.
# 도달 못 하면 START 로 후퇴 후 다음 plan (허공에 대고 close 하는 false SLIPPED 방지).
# ── 정밀도 연구용 집기 전 캡처(NERO_PICK_CAPTURE=1, 기본 꺼짐): pre-grasp 호버 자세에서 영상/깊이/카메라 자세/시뮬 정답(전체 강체 자세)을 저장한다.
#    이때 물체는 손 아래에 있어 시야가 가려지지 않고 윗면이 가까이서 거의 정면으로 보인다 -> 중심/플랜지 상대 오프셋을 다시 재는 추정기를 오프라인으로 개발/검증할 데이터. ──
_PCAP = os.environ.get("NERO_PICK_CAPTURE", "0") == "1"
_pcap = {}
if _PCAP:
    import rclpy.time
    from sensor_msgs.msg import Image as _Img, CameraInfo as _CI
    from rclpy.qos import qos_profile_sensor_data as _qsd
    from tf2_ros import Buffer as _Buf, TransformListener as _TL
    _tfb = _Buf(); _TL(_tfb, n)
    n.create_subscription(_Img, "/camera/color/image_raw", lambda m: _pcap.__setitem__("rgb", m), _qsd)
    n.create_subscription(_Img, "/camera/depth/image_raw", lambda m: _pcap.__setitem__("depth", m), _qsd)
    n.create_subscription(_CI, "/camera/camera_info", lambda m: _pcap.__setitem__("info", m), _qsd)


def capture_pregrasp(cand):
    if not _PCAP:
        return
    try:
        _spin_for(0.5); _pcap.pop("rgb", None); _pcap.pop("depth", None)
        t0_ = time.time()
        while time.time() - t0_ < 3.0 and not ("rgb" in _pcap and "depth" in _pcap and "info" in _pcap):
            rclpy.spin_once(n, timeout_sec=0.05)
        m = _pcap["rgb"]; d = _pcap["depth"]; info = _pcap["info"]
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
            _sp.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap_pose.py"), "/tmp/_pcap_gt.json"], capture_output=True, timeout=40)
            gt = json.load(open("/tmp/_pcap_gt.json"))
        except Exception as e_:
            print("[pcap] GT 스냅샷 실패", repr(e_))
        os.makedirs("/tmp/pick_cap", exist_ok=True); fn = f"/tmp/pick_cap/pg_{int(time.time()*1000)}.npz"
        np.savez_compressed(fn, rgb=rgb, depth=depth, K=K, T=T, q=np.array([st.get(j, 0.0) for j in ARM]), contact=np.array(cand.get("contact") or [0, 0, 0]), flange=np.array(cand.get("flange") or [0, 0, 0]),
                            quat=np.array(cand.get("quat") or [0, 0, 0, 1]), approach=np.array(cand.get("approach") or [0, 0, -1]), gt=json.dumps(gt), t=time.time())
        print(f"[pcap] 저장 {fn} depth 유효 {np.isfinite(depth).mean():.2f}")
    except Exception as e:
        print("[pcap] 실패", repr(e))


P = None
for pi, cand in enumerate(PLANS):
    if MERGE and not _PCAP:
        # 2026-10-08: transit(접근) 끝에서 멈췄다가 advance(하강)를 다시 쏘던 것을 한 궤적으로 이어서 실행(구간별 속도는 그대로). 중간 정지/정지 확인/재출발이 없다.
        _adv = _dedupe_join(cand["transit"], cand["advance"])
        _dts = [0.13 * _MOVE_DT] * len(cand["transit"]) + [0.20 * _MOVE_DT] * len(_adv)
        exec_traj(list(cand["transit"]) + _adv, dt=0.13 * _MOVE_DT, label=f"p{pi}-approach+descend", dts=_dts)
        _lap(f"p{pi}-approach+descend")
        err, ok = reached(cand["grasp_cfg"])
        print(f"[p{pi}] grasp_cfg 도달 오차 {err:.3f} rad -> {'OK' if ok else '미도달'} (합친 궤적)")
        if ok:
            P = cand; res["plan_used"] = pi; break
        if pi < len(PLANS) - 1:
            exec_traj([START_Q], dt=2.5 * _MOVE_DT, label=f"p{pi}-backoff")
            _lap(f"p{pi}-backoff")
        continue
    exec_traj(cand["transit"], dt=0.13 * _MOVE_DT, label=f"p{pi}-transit")
    _lap(f"p{pi}-transit")
    # 2026-09-29: transit 이 실패(-4 등)해 팔이 pre_cfg 에 못 갔는데 advance 를 그대로 쏘면
    # exec_traj 가 "현재자세 -> advance 웨이포인트" 로 앵커를 잡아 홈에서 grasp 자세로 순식간에
    # 튀어 주변 물체를 쓸어 넘어뜨린다(파일럿 seed101 육안+로그 확인). pre_cfg 근처일 때만 advance.
    _pe, _pok = reached(cand["pre_cfg"], tol=0.20)
    if not _pok:
        print(f"[p{pi}] transit 후 pre_cfg 미도달(오차 {_pe:.3f} rad) -- advance 생략(물체 충돌 방지)")
        if pi < len(PLANS) - 1:
            exec_traj([START_Q], dt=2.5 * _MOVE_DT, label=f"p{pi}-backoff")
            _lap(f"p{pi}-backoff")
        continue
    capture_pregrasp(cand)      # pre-grasp 호버에서 캡처(기본 꺼짐)
    exec_traj(cand["advance"], dt=0.20 * _MOVE_DT, label=f"p{pi}-advance")
    _lap(f"p{pi}-advance")
    err, ok = reached(cand["grasp_cfg"])
    print(f"[p{pi}] grasp_cfg 도달 오차 {err:.3f} rad -> {'OK' if ok else '미도달'}")
    if ok:
        P = cand; res["plan_used"] = pi; break
    if pi < len(PLANS) - 1:
        exec_traj([START_Q], dt=2.5 * _MOVE_DT, label=f"p{pi}-backoff")
        _lap(f"p{pi}-backoff")

if P is None:
    res["plan_used"] = None
    res["grip_after_close"] = res["grip_final"] = [0.0, 0.0]
    res["verdict"] = "NO_REACH"
    res["run_ts"] = D.get("run_ts") or time.strftime("%Y%m%d_%H%M%S")
    open("/tmp/exec_pick_res.json", "w").write(json.dumps(res, indent=1))
    try:
        _p = f"/tmp/{res['run_ts']}.exec.json"; open(_p, "w").write(json.dumps(res))
        subprocess.run(["rsync", "-q", _p, f"thor:grasp/runs/{res['run_ts']}.exec.json"], timeout=15)
    except Exception:
        pass
    print("[verdict] 모든 plan 이 grasp pose 도달 실패 -> NO_REACH")
    rclpy.shutdown(); raise SystemExit

res["at_grasp"] = {j: round(st.get(j, 0), 3) for j in ARM}
_lap("grasp_cfg 도달확인")
(_settle(1.0) if FAST else time.sleep(_H(0.4, 0.1)))      # 궤적 직후 컨트롤러 정착(FAST: 정지 확인으로 대체)
_lap("정착 sleep")
# 2026-09-17: close/squeeze 는 NO_PHYSICS 라도 속도보다 파지 신뢰성이 우선(사용자 실측 --
# transit 중 떨어뜨림). 다른 단계(open/settle/lift 후 hold)만큼 짧게 안 자르고 중간값
# 유지 + close_gripper() 의 tight->폴백 재시도까지 더해서 "완전히 닫혔는지"를 최대한
# 확보한다. NO_PHYS 아니면(실물) 기존 값 그대로.
_gc1 = close_gripper("grip-close", hold=_H(2.5, 1.0), tries=4)
_lap("grip-close")
res["grip_after_close"] = [round(st.get("gripper_joint1", 0), 4), round(st.get("gripper_joint2", 0), 4)]
# ⚠ 2026-09-17 버그이력: squeeze/hold 는 "이미 닫혀있는 상태에서 같은 명령을 다시
# 보내는 재확인"이라, grip()의 "이번 호출 동안 움직였나" 판정이 항상 False 로 나와서
# (물체 폭에 막혀 이미 정지해있으니 더 안 움직임 + tight 목표(0)과도 안 가까움) tight
# 4회 + 폴백 4회 재시도를 전부 헛돌며 9초 가까이 날렸다(실측, lap 계측으로 찾음).
# squeeze/hold 는 재시도 없이 1번만 보낸다 -- 안 움직여도 "이미 그 상태"인 게 정상.
close_gripper("grip-squeeze", hold=_H(1.2, 0.6), tries=1)          # 재-압박(1회, 재시도 없음)
_lap("grip-squeeze")
# 2026-09-30 판정 오류 수정(false SLIPPED): 첫 close 직후 값(grip_after_close)은 손가락이 아직 덜 닫힌 상태(예: 책 0.041~0.044)에서 읽히고,
# 곧이은 squeeze 가 물체 폭(책 3.3cm -> 0.0175)까지 실제로 닫는다. 예전엔 "덜 닫힌 값 -> lift 후 값" 을 비교해 1.2cm 넘게 줄었다고 SLIPPED 로 판정했다
# (빈 그리퍼는 ~0.003 이지 0.0173 이 아님. 책/병 SLIPPED 의 final 이 물체 반폭과 일치). 그래서 기준을 squeeze 이후 값으로 바꾼다.
# 원래 값은 grip_first_close 에 남긴다. NERO_VERDICT_OLD=1 이면 예전 동작.
if os.environ.get("NERO_VERDICT_OLD", "0") != "1":
    res["grip_first_close"] = res["grip_after_close"]
    res["grip_after_close"] = [round(st.get("gripper_joint1", 0), 4), round(st.get("gripper_joint2", 0), 4)]
# 여전히 많이 열려있으면(물체 폭에 막힌 게 아니라 아직 개방에 가까움) 한 번 더 시도 --
# transit 전 마지막 안전망. 이건 진짜 "아직 안 닫힘"이라 재시도 의미가 있음(tries 유지).
# 2026-09-18 (사용자 지시, waste time 제거): 문턱을 0.030 -> 0.040(OPEN_POS=0.045
# 바로 아래)으로 올림. 0.030은 HELD 판정 범위(HELD_MIN 0.006 ~ OPEN_POS 0.045) 안에
# 이미 충분히 들어가 있는 값이라, 박스처럼 물체 폭에 막혀 0.030 근처에서 멈춘 "이미
# 제대로 쥔" 케이스까지 재조임을 걸어서 3.84초를 헛돌리는 게 실측 확인됨(오늘 세션
# 반복 재현). 진짜 "거의 안 닫힘"만 여기 걸리도록 올림.
if st.get("gripper_joint1", 0.05) > 0.040:
    print(f"[grip-recheck] 아직 많이 열림({st.get('gripper_joint1',0):.3f}) -- 한 번 더 조임")
    close_gripper("grip-close2", hold=_H(1.5, 0.6), tries=2)
    _lap("grip-close2(recheck)")

# ⚠ 2026-09-17 버그이력: 여기 있던 exec_traj(P["retreat"], ...)(grasp->pre 후퇴)를
# 지운다. step6_base.py의 lift_z()는 "grasp_cfg 에서 곧바로" 올라가도록 계산돼있는데
# (주석: "후퇴 단계를 exec 에서 빼서 grasp 지점에서 바로 lift", 2026-09-11에 이미
# ros2_ws 사본에는 반영됐었으나 이 사본(nero/)엔 안 됨 -- 두 사본 분기 사고), retreat로
# 먼저 pre(보통 grasp보다 높은 위치)까지 올라간 다음 "grasp 기준"으로 계산된 lift를
# 재생하면, exec_traj가 현재 위치(pre, 이미 높음)에서 lift의 첫 waypoint(grasp 근처,
# 더 낮음)로 보간하면서 도로 내려갔다 다시 올라가는 현상이 실측 확인됨(사용자 지적).
# lift 는 grasp_cfg 에서 바로 이어지는 걸 전제로 하므로 retreat 없이 곧장 이어야 한다.

# lift: step6 가 grasp_cfg 에서 곧바로 계산한 Cartesian 수직 상승. 관절델타(j2/j4) lift 폐기
# (2026-09-10: 하드코딩 관절 lift 가 그리퍼를 기울여 물체 slide — 병 실측).
# 2026-09-18 (사용자 지시): LIFT_SCALE(_LIFT_DT) 스케일링 제거 -- lift dt는 이제
# 항상 고정 0.30(가장 빠른 검증된 값)으로, NERO_LIFT_SCALE을 뭘로 잡든 lift만 다시
# 느려지는 일이 없게 한다. LIFT_SCALE 변수 자체는 하위호환으로 남겨두되 여기선 안 씀.
#
# 2026-09-18 추가 -- "이상치" 정체 확정(사용자 지적 "조인트가 이리저리 움직이며
# 안정값 찾는 과정 같다"): 실측으로 box_1/2의 lift는 8웨이포인트, box_3는 52
# 웨이포인트였다(step6_base.py의 lift_z()가 부르는 GetCartesianPath가 grasp_cfg
# 근방 기구학이 꼬인 경우 훨씬 잘게/많이 쪼갠 경로를 반환 -- 실제로 그 52개 중
# joint1/joint3가 0.55→1.59→1.30, 0.47→-1.59→-0.80 식으로 크게 왕복함, 컨트롤러
# stall이 아니라 경로 자체가 그렇게 계산된 것). exec_traj의 dt가 "웨이포인트당
# 고정"이라 52×0.30=15.6s가 그대로 나왔다(실측 15.73s와 일치) -- 즉 dt를 고정해도
# 웨이포인트 개수가 늘면 총 시간이 그만큼 늘어나는 구조적 문제. LIFT_MAX_T로 총
# 시간에 상한을 걸어 웨이포인트가 몇 개든 lift가 항상 빠르게 끝나도록 한다.
LIFT_MAX_T = 3.0   # 초, lift 전체 목표 상한(8웨이포인트×0.30=2.4s 기준에 여유)
if P.get("lift"):
    _lift_wps = P["lift"]
    _lift_dt = min(0.30, LIFT_MAX_T / max(1, len(_lift_wps)))
    if _lift_dt < 0.30:
        print(f"[lift] 웨이포인트 {len(_lift_wps)}개 -> dt {_lift_dt:.4f}s로 압축"
              f"(총 {LIFT_MAX_T}s 상한, 그대로였으면 {len(_lift_wps)*0.30:.1f}s)")
    exec_traj(_lift_wps, dt=_lift_dt, label="lift-z")
else:
    print("[lift] Cartesian lift 없음 -> 폴백 관절 lift")
    cur = [st.get(j, 0.0) for j in ARM]
    _lf = cur[:]; _lf[1] -= 0.35; _lf[3] -= 0.20
    # 2026-09-18 (사용자 지시, 위험 감수): dt 2.0 -> 0.30 고정(LIFT_SCALE 미적용,
    # 정상 lift-z와 동일). 원래 2.0이던 이유는 2026-09-10 하드코딩 관절 lift가
    # 그리퍼를 기울여 물체를 slide시킨 사고 때문에 넣은 안전마진 -- 이 폴백 자체는
    # 여전히 같은 메커니즘이라 재발 위험 있음. 폴백 발생 원인(C단계가 Cartesian
    # lift를 못 만드는 경우)은 오늘 재현 안 됨(0/9 시도) -- 그 결과에 따라 다시 올릴 수 있음.
    exec_traj([_lf], dt=0.30, label="lift-fallback")
_lap("lift")
grip(GCLOSE, "grip-hold", hold=_H(1.5, 0.3), tries=1)   # 재확인성 재-명령, 재시도 불필요(위 버그 참고)
_lap("grip-hold")

res["final"] = {j: round(st.get(j, 0), 3) for j in ARM}
gf1 = st.get("gripper_joint1", 0.0)
res["grip_final"] = [round(gf1, 4), round(st.get("gripper_joint2", 0), 4)]
gc1 = res["grip_after_close"][0]
# 성공 판정: close 를 명령했는데 조가 "완전 폐합도 완전 개방도 아닌" 중간에 멈춤 = 손끝 사이에 물체.
# ★ 조가 완전 개방(≈0.05)이면 그리퍼가 놓아버린 것 -> held 아님 (이전 로직은 이걸 HELD 로 오판).
OPEN_POS = float(os.environ.get("NERO_OPEN_POS", "0.0492"))     # 2026-10-01 0.045 -> 0.0492: 폭 9cm 안팎의 넓은 물체(액션 피규어, 마우스)를 쥔 폭이 0.045 를 넘어 MISSED 로 오판됐다. 완전 개방은 0.0500.
held_close = bool(HELD_MIN < gc1 < OPEN_POS)
held_final = bool(HELD_MIN - 0.002 < gf1 < OPEN_POS)   # lift 후에도 물체가 조 사이에 있나
# 2026-09-29: 쥔 직후보다 조가 1.2cm 넘게 더 닫혔으면 물체가 빠진 것 (머스타드 0.035->0.0075 가 HELD 로 오판)
SLIP_DROP = float(os.environ.get("NERO_SLIP_DROP", "0.012"))
# 2026-09-30 판정 기준 v2: "닫힘 직후 대비 폭 감소(SLIP_DROP)" 기준을 폐기한다. 시뮬 정답과 대조한 461쌍에서 이 기준이 거짓 실패의 대부분(18건 중 14건)을 만들었고
# (정답 HELD 중앙값 감소 0.0026 대 정답 NOT 0.0029 -- 성공/실패를 구분하지 못함), 손떨림이나 물체가 손 안에서 돌아 눕는 경우를 슬립으로 읽었다.
# 같은 데이터에서 폭 범위만 보면 거짓 실패 4.4% -> 1.0%, 거짓 성공(안 들렸는데 HELD, 13/30)은 그대로. NERO_VERDICT_V1=1 이면 예전 기준(감소량 포함).
held_final_v1 = held_final and not (gc1 - gf1 > SLIP_DROP)
res["verdict_v1_drop_rule"] = "HELD" if (held_close and held_final_v1) else ("SLIPPED" if held_close else "MISSED")
if os.environ.get("NERO_VERDICT_V1", "0") == "1":
    held_final = held_final_v1
res["grip_series"] = GSER[-700:]
res["held"] = held_close and held_final
if held_close and held_final:
    res["verdict"] = "HELD"
elif held_close and not held_final:
    res["verdict"] = "SLIPPED"       # 닫힐 땐 물렸는데 lift 중 놓침
else:
    res["verdict"] = "MISSED"

# NERO_NO_PHYSICS=1 (2026-09-17, 사용자 지시): Isaac Sim 물체에 rigid body/충돌을
# 안 걸어서 그리퍼가 저항 없이 항상 완전폐합(GCLOSE)까지 감 -- held_close/held_final은
# "손끝 사이 중간에서 멈췄나"로 파지를 판정하므로 물리가 없으면 구조적으로 항상 False
# (=MISSED)만 나온다. 이건 파지 실패가 아니라 애초에 판정 불가능한 상황이다.
# grasp_cfg 에 실제로 도달해서(P is not None, NO_REACH 아님) 여기까지 온 것 = 계획된
# 궤적이 실제로 실행됐다는 뜻이므로 이걸 성공으로 처리한다. NO_REACH(궤적 자체가 grasp
# pose 에 못 감)는 물리와 무관한 진짜 실패라 이 오버라이드 대상이 아님(위에서 이미 분리 처리).
if NO_PHYS:
    res["verdict"] = "EXEC_OK"
    res["held"] = None   # 물리 없음 -> 실제 파지여부는 grip_after_close/grip_final 참고만
res["run_ts"] = D.get("run_ts") or time.strftime("%Y%m%d_%H%M%S")
open("/tmp/exec_pick_res.json", "w").write(json.dumps(res, indent=1))
# 계측: <ts>.exec.json 사이드카를 Thor runs/ 로 올림 (show.py 가 plan 레코드와 join)
try:
    _p = f"/tmp/{res['run_ts']}.exec.json"
    open(_p, "w").write(json.dumps(res))
    if FAST: subprocess.Popen(["rsync", "-q", _p, f"thor:grasp/runs/{res['run_ts']}.exec.json"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    # 기다리지 않는다
    else: subprocess.run(["rsync", "-q", _p, f"thor:grasp/runs/{res['run_ts']}.exec.json"], timeout=15)
except Exception as e:
    print("[runlog] 사이드카 업로드 실패:", e)
print(f"[verdict] after_close={gc1:.4f}  final={gf1:.4f}  ->  {res['verdict']}  "
      f"(HELD = 둘 다 > ~0.02; SLIPPED = 닫힘OK 후 lift 중 놓침)")
rclpy.shutdown()

