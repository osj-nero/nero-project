#!/usr/bin/env python3
"""curobo_c_server.py -- cuRobo 기반 C단계 서버 (PC GPU). step6_base.py(STOMP)의 대체.
POST /plan  JSON:
  start_q[7], candidates[{flange,pre,quat(xyzw),...}] (B 순위 순), obstacles[{id,xyz,raw,yaw}], target{xyz,dims}|null,
  margin(기본 0.01), tol(기본 0, 스윕 허용오차: 장애물 여유를 그만큼 줄임), transit_dt/advance_dt/lift_dt, lift(기본 0.10), max_cands(기본 4), diagnose(기본 true)
순서(B 순위를 존중): 후보마다 (1) 접근 자세로 계획(타깃 박스를 장애물로) -> (2) 타깃 박스를 빼고 접근->파지 직선 -> (3) 월드 +z 로 직선 lift. 처음 성공한 후보를 반환.
실패하면 diagnose: 이웃을 하나씩 뺐을 때 성공하는 물체를 blockers 로 돌려준다(discover 의 막힘 그래프용).
GET /health.   실행: venv curobo 파이썬으로 python curobo_c_server.py [--port 8020]"""
import os as _nc_os; CUROBO_DIR = _nc_os.environ.get("NERO_CUROBO_DIR") or _nc_os.path.expanduser("~/nero/curobo_nero")  # robot yml + failure dumps (override: NERO_CUROBO_DIR)
import json, math, sys, time, threading, argparse, os
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index

ROBOT = "" + CUROBO_DIR + "/nero7_open.yml"
TABLE_TOP = -0.02
LOCK = threading.Lock()
_BS = float(os.environ.get("NERO_CUROBO_BUDGET_SCALE", "1.0"))      # 2026-11-03(Thor 이관): 시간 예산(budget_s 계열) 배수. 기본 1.0 = 현행 동작 불변. 느린 GPU(Thor)에서 같은 후보 수를 보게 하려는 용도
P = None
_DEADLINE = {"t": 1e18}
_HARD = {"on": False}      # 2026-10-22(기본 꺼짐): pick 요청의 hard_deadline=True 일 때 후보 내부 루프(연속 IK/접근 사다리)도 마감에서 끊는다. solve() 가 요청마다 설정/해제
IKS = None
IKS_FREE = None
IK_BATCH = 128


def init_planner():
    cfg = MotionPlannerCfg.create(robot=ROBOT, scene_model={"cuboid": {"table": _table()}}, collision_cache={"obb": 24}, max_goalset=4)
    pl = MotionPlanner(cfg); pl.warmup(enable_graph=True, num_warmup_iterations=5)
    return pl


def _table(top=None):
    top = TABLE_TOP if top is None else top
    return {"dims": [1.6, 1.6, 0.05], "pose": [0.0, 0.0, top - 0.025, 1, 0, 0, 0]}


def _cuboids(obstacles, margin, tol, skip=(), table_top=None):
    out = {"table": _table(table_top)}
    for o in obstacles:
        if o["id"] in skip:
            continue
        raw = o.get("raw") or o["dims"]; yaw = float(o.get("yaw", 0.0))
        m = max(0.0, margin - 2.0 * tol)
        out[o["id"]] = {"dims": [raw[0] + m, raw[1] + m, raw[2] + m],
                        "pose": [o["xyz"][0], o["xyz"][1], o["xyz"][2], math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]}
    return out


def _goal(pos, quat_xyzw):
    x, y, z, w = quat_xyzw
    return GoalToolPose(tool_frames=P.tool_frames,
                        position=torch.tensor(pos, device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1, 3),
                        quaternion=torch.tensor([w, x, y, z], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1, 4))


def _js(q):
    return JointState.from_position(torch.tensor([q], device="cuda", dtype=torch.float32), joint_names=P.joint_names)


def _wps(r):
    ip = r.get_interpolated_plan()
    pos = ip.position.reshape(-1, ip.position.shape[-1])
    last = int(r.interpolated_last_tstep.view(-1)[0].item())
    return pos[: last + 1, :7].detach().cpu().numpy()      # 팔 7관절만(그리퍼 고정 관절 제외): exec/MoveIt 은 7축을 기대한다


_PRESCREEN = os.environ.get("NERO_INSERT_PRESCREEN", "1") == "1"      # 삽입 파지 후보 사전 선별
_AMIN = float(os.environ.get("NERO_APPROACH_DOT_MIN", "0.3"))      # 접근 정합(end_dot) 최소값
PLAN_VMAX = float(os.environ.get("NERO_PLAN_VMAX", "1.2")) if "os" in globals() else 1.2      # 2026-10-16: 재생 시 한 구간 최고 관절 속도 상한(rad/s). 이전엔 cuRobo 가 짧게 잡은 시간 그대로 재생해 이송이 2~3.4rad/s, 가속 60~126rad/s^2 였다


def _resample(w, dt_exec, dt_interp, t_min=0.0, vmax=None):
    w = np.asarray(w, float); T = (len(w) - 1) * dt_interp
    n = max(3, int(round(max(T, t_min) / dt_exec)) + 1)   # t_min: 재생 시간 하한(접근/들어올리기를 너무 빠르게 재생하지 않는다)
    vm = PLAN_VMAX if vmax is None else vmax
    if vm and len(w) > 1:
        m = float(np.abs(np.diff(w[:, :7], axis=0)).max()); n = max(n, int(np.ceil(m * (len(w) - 1) / (vm * dt_exec))) + 1)      # 구간당 최대 변화/재생 간격 <= vmax
    pos = np.linspace(0, len(w) - 1, n); i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, len(w) - 1); f = (pos - i0)[:, None]
    out = w[i0] * (1 - f) + w[i1] * f
    return [list(map(float, r)) for r in out], T


SANE_CFG = dict(wander=0.6, swing_short=1.0, swing_transit=3.5)      # 궤적 안정장치 기본 임계(rad). 요청의 sane_* 로 덮어쓸 수 있고 sane=False 로 끌 수 있다


def _traj_sane(phases, req=None):
    """궤적 안정장치(2026-10-02, 사용자 제보: 팔이 원을 3~4번 그리며 도는 위험한 궤적이 선택됨).
    phases: {이름: (T,>=7) 관절 궤적}. (1) 돌아감: 관절별 총변동 - 순변위(낭비)가 wander 초과, (2) 한 축 큰 회전: 순변위가 짧은 구간(하강/후퇴) swing_short, 이송 swing_transit 초과 -> 불안정.
    반환 (ok, 사유, 지표)."""
    req = req or {}
    if req.get("sane") is False:
        return True, None, {}
    cfg = dict(SANE_CFG); cfg.update({k[5:]: float(v) for k, v in req.items() if k.startswith("sane_") and k != "sane"})
    metrics = {}
    clear = float(req.get("held_clear", 0.03))       # 든 물체가 이송 중 테이블에서 떨어져야 하는 최소 높이(m). 0 이면 끔. 잠정값: 계획-실행 오차(관절 추적 오차 최대 0.17rad)와 접촉 허용 mm 단위를 고려한 여유
    if clear > 0 and "transit" in phases:
        sph_ = _held_from_req(req)
        if sph_ is not None and len(phases["transit"]) > 1:
            low = _held_min_height(phases["transit"], sph_); metrics["held_low_cm"] = round(100 * low, 1)
            if low < clear:
                return False, "이송 중 든 물체가 테이블에서 %.1fcm (< %.0fcm)" % (100 * low, 100 * clear), metrics
    for nm, w in phases.items():
        q = np.asarray(w, float)[:, :7]
        if len(q) < 2:
            continue
        tv = np.abs(np.diff(q, axis=0)).sum(0); net = np.abs(q[-1] - q[0]); wander = float((tv - net).max()); swing = float(net.max())
        metrics[nm] = dict(wander=round(wander, 2), swing=round(swing, 2), tv=round(float(tv.sum()), 2))
        if wander > cfg["wander"]:
            return False, "%s 궤적이 돌아감(관절 낭비 %.2frad)" % (nm, wander), metrics
        lim = cfg["swing_transit"] if nm == "transit" else cfg["swing_short"]
        if swing > lim:
            return False, "%s 관절 한 축이 %.2frad 크게 회전" % (nm, swing), metrics
    return True, None, metrics


PLAN_RETRY = int(os.environ.get("NERO_CUROBO_PLAN_RETRY", "3")) if "os" in globals() else 3        # 호출당 최대 시도 횟수(1 = 재시도 없음, 기존 동작)
RETRY_BUDGET = {"n": 0}                                                                          # 요청당 총 추가 시도 상한(요청 시작마다 채운다)
RETRY_BUDGET_DEFAULT = int(os.environ.get("NERO_CUROBO_RETRY_BUDGET", "0")) if "os" in globals() else 0         # 기본 0 = 재시도 없음. 연속 IK 도입 뒤에는 재시도의 측정 가능한 이득이 없어 OFF(삭제하지 않음)


def _plan_pose_r(goal, start, tries=None):
    """plan_pose 재시도 래퍼(공통: pick/place/insert). trajopt 는 확률적으로 실패(결과 None 또는 success 없음)하는데(저장된 삽입 실패 21건: 6회 중 18건이 한 번은 성공), 결정적으로 막힌
    후보에 재시도를 쓰면 시간 예산(8초)이 줄어 시도하는 후보 수가 줄어든다(일반 place N0: NO_PLAN 시 시도 후보 4~6개 -> 2~4개). 그래서 재시도는 요청당 총 예산(RETRY_BUDGET)만큼만 쓴다:
    후보가 적을 때(삽입)는 구제가 되고, 후보가 많을 때(일반 place)는 예산이 금방 바닥나 기존과 같게 움직인다."""
    n = max(1, PLAN_RETRY if tries is None else int(tries)); r = None
    for k in range(n):
        if k > 0:
            if RETRY_BUDGET["n"] <= 0:
                break
            RETRY_BUDGET["n"] -= 1
        r = P.plan_pose(goal, start)
        if _ok(r) or time.time() > _DEADLINE["t"]:
            break
    return r


def _reset_retry_budget(req):
    RETRY_BUDGET["n"] = int(req.get("retry_budget", RETRY_BUDGET_DEFAULT))


def _adaptive_retry_budget(n_feasible, req):
    """공통 규칙: 대안 후보가 적을수록 재시도를 쓴다(후보가 많으면 다른 후보를 시도하는 것이 낫다). 예산 = 8 - IK 통과 후보 수, 0~RETRY_BUDGET_DEFAULT.
    일반 place(IK 통과 16~60개) -> 0, 삽입(1~8개) -> 최대 6. 요청의 retry_budget 으로 고정할 수 있다."""
    RETRY_BUDGET["n"] = int(req["retry_budget"]) if "retry_budget" in req else max(0, min(RETRY_BUDGET_DEFAULT, 8 - int(n_feasible)))


def _ik_chain(chains, R=8, max_jump=0.6):
    """연속 IK(공통): 웨이포인트마다 독립으로 풀면 7축 팔의 해 분기가 달라 연속 경로가 안 된다. 웨이포인트마다 해를 R개씩 받아(return_seeds) 체인별로 관절 점프가 최소인 조합을
    동적계획법으로 고른다(공개 인자만 사용: current_state/seed_config 를 직접 주면 시드 수가 안 맞아 CUDA 오류). chains: [[{flange,quat}, ...]]. 반환: 체인별 [q|None...] (점프가 max_jump 를 넘으면 None)."""
    B = IK_BATCH; n = len(chains); out = [None] * n
    for i0 in range(0, n, B):
        grp = chains[i0:i0 + B]; m = len(grp); L = len(grp[0]); tf = IKS.tool_frames[0]; sols = [[None] * L for _ in range(m)]
        for k in range(L):
            poses = [grp[i][k] for i in range(m)]; ch = poses + [poses[-1]] * (B - m)
            pos = torch.tensor([p_["flange"] for p_ in ch], device="cuda", dtype=torch.float32)
            qu = torch.tensor([[p_["quat"][3], p_["quat"][0], p_["quat"][1], p_["quat"][2]] for p_ in ch], device="cuda", dtype=torch.float32)
            r = IKS.solve_pose(GoalToolPose.from_poses({tf: Pose(position=pos, quaternion=qu)}, num_goalset=1), return_seeds=R)
            ok = r.success.reshape(B, -1).cpu().numpy(); sol = r.solution.reshape(B, ok.shape[1], -1)[:, :, :7].detach().cpu().numpy()
            for i in range(m):
                sols[i][k] = [sol[i, j] for j in range(ok.shape[1]) if ok[i, j]]
        for i in range(m):
            if any(len(x) == 0 for x in sols[i]):
                continue
            cost = [np.zeros(len(sols[i][0]))]; back = []
            for k in range(1, L):
                A = np.array(sols[i][k - 1]); Bm = np.array(sols[i][k]); dd = np.abs(Bm[:, None, :] - A[None, :, :]).max(axis=2)
                tot = dd + cost[-1][None, :]; j = tot.argmin(axis=1); back.append(j); cost.append(tot[np.arange(len(Bm)), j])
            kk = int(np.argmin(cost[-1])); path = [kk]
            for k in range(L - 2, -1, -1):
                kk = int(back[k][kk]); path.append(kk)
            path = path[::-1]; qs = [sols[i][k][path[k]] for k in range(L)]
            if L < 2 or float(np.abs(np.diff(np.array(qs), axis=0)).max()) <= max_jump:
                out[i0 + i] = [[float(x) for x in q] for q in qs]
    return out


_JLIM = {"v": None}


def _chain_ik_one(wps, quat, start_q=None, rep=3, cap=96, max_jump=0.9, margin_w=2.0, margin_ok=0.2, start_w=1.0):
    """2026-10-08 pick 용 연속 IK(접근 -> 파지 -> 들어올림). wps: [{flange}, ...] 같은 자세(quat)로 웨이포인트마다 IK 해를 모아(배치 복사본 전체 x rep 회) 동적계획법으로 고른다:
    인접 웨이포인트 해의 최대 관절차(max_jump 이하만)가 작고, 관절 한계에서 margin_ok(rad) 이내로 붙은 해는 벌점(margin_w x 모자란 만큼).
    목적: 파지 자세를 정적 IK 한 개로만 고르면(7축 여유 자유도의 임의 분기) 그 분기에서 자세를 유지한 채 위로 들어올릴 연속 경로가 없을 수 있다(손목 j6/j7 이 한계 0.15rad 이내). 반환 [q_pre, q_grasp, q_lift] 또는 None."""
    B = IK_BATCH; tf = IKS.tool_frames[0]; sols = []; rng = np.random.default_rng(0)
    if _JLIM["v"] is None:
        kc = P.kinematics.kinematics_config; _JLIM["v"] = kc.joint_limits.position.detach().cpu().numpy()[:, :7]
    lo, hi = _JLIM["v"][0], _JLIM["v"][1]
    for w in wps:
        pos = torch.tensor([list(w["flange"])] * B, device="cuda", dtype=torch.float32); qu = torch.tensor([[quat[3], quat[0], quat[1], quat[2]]] * B, device="cuda", dtype=torch.float32); acc = []
        for _ in range(rep):
            r = IKS.solve_pose(GoalToolPose.from_poses({tf: Pose(position=pos, quaternion=qu)}, num_goalset=1), return_seeds=8)
            ok = r.success.reshape(B, -1).cpu().numpy(); so = r.solution.reshape(B, ok.shape[1], -1)[:, :, :7].detach().cpu().numpy(); acc.append(so[ok])
        S = np.concatenate(acc, axis=0)
        if len(S) == 0:
            return None
        if len(S) > cap:
            S = S[rng.choice(len(S), cap, replace=False)]
        sols.append(S)
    pen = [margin_w * np.maximum(0.0, margin_ok - np.minimum(S - lo, hi - S).min(axis=1)) for S in sols]
    if start_q is not None:      # 현재 팔 자세에서 가까운 분기의 접근 구성을 선호한다(멀리 돌아가는 분기는 접근 경로가 물체를 스친다)
        pen[0] = pen[0] + start_w * np.abs(sols[0] - np.asarray(start_q[:7], float)[None, :]).max(axis=1)
    cost = [pen[0]]; back = []
    for k in range(1, len(sols)):
        dd = np.abs(sols[k][:, None, :] - sols[k - 1][None, :, :]).max(axis=2); dd = np.where(dd > max_jump, 1e6, dd)
        tot = dd + cost[-1][None, :]; j = tot.argmin(axis=1); back.append(j); cost.append(tot[np.arange(len(sols[k])), j] + pen[k])
    kk = int(np.argmin(cost[-1]))
    if cost[-1][kk] >= 1e5:
        return None
    path = [kk]
    for k in range(len(sols) - 2, -1, -1):
        kk = int(back[k][kk]); path.append(kk)
    path = path[::-1]
    return [sols[k][path[k]].tolist() for k in range(len(sols))]


def _slot_blocker(slot, rad, obstacles, dest_ids, min_h=0.008):
    """슬롯(xy) 위에 든 물체(반경 rad)를 놓으면 겹치는 장애물 id (없으면 None). 장애물 윤곽(회전 사각형)까지의 2D 거리 < rad 이면 겹친다.
    도착지 자신(dest_ids)과 높이가 min_h 미만인 납작한 것은 무시한다. 일반 기하 규칙: 다른 물체가 있는 자리에는 놓지 않는다."""
    for o in obstacles:
        if o["id"] in dest_ids:
            continue
        raw = o.get("raw") or o["dims"]
        if float(raw[2]) < min_h:
            continue
        yaw = float(o.get("yaw", 0.0)); dx = slot[0] - o["xyz"][0]; dy = slot[1] - o["xyz"][1]
        c, s_ = math.cos(-yaw), math.sin(-yaw); lx = c * dx - s_ * dy; ly = s_ * dx + c * dy
        qx = max(abs(lx) - raw[0] / 2.0, 0.0); qy = max(abs(ly) - raw[1] / 2.0, 0.0)
        if math.hypot(qx, qy) < rad:
            return o["id"]
    return None


def _held_from_req(req):
    """든 물체를 덮는 구들(플랜지 좌표계, (N,4))을 요청의 파지/막대(삽입) 또는 물체(place)에서 다시 만든다. 계획기 파일을 건드리지 않고 안전장치가 쓰게 하려는 용도."""
    try:
        g = req.get("grasp")
        if not g:
            return None
        R0 = Rot.from_quat([float(x) for x in g["quat"]]); FL0 = np.array(g["flange"], float); pg = req.get("peg"); pad = float(req.get("sphere_pad", 0.0 if pg else 0.004))
        if pg and pg.get("axis") is not None:
            ax = np.array(pg["axis"], float); ax /= np.linalg.norm(ax); L = float(pg["length"]); c = np.array(pg["center"], float); r = float(np.clip(max(pg.get("cross", (0.03, 0.03))) / 2.0 + pad, 0.012, 0.035))
            span = max(L / 2.0 - r, 0.0); ts = np.linspace(-span, span, max(1, int(math.ceil(2 * span / (1.2 * r))) + 1)) if span > 0 else np.zeros(1)
            pf = R0.inv().apply(c[None, :] + ts[:, None] * ax[None, :] - FL0[None, :]); return np.concatenate([pf, np.full((len(pf), 1), r)], axis=1)
        ob = req.get("obj")
        if ob:
            c = ob["centre"]; return _held_spheres(R0, FL0, [c[0], c[1], c[2]], ob["dims"], float(ob.get("yaw", 0.0)), pad=pad, cover=req.get("held_cover", "old"))
    except Exception:
        return None
    return None


def _held_min_height(q, sph):
    """관절 경로 q (T,>=7) 위에서 든 물체 구들의 최저 높이(구 아래 끝, m)."""
    qt = torch.tensor(np.asarray(q, float)[:, :7], device="cuda", dtype=torch.float32)
    st = P.kinematics.compute_kinematics(JointState.from_position(qt, joint_names=P.joint_names)); lp = st.tool_poses.get_link_pose(P.tool_frames[0])
    pos = lp.position.reshape(-1, 3).detach().cpu().numpy(); qq = lp.quaternion.reshape(-1, 4).detach().cpu().numpy(); R = Rot.from_quat(np.c_[qq[:, 1:], qq[:, 0]])
    low = 9.0
    for i in range(len(pos)):
        z = pos[i][2] + R[i].apply(sph[:, :3])[:, 2] - sph[:, 3]; low = min(low, float(z.min()))
    return low


def _end_state(r):
    """plan_grasp 와 같은 방식으로 다음 단계 시작 상태를 만든다(활성 관절만)."""
    return P.kinematics.get_active_js(get_joint_state_at_horizon_index(r.js_solution, -1).squeeze(0))


def _reject_code(w):
    """why 문구 -> 거부 단계 코드(분석용 분류, 동작에 쓰지 않음)."""
    w = str(w)
    for key, code in (("연속 IK 없음", "chain_ik"), ("접근 자세 계획 실패", "approach_plan"), ("직선 계획 실패", "linear_plan"), ("대상과 충돌", "approach_target"), ("링크", "link_target"), ("접근 정합 낮음", "end_dot"), ("place 불가", "place"), ("시간 예산 초과", "deadline"), ("들어올림", "lift")):
        if key in w: return code
    return "other"


def _ok(r):
    return r is not None and r.success is not None and bool(r.success.any())


def _reach_order(variants, quat, enabled=True):
    """사다리 병렬화(2026-10-04): 단계(variants: [(이름, flange 위치)] 선호 순서)마다 순차로 궤적 계획(실패 한 번에 수백 ms~초)을 시도하는 대신, 모든 단계의 끝 자세를 IK 배치 한 번으로 풀어
    도달 가능한 단계를 앞으로, 불가능한 단계를 뒤로 보낸다(선호 순서는 각 그룹 안에서 유지, 불가능 단계도 지우지 않고 마지막에 시도). 반환: 재정렬된 인덱스 목록."""
    n = len(variants)
    if not enabled or n < 2 or os.environ.get("NERO_LADDER_SCREEN", "1") != "1" or os.path.exists("/tmp/nero_ladder_off"):      # 파일 /tmp/nero_ladder_off 가 있으면 끔(조건 전환용, 서버 재시작 불필요)
        return list(range(n))
    try:
        ik = solve_ik({"poses": [dict(flange=[float(x) for x in p], quat=[float(x) for x in quat]) for _, p in variants], "check_collision": False})["q"]
    except Exception as e:
        print("사다리 IK 선별 실패(무시)", repr(e), flush=True); return list(range(n))
    good = [i for i in range(n) if ik[i] is not None]; bad = [i for i in range(n) if ik[i] is None]
    return good + bad


_LINK_MASK = {"m": None}
_LASTMIN = {"i": None}


def _link_target_min_dist(W, box_center, box_dims, fingers=False):
    """2026-10-08: 파지 직전 직선 접근(대상을 월드에서 뺀 구간)에서 '손가락(gripper_link1/2) 외' 링크 충돌 구체가 대상 상자와 얼마나 가까운지.
    W: 관절 배치 목록. 반환: 모든 웨이포인트·금지 구체에 대한 부호 있는 최소 거리(m, 음수=상자 안으로 파고듦). 손가락과 attached_object 구체는 제외한다."""
    kc = P.kinematics.kinematics_config
    _mk = "mf" if fingers else "m"      # fingers=True: 손가락 구체도 포함(접근 구간 r1 검사용)
    if _LINK_MASK.get(_mk) is None:
        inv = {v: k for k, v in kc.link_name_to_idx_map.items()}
        ok_links = {"attached_object", "world"} if fingers else {"gripper_link1", "gripper_link2", "attached_object", "world"}
        _LINK_MASK[_mk] = torch.tensor([inv.get(int(i), "?") not in ok_links for i in kc.link_sphere_idx_map.tolist()], device="cuda")
    q = torch.tensor([list(map(float, w[:7])) for w in W], device="cuda", dtype=torch.float32)
    st = P.kinematics.compute_kinematics(JointState.from_position(q, joint_names=P.joint_names))
    rs = st.robot_spheres.reshape(q.shape[0], -1, 4)
    sel = _LINK_MASK[_mk] & (rs[0, :, 3] > 0.0)
    cen = rs[:, sel, :3]; rad = rs[:, sel, 3]
    c = torch.tensor(box_center, device="cuda", dtype=torch.float32); h = torch.tensor(box_dims, device="cuda", dtype=torch.float32) / 2.0
    qd = (cen - c).abs() - h
    d = qd.clamp(min=0).norm(dim=-1) + qd.max(dim=-1).values.clamp(max=0) - rad
    _LASTMIN["i"] = None
    try:
        _fl = int(d.argmin()); _w, _s = divmod(_fl, d.shape[1]); _ids = sel.nonzero().flatten(); _sid = int(_ids[_s]); inv_ = {v: k for k, v in kc.link_name_to_idx_map.items()}
        _LASTMIN["i"] = dict(wp=_w, n_wp=int(d.shape[0]), link=inv_.get(int(kc.link_sphere_idx_map[_sid]), "?"))      # 디버그: 가장 파고든 웨이포인트/링크
    except Exception:
        pass
    return float(d.min())


def _dbg_lift_variants(c, start3, quat, scene_no_target, r2):
    """lift 실패 진단: 같은 시작 상태(start3)에서 세계/목표를 바꿔 가며 계획해 본다."""
    import copy as _cp
    fl = np.array(c["flange"], float); a = np.array(c["approach"], float); a = a / (np.linalg.norm(a) + 1e-9); out = {}
    def tryplan(tag, scene, goal_pos, linear=False):
        P.update_world(Scene.create(scene))
        if linear: P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=False) for k in P.tool_frames})
        try: r = _plan_pose_r(_goal(list(goal_pos), quat), start3, tries=1)
        finally: P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
        out[tag] = bool(_ok(r))
    base = _cp.deepcopy(scene_no_target); cub = base["cuboid"]
    tryplan("현상태_제자리(0cm)", base, fl)
    tryplan("현상태_위5cm_비직선", base, fl + [0, 0, 0.05])
    tryplan("현상태_위2cm_비직선", base, fl + [0, 0, 0.02])
    tryplan("현상태_접근반대5cm_비직선", base, fl - a * 0.05)
    t_only = _cp.deepcopy(base); t_only["cuboid"] = {"table": cub["table"]}
    tryplan("테이블만_위5cm", t_only, fl + [0, 0, 0.05]); tryplan("테이블만_위10cm", t_only, fl + [0, 0, 0.10])
    low = _cp.deepcopy(t_only); low["cuboid"]["table"]["pose"][2] -= 0.04
    tryplan("테이블4cm내림_위5cm", low, fl + [0, 0, 0.05]); tryplan("테이블4cm내림_제자리", low, fl)
    far = {"cuboid": {"far": {"dims": [0.05, 0.05, 0.05], "pose": [3.0, 3.0, 1.0, 1, 0, 0, 0]}}}
    tryplan("세계없음_위5cm", far, fl + [0, 0, 0.05]); tryplan("세계없음_위10cm", far, fl + [0, 0, 0.10]); tryplan("세계없음_제자리", far, fl)
    def _fkq(q7):
        st_ = P.kinematics.compute_kinematics(JointState.from_position(torch.tensor([list(q7)], device="cuda", dtype=torch.float32), joint_names=P.joint_names)); lp_ = st_.tool_poses.get_link_pose(P.tool_frames[0])
        return lp_.position.reshape(-1).detach().cpu().numpy(), lp_.quaternion.reshape(-1).detach().cpu().numpy()
    q0_ = start3.position.reshape(-1)[:7].tolist(); p0_, qq0_ = _fkq(q0_)
    for w in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02):      # 진단: 위 10cm, 회전 가중치 w (위치 가중치 1). 되면 실제로 얼마나 기울었는지 FK 로 잰다
        tag = f"회전가중치{w}_위10cm"
        try:
            P.update_world(Scene.create(base)); P.update_tool_pose_criteria({k: ToolPoseCriteria(terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, w, w, w]) for k in P.tool_frames})
            r_ = _plan_pose_r(_goal((fl + [0, 0, 0.10]).tolist(), quat), start3, tries=1); ok_ = bool(_ok(r_))
            if ok_:
                pe_, qe_ = _fkq(_end_state(r_).position.reshape(-1)[:7].tolist()); out[tag] = dict(ok=True, 기울기도=round(math.degrees(2 * math.acos(min(1.0, abs(float(qq0_ @ qe_))))), 1), 위치변화cm=[round(float(x) * 100, 1) for x in (pe_ - p0_)])
            else: out[tag] = dict(ok=False)
        except Exception as e_: out[tag] = "예외 " + repr(e_)[:50]
        finally: P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
    try:      # 진단: 파지 자세 IK 해를 많이(64) 받아, 위 10cm 자세의 해와 관절 거리(최대 한 축)가 가장 가까운 짝이 얼마나 가까운지 -- 연속 경로의 존재 여부
        tf_ = IKS.tool_frames[0]; B_ = IK_BATCH; kc = P.kinematics.kinematics_config; lim = kc.joint_limits.position.detach().cpu().numpy() if hasattr(kc, "joint_limits") else None
        def _sols(pos_, R_):      # return_seeds 는 기존 사용값(8)만 쓴다(다른 값은 CUDA 오류로 서버 컨텍스트가 깨진 전례). 여러 번 불러 해를 모은다
            pos_t = torch.tensor([pos_] * B_, device="cuda", dtype=torch.float32); qu_t = torch.tensor([[quat[3], quat[0], quat[1], quat[2]]] * B_, device="cuda", dtype=torch.float32); acc = []
            for _ in range(6):
                r_ = IKS.solve_pose(GoalToolPose.from_poses({tf_: Pose(position=pos_t, quaternion=qu_t)}, num_goalset=1), return_seeds=8)
                ok_ = r_.success.reshape(B_, -1).cpu().numpy(); so_ = r_.solution.reshape(B_, ok_.shape[1], -1)[:, :, :7].detach().cpu().numpy(); acc.append(so_[ok_])
            return np.concatenate(acc, axis=0)
        G_ = _sols(fl.tolist(), 8); U_ = _sols((fl + [0, 0, 0.10]).tolist(), 8)
        if len(G_) and len(U_):
            D_ = np.abs(U_[:, None, :] - G_[None, :, :]).max(axis=2); i_, j_ = np.unravel_index(D_.argmin(), D_.shape)
            lo_, hi_ = (lim[0][:7], lim[1][:7]) if lim is not None else (None, None)
            out["IK해수(파지/위10cm)"] = [len(G_), len(U_)]; out["가장가까운_짝의_최대관절차(rad)"] = round(float(D_.min()), 2)
            out["그_파지해_한계여유"] = [round(float(min(G_[j_][k] - lo_[k], hi_[k] - G_[j_][k])), 2) for k in range(7)] if lo_ is not None else None
            out["시작상태(start3)_와_가장가까운_파지해_차"] = round(float(np.abs(G_ - np.array(q0_)).max(axis=1).min()), 2)
    except Exception as e_: out["IK짝진단예외"] = repr(e_)[:80]
    try:      # 진단: 파지 자세 관절의 한계 여유와, [접근, 파지, 위 10cm] 를 잇는 연속 IK 해가 있는지
        kc = P.kinematics.kinematics_config; lim = kc.joint_limits.position.detach().cpu().numpy() if hasattr(kc, "joint_limits") else None
        q3_ = np.array(start3.position.reshape(-1)[:7].tolist())
        if lim is not None:
            lo, hi = lim[0][:7], lim[1][:7]; out["파지자세_한계여유(rad)"] = [round(float(min(q3_[i] - lo[i], hi[i] - q3_[i])), 2) for i in range(7)]
        ch = _ik_chain([[dict(flange=list(c["pre"]), quat=list(quat)), dict(flange=fl.tolist(), quat=list(quat)), dict(flange=(fl + [0, 0, 0.10]).tolist(), quat=list(quat))]])[0]
        out["연속IK(접근,파지,위10cm)"] = None if ch is None else [[round(float(x), 2) for x in q] for q in ch]
        if ch is not None and lim is not None:
            out["연속IK_파지_한계여유"] = [round(float(min(ch[1][i] - lo[i], hi[i] - ch[1][i])), 2) for i in range(7)]
    except Exception as e_:
        out["진단예외"] = repr(e_)[:80]
    jl = P.kinematics.get_joint_limits() if hasattr(P.kinematics, "get_joint_limits") else None
    q3 = [round(float(x), 3) for x in (start3.position.reshape(-1)[:7].tolist() if hasattr(start3, "position") else [])]
    print("DBG_LIFT", json.dumps(dict(flange=[round(float(x), 3) for x in fl], approach=[round(float(x), 2) for x in a], start3_q=q3, res=out), ensure_ascii=False), flush=True)


def _tcp_path(w):
    q_ = torch.tensor(np.asarray(w)[:, :7], device="cuda", dtype=torch.float32)
    return P.kinematics.compute_kinematics(JointState.from_position(q_, joint_names=P.joint_names)).tool_poses.get_link_pose(P.tool_frames[0]).position.reshape(-1, 3).detach().cpu().numpy()


def _approach_quality(w, a, tail=0.06):
    """2026-10-16: 이송 경로의 '접근 정합' 지표. end_dot = 끝 tail(m) 구간의 TCP 진행 방향과 접근 벡터의 내적(1 = 접근 방향으로 들어옴, <0 = 반대로 들어옴), cusps = 진행 방향이 90도 넘게 꺾이는 횟수(뚝 끊기는/되돌아가는 선), len_ratio = 경로 길이/직선 거리."""
    try:
        X = _tcp_path(w); d = np.linalg.norm(np.diff(X, axis=0), axis=1); L = float(d.sum())
        if L < 1e-4: return None
        cum = np.concatenate([[0], np.cumsum(d)]); k = int(np.searchsorted(cum, L - tail)); k = min(k, len(X) - 2)
        v = X[-1] - X[k]; end_dot = float(np.dot(v / (np.linalg.norm(v) + 1e-9), a))
        Xs = X[np.unique(np.searchsorted(cum, np.arange(0, L, 0.03)))]
        T_ = np.diff(Xs, axis=0); T_ = T_ / (np.linalg.norm(T_, axis=1, keepdims=True) + 1e-9)
        cusps = int(((T_[1:] * T_[:-1]).sum(1) < 0.0).sum()) if len(T_) > 1 else 0
        return dict(end_dot=round(end_dot, 2), cusps=cusps, len_ratio=round(float(L / (np.linalg.norm(X[-1] - X[0]) + 1e-9)), 2), path_cm=round(float(100 * L), 1))
    except Exception as e:
        print("접근 지표 예외", repr(e), flush=True); return None


def _try_candidate(c, start_q, scene_all, scene_no_target, lift, dts, tmin):
    """2026-10-16: 먼저 긴 직선 진입(far)으로 시도하고, 안 되면 예전 사다리(표준 거리부터)로 한 번 더. far 만 막힌 후보를 버리지 않는다."""
    _far = float(os.environ.get("NERO_PICK_FAR", "1.0"))
    _amin = float(os.environ.get("NERO_APPROACH_DOT_MIN", "0.3")); best = None
    if _far > 1.0:
        ok_, pl_ = _try_candidate_impl(dict(c, _far=_far), start_q, scene_all, scene_no_target, lift, dts, tmin)
        if ok_:
            if ((pl_.get("approach_quality") or {}).get("end_dot", 1.0)) >= _amin: return ok_, pl_
            best = pl_
    ok2, pl2 = _try_candidate_impl(dict(c, _far=1.0), start_q, scene_all, scene_no_target, lift, dts, tmin)
    if ok2 and (best is None or ((pl2.get("approach_quality") or {}).get("end_dot", 1.0)) > ((best.get("approach_quality") or {}).get("end_dot", 1.0))): return ok2, pl2
    if best is not None: return True, best
    return ok2, pl2


def _try_candidate_impl(c, start_q, scene_all, scene_no_target, lift, dts, tmin):
    """한 후보의 3단계. (ok, plan_dict | reason)"""
    dt_i = P.trajopt_solver.config.interpolation_dt
    quat = c["quat"]
    _r1nt = os.environ.get("NERO_R1_NO_TARGET", "0") == "1" and bool((scene_all.get("cuboid") or {}).get("target_obj"))      # 2026-10-08: 접근(r1)을 대상 상자 없이 계획하고 뒤에서 링크 거리로 검사(상자 근처 파지 직전 자세가 플래너 여유 때문에 막히는 문제)
    P.update_world(Scene.create(scene_no_target if _r1nt else scene_all))
    a = np.array(c["approach"], float); a = a / (np.linalg.norm(a) + 1e-9); fl = np.array(c["flange"], float); so = float(c.get("standoff", 0.07))
    # 접근 거리 사다리(1.0 -> 0.7/0.5/0.35): 7cm 뒤 자세가 팔 도달 한계 밖인 후보(비COCO 시드 21/22)가 있다. 끝 자세 IK 를 배치로 한 번에 풀어 도달 가능한 가장 긴 접근부터 계획한다.
    _far = float(c.get("_far", 1.0))      # 2026-10-16: 접근 방향으로 더 긴 직선 진입(기본 표준 거리의 1.7배 = 약 12cm 뒤에서 시작). 이송 끝 구간이 접근 벡터 방향으로 들어오게 한다(도달 안 되면 아래 사다리로 1.0/0.7... 로 내려간다)
    _fr = ((_far,) if _far > 1.0 else ()) + (1.0, 0.7, 0.5, 0.35); _vars = [(f, list(c["pre"]) if f == 1.0 else list(fl - a * so * f)) for f in _fr]
    r1 = None; pre_used = 1.0; _chain = None; _hard_hit = False
    if os.environ.get("NERO_PICK_CHAIN", "1") == "1":      # 2026-10-08: 접근-파지-들어올림을 같은 분기로 잇는 연속 IK 를 먼저 구하고, 접근 계획의 목표를 그 분기의 관절 구성으로 준다
        try:
            _g = dict(flange=fl.tolist()); _lf = None      # 1) 파지-들어올림 두 점만으로 먼저 가능성을 싸게 본다(없으면 접근 사다리를 돌 필요 없이 바로 거부)
            for lf_ in (lift, 0.07, 0.05):
                _up = dict(flange=[fl[0], fl[1], fl[2] + lf_])
                _c2 = _chain_ik_one([_g, _up], quat, rep=3, cap=600)
                if _c2 is not None:
                    _lf = lf_; break
            if os.environ.get("NERO_DBG_LIFT") == "1": print("DBG_CHAIN 1단계(파지-들어올림) 들어올림 거리 =", _lf, flush=True)
            if _lf is not None:      # 2) 접근 자세까지 잇는 3점 체인 (현재 팔 자세에 가까운 분기를 선호). 접근 사다리(1.0/0.7/0.5/0.35)와 들어올림 거리(1단계에서 된 값부터 짧은 쪽)를 돌며 처음 이어지는 것
                for _i in _reach_order(_vars, quat):
                    if _HARD["on"] and time.time() > _DEADLINE["t"]: _hard_hit = True; break
                    f, pp_ = _vars[_i]
                    for lf_ in [x for x in (lift, 0.07, 0.05) if x <= _lf + 1e-9]:
                        if _HARD["on"] and time.time() > _DEADLINE["t"]: _hard_hit = True; break
                        _chain = _chain_ik_one([dict(flange=pp_), _g, dict(flange=[fl[0], fl[1], fl[2] + lf_])], quat, start_q=start_q)
                        if os.environ.get("NERO_DBG_LIFT") == "1": print("DBG_CHAIN 2단계 pre_f=%s lift=%.2f 체인=%s" % (f, lf_, "있음" if _chain is not None else "없음"), flush=True)
                        if _chain is not None: break
                    if _chain is not None: break
        except Exception as e_:
            print("연속 IK 예외", repr(e_), flush=True); _chain = None
        if _chain is None and _hard_hit: return False, "시간 예산 초과(후보 내부, 연속 IK 탐색 중)"
        if _chain is None and os.environ.get("NERO_PICK_CHAIN_STRICT", "1") == "1":      # 접근-파지-들어올림을 같은 분기로 잇는 연속 IK 가 없다 = 이 파지는 자세를 유지한 채 들어올릴 수 없다 -> 궤적 최적화를 3~5초 헛돌리지 않고 바로 거부한다
            return False, "파지 후 들어올림까지 이어지는 연속 IK 없음(손목 관절 한계)"
        if os.environ.get("NERO_DBG_LIFT") == "1" and _chain is not None:
            try:
                res_ = {}
                for nm_, sc_ in (("scene_all", scene_all), ("no_target", scene_no_target)):
                    P.update_world(Scene.create(sc_)); r_ = P.plan_cspace(_js(_chain[0]), _js(start_q)); res_["접근(관절목표) " + nm_] = bool(_ok(r_))
                P.update_world(Scene.create(scene_no_target)); rr1 = P.plan_cspace(_js(_chain[0]), _js(start_q))
                if _ok(rr1):
                    st_ = _end_state(rr1); P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True) for k in P.tool_frames})
                    try: rr2 = _plan_pose_r(_goal(c["flange"], quat), st_, tries=1)
                    finally: P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
                    res_["직선 파지"] = bool(_ok(rr2))
                    if _ok(rr2):
                        st2_ = _end_state(rr2); rr3 = _plan_pose_r(_goal([fl[0], fl[1], fl[2] + lift], quat), st2_, tries=1); res_["들어올림(자세유지 10cm)"] = bool(_ok(rr3))
                        rr4 = P.plan_cspace(_js(_chain[2]), st2_); res_["들어올림(체인 관절목표)"] = bool(_ok(rr4))
                print("DBG_CHAIN_DETAIL", json.dumps(res_, ensure_ascii=False), "pre관절", [round(x, 2) for x in _chain[0]], "파지", [round(x, 2) for x in _chain[1]], "들", [round(x, 2) for x in _chain[2]], flush=True)
            except Exception as e_: print("DBG_CHAIN_DETAIL 예외", repr(e_), flush=True)
        if os.environ.get("NERO_DBG_LIFT") == "1": print("DBG_CHAIN", "체인 있음 pre_f=%s" % pre_used if _chain is not None else "체인 없음", flush=True)
        if _chain is not None:
            try:      # 체인 분기의 접근은 대상 상자 없이 계획한다(손가락이 상자 옆 6mm 로 지나가는 파지에서 대상 상자를 장애물로 두면 접근 자체가 막힌다). 대신 팔 링크(손가락 제외)가 대상 상자를 침범하지 않는지 뒤에서 검사한다
                P.update_world(Scene.create(scene_no_target))
                r1 = P.plan_cspace(_js(_chain[0]), _js(start_q))
                if os.environ.get("NERO_DBG_LIFT") == "1": print("DBG_CHAIN r1 cspace(no_target) ok=", bool(_ok(r1)), flush=True)
                if _ok(r1):
                    _tg = (scene_all.get("cuboid") or {}).get("target_obj")
                    if _tg:
                        _d1 = _link_target_min_dist(list(_wps(r1)), _tg["pose"][:3], _tg["dims"])      # 손가락 제외 링크의 대상 상자까지 부호 거리(음수 = 침범)
                        if os.environ.get("NERO_DBG_LIFT") == "1": print("DBG_CHAIN r1 팔링크-대상 최소거리 %.1fcm" % (100 * _d1), _LASTMIN["i"], flush=True)
                        if _d1 < float(os.environ.get("NERO_CHAIN_ARM_MARGIN", "-0.005")): r1 = None
                    if r1 is not None: pre_used = f
                else: r1 = None
                P.update_world(Scene.create(scene_all))
            except Exception as e_:
                print("연속 IK 접근 계획 예외", repr(e_), flush=True); r1 = None
    if r1 is None:
      for _i in _reach_order(_vars, quat):
        f, pp_ = _vars[_i]
        if (f != 1.0 or _HARD["on"]) and time.time() > _DEADLINE["t"]:
            if _HARD["on"]: _hard_hit = True
            break
        r1 = _plan_pose_r(_goal(pp_, quat), _js(start_q))
        if _ok(r1):
            pre_used = f; break
    if not _ok(r1) and c.get("ik_cfg"):
        # 폴백: 직교 목표 IK 로는 접근 자세를 못 찾는 후보(관절 한계 근처/손목 자세). B단계가 이미 구한 파지 관절해(ik_cfg)에서
        # 직선으로 접근 자세까지 물러나 pre 관절해를 얻고(타깃 박스 없이), 거기로 관절 공간 계획한다.
        try:
            P.update_world(Scene.create(scene_no_target))
            P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True) for k in P.tool_frames})
            try:
                rb = _plan_pose_r(_goal(c["pre"], quat), P.kinematics.get_active_js(_js([float(x) for x in c["ik_cfg"]])))
            finally:
                P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
            if _ok(rb):
                pre_cfg = [float(x) for x in _wps(rb)[-1]]
                P.update_world(Scene.create(scene_no_target if _r1nt else scene_all))
                r1 = P.plan_cspace(_js(pre_cfg), _js(start_q))
        except Exception as e:      # 폴백 자체가 실패해도 후보를 버리고 다음으로 간다
            print("폴백 예외", repr(e), flush=True)
    if not _ok(r1):
        if _hard_hit: return False, "시간 예산 초과(후보 내부, 접근 사다리 중)"
        return False, "접근 자세 계획 실패"
    w1 = _wps(r1)
    _r1d = None
    if _r1nt:
        _t1 = scene_all["cuboid"]["target_obj"]
        try:
            _r1d = _link_target_min_dist(list(w1), _t1["pose"][:3], _t1["dims"], fingers=True)
            if _r1d < float(os.environ.get("NERO_R1_TARGET_MARGIN", "-0.005")):
                return False, f"접근 자세 경로가 대상과 충돌(손가락 포함, 최소 {100 * _r1d:.1f}cm, {_LASTMIN['i']})"
        except Exception as e:
            print("r1 링크-대상 검사 예외", repr(e), flush=True)
    P.update_world(Scene.create(scene_no_target))
    lin = ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True)
    P.update_tool_pose_criteria({k: lin for k in P.tool_frames})
    try:
        r2 = _plan_pose_r(_goal(c["flange"], quat), _end_state(r1))
    finally:
        P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
    if not _ok(r2):
        return False, "접근->파지 직선 계획 실패"
    w2 = _wps(r2)
    _lt_d = None
    _tgt = (scene_all.get("cuboid") or {}).get("target_obj")
    _f2_d = None
    if _tgt and (os.environ.get("NERO_R2_DIAG", "0") == "1" or os.environ.get("NERO_R2_FINGER_CHECK", "0") == "1"):      # 2026-10-12 진단(기본 꺼짐: 환경변수 NERO_R2_DIAG=1 일 때만 계산해 기록): 파지 직전 직선 접근 구간에서 손가락 포함 모든 링크와 대상 상자의 부호 거리. 손가락이 대상을 스치는 계획을 가려내려는 것(MISSED 분석)
        try:
            _f2_d = _link_target_min_dist(list(w2), _tgt["pose"][:3], _tgt["dims"], fingers=True)
            if os.environ.get("NERO_R2_FINGER_CHECK", "0") == "1" and _f2_d < float(os.environ.get("NERO_R2_FINGER_MARGIN", "0.002")):
                return False, f"접근 구간 손가락이 대상 상자와 가까움(최소 {100 * _f2_d:.1f}cm, {_LASTMIN['i']})"
        except Exception as e:
            print("r2 손가락-대상 검사 예외", repr(e), flush=True)
    if _tgt and os.environ.get("NERO_LINK_TARGET_FILTER", "0") == "1":
        try:
            _lt_d = _link_target_min_dist(list(w2), _tgt["pose"][:3], _tgt["dims"])
            if _lt_d < float(os.environ.get("NERO_LINK_TARGET_MARGIN", "-0.01")):       # 손가락 외 링크가 대상 상자 안으로 1cm 넘게 파고들면 이 후보는 버린다(팔이 대상을 쳐서 쓰러뜨리는 해)
                return False, f"접근 구간 팔 링크가 대상과 충돌(손가락 제외, 최소 {100 * _lt_d:.1f}cm)"
        except Exception as e:
            print("링크-대상 검사 예외", repr(e), flush=True)
    # lift: 직선 lift 가 안 되는 자세(팔 도달 한계)가 있어 거리를 줄이거나 직선 제약 없이 시도한다(책 pick 실패 사례, 시드 120/123).
    start3 = _end_state(r2); r3 = None; lift_used = None
    _lv = [(lift, True), (min(lift, 0.07), True), (lift, False), (0.05, True)]
    _ord = _reach_order([(i, [c["flange"][0], c["flange"][1], c["flange"][2] + _lv[i][0]]) for i in range(len(_lv))], quat)      # 들어올리기 사다리: 끝 자세 IK 를 배치로 한 번에 풀어 도달 가능한 단계부터
    for lf_d, linear in [_lv[i] for i in _ord]:
        if lift_used is None and r3 is not None and time.time() > _DEADLINE["t"]:
            break
        up = [c["flange"][0], c["flange"][1], c["flange"][2] + lf_d]
        if linear:
            P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=False) for k in P.tool_frames})
        try:
            r3 = _plan_pose_r(_goal(up, quat), start3)
        finally:
            P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
        if _ok(r3):
            lift_used = (lf_d, linear); break
    if lift_used is None and _chain is not None:      # 체인 분기의 들어올림 구성으로 관절 공간 계획(자세 유지 + 위치 이동이 같은 분기에서 가능함이 IK 로 보장된 구성)
        try:
            P.update_world(Scene.create(scene_no_target)); r3 = P.plan_cspace(_js(_chain[2]), start3)
            if _ok(r3): lift_used = (lift, False)
            if os.environ.get("NERO_DBG_LIFT") == "1": print("DBG_CHAIN 들어올림 cspace ok=", bool(_ok(r3)), flush=True)
        except Exception as e_: print("체인 들어올림 계획 예외", repr(e_), flush=True)
    if lift_used is None:
        if os.environ.get("NERO_DBG_LIFT") == "1":      # 진단(2026-10-08): lift 실패 후보에 대해 변인을 하나씩 바꿔 어디서 막히는지 서버 로그(DBG_LIFT)에 남긴다. 기본 꺼짐
            try: _dbg_lift_variants(c, start3, quat, scene_no_target, r2)
            except Exception as e_: print("DBG_LIFT 예외", repr(e_), flush=True)
        return False, "lift 계획 실패(직선 10/7cm, 비직선, 5cm 모두)"
    w3 = _wps(r3)
    tr, T1 = _resample(w1, dts[0], dt_i, tmin[0]); ad, T2 = _resample(w2, dts[1], dt_i, tmin[1]); lf, T3 = _resample(w3, dts[2], dt_i, tmin[2])
    _aq = _approach_quality(w1, a); print("접근 지표", _aq, "pre_used", pre_used, flush=True)
    plan = {k: c[k] for k in ("quat", "approach", "contact", "flange", "opening", "radius", "grip_close", "kind", "twin", "score",
                              "approach_theta", "min_margin", "j6_ik", "cost") if k in c}
    plan.update(dict(transit=tr, advance=ad, retreat=list(reversed(ad)), lift=lf, grasp_cfg=ad[-1], pre_cfg=tr[-1], cartesian_fraction=1.0,
                     j6=c.get("j6_ik"), j1_travel=float(abs(tr[-1][0] - start_q[0])), clearance=0.05, planner="curobo",
                     durations=dict(transit=T1, advance=T2, lift=T3), lift_used=dict(dist=lift_used[0], linear=lift_used[1]), pre_used=pre_used, approach_quality=_aq, link_target_min_dist=_lt_d, r1_target_min_dist=_r1d, r2_finger_min_dist=_f2_d))
    return True, plan


def _expand_place_spec(pr):
    """pr(놓기 요청)에 place_spec 이 있으면 서버에서 후보 목록(obj_rot_list)을 만든다. 만든 목록 이름을 돌려줌."""
    ps = (pr or {}).get("place_spec")
    if not ps: return None
    import place_spec_core as _PSC
    pr["obj_rot_list"] = _PSC.pose_entries(dict(hard=ps.get("hard") or [], manner=["place"]), ps["asset"], float(ps["yaw0"]), ps.get("order")); pr.pop("obj_rot", None)
    pr.pop("place_spec", None)      # 2026-11-10: 펼친 뒤 조건을 빼야 함 -- 남기면 자세별 안쪽 놓기 검사(solve_place)가 전체 목록으로 다시 펼쳐 되풀이(서버 후보 모드 /plan 전부 실패하던 원인)
    return [e["name"] for e in pr["obj_rot_list"]]


def solve_candidates(req):
    """POST /candidates {hard, asset, yaw0} -> 조건을 만족하는 자세 후보(이름, yaw 목록, 놓을 점). 판단 쪽(Gemma 순서 매기기)이 쓴다."""
    import place_spec_core as _PSC
    t_ = time.time(); cs = _PSC.candidates(dict(hard=req.get("hard") or [], manner=["place"]), req["asset"], float(req["yaw0"]))
    return dict(ok=True, cands=cs, sec=round(time.time() - t_, 4))


import contextlib
_nullctx = contextlib.nullcontext
_MEMH = {"n": 0}


def _mem_hygiene():
    """2026-11-10: 요청이 쌓일수록 이 서버의 GPU 메모리가 늘어(2.4 -> 6.2GB) Thor 통합 메모리가 빠듯해지고 CGN 이 10배 느려짐(측정).
    요청마다 gc + torch 캐시 반환(NERO_EMPTY_CACHE=0 이면 끔). 20요청마다 할당/예약량 기록(진짜 누수=할당 증가, 캐시=예약만 증가)."""
    try:
        with LOCK:
            if os.environ.get("NERO_EMPTY_CACHE", "1") == "1":
                import gc; gc.collect(); torch.cuda.empty_cache()
            if _MEMH["n"] % 20 == 0:
                print("MEMH n=%d alloc=%.0fMB reserved=%.0fMB" % (_MEMH["n"], torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6), flush=True)
                if os.environ.get("NERO_MEMH_TENSORS", "0") == "1":      # 진단 전용: 살아 있는 CUDA 텐서를 (모양, dtype) 별로 세어 이전과 비교
                    import gc, collections
                    cnt = collections.Counter(); byt = collections.Counter()
                    for o_ in gc.get_objects():
                        try:
                            if torch.is_tensor(o_) and o_.is_cuda: k_ = (tuple(o_.shape), str(o_.dtype)); cnt[k_] += 1; byt[k_] += o_.element_size() * o_.nelement()
                        except Exception: pass
                    prev = _MEMH.get("cnt") or collections.Counter(); _MEMH["cnt"] = cnt
                    grow = sorted(((cnt[k] - prev.get(k, 0), k, byt[k]) for k in cnt if cnt[k] > prev.get(k, 0)), reverse=True)[:8]
                    print("MEMH tensors=%d bytes=%.0fMB grow=%s" % (sum(cnt.values()), sum(byt.values()) / 1e6, grow), flush=True)
                    try:
                        _ts = [o_ for o_ in gc.get_objects() if torch.is_tensor(o_) and o_.is_cuda and tuple(o_.shape) == (4096, 7)]
                        _t = _ts[-1]; _lv = [_t]
                        for _d in range(4):
                            _rs = [r_ for r_ in gc.get_referrers(_lv[-1]) if r_ is not _ts and r_ is not _lv and type(r_).__name__ != "frame"]
                            print("MEMH lvl%d referrer types=%s" % (_d, [type(r_).__name__ for r_ in gc.get_referrers(_lv[-1])][:10]), flush=True)
                            if not _rs: break
                            _r = _rs[0]; _desc = type(_r).__name__ + (":" + ",".join(list(_r.keys())[:6]) if isinstance(_r, dict) else (":len%d" % len(_r) if isinstance(_r, (list, tuple)) else ""))
                            print("MEMH ref%d %s n_ref=%d" % (_d, _desc[:200], len(_rs)), flush=True); _lv.append(_r)
                    except Exception as _x: print("MEMH ref 예외", repr(_x), flush=True)
    except Exception as _x: print("mem_hygiene 예외", repr(_x), flush=True)


_NBF_ENV = os.environ.get("NERO_NB_FIRST", "0") == "1"      # 2026-11-10 nb_first 기본값(요청의 nb_first 가 우선)


def solve(req):
    t0 = time.time()
    _psn = _expand_place_spec((req.get("place_check") or {}).get("req"))      # 2026-11-10: 조건 -> 후보는 서버에서
    _reset_retry_budget(dict(req, retry_budget=req.get("retry_budget", 0)))       # pick 은 일단 재시도 없음(기존 동작). 따로 평가한다
    start_q = [float(x) for x in req["start_q"]]
    margin = float(req.get("margin", 0.01)); tol = float(req.get("tol", 0.0))
    obstacles = req.get("obstacles", []); target = req.get("target")
    _added_obs = []
    if req.get("pick_with_place_obstacles") and (req.get("place_check") or {}).get("req"):      # 2026-11-07(복사본, 기본 꺼짐): 놓기 월드에만 있는 장애물(작업대 등)을 집기 월드에도 넣는다.
        _ids = {o["id"] for o in obstacles}                                                         # 집기가 그 장애물 속으로 들어간 뒤 transit 시작 자세 충돌로 후보를 다 버리는 일을 막는다(장애물을 더하기만 하므로 물리적으로 안 되는 해만 빠진다)
        _added_obs = [o for o in (req["place_check"]["req"].get("obstacles") or []) if o["id"] not in _ids]
        obstacles = list(obstacles) + _added_obs
    lift = float(req.get("lift", 0.10)); K = int(req.get("max_cands", 4))
    dts = (float(req.get("transit_dt", 0.1625)), float(req.get("advance_dt", 0.25)), float(req.get("lift_dt", 0.3)))
    tmin = (float(req.get("transit_tmin", 1.0)), float(req.get("advance_tmin", 1.5)), float(req.get("lift_tmin", 1.2)))
    cands = req["candidates"][:K]
    _pre_note = None
    if req.get("prescreen") and len(req["candidates"]) > 1:      # 2026-10-22(기본 꺼짐): 후보(앞 screen_n 개)를 파지-들어올림 연속 IK 로 먼저 선별해 연결 가능한 후보를 앞에 둔다(안정 정렬, 안 본 후보는 원래 순서). 선별 시간은 요청 예산에 포함
        _t_p = time.time(); _okc = {}
        for _i, _c in enumerate(req["candidates"][:int(req.get("screen_n", 12))]):
            if time.time() - _t_p > float(req.get("screen_budget_s", 1.5)) * _BS: break
            try:
                _fl = [float(x) for x in _c["flange"]]; _qq = [float(x) for x in _c["quat"]]
                _okc[_i] = _chain_ik_one([dict(flange=_fl), dict(flange=[_fl[0], _fl[1], _fl[2] + lift])], _qq, rep=2, cap=200) is not None
            except Exception as _e:
                print("후보 사전 선별 예외", repr(_e), flush=True); break
        _ord = [i for i in range(len(req["candidates"])) if _okc.get(i, True)] + [i for i in range(len(req["candidates"])) if not _okc.get(i, True)]
        cands = [req["candidates"][i] for i in _ord][:K]; _pre_note = dict(order=_ord[:K], ok=[int(_okc[i]) for i in sorted(_okc)], sec=round(time.time() - _t_p, 3))
        print("후보 사전 선별: 연결 가능 %s -> 순서 %s (%.2fs)" % ([int(_okc[i]) for i in sorted(_okc)], _ord[:K], time.time() - _t_p), flush=True)
    _TM = dict(pick=0.0, place=0.0, prescreen=0.0, n_pick=0, n_place=0, n_run=0, n_skip=0)      # 2026-11-09 단계별 시간
    _PSE = {"req": None}; _NBF = bool(req.get("nb_first", _NBF_ENV))
    _pc0 = req.get("place_check")
    if _pc0 and _pc0.get("kind") == "insert" and _PRESCREEN and len(req["candidates"]) > 1:
        # 2026-10-16: 삽입(꽂기) 파지 후보를 CGN 점수순 상위 K 개만 보지 않고, 손-홀더 여유(IK 만, 궤적 계획 없음)로 먼저 선별해 여유가 큰 후보부터 본다.
        # 실패 재현(손전등->컵): 점수순 상위 3개는 전부 여유 -10~-18mm 로 실패했고 4번째(여유 +5.7mm)만 성공 가능했으나 시도되지 않았다.
        _t_s = time.time(); _sc = []
        try:
            import insert_planner as _ip
            _rq = []
            for _c in req["candidates"][:int(req.get("screen_n", 12))]:
                _r = json.loads(json.dumps(_pc0["req"])); _r["grasp"] = dict(quat=[float(x) for x in _c["quat"]], approach=[float(x) for x in _c["approach"]], flange=[float(x) for x in _c["flange"]]); _rq.append(_r)
            _ip.prewarm(_rq, sys.modules[__name__])      # 후보 전체의 IK 체인을 한 번에(배치 128) 풀어 캐시에 넣는다
        except Exception as _e:
            print("삽입 IK 일괄 계산 예외(개별 계산으로 진행)", repr(_e), flush=True)
        for _i, _c in enumerate(req["candidates"][:int(req.get("screen_n", 12))]):
            try:
                _r = json.loads(json.dumps(_pc0["req"])); _r["grasp"] = dict(quat=[float(x) for x in _c["quat"]], approach=[float(x) for x in _c["approach"]], flange=[float(x) for x in _c["flange"]])
                _r["start_q"] = [float(x) for x in req["start_q"]]; _r["screen_only"] = True
                _o = _solve_insert(_r); _sc.append((float(_o.get("best_clear", -1.0)) if _o.get("ok") else -1.0, _i))
            except Exception as _e:
                print("삽입 사전 선별 예외", repr(_e), flush=True); _sc.append((-1.0, _i))
            if time.time() - _t_s > float(req.get("screen_budget_s", 6.0)) * _BS: break
        _done = {i for _, i in _sc}; _order = [i for _, i in sorted(_sc, key=lambda x: (-x[0], x[1]))] + [i for i in range(len(req["candidates"])) if i not in _done]
        cands = [req["candidates"][i] for i in _order][:K]
        print("삽입 사전 선별: 후보 %d개 %.1fs, 여유(mm) %s -> 순서 %s" % (len(_sc), time.time() - _t_s, [round(1000 * c_, 1) for c_, _ in _sc], _order[:K]), flush=True)

    _pps = None
    if req.get("place_prescreen") and _pc0 and _pc0.get("kind") != "insert" and cands:      # 2026-11-06(복사본, 기본 꺼짐): 놓기 IK 사전 선별 -- 후보 순서만 바꾼다(drop 은 측정용)
        try:
            _cands_before = list(cands); cands, _pps, _pps_poses = _place_prescreen(req, cands)
            print("놓기 IK 사전 선별: 통과 %d/%d, %.2fs -> 순서 %s%s" % (_pps["n_good"], len(_cands_before), _pps["sec"], _pps["order"], (" (뺌 %s)" % _pps["dropped"]) if _pps["dropped"] else ""), flush=True)
        except Exception as _e:
            print("놓기 IK 사전 선별 예외(원래 순서로 진행)", repr(_e), flush=True); _pps = dict(error=repr(_e))
    deadline = t0 + float(req.get("budget_s", 4.0)) * _BS; _DEADLINE["t"] = deadline
    _HARD["on"] = bool(req.get("hard_deadline")); _cs = []

    def run(skip):
        sc_no = _cuboids(obstacles, margin, tol, skip, req.get("table_top"))
        sc_all = {"cuboid": dict(sc_no)}
        if target:
            tm = float(req.get("target_margin", 0.0))
            sc_all["cuboid"]["target_obj"] = {"dims": [target["dims"][0] + tm, target["dims"][1] + tm, target["dims"][2] + tm], "pose": [*target["xyz"], 1, 0, 0, 0]}
        sc_no_t = {"cuboid": dict(sc_no)}
        why = []; _poor = []
        if not skip: _cs.clear()
        _TM["n_run"] += 1
        for i, c in enumerate(cands):
            if time.time() > deadline and i > 0:
                why.append("시간 예산 초과"); break
            if _PSE.get("nb_blocked"):
                _TM["n_skip"] += 1; _TM["n_nbskip"] = _TM.get("n_nbskip", 0) + 1; why.append(f"cand{i}: pick 미확인, place 불가 (IK 후보 0개, 놓을 자리에 다른 물체가 있어 놓을 수 없음(놓은 물체가 이웃과 겹침))"); continue
            if _PSE["req"] is not None:
                _t_ps = time.time()
                try:
                    _ppi = _place_prescreen(dict(place_check=dict(req["place_check"], req=_PSE["req"])), [c])[1]; _nz = _ppi["place_ik_n"][0] == 0
                except Exception as _x:
                    _nz = False; print("파지 사전 선별 예외(계획 진행)", repr(_x), flush=True)
                _TM["prescreen"] += time.time() - _t_ps
                if _nz:
                    _TM["n_skip"] += 1; why.append(f"cand{i}: pick 미확인, place 불가 (IK 후보 0개, 사전 선별: 이 파지로 놓기 IK 0)"); continue      # 'place 불가' 문구 = 놓기 실패로 분류(장애물 빼 보기 진단을 돌리지 않음, Gemma 요약도 같은 분류)
            _tc0 = time.time()
            ok, res = _try_candidate(c, start_q, sc_all, sc_no_t, lift, dts, tmin)
            if not skip: _cs.append(round(time.time() - _tc0, 2))
            _TM["pick"] += time.time() - _tc0; _TM["n_pick"] += 1
            if ok and req.get("place_check"):      # 파지-place 동시 선택: pick 이 되는 후보라도 그 파지로 place/삽입이 안 되면 다음 후보로 간다(집은 뒤에 place 궤적이 없어 실패하는 일을 없앤다)
                pc = req["place_check"]; pr_req = json.loads(json.dumps(pc["req"]))
                pr_req["grasp"] = dict(quat=[float(x) for x in c["quat"]], approach=[float(x) for x in c["approach"]], flange=[float(x) for x in c["flange"]])
                pr_req["start_q"] = [float(x) for x in res["lift"][-1]]; pr_req["budget_s"] = min(float(pr_req.get("budget_s", 8.0)), float(req.get("place_budget_s", 6.0)))
                if req.get("start_collision_report"): pr_req["start_collision_report"] = True      # 2026-11-07(기본 꺼짐): place 시작 자세(lift 끝) 충돌 보고
                saved_dl = _DEADLINE["t"]
                try:
                    _tp0 = time.time(); pr = _solve_insert(pr_req) if pc.get("kind") == "insert" else solve_place(pr_req); _TM["place"] += time.time() - _tp0; _TM["n_place"] += 1
                except Exception as e_:
                    pr = {"ok": False, "why": ["place 확인 예외 %r" % (e_,)]}
                finally:
                    _DEADLINE["t"] = saved_dl
                if not (pr or {}).get("ok"):
                    tr_ = (pr or {}).get("tried") or []
                    why.append(f"cand{i}: pick 가능, place 불가 (IK 후보 {(pr or {}).get('n_ik_ok')}개, " + str((tr_[0].get('fail') or (tr_[0].get('fails') or ['?'])[0]) if tr_ else (lambda w_: w_ if isinstance(w_, str) else w_[0])((pr or {}).get('why') or ['후보 없음']))[:60] + ")")
                    if req.get("start_collision_report"):      # 기본 꺼짐: 완화 단계 전체 실패 사유와 시작 자세 충돌(장애물: mm, 링크)을 덧붙인다
                        why[-1] += " [단계 %s] [시작 자세 충돌 %s]" % ("/".join((tr_[0].get("fails") or []) + ([tr_[0]["fail"]] if tr_ and tr_[0].get("fail") else [])) if tr_ else "-", (pr or {}).get("start_collision") or "없음")
                    continue
                res["place_plan"] = pr; res["place_kind"] = pc.get("kind")
            if ok:
                res["cand_index"] = i
                _ed = (res.get("approach_quality") or {}).get("end_dot", 1.0)
                if _ed < _AMIN and i < len(cands) - 1 and time.time() < deadline:      # 2026-10-16: 접근 벡터 반대로 들어오는 궤적은 보류하고 다음 후보를 먼저 본다(없으면 보류한 것 중 가장 나은 것)
                    _poor.append((_ed, res)); why.append(f"cand{i}: 접근 정합 낮음(end_dot {_ed}) -- 다음 후보 확인"); continue
                return True, res, why
            why.append(f"cand{i}: {res}")
        if _poor:
            _b = max(_poor, key=lambda x: x[0]); return True, _b[1], why
        return False, None, why

    _rl = ((req.get("place_check") or {}).get("req") or {}).get("obj_rot_list")
    _cands_all = list(cands)
    if _rl and (len(_rl) > 1 or req.get("place_prescreen_pose")) and req["place_check"].get("kind") != "insert":      # 2026-11-09: 자세 순위 우선(자세 바깥, 파지 안쪽 루프) -- 순위 1 자세를 모든 파지로 먼저 확인하고, 안 되면 순위 2 ...
        _base = req["place_check"]["req"]; _why_all = []; ok, plan = False, None; why = []
        for _k, _e in enumerate(_rl):
            if req.get("total_budget_s") and time.time() - t0 > float(req["total_budget_s"]):
                _why_all.append(["%s(순위 %d)" % (_e.get("name"), _k), "시간 예산 초과(요청 전체 %.0f초 상한 -- 이 자세는 계획하지 못함)" % float(req["total_budget_s"])]); continue
            req["place_check"]["req"] = dict(_base, obj_rot_list=[_e]); deadline = time.time() + float(req.get("budget_s", 4.0)) * _BS
            if req.get("total_budget_s"): deadline = min(deadline, t0 + float(req["total_budget_s"]))
            _DEADLINE["t"] = deadline
            if req.get("place_prescreen_pose"):      # 이 자세의 놓기 요청(회전·각도·자리) -- run() 이 파지마다 집기 계획 직전에 놓기 IK 를 확인
                _prq = {k_: v_ for k_, v_ in _base.items() if k_ not in ("obj_rot_list", "obj_rot", "yaws")}
                if _e.get("rot") is not None: _prq["obj_rot"] = _e["rot"]
                if _e.get("yaws"): _prq["yaws"] = _e["yaws"]
                if _e.get("slots"): _prq["slots"] = _e["slots"]
                _PSE["req"] = _prq; _PSE["nb_blocked"] = False
                if _NBF and _prq.get("neighbor_reject") and cands:      # 2026-11-10 nb_first(사용자 지시): 놓은 물체-이웃 겹침은 파지와 무관(최종 물체 자세만으로 정해짐) -> 이 자세의 모든 (자리, yaw) 가 겹치면
                    _t_nb = time.time()                               #   어떤 파지로도 놓기 불가이므로 파지마다 집기 경로를 계획하지 않고 바로 놓기 불가로 적는다(성공/실패·고른 후보는 그대로, 집기 실패 사유 개수만 달라질 수 있음)
                    try:
                        _c0 = cands[0]; _nbq = json.loads(json.dumps(_prq)); _nbq.update(_nb_only=True, start_q=[float(x) for x in start_q], grasp=dict(quat=[float(x) for x in _c0["quat"]], approach=[float(x) for x in _c0["approach"]], flange=[float(x) for x in _c0["flange"]]))
                        _nbr = _solve_place_impl(_nbq); _PSE["nb_blocked"] = bool(_nbr.get("nb_only") and _nbr.get("n_all", 0) > 0 and _nbr.get("n_left", 1) == 0)
                    except Exception as _x: print("nb_first 예외(검사 생략)", repr(_x), flush=True)
                    _TM["nb_first"] = _TM.get("nb_first", 0.0) + (time.time() - _t_nb)
            ok, plan, why = run(set()); _why_all.append(["%s(순위 %d)" % (_e.get("name"), _k)] + list(why or []))
            if ok: plan["pose_rank"] = _k; plan["pose_name"] = _e.get("name"); break
        req["place_check"]["req"] = _base; why = why if ok else [w_ for g_ in _why_all for w_ in g_]
        print("자세 순위 계획: %s" % ("순위 %d %s 성공" % (plan["pose_rank"], plan["pose_name"]) if ok else "모든 자세 실패(%d개)" % len(_rl)), flush=True)
    else:
        ok, plan, why = run(set())
    out = {"ok": ok, "why": why}
    out["timing"] = {k_: (round(v_, 2) if isinstance(v_, float) else v_) for k_, v_ in _TM.items()}; out["timing"]["total"] = round(time.time() - t0, 2)
    if _added_obs: out["pick_added_obstacles"] = [o["id"] for o in _added_obs]
    if _pps is not None:      # place_prescreen 켰을 때만(꺼짐이면 출력 동일)
        out["place_prescreen"] = _pps
        if ok and "order" in _pps:
            _ci = plan.get("cand_index"); _pps["chosen_orig_index"] = _pps["order"][_ci] if _ci is not None and _ci < len(_pps["order"]) else None
            try:      # 선택된 놓기 플랜지가 사전 선별 자세 목록에 있는지(같은 식인지 자가 점검, mm)
                _pf = np.array((plan.get("place_plan") or {}).get("flange"), float); _L = np.array(_pps_poses[_pps["order"][_ci]], float)      # _pps_poses 는 원래 후보 번호 기준
                _pps["chosen_pose_match_mm"] = round(1000 * float(np.linalg.norm(_L - _pf[None, :], axis=1).min()), 3)
            except Exception:
                pass
    if req.get("reject_log"): out["cand_reject"] = [dict(cand=i_, code=_reject_code(w_), why=w_) for i_, w_ in enumerate(why)]      # 2026-10-23(기본 꺼짐): 후보별 거부 단계 코드
    if req.get("log_extra"): out["cand_secs"] = list(_cs); out["n_cands_used"] = len(cands); out["budget_s"] = float(req.get("budget_s", 4.0)); out["prescreen"] = _pre_note
    if ok:
        out["plan"] = plan
    elif req.get("diagnose", True) and obstacles and not any("place 불가" in w_ for w_ in why):      # 2026-10-16: pick 은 되는데 place(삽입)가 안 되는 실패는 이웃 물체를 빼 봐도 달라지지 않는다 -- 후보 루프(삽입 계획 포함)를 최대 5번 반복하던 진단을 건너뛴다
        tx = (target or {}).get("xyz") or cands[0]["flange"]
        near = sorted([o for o in obstacles if o not in _added_obs], key=lambda o: math.hypot(o["xyz"][0] - tx[0], o["xyz"][1] - tx[1]))[:4]
        blk = []
        for o in near:
            if time.time() > deadline + 3.0:
                break
            if run({o["id"]})[0]:
                blk.append(o["id"])
        if not blk and run({o["id"] for o in near})[0]:
            blk = [o["id"] for o in near[:2]]
        out["blockers"] = blk
    if not ok:
        try:      # 실패한 pick 계획 요청 저장(오프라인 재생용, place 실패 저장과 같은 방식). 2026-10-03: pick 단계 실패(예산 2초 안에 후보 소진 못 함)를 분석하려고
            os.makedirs("" + CUROBO_DIR + "/fail_goalpose", exist_ok=True)
            json.dump(req, open("" + CUROBO_DIR + "/fail_goalpose/%d_pick.json" % int(time.time() * 1000), "w"))
        except Exception:
            pass
    out["sec"] = round(time.time() - t0, 3)
    _HARD["on"] = False
    if not ok:       # 실패한 요청은 저장해 오프라인 재현에 쓴다
        try:
            import os
            os.makedirs("" + CUROBO_DIR + "/fail_goalpose", exist_ok=True)
            tag = (cands[0].get("object") or "x") if cands else "x"
            json.dump(req, open("" + CUROBO_DIR + "/fail_goalpose/%d_%s.json" % (int(time.time() * 1000), tag), "w"))
        except Exception:
            pass
    return out


# ── place (2026-10-01): 손에 든 물체를 충돌 모델에 붙이고 슬롯 위로 옮겨 내려놓는 계획 ─────────────────────────────────
D_TCP = 0.1358
J_LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5707963, 1.5707963)]
PLACE_YAWS = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]


def _jmargin(q):
    return min(min(qi - lo, hi - qi) for qi, (lo, hi) in zip(q, J_LIM))


def _held_spheres(R0, FL0, centre_w, dims, yaw, n_max=32, pad=0.004, cover="old"):
    """집은 물체(OBB)를 채우는 구들을 그리퍼 플랜지 좌표계로. -> (N,4) numpy"""
    d = np.maximum(np.asarray(dims, float), 0.02) + 2 * pad
    if cover == "bound":      # 2026-10-14(기본 꺼짐, req["held_cover"]="bound"): 구 간격 격자의 칸마다 그 칸을 통째로 덮는 외접구를 쓴다. 기존 방식은 상자 모서리(예: 6x10x15cm 통의 네 귀퉁이)가 구 밖으로 2~3cm 삐져나와 그 모서리로 이웃을 치고 지나갈 수 있었다
        best = None
        for nx in range(1, 9):
            for ny in range(1, 9):
                for nz in range(1, 9):
                    if nx * ny * nz > n_max: continue
                    h = d / (2.0 * np.array([nx, ny, nz])); rr = float(np.linalg.norm(h))
                    if best is None or rr < best[0] - 1e-6 or (abs(rr - best[0]) < 1e-6 and nx * ny * nz < best[1]): best = (rr, nx * ny * nz, (nx, ny, nz))
        rr, _, (nx, ny, nz) = best
        axes = [(np.arange(n) - (n - 1) / 2.0) * (di / n) for n, di in zip((nx, ny, nz), d)]
        g = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        pw = np.asarray(centre_w, float) + Rot.from_euler("z", yaw).apply(g); pf = R0.inv().apply(pw - np.asarray(FL0, float))
        return np.concatenate([pf, np.full((len(pf), 1), rr)], axis=1)
    r = float(np.clip(d.min() / 2.0, 0.012, 0.035))
    while True:
        axes = []
        for di in d:
            span = di / 2.0 - r
            n = 1 if span <= 0 else int(math.ceil(2 * span / (1.2 * r))) + 1
            axes.append(np.linspace(-span, span, n) if span > 0 else np.zeros(1))
        if np.prod([len(a) for a in axes]) <= n_max or r > 0.08:
            break
        r *= 1.2
    g = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    Rb = Rot.from_euler("z", yaw)
    pw = np.asarray(centre_w, float) + Rb.apply(g)
    pf = R0.inv().apply(pw - np.asarray(FL0, float))
    return np.concatenate([pf, np.full((len(pf), 1), r)], axis=1)


def _pre_candidates(qp, qpre_list, ika, branch_max=0.8):
    """2026-11-04 pre_mode: 후보 하나의 사전 자세 목록과 탈락 사유. qp=수직 사전 자세 체인에서 나온 놓기 해(없으면 None), qpre_list=[(dz, q_pre or None)...](수직, 사전 높이 순),
    ika=(q_pre_a, q_pl_a, dz_a) 또는 None(접근축 사전 자세). -> (pres 목록[{kind,dz,q_pre,q_pl}], why 또는 None). 수직이 먼저, 접근축은 뒤(수직이 가능하면 첫 항목은 현행과 동일)."""
    pres = []; why = None
    if qp is None:
        why = "IK(place) 없음"
    else:
        qpre = dzu = None
        for dz, q_ in qpre_list:
            if q_ is not None:
                qpre, dzu = q_, dz; break
        if qpre is None: why = "IK(pre) 없음"
        elif max(abs(x - y) for x, y in zip(qpre, qp)) > branch_max: why = "pre/place 해 분기 다름"
        else: pres.append(dict(kind="vertical", dz=dzu, q_pre=qpre, q_pl=qp))
    if ika is not None:
        pres.append(dict(kind="approach", dz=ika[2], q_pre=ika[0], q_pl=ika[1]))
    return pres, (None if pres else why)


_GRIP_SPH = {"v": None}


def _gripper_sphere_offsets():
    """2026-11-06 auto_clear(복사본 전용): 로봇 충돌 구 중 그리퍼 쪽 구를 도구(플랜지) 좌표계로. 그리퍼 관절은 모델에서 열린 상태로 고정(nero7_open)이라 플랜지에 강체로 붙어 있다.
    -> {"finger": (N,4), "palm": (M,4)}  finger = gripper_link1/2, palm = gripper_base/gripper_flange. 한 번 계산해 캐시."""
    if _GRIP_SPH["v"] is None:
        kc = P.kinematics.kinematics_config; inv = {v: k for k, v in kc.link_name_to_idx_map.items()}
        names = [inv.get(int(i), "?") for i in kc.link_sphere_idx_map.tolist()]
        q = torch.zeros((1, 7), device="cuda", dtype=torch.float32)
        st = P.kinematics.compute_kinematics(JointState.from_position(q, joint_names=P.joint_names))
        rs = st.robot_spheres.reshape(-1, 4).detach().cpu().numpy(); lp = st.tool_poses.get_link_pose(P.tool_frames[0])
        fl = lp.position.reshape(3).detach().cpu().numpy(); w, x, y, z = lp.quaternion.reshape(4).detach().cpu().numpy(); Rt = Rot.from_quat([x, y, z, w])
        def grp(keys):
            m = np.array([n in keys for n in names]) & (rs[:, 3] > 0)
            return np.concatenate([Rt.inv().apply(rs[m, :3] - fl), rs[m, 3:4]], axis=1)
        _GRIP_SPH["v"] = dict(finger=grp({"gripper_link1", "gripper_link2"}), palm=grp({"gripper_base", "gripper_flange"}), f1=grp({"gripper_link1"}), f2=grp({"gripper_link2"}), cam=grp({"camera_link", "camera_stand_link"}), l7=grp({"link7"}))
        print("auto_clear: 그리퍼 구 손가락 %d개, 손바닥 %d개 (플랜지 좌표계)" % (len(_GRIP_SPH["v"]["finger"]), len(_GRIP_SPH["v"]["palm"])), flush=True)
    return _GRIP_SPH["v"]


def _sph_min_z(sph, fl, quat):
    """플랜지 좌표계 구들(N,4)을 플랜지 자세(fl, quat xyzw)에 놓았을 때 구 아래 끝의 최저 높이."""
    if sph is None or len(sph) == 0: return 9.0
    return float((np.asarray(fl, float)[2] + Rot.from_quat(quat).apply(sph[:, :3])[:, 2] - sph[:, 3]).min())


def _auto_clear_raise(pr, fl, quat, tz, clear):
    """auto_clear: 손가락 구 최저점이 받침면(tz)보다 margin 이상 위가 되도록 올릴 양(m). clear(요청값)가 하한, clear+올림 <= auto_clear_max(기본 0.03).
    -> (올림, 손가락 최저점-받침면(올리기 전), 필요량이 상한을 넘었는지)."""
    g = _gripper_sphere_offsets(); zf = _sph_min_z(g["finger"], fl, quat) - tz
    need = max(0.0, float(pr.get("auto_clear_margin", 0.005)) - zf); room = max(0.0, float(pr.get("auto_clear_max", 0.03)) - clear)
    capped = need > room + 1e-9
    if not pr.get("auto_clear"): return 0.0, zf, capped
    if capped and not pr.get("auto_clear_partial"): return 0.0, zf, capped      # 보수적: 상한 안에서 다 못 고치면 올리지 않는다(자세 = 끔과 동일, 보고만). 상한까지 일부만 올리던 1차판은 x90 눕힘에서 계획 실패를 늘렸다
    return min(need, room), zf, capped


def _fp_hits(cx, cy, R3, halfs, obs, tol=0.002):
    """물체 상자(축 R3 열, 반길이 halfs)의 바닥 사각형(연직에 가장 가까운 축을 뺀 두 축)이 중심 (cx,cy) 일 때 obs[(ox,oy,ux,uy)] 사각형과 겹치는가(2D 분리축)."""
    k = int(np.argmax(np.abs(R3[2, :]))); u = [R3[:2, i] * halfs[i] for i in range(3) if i != k]
    for ox, oy, ux, uy in obs:
        d = np.array([cx - ox, cy - oy]); sep = False
        for ax in (u[0], u[1], ux, uy):
            n_ = float(np.linalg.norm(ax))
            if n_ < 1e-9: continue
            a_ = ax / n_
            if abs(float(d @ a_)) >= abs(float(u[0] @ a_)) + abs(float(u[1] @ a_)) + abs(float(ux @ a_)) + abs(float(uy @ a_)) - tol: sep = True; break
        if not sep: return True
    return False


def _place_goal_poses(pr, grasp):
    """2026-11-06 place_prescreen(복사본 전용): _solve_place_impl 의 후보 생성과 같은 식으로 놓기 목표 플랜지 자세 목록 [(flange, quat)] 을 만든다(IK 사전 선별용, 계획은 하지 않는다).
    모드: obj_rot 있으면 goal_rot 하나, mode 지정이면 그 모드, 없으면 solve_place 와 같이 same + (접근각 < plumb_deg 이면) plumb."""
    Q0 = np.array(grasp["quat"], float); A0 = np.array(grasp["approach"], float); A0 /= np.linalg.norm(A0); FL0 = np.array(grasp["flange"], float)
    ob = pr["obj"]; CTR = np.array(ob["centre"], float); BASE_Z = float(ob.get("base_z", 0.0)); OBJ_H = float(ob["height"]); dims = ob.get("dims") or [0.06, 0.06, OBJ_H]
    CONTACT0 = FL0 + D_TCP * A0; HELD_H = float(CONTACT0[2] - BASE_Z); O_W0 = CONTACT0 - np.array([CTR[0], CTR[1], BASE_Z + OBJ_H / 2.0]); R0 = Rot.from_quat(Q0)
    theta0 = math.degrees(math.acos(max(-1.0, min(1.0, -A0[2]))))
    top_z = float(pr["top_z"]); clear = float(pr.get("clear", 0.006)); yaws = pr.get("yaws") or PLACE_YAWS
    R_REL = _parse_obj_rot(pr.get("obj_rot")) if pr.get("obj_rot") not in (None, "", "none") else None
    out = []
    if R_REL is not None:
        _dims_h = np.array([float(dims[0]), float(dims[1]), OBJ_H]); _Robj = (R_REL * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_matrix()
        HZ = 0.5 * float(np.abs(_Robj[2, :]) @ _dims_h); O_R = CONTACT0 - np.array([CTR[0], CTR[1], BASE_Z + OBJ_H / 2.0])
        for slot in pr["slots"]:
            tz = float(slot[2]) if len(slot) > 2 else top_z
            for yw in yaws:
                Rg = Rot.from_euler("z", yw, degrees=True) * R_REL; a = Rg.apply(A0)
                if pr.get("_fp_obs") and _fp_hits(float(slot[0]), float(slot[1]), (Rg * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_matrix(), _dims_h / 2.0, pr["_fp_obs"]): continue      # 이 자리·각도면 물체가 다른 물체와 겹침
                tcp = np.array([float(slot[0]), float(slot[1]), tz + clear + HZ]) + Rg.apply(O_R); fl_ = tcp - D_TCP * a; q_ = (Rg * R0).as_quat()
                if pr.get("auto_clear"): fl_ = fl_ + np.array([0.0, 0.0, _auto_clear_raise(pr, fl_, q_, tz, clear)[0]])
                out.append((fl_.tolist(), q_.tolist()))
        return out
    if pr.get("mode"): modes = [pr["mode"]]
    else: modes = ["same"] + (["plumb"] if theta0 < float(pr.get("plumb_deg", 45)) else [])
    for md in modes:
        QA = Rot.align_vectors([[0.0, 0.0, -1.0]], [A0])[0] if md == "plumb" else Rot.identity()
        for slot in pr["slots"]:
            tz = float(slot[2]) if len(slot) > 2 else top_z
            for yw in yaws:
                Rz = Rot.from_euler("z", yw, degrees=True)
                if md == "same" and pr.get("_fp_obs") and _fp_hits(float(slot[0]), float(slot[1]), (Rz * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_matrix(), np.array([float(dims[0]), float(dims[1]), OBJ_H]) / 2.0, pr["_fp_obs"]): continue
                q = (Rz * QA * R0).as_quat(); a = Rz.apply(QA.apply(A0)); o = Rz.apply(O_W0)
                tcp = np.array([slot[0] + o[0], slot[1] + o[1], tz + clear + HELD_H]); fl_ = tcp - D_TCP * a
                if pr.get("auto_clear"): fl_ = fl_ + np.array([0.0, 0.0, _auto_clear_raise(pr, fl_, q, tz, clear)[0]])
                out.append((fl_.tolist(), q.tolist()))
    return out


def _hand_in_obs_fq(fl, quat, obs):
    """손 전체(손가락 포함) 구가 obs 상자들 속으로 파고든 최대 깊이(mm)."""
    _g = _gripper_sphere_offsets(); _hs = np.concatenate([_g["finger"], _g["cam"], _g["palm"], _g["l7"]], axis=0)
    _P = np.asarray(fl, float) + Rot.from_quat(quat).apply(_hs[:, :3]); _r = _hs[:, 3]; best = 0.0
    for o in obs:
        raw = o.get("raw") or o["dims"]; _L = Rot.from_euler("z", -float(o.get("yaw", 0.0))).apply(_P - np.asarray(o["xyz"], float))
        pen = (np.asarray(raw, float) / 2.0 + _r[:, None]) - np.abs(_L); inside = (pen > 0).all(axis=1)
        if inside.any(): best = max(best, float(pen[inside].min(axis=1).max()))
    return 1000.0 * best


def _hand_in_dest_fq(fl, quat, dest_obs):
    """놓는 플랜지 자세(fl, quat xyzw)에서 카메라/손바닥/link7 구가 도착지 상자 속으로 파고든 깊이(mm). _solve_place_impl 의 _hand_in_dest 와 같은 식."""
    _g = _gripper_sphere_offsets(); _hs = np.concatenate([_g["cam"], _g["palm"], _g["l7"]], axis=0)
    _P = np.asarray(fl, float) + Rot.from_quat(quat).apply(_hs[:, :3]); _r = _hs[:, 3]; best = 0.0
    for o in dest_obs:
        raw = o.get("raw") or o["dims"]; yw = float(o.get("yaw", 0.0)); _L = Rot.from_euler("z", -yw).apply(_P - np.asarray(o["xyz"], float))
        pen = (np.asarray(raw, float) / 2.0 + _r[:, None]) - np.abs(_L); inside = (pen > 0).all(axis=1)
        if inside.any(): best = max(best, float(pen[inside].min(axis=1).max()))
    return 1000.0 * best


def _place_prescreen(req, cands):
    """2026-11-06 place_prescreen(기본 꺼짐, req["place_prescreen"]=True): 후보마다 (a) 집기 플랜지 자세 IK, (b) 놓기 목표 플랜지 자세 IK(슬롯 x yaw (x 모드/obj_rot)) 를
    한 번의 배치 IK(solve_ik, 충돌 포함)로 본다. 순서 = (집기 IK 가능 & 놓기 IK 1개 이상) 후보 먼저, 나머지는 원래 순서 그대로 뒤에(안정 분할: 후보는 하나도 빼지 않는다).
    IK 는 확률적이라 한 번 실패가 불가능의 증거가 아니므로 빼지 않는 것이 기본. req["place_prescreen_drop"]=True(측정용)면 놓기 IK 0개 후보를 뺀다(전부 빠지면 원래 순서)."""
    t_ = time.time(); pr = req["place_check"]["req"]; poses = []; idx = []
    _hdr = bool(pr.get("hand_dest_reject")); _hdm = float(pr.get("hand_dest_margin_mm", 2.0)); _dids = set(pr.get("dest_ids", [])); _dobs = [o for o in pr.get("obstacles", []) if o["id"] in _dids]; _pobs = []
    try:
        _rmin = 0.5 * float(min([float(v) for v in (pr["obj"].get("dims") or [0.05, 0.05])[:2]] + [float(pr["obj"]["height"])]))      # 물체가 어떤 자세·yaw 여도 바닥 윤곽 안에 드는 원의 반지름(가장 짧은 반폭)
        _oth = [o for o in pr.get("obstacles", []) if o["id"] not in _dids and float((o.get("raw") or o["dims"])[2]) >= 0.008]
        def _blocked(sl):
            for o in _oth:
                raw = o.get("raw") or o["dims"]; L = Rot.from_euler("z", -float(o.get("yaw", 0.0))).apply(np.array([float(sl[0]) - o["xyz"][0], float(sl[1]) - o["xyz"][1], 0.0]))
                if abs(L[0]) < raw[0] / 2.0 + _rmin and abs(L[1]) < raw[1] / 2.0 + _rmin and (o["xyz"][2] - raw[2] / 2.0) < float(sl[2] if len(sl) > 2 else pr["top_z"]) + 0.05: return True
            return False
        _sl = [sl for sl in pr["slots"] if not _blocked(sl)]
        _tz = float(pr["top_z"]); _fo = []
        for o in _oth:
            raw = o.get("raw") or o["dims"]
            if (o["xyz"][2] - raw[2] / 2.0) < _tz + 0.05 and (o["xyz"][2] + raw[2] / 2.0) > _tz + 0.005:      # 받침면 위에 놓인 것
                c_, s_ = math.cos(float(o.get("yaw", 0.0))), math.sin(float(o.get("yaw", 0.0)))
                _fo.append((float(o["xyz"][0]), float(o["xyz"][1]), np.array([c_, s_]) * raw[0] / 2.0, np.array([-s_, c_]) * raw[1] / 2.0))
        if _fo: pr = dict(pr, _fp_obs=_fo)
        _pobs = [o for o in _oth if (o["xyz"][2] - (o.get("raw") or o["dims"])[2] / 2.0) < _tz + 0.05 and (o["xyz"][2] + (o.get("raw") or o["dims"])[2] / 2.0) > _tz + 0.005]
        if len(_sl) != len(pr["slots"]): pr = dict(pr, slots=_sl)
    except Exception as _x:
        print("사전 선별 자리 점유 검사 예외(그대로 진행)", repr(_x), flush=True)
    for i, c in enumerate(cands):
        g = dict(quat=[float(x) for x in c["quat"]], approach=[float(x) for x in c["approach"]], flange=[float(x) for x in c["flange"]])
        poses.append(dict(flange=g["flange"], quat=g["quat"])); idx.append((i, "pick"))
        for fl_, q_ in _place_goal_poses(pr, g):
            if _hdr and _hand_in_dest_fq(fl_, q_, _dobs) > _hdm: continue
            if _pobs and _hand_in_obs_fq(fl_, q_, _pobs) > 2.0: continue      # 손이 받침대 위 다른 물체에 닿는 놓기      # 실제 놓기 단계에서 손 거르기로 버려질 자세는 IK 전에 뺀다(같은 식)
            poses.append(dict(flange=[float(x) for x in fl_], quat=[float(x) for x in q_])); idx.append((i, "place"))
    q = solve_ik({"poses": poses, "check_collision": True})["q"]
    pick_ok = [False] * len(cands); n_pl = [0] * len(cands); n_tot = [0] * len(cands); pl_poses = [[] for _ in cands]
    for (i, kind), qq, p_ in zip(idx, q, poses):
        if kind == "pick": pick_ok[i] = qq is not None
        else:
            n_tot[i] += 1; pl_poses[i].append(p_["flange"])
            if qq is not None: n_pl[i] += 1
    good = [i for i in range(len(cands)) if pick_ok[i] and n_pl[i] > 0]; rest = [i for i in range(len(cands)) if i not in good]
    order = good + rest; dropped = []
    if req.get("place_prescreen_drop"):
        keep = [i for i in order if n_pl[i] > 0]
        if keep: dropped = [i for i in order if n_pl[i] == 0]; order = keep
    info = dict(pick_ik=[int(x) for x in pick_ok], place_ik_n=n_pl, place_poses_n=n_tot, order=order, dropped=dropped, n_good=len(good), n_poses=len(poses), sec=round(time.time() - t_, 3))
    return [cands[i] for i in order], info, pl_poses


def _parse_obj_rot(v):
    """2026-11-06 goalpose: obj_rot -> scipy Rotation 또는 None. 허용: None/""/"none", [x,y,z,w], {"quat":[...]}, "x90"/"x-90"/"y90"/"x180" 같은 축+각도(도) 문자열(여러 개는 '_' 또는 ',' 로 이어 왼쪽부터 차례로 적용)."""
    if v is None: return None
    if isinstance(v, dict): v = v.get("quat")
    if isinstance(v, str):
        s_ = v.strip().lower()
        if s_ in ("", "none", "0", "id", "identity"): return None
        if s_.startswith("[") or s_.startswith("{"): return _parse_obj_rot(json.loads(s_))
        R_ = Rot.identity()
        for tok in s_.replace(",", "_").split("_"):
            tok = tok.strip()
            if not tok: continue
            if tok[0] not in "xyz": raise ValueError("obj_rot 해석 불가: %r" % v)
            R_ = Rot.from_euler(tok[0], float(tok[1:]), degrees=True) * R_
        return R_
    q_ = np.asarray(v, float).reshape(-1)
    if q_.size != 4 or np.linalg.norm(q_) < 1e-9: raise ValueError("obj_rot 쿼터니언(xyzw) 4개 필요: %r" % (v,))
    return Rot.from_quat(q_ / np.linalg.norm(q_))


def _solve_place_impl(req):
    t0 = time.time(); why = []
    _reset_retry_budget(req)
    start_q = [float(x) for x in req["start_q"]]
    g = req["grasp"]; ob = req["obj"]
    Q0 = np.array(g["quat"], float); A0 = np.array(g["approach"], float); A0 /= np.linalg.norm(A0); FL0 = np.array(g["flange"], float)
    CTR = np.array(ob["centre"], float); BASE_Z = float(ob.get("base_z", 0.0)); OBJ_H = float(ob["height"]); dims = ob.get("dims") or [0.06, 0.06, OBJ_H]
    CONTACT0 = FL0 + D_TCP * A0
    HELD_H = float(CONTACT0[2] - BASE_Z)
    O_W0 = CONTACT0 - np.array([CTR[0], CTR[1], BASE_Z + OBJ_H / 2.0])
    R0 = Rot.from_quat(Q0)
    theta0 = math.degrees(math.acos(max(-1.0, min(1.0, -A0[2]))))
    mode = req.get("mode") or ("plumb" if theta0 < float(req.get("plumb_deg", 45)) else "same")
    QA = Rot.align_vectors([[0.0, 0.0, -1.0]], [A0])[0] if mode == "plumb" else Rot.identity()
    top_z = float(req["top_z"]); clear = float(req.get("clear", 0.006)); slots = req["slots"]
    # 2026-11-06 goalpose(복사본 전용): req["obj_rot"] = 물체에 월드 기준으로 적용할 회전(xyzw 쿼터니언 또는 "x90" 같은 문자열, 물체 중심 둘레, yaw 보다 먼저).
    # 없으면 아래 기존 경로와 완전히 같다. 있으면 QA(plumb) 를 쓰지 않고(same 의미) 손을 물체와 함께 Rg = Rz(yaw)*R_rel 로 돌린다.
    R_REL = _parse_obj_rot(req.get("obj_rot"))
    if R_REL is not None:
        mode = "goal_rot"; QA = Rot.identity()
        _dims_h = np.array([float(dims[0]), float(dims[1]), OBJ_H])            # 중심 높이 정의(BASE_Z+OBJ_H/2)와 맞추려고 높이 축은 OBJ_H 를 쓴다
        _Robj = (R_REL * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_matrix()      # 물체 박스 축(서 있는 상태, ob.yaw) -> 회전 후 축
        HZ_ROT = 0.5 * float(np.abs(_Robj[2, :]) @ _dims_h)                    # 회전 후 수직 반높이(축정렬 박스 근사). yaw 는 z 성분을 바꾸지 않는다
        _cn = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * (_dims_h / 2.0)
        RAD_ROT = float(np.max(np.hypot(*(_cn @ _Robj.T)[:, :2].T)))           # 회전 후 바닥 윤곽의 외접 반경(yaw 무관)
        CTR3 = np.array([CTR[0], CTR[1], BASE_Z + OBJ_H / 2.0]); O_W0R = CONTACT0 - CTR3      # 물체 중심 -> 접촉점(월드, 집은 자세)
    if req.get("slot_filter", True):      # 다른 물체와 겹치는 슬롯은 뒤로 미룬다(빈 슬롯부터 시도). 이전엔 슬롯 0(접시 중심) 아래 후보 6개가 전부 장애물 위라 다른 슬롯은 시도도 못 했다
        _rad = 0.5 * math.hypot(float(req["obj"]["dims"][0]), float(req["obj"]["dims"][1])) + 0.01; _dids = set(req.get("dest_ids", []))
        if R_REL is not None: _rad = RAD_ROT + 0.01          # goalpose: 눕히면 바닥 윤곽이 커진다
        _blk = [_slot_blocker(sl, _rad, req.get("obstacles", []), _dids) for sl in slots]
        slots = [sl for sl, b in zip(slots, _blk) if b is None] + [sl for sl, b in zip(slots, _blk) if b is not None]
        _n_blocked = sum(b is not None for b in _blk)
    else:
        _n_blocked = 0
    yaws = req.get("yaws") or PLACE_YAWS; K = int(req.get("max_cands", 6)); pre_dzs = (0.08, 0.04)
    ret_len = float(req.get("retreat", 0.10)); margin = float(req.get("margin", 0.005)); budget = float(req.get("budget_s", 8.0)) * _BS
    deadline = t0 + budget; _DEADLINE["t"] = deadline
    obstacles = req.get("obstacles", []); dest_ids = set(req.get("dest_ids", []))
    _start_col = None
    if req.get("start_collision_report") and obstacles:      # 2026-11-07(기본 꺼짐): 시작 자세에서 로봇 충돌 구가 장애물 속에 있는지(mm, 음수=침범). transit 이 'Start state in collision' 으로 실패하는지 가리는 용도
        try:
            _sd = solve_sphere_dist({"q": [start_q], "boxes": [dict(id=o["id"], xyz=o["xyz"], dims=(o.get("raw") or o["dims"]), yaw=float(o.get("yaw", 0.0))) for o in obstacles]})["per_box"]
            _start_col = {k: [round(1000 * v["min"][0], 1), v["link"][0]] for k, v in _sd.items() if v["min"][0] < 0}
        except Exception as _e:
            _start_col = {"error": repr(_e)}
    # 1) 후보 생성: (슬롯 우선순위, yaw) -> 플랜지 자세
    cands = []
    for si, slot in enumerate(slots):
        for yw in yaws:
            Rz = Rot.from_euler("z", yw, degrees=True)
            tz = float(slot[2]) if len(slot) > 2 else top_z       # 슬롯별 면 높이(오목한 도착지 등)
            if R_REL is not None:      # goalpose: 목표 물체 중심 c_goal, 손은 물체와 강체로 같이 돈다
                Rg = Rz * R_REL
                q = (Rg * R0).as_quat(); a = Rg.apply(A0)
                c_goal = np.array([float(slot[0]), float(slot[1]), tz + clear + HZ_ROT])
                tcp = c_goal + Rg.apply(O_W0R); fl = tcp - D_TCP * a
                cands.append(dict(si=si, slot=list(map(float, slot))[:2], tz=tz, yaw=yw, quat=q, a=a, tcp=tcp, fl=fl, c_goal=c_goal, Rg=Rg))
                continue
            q = (Rz * QA * R0).as_quat(); a = Rz.apply(QA.apply(A0)); o = Rz.apply(O_W0)
            tcp = np.array([slot[0] + o[0], slot[1] + o[1], tz + clear + HELD_H]); fl = tcp - D_TCP * a
            cands.append(dict(si=si, slot=list(map(float, slot))[:2], tz=tz, yaw=yw, quat=q, a=a, tcp=tcp, fl=fl))
    if req.get("auto_clear") or req.get("clear_report"):      # 2026-11-06(복사본, 기본 꺼짐): auto_clear = 손가락이 받침면에 걸리지 않게 후보마다 놓는 높이를 올린다 / clear_report = 측정만(자세 불변)
        for c in cands:
            _rz, _zf, _cap = _auto_clear_raise(req, c["fl"], c["quat"], c["tz"], clear)
            if _rz > 0:
                _dz = np.array([0.0, 0.0, _rz]); c["tcp"] = c["tcp"] + _dz; c["fl"] = c["fl"] + _dz
                if c.get("c_goal") is not None: c["c_goal"] = c["c_goal"] + _dz
            c["ac"] = dict(clear_used=clear + _rz, raise_=_rz, finger_before=_zf, finger_after=_zf + _rz, capped=bool(_cap and req.get("auto_clear")),
                           palm_after=_sph_min_z(_gripper_sphere_offsets()["palm"], c["fl"], c["quat"]) - c["tz"])
    if req.get("finger_reject") and not req.get("_nb_only"):      # 2026-11-08(복사본, 기본 꺼짐): 하강(r2)은 도착지를 충돌에서 뺀 월드로 계획되므로, 놓는 자세에서 손가락이 받침면 아래로 파고드는 후보가 걸러지지 않았다
        # (실측: 물체를 굴려 눕히는 회전에서 손가락 닫힘 축이 연직이 되어 아래 손가락이 물체와 면 사이에 끼고 물체가 25~55mm 떠서 손에 걸림 12/12). 그런 후보는 버린다(올리지 않음).
        _m = float(req.get("finger_reject_margin", 0.005)); _n0 = len(cands); _g = _gripper_sphere_offsets(); _kept = []
        _sph_all = np.concatenate([_g["finger"], _g["palm"]], axis=0); _n_under = 0; _n_pen = 0
        _rep = (req.get("finger_reject") == "report")      # report: 거르지 않고 후보별 진단만 기록(원인 분석용)
        _cl0 = _g["f1"][:, :3].mean(0) - _g["f2"][:, :3].mean(0); _cl0 = _cl0 / (np.linalg.norm(_cl0) or 1.0)      # 손가락 닫힘 축(플랜지 좌표)
        _dest_obs = [o for o in obstacles if o["id"] in dest_ids]; _hs = np.concatenate([_g["cam"], _g["palm"], _g["l7"]], axis=0); _n_hand = 0      # 2026-11-09: 손가락 외 손(카메라/손바닥/link7) 구
        def _hand_in_dest(c):
            # 하강(r2)·후퇴는 도착지를 뺀 월드로 계획 -> 놓는 자세에서 손가락 외 손 구가 도착지 상자(실제 크기, 여유 없음) 안으로 파고드는지 직접 검사. 파고든 깊이(mm, 0 = 안 닿음)
            _P = np.asarray(c["fl"], float) + Rot.from_quat(c["quat"]).apply(_hs[:, :3]); _r = _hs[:, 3]; best = 0.0
            for o in _dest_obs:
                raw = o.get("raw") or o["dims"]; yw = float(o.get("yaw", 0.0)); _L = Rot.from_euler("z", -yw).apply(_P - np.asarray(o["xyz"], float))
                pen = (np.asarray(raw, float) / 2.0 + _r[:, None]) - np.abs(_L)      # 모든 축에서 양수면 구가 상자와 겹침
                inside = (pen > 0).all(axis=1)
                if inside.any(): best = max(best, float(pen[inside].min(axis=1).max()))
            return round(1000 * best, 1)
        for c in cands:
            _zf = _sph_min_z(_g["finger"], c["fl"], c["quat"]) - c["tz"]
            _clz = float(Rot.from_quat(c["quat"]).apply(_cl0)[2]); _under = None
            if c.get("c_goal") is not None and R_REL is not None:      # 받침 검사: 손(손가락+손바닥) 구 중심이 물체 바닥면보다 아래이면서 물체 바닥 윤곽 안에 있으면, 손을 펴도 물체가 손 위에 얹힌다(각도가 아니라 위치 관계로 판정)
                _Rob = (c["Rg"] * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_matrix(); _hd = np.array([float(dims[0]), float(dims[1]), OBJ_H]) / 2.0
                _P = np.asarray(c["fl"], float) + Rot.from_quat(c["quat"]).apply(_sph_all[:, :3]); _L = (_P - c["c_goal"]) @ _Rob      # 물체 좌표
                _vz = int(np.argmax(np.abs(_Rob[2, :]))); _hz = [i for i in range(3) if i != _vz]
                _below = _P[:, 2] < (c["c_goal"][2] - HZ_ROT)
                _inside = (np.abs(_L[:, _hz[0]]) < _hd[_hz[0]]) & (np.abs(_L[:, _hz[1]]) < _hd[_hz[1]])
                _under = bool(np.any(_below & _inside))
            c["fdiag"] = dict(a_z=round(float(c["a"][2]), 3), close_z=round(_clz, 3), finger_min_mm=round(1000 * _zf, 1), under_obj=_under, yaw=c["yaw"],
                             cam_min_mm=round(1000 * (_sph_min_z(_g["cam"], c["fl"], c["quat"]) - c["tz"]), 1), palm_min_mm=round(1000 * (_sph_min_z(_g["palm"], c["fl"], c["quat"]) - c["tz"]), 1),
                             l7_min_mm=round(1000 * (_sph_min_z(_g["l7"], c["fl"], c["quat"]) - c["tz"]), 1), hand_in_dest_mm=_hand_in_dest(c))
            if req.get("hand_dest_reject") and c["fdiag"]["hand_in_dest_mm"] > float(req.get("hand_dest_margin_mm", 2.0)): _n_hand += 1; continue      # 카메라 등이 받침대 속으로 들어가는 놓기는 실제로 부딪혀 덜 내려가고 후퇴도 막힘(실측 2/2, -26~-28mm) -> 버림      # 2026-11-09 진단: 하강(r2)은 도착지를 뺀 월드라 카메라/손바닥/link7 이 받침면 아래로 들어가도 안 걸림(상자 눕히기에서 카메라가 작업대에 걸림 관찰)
            if not _rep:      # 2026-11-08 진단 8장면 근거: 손에 걸림(+25~38mm) 4/4 = 닫힘 축 |z|>=0.98(ㄷ자 아래턱이 물체 밑), 정상 2/2 = |z|<=0.06. 접근 방향 z(위/아래)는 -0.08~+0.08 로 가르지 못함.
                if abs(_clz) > float(req.get("close_z_max", 0.7)): _n_under += 1; continue      # 좁은 규칙: 닫힘 축이 연직에 가까운 놓기(ㄷ자 턱 하나가 물체 밑)만 버린다. 파고듦/물체 밑 판정은 진단으로만 남김
            _kept.append(c)
        print("finger_reject: 후보 %d -> %d (닫힘축 연직(ㄷ자 턱이 물체 밑) %d 제외, 손(카메라 등)이 도착지 속 %d 제외)" % (_n0, len(_kept), _n_under, _n_hand), flush=True)
        cands = _kept
        if not cands:
            return dict(ok=False, mode=mode, n_cands=0, n_ik_ok=0, ik_rejects={"finger_reject": _n0}, tried=[], slots_blocked=0, why="finger_reject: 모든 후보가 걸러짐(닫힘축 연직 또는 손이 도착지 속)", sec=round(time.time() - t0, 3))
    if req.get("neighbor_reject"):      # 2026-11-10(기본 꺼짐, 8021 사본): 놓은 뒤 물체 상자(회전 상자, 실제 치수)가 이웃 장애물(도착지 제외)과 겹치는 후보를 버린다.
        # 근거: 든 물체를 구로 근사(held_cover old)하면 상자 모서리가 빠져 블록과 3mm 겹치는 놓기가 통과(run 007/011), 외접구(bound)는 바닥 구가 받침면에 걸려 되던 놓기를 막음(A/B). -> 최종 자세만 정확히 검사.
        _nm = float(req.get("neighbor_margin", 0.002)); _nb = [o for o in obstacles if o["id"] not in dest_ids and float((o.get("raw") or o["dims"])[2]) >= 0.008]; _kept = []; _n_nb = 0
        def _sat(c1, R1, h1, c2, R2, h2):
            d = c2 - c1; ax = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)] + [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]; pen = 1e9
            for a_ in ax:
                n_ = float(np.linalg.norm(a_))
                if n_ < 1e-6: continue
                a_ = a_ / n_; pen = min(pen, float(np.abs(R1.T @ a_) @ h1 + np.abs(R2.T @ a_) @ h2 - abs(a_ @ d)))
            return pen
        _hd = np.array([float(dims[0]), float(dims[1]), OBJ_H]) / 2.0
        for c in cands:
            if c.get("c_goal") is not None and c.get("Rg") is not None: _cg = np.asarray(c["c_goal"], float); _Ro = (c["Rg"] * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_matrix()
            else: _cg = np.array([c["slot"][0], c["slot"][1], c["tz"] + clear + OBJ_H / 2.0]); _Ro = Rot.from_euler("z", float(c["yaw"]), degrees=True).as_matrix() @ Rot.from_euler("z", float(ob.get("yaw", 0.0))).as_matrix()
            hit = False
            for o in _nb:
                raw = np.asarray(o.get("raw") or o["dims"], float); _Rn = Rot.from_euler("z", float(o.get("yaw", 0.0))).as_matrix()
                if _sat(_cg, _Ro, _hd, np.asarray(o["xyz"], float), _Rn, raw / 2.0) > -_nm: hit = True; break
            if hit: _n_nb += 1; continue
            _kept.append(c)
        if req.get("_nb_only"): return dict(nb_only=True, n_all=len(cands), n_left=len(_kept))      # 2026-11-10 nb_first: 이웃 겹침만 보는 마른 호출(파지와 무관한 검사)
        print("neighbor_reject: 후보 %d -> %d (놓은 물체가 이웃과 겹침 %d 제외)" % (len(cands), len(_kept), _n_nb), flush=True)
        cands = _kept
        if not cands:
            return dict(ok=False, mode=mode, n_cands=0, n_ik_ok=0, ik_rejects={"neighbor_reject": _n_nb}, tried=[], slots_blocked=0, why="놓을 자리에 다른 물체가 있어 놓을 수 없음(놓은 물체가 이웃과 겹침)", sec=round(time.time() - t0, 3))
    poses = []
    for c in cands:
        poses.append(dict(flange=[float(x) for x in c["fl"]], quat=[float(x) for x in c["quat"]]))
        for dz in pre_dzs:
            poses.append(dict(flange=[float(c["fl"][0]), float(c["fl"][1]), float(c["fl"][2] + dz)], quat=[float(x) for x in c["quat"]]))
    if req.get("ik_mode", "chain") == "chain":
        # 사전 자세(pre) -> 놓는 자세(place) 쌍을 연속 해로 푼다(독립 IK 는 두 해의 분기가 달라 "pre/place 해 분기 다름"으로 버려지는 후보가 많다)
        ik = [None] * len(poses); _chs = []
        for i, c in enumerate(cands):
            for k, dz in enumerate(pre_dzs):
                _chs.append([poses[i * 3 + 1 + k], poses[i * 3]])
        _res = _ik_chain(_chs, max_jump=0.8)
        for i in range(len(cands)):
            for k in range(len(pre_dzs)):
                rr = _res[i * len(pre_dzs) + k]
                if rr is not None:
                    ik[i * 3] = rr[1]; ik[i * 3 + 1 + k] = rr[0]; break          # 첫 번째로 성공한 (pre, place) 쌍만 채워 아래 로직과 어긋나지 않게 한다
    else:
        ik = solve_ik({"poses": poses, "check_collision": True})["q"]
    pre_mode = req.get("pre_mode", "vertical")      # 2026-11-04: 'vertical'(기본, 현행) | 'auto'(수직 사전 자세가 충돌/불가일 때만 접근축 반대 사전 자세 pre = flange − dz·approach 도 시도)
    ikA = [None] * len(cands)
    if pre_mode == "auto" and req.get("ik_mode", "chain") == "chain":
        _chsA = []; _idxA = []
        for i, c in enumerate(cands):
            if abs(float(c["a"][2])) > 0.95: continue          # 접근축이 거의 수직이면 접근축 사전 자세 = 수직 사전 자세(중복)
            for k, dz in enumerate(pre_dzs):
                _chsA.append([dict(flange=[float(x) for x in (c["fl"] - dz * c["a"])], quat=[float(x) for x in c["quat"]]), poses[i * 3]]); _idxA.append((i, dz))
        for (i, dz), rr in zip(_idxA, _ik_chain(_chsA, max_jump=0.8) if _chsA else []):
            if rr is not None and ikA[i] is None: ikA[i] = (rr[0], rr[1], dz)
    feas = []
    for i, c in enumerate(cands):
        pres, why_ = _pre_candidates(ik[i * 3], [(dz, ik[i * 3 + 1 + k]) for k, dz in enumerate(pre_dzs)], ikA[i])
        if not pres:
            c["why"] = why_; continue
        qp, qpre, dzu = pres[0]["q_pl"], pres[0]["q_pre"], pres[0]["dz"]
        if _jmargin(qp) < float(req.get("min_margin", 0.05)):
            c["why"] = "관절 한계 근접"; continue
        c.update(q_pl=qp, q_pre=qpre, dz=dzu, pres=pres,
                 cost=float(np.sum(np.abs(np.array(qp) - start_q))) + 0.5 * float(np.sum(np.abs(np.array(qpre) - start_q))) - 0.5 * _jmargin(qp) + 0.03 * abs(c["yaw"]) / 180.0)
        feas.append(c)
    feas.sort(key=lambda c: (c["si"], c["cost"]))
    _adaptive_retry_budget(len(feas), req)
    order = feas[:K + int(req.get("sane_extra", 4))]      # 안정장치가 후보를 버려도 대체 후보를 시도할 수 있게 여유 4개(시간 예산이 상한)
    n_why = {}
    for c in cands:
        if c.get("why"):
            n_why[c["why"]] = n_why.get(c["why"], 0) + 1
    out = dict(ok=False, mode=mode, n_cands=len(cands), n_ik_ok=len(feas), ik_rejects=n_why, tried=[], slots_blocked=_n_blocked)
    if _start_col is not None: out["start_collision"] = _start_col
    if cands and cands[0].get("ac") is not None:      # auto_clear/clear_report 켰을 때만: 후보 전체의 [손가락 최저점-받침면(올리기 전), 올린 양, 상한 걸림] (mm)
        out["auto_clear_cands"] = [[round(1000 * c["ac"]["finger_before"], 1), round(1000 * c["ac"]["raise_"], 1), int(c["ac"]["capped"])] for c in cands]
    # 2) 계획: 손에 든 물체를 붙이고 transit -> 하강 -> (놓고) 후퇴. 후보마다 완화 사다리(원래 -> 테이블 낮춤+이웃 여유 0 -> 거기에 직선 하강 해제)로 시도한다.
    #    놓기는 본래 지지면에 닿는 동작이라 든 물체 구가 테이블 근처(여유 2cm)와 이웃에 과하게 걸리면 계획이 안 된다(실패 요청 재생: 부착 시 2/14 -> 테이블 낮춤 7/14).
    sph = _held_spheres(R0, FL0, [CTR[0], CTR[1], BASE_Z + OBJ_H / 2.0], dims, float(ob.get("yaw", 0.0)), pad=float(req.get("sphere_pad", 0.004)), cover=req.get("held_cover", "old"))
    dt_i = P.trajopt_solver.config.interpolation_dt
    dts = (float(req.get("transit_dt", 0.1625)), float(req.get("descend_dt", 0.25)), float(req.get("retreat_dt", 0.25)))
    tmin = (float(req.get("transit_tmin", 1.0)), float(req.get("descend_tmin", 1.2)), float(req.get("retreat_tmin", 1.0)))
    relax_tt = float(req.get("relax_table_top", -0.06))
    LADDER = req.get("ladder") or [dict(name="L0", tt=req.get("table_top"), mg=margin, lin=True),
                                   dict(name="L1", tt=relax_tt, mg=0.0, lin=True),
                                   dict(name="L2", tt=relax_tt, mg=0.0, lin=False)]
    held = {"on": False}
    ret_used = {"v": ret_len}

    def attach():
        if not req.get("no_attach", False):
            P.attachment_manager.update(torch.tensor(sph, device="cuda", dtype=torch.float32), _js(start_q), "attached_object"); held["on"] = True

    def detach():
        if held["on"]:
            P.attachment_manager.detach("attached_object"); held["on"] = False

    def plan_c(c, lv):
        """사전 자세 목록(c['pres']: 수직 먼저, auto 면 접근축 반대 뒤)을 순서대로 시도해 첫 성공을 쓴다. 목록이 하나(수직)면 현행과 동일."""
        pres = c.get("pres") or [dict(kind="vertical", dz=c["dz"])]
        err = None
        for pr_ in pres:
            res_, err = _plan_c_one(c, lv, pr_["kind"], pr_["dz"])
            if res_ is not None:
                c["pre_kind_used"] = pr_["kind"]; c["pre_dz_used"] = pr_["dz"]; return res_, None
            detach()
            if len(pres) > 1 and time.time() > deadline: break
        return None, (err if len(pres) == 1 else "%s [사전자세 %s]" % (err, "+".join(p["kind"] for p in pres)))

    def _plan_c_one(c, lv, pre_kind, pre_dz):
        quat = [float(x) for x in c["quat"]]; fl = c["fl"]
        pre_fl = ([float(fl[0]), float(fl[1]), float(fl[2] + pre_dz)] if pre_kind == "vertical" else [float(x) for x in (fl - pre_dz * c["a"])])
        sc_all = {"cuboid": _cuboids(obstacles, lv["mg"], 0.0, (), lv["tt"])}; sc_nodest = {"cuboid": _cuboids(obstacles, lv["mg"], 0.0, dest_ids, lv["tt"])}
        attach()
        P.update_world(Scene.create(sc_all))
        r1 = _plan_pose_r(_goal(pre_fl, quat), _js(start_q))
        if not _ok(r1):
            return None, "transit 계획 실패"
        _dd = req.get("descend_with_dest")      # 2026-11-09 v4(사용자 보고: 옮기다 집은 물체가 받침대에 걸림): 하강을 도착지를 뺀 월드로 계획하면 물체/손이 받침대 모서리를 쓸고 지나가도 안 걸린다.
        # -> 도착지를 넣은(여유 5mm 이하) 월드로 하강을 계획. "strict" = 그게 안 되면 이 후보를 버림, 그 밖의 참 값 = 안 되면 예전처럼 도착지를 빼고 계획.
        P.update_world(Scene.create({"cuboid": _cuboids(obstacles, min(lv["mg"], 0.005), 0.0, (), lv["tt"])} if _dd else sc_nodest))
        if mode == "plumb" and lv["lin"]:
            P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True) for k in P.tool_frames})
        try:
            r2 = _plan_pose_r(_goal([float(x) for x in fl], quat), _end_state(r1))
        finally:
            P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
        if _dd and not _ok(r2) and _dd != "strict":
            P.update_world(Scene.create(sc_nodest))
            if mode == "plumb" and lv["lin"]:
                P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True) for k in P.tool_frames})
            try:
                r2 = _plan_pose_r(_goal([float(x) for x in fl], quat), _end_state(r1))
            finally:
                P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
        if not _ok(r2):
            return None, ("하강 계획 실패(도착지 포함)" if _dd == "strict" else "하강 계획 실패")
        detach()
        P.update_world(Scene.create(sc_nodest))        # 후퇴 시작 자세는 손가락이 도착지 면 바로 위라(여유 안) 도착지는 충돌에서 뺀다. 테이블/이웃은 그대로
        def _retreat():
            r3 = None
            if req.get("retreat_mode") == "up":
                # 2026-10-02 실험: 접근축(기울어진 손에서는 옆으로 쓸린다) 대신 월드 수직으로 먼저 올라간다. 경로가 직선에서 12mm 넘게 벗어나면 접근축 후퇴로 되돌린다.
                r3 = _plan_pose_r(_goal([float(fl[0]), float(fl[1]), float(fl[2] + ret_len)], quat), _end_state(r2))
                if _ok(r3):
                    w3_ = _wps(r3)
                    q_ = torch.tensor(w3_[:, :7], device="cuda", dtype=torch.float32)
                    pos_ = P.kinematics.compute_kinematics(JointState.from_position(q_, joint_names=P.joint_names)).tool_poses.get_link_pose(P.tool_frames[0]).position.reshape(-1, 3).detach().cpu().numpy()
                    a_, b_ = pos_[0], pos_[-1]; ab = b_ - a_; nab = float(np.linalg.norm(ab)) + 1e-9
                    dev = float(np.max(np.linalg.norm(np.cross(pos_ - a_, ab / nab), axis=1)))
                    if dev > 0.012:
                        r3 = None
            if r3 is None or not _ok(r3):
                P.update_tool_pose_criteria({k: ToolPoseCriteria.linear_motion(axis="z", non_terminal_scale=1.0, project_distance_to_goal=True) for k in P.tool_frames})
                try:
                    # 후퇴 길이 사다리(2026-10-03): 기본 10cm 후퇴가 막히는 요청이 많다(저장된 후퇴 실패 22건: 10cm 1건, 5cm 22건 복구). 손이 물체에서 빠질 만큼만 가면 되므로 줄여 가며 시도한다(pick lift 와 같은 방식).
                    _lens = [ret_len] + ([x for x in (0.07, 0.05, 0.03) if x < ret_len] if req.get("retreat_ladder", True) else [])
                    for rl_ in [_lens[i] for i in _reach_order([(i, [float(x) for x in (fl - _lens[i] * c["a"])]) for i in range(len(_lens))], quat)]:
                        r3 = _plan_pose_r(_goal([float(x) for x in (fl - rl_ * c["a"])], quat), _end_state(r2))
                        if _ok(r3):
                            ret_used["v"] = rl_; break
                finally:
                    P.update_tool_pose_criteria({k: ToolPoseCriteria() for k in P.tool_frames})
            return r3
        r3 = None; ret_used["dest"] = False
        if req.get("retreat_with_dest", True):      # 2026-10-15: 후퇴도 도착지(용기 벽 등)를 충돌에 넣어 먼저 계획한다. 시작 자세가 도착지 여유 안이라 막히면 예전처럼 도착지를 뺀 월드로 되돌린다
            P.update_world(Scene.create({"cuboid": _cuboids(obstacles, 0.0, 0.0, (), lv["tt"])}))
            r3 = _retreat()
            if r3 is not None and _ok(r3): ret_used["dest"] = True
            else:
                r3 = None; ret_used["v"] = ret_len; P.update_world(Scene.create(sc_nodest))
        if r3 is None: r3 = _retreat()
        if not _ok(r3):
            return None, "후퇴 계획 실패"
        return (r1, r2, r3), None

    try:
        for c in order:
            if time.time() > deadline and out["tried"]:
                why.append("시간 예산 초과"); break
            tag = dict(slot=c["slot"], yaw=c["yaw"], cost=round(c["cost"], 2)); got = None
            for lv in LADDER:
                if time.time() > deadline and out["tried"]:
                    break
                res, err = plan_c(c, lv); detach()
                if res is not None:
                    got = (res, lv); break
                tag.setdefault("fails", []).append(f"{lv['name']}:{err}")
            if got is None:
                out["tried"].append(tag); continue
            (r1, r2, r3), lv = got; quat = [float(x) for x in c["quat"]]; fl = c["fl"]
            w1, w2, w3 = _wps(r1), _wps(r2), _wps(r3)
            sane_ok, sane_why, sane_m = _traj_sane(dict(transit=w1, descend=w2, retreat=w3), req)
            if not sane_ok:
                tag["fail"] = "궤적 불안정: " + sane_why; tag["sane"] = sane_m; out["tried"].append(tag); continue
            tr, T1 = _resample(w1, dts[0], dt_i, tmin[0]); ds, T2 = _resample(w2, dts[1], dt_i, tmin[1]); rt, T3 = _resample(w3, dts[2], dt_i, tmin[2])
            out.update(fdiag=c.get("fdiag")); out.update(ok=True, slot=c["slot"], yaw_deg=c["yaw"], quat=quat, approach=[float(x) for x in c["a"]], tcp=[float(x) for x in c["tcp"]],
                       flange=[float(x) for x in fl], place_cfg=ds[-1], pre_cfg=tr[-1], transit=tr, descend=ds, retreat=rt, cost=c["cost"], planner="curobo",
                       durations=dict(transit=T1, descend=T2, retreat=T3), slot_index=c["si"], slot_z=c["tz"], relax_level=lv["name"], traj_metrics=sane_m, retreat_len=ret_used["v"], retreat_dest=ret_used.get("dest"))
            tag["relax"] = lv["name"]; out["tried"].append(tag)
            if pre_mode != "vertical": out.update(pre_mode=pre_mode, pre_kind=c.get("pre_kind_used", "vertical"), pre_dz=c.get("pre_dz_used", c.get("dz")))
            if c.get("ac") is not None:      # auto_clear / clear_report 결과(켰을 때만)
                _ac = dict(c["ac"]); _ac["held_sph_min_above"] = _sph_min_z(sph, c["fl"], c["quat"]) - c["tz"]
                out.update(auto_clear=bool(req.get("auto_clear")), auto_clear_used=_ac["clear_used"], finger_min_above_support=dict(before=_ac["finger_before"], after=_ac["finger_after"]), auto_clear_info=_ac)
            if R_REL is not None:      # goalpose: 목표 물체 자세(중심, 월드 회전 = Rg * Rz(ob.yaw)), 회전 후 반높이. slot_z 는 그대로 '놓는 면 높이'
                out.update(obj_rot=[float(x) for x in R_REL.as_quat()], half_h_rot=HZ_ROT, cand_rank=next(i_ for i_, x_ in enumerate(order) if x_ is c),
                           obj_goal=dict(centre=[float(x) for x in c["c_goal"]], quat=[float(x) for x in (c["Rg"] * Rot.from_euler("z", float(ob.get("yaw", 0.0)))).as_quat()]))
            break
    finally:
        detach()
    out["why"] = why; out["plan_sec"] = out["sec"] = round(time.time() - t0, 3)
    if not out["ok"]:
        try:
            import os
            os.makedirs("" + CUROBO_DIR + "/fail_goalpose", exist_ok=True)
            json.dump(req, open("" + CUROBO_DIR + "/fail_goalpose/%d_place.json" % int(time.time() * 1000), "w"))
        except Exception:
            pass
    return out


def solve_sphere_dist(req):
    """2026-10-16 진단: 관절 배치 목록 q(N,7) 과 상자들 boxes[{id,xyz,dims,yaw}] 에 대해, 링크별(구 단위) 상자까지 부호 거리(m, 음수=침범)의 최솟값. 어떤 링크가 어느 시각에 상자에 닿았나 가리는 용도(충돌 모델 검증).
    반환: per_box[id] = dict(min=[N개: 그 시각의 최소 거리], link=[N개: 그 구가 속한 링크])."""
    kc = P.kinematics.kinematics_config; inv = {v: k for k, v in kc.link_name_to_idx_map.items()}
    names = [inv.get(int(i), "?") for i in kc.link_sphere_idx_map.tolist()]
    q = torch.tensor([list(map(float, w[:7])) for w in req["q"]], device="cuda", dtype=torch.float32)
    st = P.kinematics.compute_kinematics(JointState.from_position(q, joint_names=P.joint_names))
    rs = st.robot_spheres.reshape(q.shape[0], -1, 4); cen = rs[:, :, :3]; rad = rs[:, :, 3]; valid = rad > 0.0
    out = {}
    for b in req["boxes"]:
        yaw = float(b.get("yaw", 0.0)); c = torch.tensor(b["xyz"], device="cuda", dtype=torch.float32); h = torch.tensor(b["dims"], device="cuda", dtype=torch.float32) / 2.0
        R = torch.tensor([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]], device="cuda", dtype=torch.float32)
        loc = (cen - c) @ R.T; qd = loc.abs() - h
        d = qd.clamp(min=0).norm(dim=-1) + qd.max(dim=-1).values.clamp(max=0) - rad
        d = torch.where(valid, d, torch.full_like(d, 9.0)); m, idx = d.min(dim=1)
        out[b["id"]] = dict(min=[round(float(x), 4) for x in m.tolist()], link=[names[int(i)] for i in idx.tolist()])
    return dict(ok=True, per_box=out)


def solve_jmove(req):
    """2026-10-15: 관절 목표까지 충돌 회피 이동 계획(놓은 직후 다음 물체를 보러 j1 을 크게 돌릴 때, MoveIt 은 용기를 몰라 쳐서 날렸다). 장애물에는 용기/놓은 물체까지 모두 넣는다."""
    t0 = time.time(); start = [float(x) for x in req["start_q"]]; goal = [float(x) for x in req["goal_q"]]
    try: P.attachment_manager.detach("attached_object")
    except Exception: pass
    P.update_world(Scene.create({"cuboid": _cuboids(req.get("obstacles", []), float(req.get("margin", 0.015)), 0.0, (), req.get("table_top"))}))
    _DEADLINE["t"] = t0 + float(req.get("budget_s", 6.0)) * _BS
    r = None
    for _ in range(int(req.get("tries", 3))):
        r = P.plan_cspace(_js(goal), _js(start))
        if _ok(r) or time.time() > _DEADLINE["t"]: break
    if not _ok(r): return dict(ok=False, why="plan_cspace 실패", sec=round(time.time() - t0, 3))
    w = _wps(r); dt_i = P.trajopt_solver.config.interpolation_dt
    tr, T = _resample(w, float(req.get("dt_exec", 0.1)), dt_i, float(req.get("t_min", 1.5)))
    return dict(ok=True, traj=tr, T=max(T, float(req.get("t_min", 1.5))), n=len(tr), sec=round(time.time() - t0, 3))


def solve_push(req):
    """2026-10-15: 붙어 있는 물체를 옆으로 밀어 틈을 벌린다(집기 계획이 이웃 때문에 막힐 때). 닫은 그리퍼를 탑다운으로 세워 물체 뒤(밀 반대쪽)로 내려 수평으로 밀고 올라온다.
    장애물은 밀 물체를 뺀 전부(요청이 뺀다). 수평 구간은 2cm 단위로 plan_pose 를 이어 직선에 가깝게 한다."""
    t0 = time.time(); start = [float(x) for x in req["start_q"]]
    c = np.array(req["center"][:2], float); u = np.array(req["dir"][:2], float); u /= (np.linalg.norm(u) + 1e-9)
    r = float(req.get("radius", 0.04)); d = float(req.get("dist", 0.07)); zp = float(req.get("z_push", 0.035)); lead = float(req.get("lead", 0.05)); zh = float(req.get("z_hi", 0.16)); hop = float(req.get("hop", 0.02))
    s_ = c - u * (r + lead); n_h = max(1, int(round((lead + d) / hop))); e_ = s_ + u * (lead + d)
    pts = [("hi_s", s_, zh, 1.0), ("lo_s", s_, zp, 1.0)] + [("push%d" % k, s_ + (e_ - s_) * (k / n_h), zp, max(0.5, hop / 0.05)) for k in range(1, n_h + 1)] + [("hi_e", e_, zh, 1.0)]
    try: P.attachment_manager.detach("attached_object")
    except Exception: pass
    P.update_world(Scene.create({"cuboid": _cuboids(req.get("obstacles", []), float(req.get("margin", 0.01)), 0.0, (), req.get("table_top"))}))
    _DEADLINE["t"] = t0 + float(req.get("budget_s", 10.0)) * _BS; dt_i = P.trajopt_solver.config.interpolation_dt
    why_ = None
    for yoff in (0.0, math.pi):      # 닫은 그리퍼는 180도 돌려도 같다 -> 손목 한계/도달이 막히면 반대 yaw 로 한 번 더
        quat = (Rot.from_euler("z", math.atan2(u[1], u[0]) + yoff) * Rot.from_quat([0.0, 1.0, 0.0, 0.0])).as_quat().tolist()
        cur = _js(start); traj = []; times = []; tt = 0.0; why_ = None
        for name, p, z, tmin in pts:
            res = None
            for _ in range(3):
                res = P.plan_pose(_goal([float(p[0]), float(p[1]), float(z + D_TCP)], quat), cur)
                if _ok(res) or time.time() > _DEADLINE["t"]: break
            if not _ok(res): why_ = "push 구간 계획 실패: " + name; break
            w = _wps(res); seg, T = _resample(w, 0.1, dt_i, tmin)
            for i, q in enumerate(seg):
                if traj and i == 0: continue
                traj.append(q); times.append(tt + T * i / max(1, len(seg) - 1))
            tt += max(T, tmin); cur = _end_state(res)
        if why_ is None: break
    if why_ is not None: return dict(ok=False, why=why_, sec=round(time.time() - t0, 3))
    return dict(ok=True, traj=traj, times=times, T=tt, n=len(traj), sec=round(time.time() - t0, 3), start=[float(x) for x in s_], end=[float(x) for x in e_], yaw_flip=bool(yoff))


def solve_place(req):
    _expand_place_spec(req)      # 2026-11-10: /place 에 place_spec 이 오면 서버에서 후보 생성
    if req.get("obj_rot_list"):      # 2026-11-09(기본 꺼짐): 순위 있는 놓기 자세 목록 -- 순서대로 풀어 처음 되는 자세를 돌려준다(이 파지 기준)
        _last = None; _fails = []
        for _k, _e in enumerate(req["obj_rot_list"]):
            _r = {k_: v_ for k_, v_ in req.items() if k_ not in ("obj_rot_list", "obj_rot", "yaws")}
            if _e.get("rot") is not None: _r["obj_rot"] = _e["rot"]
            if _e.get("yaws"): _r["yaws"] = _e["yaws"]
            if _e.get("slots"): _r["slots"] = _e["slots"]      # 구역 조건(왼쪽 등)이 있으면 그 자리
            _o = solve_place(_r); _o["pose_rank"] = _k; _o["pose_name"] = _e.get("name")
            if _o.get("ok"): _o["pose_fails"] = _fails; return _o
            _fails.append(dict(name=_e.get("name"), n_ik_ok=_o.get("n_ik_ok"))); _last = _o
        _last = _last or dict(ok=False, why=["자세 목록 비어 있음"]); _last["pose_fails"] = _fails; _last["pose_rank"] = None; _last["pose_name"] = None; return _last
    """모드 우선순위: same(집을 때 손 기울기 유지 -> 쥔 물체가 처음 서 있던 자세 그대로) 먼저, 안 되면 plumb(손을 수직으로 세움; 단단히 쥔 물체는 접근각만큼 같이 기운다).
    진단(28시행 짝지은 비교): plumb 쓰러짐 33% 대 same 17%, 놓기 직전 기울기가 접근각과 거의 같았다. req["mode"] 를 주면 그 모드만 쓴다."""
    if req.get("obj_rot") not in (None, "", "none"):      # 2026-11-06 goalpose: 목표 회전이 있으면 손 자세는 Rg*R0 하나로 정해진다(plumb 재시도는 의미 없음)
        r_ = _solve_place_impl(req); r_["mode_tried"] = ["goal_rot"]; return r_
    if req.get("mode"):
        return _solve_place_impl(req)
    r1 = _solve_place_impl(dict(req, mode="same")); r1["mode_tried"] = ["same"]
    if r1["ok"]:
        return r1
    A0 = np.array(req["grasp"]["approach"], float); A0 /= np.linalg.norm(A0)
    theta0 = math.degrees(math.acos(max(-1.0, min(1.0, -A0[2]))))
    if theta0 >= float(req.get("plumb_deg", 45)):
        return r1                      # 접근이 수평에 가까우면 plumb 은 원래 후보가 아니다
    r2 = _solve_place_impl(dict(req, mode="plumb", budget_s=float(req.get("budget_s", 8.0)) * 0.6)); r2["mode_tried"] = ["same", "plumb"]
    if r2["ok"]:
        r2["fallback_from_same"] = True; r2["same_fail"] = dict(ik_ok=r1.get("n_ik_ok"), tried=r1.get("tried"))
        return r2
    r1["mode_tried"] = ["same", "plumb"]; r1["plumb_tried"] = r2.get("tried")
    return r1


def _solve_insert(req):
    """긴 직육면체 삽입 계획(insert_planner.py, 요청 때마다 reload)."""
    _reset_retry_budget(req)
    if _BS != 1.0 and "budget_s" in req: req = dict(req, budget_s=float(req["budget_s"]) * _BS)      # insert_planner 는 수정하지 않고 요청 단계에서 배수 적용
    import importlib, sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # goalpose 복사본: 자기 폴더(phase_c_goalpose)의 insert_planner 를 쓴다
    import insertion_geometry, insert_planner
    importlib.reload(insertion_geometry); importlib.reload(insert_planner)
    return insert_planner.solve_insert(req, _sys.modules[__name__])


def init_ik(collision=True):
    """B단계 도달성(IK) 배치 솔버. 테이블/자기충돌 고려. 튜닝 env: NERO_CUROBO_IK_SEEDS, NERO_CUROBO_IK_POSTOL(m), NERO_CUROBO_IK_ROTTOL(rad)."""
    import os
    cfg = InverseKinematicsCfg.create(robot=ROBOT, scene_model=({"cuboid": {"table": _table()}} if collision else None), num_seeds=int(os.environ.get("NERO_CUROBO_IK_SEEDS", "32")),
                                      max_batch_size=IK_BATCH, self_collision_check=collision,
                                      position_tolerance=float(os.environ.get("NERO_CUROBO_IK_POSTOL", "0.005")),
                                      orientation_tolerance=float(os.environ.get("NERO_CUROBO_IK_ROTTOL", "0.05")))
    return InverseKinematics(cfg)


IK_TABLE = {"top": None}


def solve_ik(req):
    t0 = time.time(); poses = req["poses"]; n = len(poses); out = [None] * n
    solver = IKS if req.get("check_collision", True) else IKS_FREE
    tf = solver.tool_frames[0]
    tt = req.get("table_top", None) if solver is IKS else IK_TABLE["top"]
    if solver is IKS and tt != IK_TABLE["top"]:       # 튜닝/시험용: 요청마다 테이블 높이를 바꿀 수 있다(None=서버 기본). 같은 값이면 월드를 다시 올리지 않는다.
        IKS.update_world(Scene.create({"cuboid": {"table": _table(tt)}})); IK_TABLE["top"] = tt
    for i in range(0, n, IK_BATCH):
        ch = poses[i:i + IK_BATCH]; m = len(ch); ch = ch + [ch[-1]] * (IK_BATCH - m)
        pos = torch.tensor([p["flange"] for p in ch], device="cuda", dtype=torch.float32)
        qu = torch.tensor([[p["quat"][3], p["quat"][0], p["quat"][1], p["quat"][2]] for p in ch], device="cuda", dtype=torch.float32)
        r = solver.solve_pose(GoalToolPose.from_poses({tf: Pose(position=pos, quaternion=qu)}, num_goalset=1))
        ok = r.success.reshape(-1).cpu().numpy(); sol = r.solution.reshape(IK_BATCH, -1)[:, :7].detach().cpu().numpy()
        for j in range(m):
            if ok[j]:
                out[i + j] = [float(x) for x in sol[j]]
    return {"q": out, "sec": round(time.time() - t0, 3)}


def solve_fk(req):
    """관절 구성 목록 -> 플랜지(tool frame) 위치 목록(UI 의 경로 표시용)."""
    qs = torch.tensor([q[:7] for q in req["q"]], device="cuda", dtype=torch.float32)
    st = P.kinematics.compute_kinematics(JointState.from_position(qs, joint_names=P.joint_names))
    lp = st.tool_poses.get_link_pose(P.tool_frames[0])
    pos = lp.position.reshape(-1, 3).detach().cpu().numpy()
    out = {"pos": [[round(float(v), 4) for v in p] for p in pos]}
    try:      # 자세(wxyz)도 같이 준다(2026-10-03: 이송 경로에서 든 물체 양 끝의 높이를 계산하려고). 기존 pos 출력은 그대로
        out["quat_wxyz"] = [[round(float(v), 5) for v in q] for q in lp.quaternion.reshape(-1, 4).detach().cpu().numpy()]
    except Exception:
        pass
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"status": "ok"} if self.path.startswith("/health") else {"error": "unknown"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK, (torch.no_grad() if os.environ.get("NERO_NO_GRAD", "0") == "1" else _nullctx()):      # 2026-11-11 진단: 누수 텐서가 autograd 그래프에 잡혀 있는지
                try: _MEMH["n"] += 1
                except Exception: pass
                self._send(200, solve_candidates(req) if self.path.startswith("/candidates") else solve_sphere_dist(req) if self.path.startswith("/sphere_dist") else solve_push(req) if self.path.startswith("/push") else solve_jmove(req) if self.path.startswith("/jmove") else solve_ik(req) if self.path.startswith("/ik") else (solve_fk(req) if self.path.startswith("/fk") else (solve_place(req) if self.path.startswith("/place") else (_solve_insert(req) if self.path.startswith("/insert") else solve(req)))))
        except Exception as e:
            import traceback; traceback.print_exc(); self._send(500, {"ok": False, "error": repr(e)})
        finally:
            _mem_hygiene()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8020); a = ap.parse_args()
    P = init_planner(); IKS = init_ik(); IKS_FREE = init_ik(False); solve_ik({"poses": [{"flange": [0.3, 0.0, 0.2], "quat": [0, 1, 0, 0]}] * 4}); print("cuRobo C 서버 준비", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
