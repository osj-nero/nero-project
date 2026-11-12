import os
#!/usr/bin/env python3
"""insert_planner.py -- cuRobo 서버의 /insert 구현(서버 모듈 S = curobo_c_server 의 함수/전역을 빌려 쓴다. 서버 재시작 없이 reload 가능).
요청: start_q[7], grasp{quat(xyzw), flange[3], approach[3]}, peg{center[3], dims[3], yaw}, E[3](홀 입구), d[3](삽입 방향 단위벡터),
      holder{inner, wall, len}(홀더 벽 충돌체 생성), obstacles[](선택), pre(0.08), depth(0.06), step(0.02), rolls[](기본 0..315 step 45), budget_s
동작: (팁 방향 +/- ) x 롤 후보마다 입구 축 위 웨이포인트(손 자세)를 만들고 배치 IK 로 전 웨이포인트 도달 가능한 것만 남긴 뒤(비용 순),
      물체 구를 붙인 채 transit -> 웨이포인트 사슬(구간 plan_pose, 이전 끝 상태에서 시작)로 삽입 궤적을 만든다. 후퇴 = 삽입 관절 경로의 역순(놓은 뒤라 물체 없음, 충돌 안 난다).
응답은 /place 와 같은 형식: ok, slot, yaw_deg(롤), tcp, flange, place_cfg, pre_cfg, transit, descend(=삽입), retreat, mode='insert' ..."""
import math, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import insertion_geometry as G


def _holder_cuboids(E, d, inner, wall, length, rim=0.0):
    """홀더 벽 4 + 바닥의 cuRobo cuboid 정의(월드). 로컬: 원점=입구 중심, z=-d(밖), 벽은 z in [-length, 0] (= d 방향으로 안쪽)."""
    d = np.asarray(d, float) / np.linalg.norm(d); E = np.asarray(E, float)
    u = -d; zax = np.array([0, 0, 1.0]); v = np.cross(zax, u); s = np.linalg.norm(v); c = float(zax @ u)
    R = Rot.identity() if s < 1e-9 and c > 0 else (Rot.from_rotvec([math.pi, 0, 0]) if s < 1e-9 else Rot.from_rotvec(v / s * math.atan2(s, c)))
    q = R.as_quat(); qw = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    h = inner / 2.0; out = {}
    m = float(rim)      # 손 여유(2026-10-04): 벽의 바깥면과 입구 위쪽으로 m 만큼 부풀려 손(링크7/손가락)이 테두리에 눌리지 않게 한다(안쪽 공동 크기는 그대로)
    L2 = length + m; z2 = -length / 2 + m / 2
    parts = {"wall_px": ((wall + m, inner + 2 * wall + 2 * m, L2), (h + (wall + m) / 2, 0, z2)), "wall_nx": ((wall + m, inner + 2 * wall + 2 * m, L2), (-h - (wall + m) / 2, 0, z2)),
             "wall_py": ((inner, wall + m, L2), (0, h + (wall + m) / 2, z2)), "wall_ny": ((inner, wall + m, L2), (0, -h - (wall + m) / 2, z2)),
             "bottom": ((inner + 2 * wall, inner + 2 * wall, wall), (0, 0, -length - wall / 2))}
    for nm, (dims, loc) in parts.items():
        out["ins_" + nm] = {"dims": [float(x) for x in dims], "pose": [float(x) for x in (E + R.apply(loc))] + qw}
    return out


RIM_RUNGS = (0.012, 0.008, 0.004, 0.0)


def _hand_clearance(S, qs_all, walls):
    """GPU 배치로 모든 후보의 모든 웨이포인트에서 로봇 구(손/링크)와 홀더 벽(실제 크기) 사이 최소 부호거리를 한 번에 계산한다. qs_all: [[q(7)...] 후보별] -> 후보별 최소 여유(m)."""
    flat = [q for qs in qs_all for q in qs]; ln = [len(qs) for qs in qs_all]
    t = S.torch.tensor([q[:7] for q in flat], device="cuda", dtype=S.torch.float32)
    st = S.P.kinematics.compute_kinematics(S.JointState.from_position(t, joint_names=S.P.joint_names))
    sp = st.robot_spheres.detach().reshape(len(flat), -1, 4)
    pos = sp[..., :3]; rad = sp[..., 3]; best = S.torch.full((len(flat),), 1e3, device="cuda")
    for w in walls.values():
        h = S.torch.tensor(w["dims"], device="cuda", dtype=S.torch.float32) / 2.0; p = w["pose"]
        R = S.torch.tensor(Rot.from_quat([p[4], p[5], p[6], p[3]]).as_matrix(), device="cuda", dtype=S.torch.float32)
        loc = (pos - S.torch.tensor(p[:3], device="cuda", dtype=S.torch.float32)) @ R           # 월드 -> 상자 로컬(R^T x)
        q_ = loc.abs() - h; out_ = S.torch.clamp(q_, min=0).norm(dim=-1); ins_ = S.torch.clamp(q_.max(dim=-1).values, max=0)
        dist = (out_ + ins_) - rad                                                                # 구 표면까지의 부호거리
        dist = S.torch.where(rad > 0, dist, S.torch.full_like(dist, 1e3))
        best = S.torch.minimum(best, dist.min(dim=1).values)
    best = best.cpu().numpy(); out = []; i = 0
    for n_ in ln:
        out.append(float(best[i:i + n_].min())); i += n_
    return out


def _ik_chained(S, chains, start_q, R=8, max_jump=0.6):
    """연속 IK(2026-10-02): 웨이포인트마다 독립으로 풀면 해 분기가 달라 연속 경로가 안 된다(IK 후보 0개 15건 중 점프 탈락이 요청당 7~24개).
    웨이포인트마다 해를 R개씩 받아(return_seeds) 후보별로 '연속 관절 이동이 최소'인 조합을 동적계획법으로 고른다. (공개 인자만 사용: current_state/seed_config 를 직접 주면 시드 수가 안 맞아 CUDA 오류가 났다)
    chains: [[{flange,quat}, ...] 후보별]. 반환: 후보별 [q|None, ...]."""
    B = S.IK_BATCH; n = len(chains); L = len(chains[0]); tf = S.IKS.tool_frames[0]; sols = [[None] * L for _ in range(n)]
    for k in range(L):
        poses = [chains[i][k] for i in range(n)]; ch = poses + [poses[-1]] * (B - n)
        pos = S.torch.tensor([p_["flange"] for p_ in ch], device="cuda", dtype=S.torch.float32)
        qu = S.torch.tensor([[p_["quat"][3], p_["quat"][0], p_["quat"][1], p_["quat"][2]] for p_ in ch], device="cuda", dtype=S.torch.float32)
        r = S.IKS.solve_pose(S.GoalToolPose.from_poses({tf: S.Pose(position=pos, quaternion=qu)}, num_goalset=1), return_seeds=R)
        ok = r.success.reshape(B, -1).cpu().numpy(); sol = r.solution.reshape(B, ok.shape[1], -1)[:, :, :7].detach().cpu().numpy()
        for i in range(n):
            sols[i][k] = [sol[i, j] for j in range(ok.shape[1]) if ok[i, j]]
    out = [[None] * L for _ in range(n)]
    for i in range(n):
        if any(len(x) == 0 for x in sols[i]):
            continue
        cost = [np.zeros(len(sols[i][0]))]; back = []
        for k in range(1, L):
            A = np.array(sols[i][k - 1]); Bm = np.array(sols[i][k]); d = np.abs(Bm[:, None, :] - A[None, :, :]).max(axis=2)       # (Nk, Nk-1) 관절별 최대 변화
            tot = d + cost[-1][None, :]; j = tot.argmin(axis=1); back.append(j); cost.append(tot[np.arange(len(Bm)), j])
        kk = int(np.argmin(cost[-1])); path = [kk]
        for k in range(L - 2, -1, -1):
            kk = int(back[k][kk]); path.append(kk)
        path = path[::-1]; qs = [sols[i][k][path[k]] for k in range(L)]
        if max(float(np.abs(np.diff(np.array(qs), axis=0)).max()), 0.0) <= max_jump:
            out[i] = [[float(x) for x in q] for q in qs]
    return out


SKIP_NEG = os.environ.get("NERO_INSERT_SKIP_NEG", "1") == "1"; NEG_MM = float(os.environ.get("NERO_INSERT_NEG_MM", "-5"))


def _ckey(chain):
    return tuple(round(float(v), 5) for p_ in chain for v in (list(p_["flange"]) + list(p_["quat"])))


def _ik_cached(S, chains):
    """2026-10-16: 같은 웨이포인트 체인의 연속 IK 는 장면과 무관하다(IK 해결기는 고정 월드). 사전 선별/place 확인/진단 재실행에서 같은 파지의 IK(1.4초)를 반복 계산하던 것을 서버 모듈에 캐시한다."""
    cache = S.__dict__.setdefault("_IK_CACHE", {})
    if len(cache) > 4096: cache.clear()
    miss = [i for i, c in enumerate(chains) if _ckey(c) not in cache]
    if miss:
        by_len = {}
        for i in miss: by_len.setdefault(len(chains[i]), []).append(i)
        for L_, ids in by_len.items():
            res = S._ik_chain([chains[i] for i in ids])
            for i, r_ in zip(ids, res): cache[_ckey(chains[i])] = r_
    return [cache[_ckey(c)] for c in chains]


def _chains_for(req):
    g = req["grasp"]; peg = req["peg"]; E = np.array(req["E"], float); d = np.array(req["d"], float); d /= np.linalg.norm(d)
    pre = float(req.get("pre", 0.08)); depth = float(req.get("depth", 0.06)); step = float(req.get("step", 0.02)); rolls = req.get("rolls") or [0, 45, 90, 135, 180, 225, 270, 315]
    grasp = dict(flange=g["flange"], quat=g["quat"]); out = []
    for sign in (+1, -1):
        for rl in rolls:
            r = G.insertion_waypoints(grasp, peg, E, d, tip_sign=sign, roll_deg=rl, pre=pre, depth=depth, step=step)
            out.append([dict(flange=[float(x) for x in xf], quat=[float(x) for x in q]) for (xf, q) in r["waypoints"]])
    return out


def prewarm(reqs, S):
    """여러 파지 후보의 IK 체인을 한 번에(128 배치) 풀어 캐시에 넣는다. 후보를 하나씩 풀면 후보당 1.4초(배치 128 중 16 만 사용)."""
    allc = []
    for r in reqs: allc += _chains_for(r)
    _ik_cached(S, allc)


def _solve_insert_once(req, S):
    _pp = getattr(S, "_plan_pose_r", None) or S.P.plan_pose        # 공통 재시도 래퍼(서버가 구버전이면 원래 호출)
    t0 = time.time(); why = []
    start_q = [float(x) for x in req["start_q"]]
    g = req["grasp"]; peg = req["peg"]; E = np.array(req["E"], float); d = np.array(req["d"], float); d /= np.linalg.norm(d)
    pre = float(req.get("pre", 0.08)); depth = float(req.get("depth", 0.06)); step = float(req.get("step", 0.02))
    rolls = req.get("rolls") or [0, 45, 90, 135, 180, 225, 270, 315]; K = int(req.get("max_cands", 6)); budget = float(req.get("budget_s", 10.0)); S._DEADLINE["t"] = t0 + budget
    hold = req.get("holder") or {}; walls = _holder_cuboids(E, d, hold.get("inner", 0.06), hold.get("wall", 0.01), hold.get("len", 0.14), float(req.get("rim_clear", os.environ.get("NERO_RIM_CLEAR", "0.0"))))
    grasp = dict(flange=g["flange"], quat=g["quat"]); R0 = Rot.from_quat(g["quat"]); FL0 = np.array(g["flange"], float)
    cands = []; poses = []
    for sign in (+1, -1):
        for rl in rolls:
            r = G.insertion_waypoints(grasp, peg, E, d, tip_sign=sign, roll_deg=rl, pre=pre, depth=depth, step=step)
            idx = []
            for (xf, q) in r["waypoints"]:
                idx.append(len(poses)); poses.append(dict(flange=[float(x) for x in xf], quat=[float(x) for x in q]))
            cands.append(dict(sign=sign, roll=rl, idx=idx, res=r))
    if req.get("ik_mode", "chain") == "chain":      # 기본 = 연속 IK(DP). "independent" 로 기존 독립 IK(OFF 스위치, 삭제하지 않음)
        try:
            _res = (_ik_cached(S, [[poses[i] for i in c["idx"]] for c in cands]) if hasattr(S, "_ik_chain") else _ik_chained(S, [[poses[i] for i in c["idx"]] for c in cands], start_q)); ik = [None] * len(poses)
            for c, rr in zip(cands, _res):
                if rr is None:       # 서버 공통 _ik_chain 은 연속 해가 없는 후보를 None 으로 준다(예전 _ik_chained 는 [None..]). 이전에 여기서 TypeError 로 매번 독립 IK 로 폴백했다
                    continue
                for i_, q_ in zip(c["idx"], rr): ik[i_] = q_
        except Exception as e_:
            why.append("연속 IK 실패 -> 독립 IK: %r" % (e_,)); ik = S.solve_ik({"poses": poses, "check_collision": True})["q"]
    else:
        ik = S.solve_ik({"poses": poses, "check_collision": True})["q"]
    _tk = time.time() - t0
    feas = []
    for c in cands:
        qs = [ik[i] for i in c["idx"]]
        if any(x is None for x in qs):
            continue
        qa = np.array(qs); jump = float(np.abs(np.diff(qa, axis=0)).max()) if len(qa) > 1 else 0.0
        if jump > 0.9:                                                  # 웨이포인트 사이 IK 해 분기가 다르면 연속 경로가 아니다
            continue
        c["qs"] = qs; c["cost"] = float(np.abs(qa[0] - np.array(start_q)).sum()) + 0.5 * jump; feas.append(c)
    if feas and str(req.get("rim_screen", os.environ.get("NERO_RIM_SCREEN", "1"))) == "1" and not req.get("no_walls"):
        try:      # 손-홀더 여유 사다리를 GPU 한 번에: 후보마다 가능한 가장 큰 여유 단계를 구해 큰 단계 후보부터 계획한다(같은 단계 안에서는 기존 비용순)
            _cl = _hand_clearance(S, [c["qs"] for c in feas], walls)
            for c, v in zip(feas, _cl):
                c["clear"] = v; c["rung"] = next((i for i, r_ in enumerate(RIM_RUNGS) if v >= r_), len(RIM_RUNGS))
            feas.sort(key=lambda c: (c["rung"], c["cost"]))
        except Exception as e_:
            why.append("여유 선별 실패(무시): %r" % (e_,)); feas.sort(key=lambda c: c["cost"])
    else:
        feas.sort(key=lambda c: c["cost"])
    _tc = time.time() - t0
    if req.get("prof"): print("[prof-insert] IK체인 %.2fs, 여유선별 포함 누계 %.2fs, 후보 %d 중 IK %d" % (_tk, _tc, len(cands), len(feas)), flush=True)
    if req.get("screen_only"):      # 2026-10-16: 파지 후보 사전 선별용 -- IK+손-홀더 여유만 보고 반환(궤적 계획 없음). best_clear: 가능한 IK 후보 중 최대 여유(m, 음수 = 손이 홀더 벽을 파고듦)
        return dict(ok=bool(feas), screen=True, n_cands=len(cands), n_ik_ok=len(feas), best_clear=max((float(c.get("clear", -1.0)) for c in feas), default=-1.0), plan_sec=round(time.time() - t0, 3))
    if hasattr(S, "_adaptive_retry_budget"):
        S._adaptive_retry_budget(len(feas), req)       # 공통 규칙: 후보가 적을수록 재시도
    out = dict(ok=False, mode="insert", n_cands=len(cands), n_ik_ok=len(feas), tried=[])
    obstacles = req.get("obstacles", []); margin = float(req.get("margin", 0.005))
    ctr = np.array(peg["center"], float)
    if peg.get("axis") is not None:       # PCA 축이 있으면 구를 장축 위에 일렬로(자세 일반). 없으면 구 방식(OBB)
        ax_, L_, cr_ = G.peg_frame(peg); r_ = float(np.clip(max(cr_) / 2.0 + float(req.get("sphere_pad", 0.0)), 0.012, 0.035)); span_ = max(L_ / 2.0 - r_, 0.0)
        ts_ = np.linspace(-span_, span_, max(1, int(math.ceil(2 * span_ / (1.2 * r_))) + 1)) if span_ > 0 else np.zeros(1)
        pf_ = R0.inv().apply(ctr[None, :] + ts_[:, None] * ax_[None, :] - np.asarray(FL0, float)); sph = np.concatenate([pf_, np.full((len(pf_), 1), r_)], axis=1)
    else:
        sph = S._held_spheres(R0, FL0, [ctr[0], ctr[1], ctr[2]], peg["dims"], float(peg.get("yaw", 0.0)), pad=float(req.get("sphere_pad", 0.0)), cover=req.get("held_cover", "old"))
    base = S._cuboids(obstacles, margin, 0.0, (), req.get("table_top")); base.update({} if req.get("no_walls") else walls)       # no_walls: 진단용(홀더 벽 충돌체 제거)
    sc_all = {"cuboid": base}
    dt_i = S.P.trajopt_solver.config.interpolation_dt
    dts = (float(req.get("transit_dt", 0.1625)), float(req.get("descend_dt", 0.30)), float(req.get("retreat_dt", 0.30)))
    tmin = (1.0, 1.5, 1.0)
    held = {"on": False}
    def attach():
        S.P.attachment_manager.update(S.torch.tensor(sph, device="cuda", dtype=S.torch.float32), S._js(start_q), "attached_object"); held["on"] = True
    def detach():
        if held["on"]:
            S.P.attachment_manager.detach("attached_object"); held["on"] = False
    try:
        for c in feas[:K + int(req.get("sane_extra", 4))]:       # 안정장치가 버린 후보를 대신할 여유
            if time.time() > t0 + budget and out["tried"]:
                why.append("시간 예산 초과"); break
            tag = dict(sign=c["sign"], roll=c["roll"], cost=round(c["cost"], 2), clear_mm=(None if "clear" not in c else round(1000 * c["clear"], 1))); wp = c["res"]["waypoints"]
            if SKIP_NEG and c.get("clear") is not None and c["clear"] < NEG_MM / 1000.0:      # 2026-10-16: 손-홀더 여유가 음수(손이 벽을 파고듦)인 후보는 계획해도 실패했다(실패 재현 8후보 중 성공은 여유 +5.7mm 뿐). 시간만 쓰므로 건너뛴다
                tag["fail"] = "손-홀더 여유 %.1fmm (< %.0fmm) -- 건너뜀" % (1000 * c["clear"], NEG_MM); out["tried"].append(tag); continue
            _tcand = time.time(); attach(); S.P.update_world(S.Scene.create(sc_all))
            r1 = _pp(S._goal([float(x) for x in wp[0][0]], [float(x) for x in wp[0][1]]), S._js(start_q))
            if not S._ok(r1):
                tag["fail"] = "transit 계획 실패"; out["tried"].append(tag); detach(); continue
            chain = []; prev = S._end_state(r1); okc = True
            for kk, (xf, q) in enumerate(wp[1:]):
                rk = _pp(S._goal([float(x) for x in xf], [float(x) for x in q]), prev)
                if not S._ok(rk):
                    okc = False
                    try:       # 진단(2026-10-02): 실패 구간에서 목표 IK 해와 현재 끝 상태의 관절 거리, 솔버 오차
                        qprev = np.array(S._wps(chain[-1])[-1][:7] if False else chain[-1][-1][:7]) if chain else np.array(S._wps(r1)[-1][:7])
                        tag["rk_type"] = str(type(rk))
                        _at = [a_ for a_ in dir(rk) if not a_.startswith("_")]
                        _val = {}
                        for a_ in ("success", "status", "solve_time", "position_error", "rotation_error", "goal_distance", "cspace_error", "metrics", "optimized_dt", "total_time"):
                            if hasattr(rk, a_):
                                try: _val[a_] = str(getattr(rk, a_))[:200]
                                except Exception as e2_: _val[a_] = "err " + repr(e2_)[:40]
                        tag["rk"] = dict(attrs=_at[:60], val=_val)
                        tag["dbg"] = dict(k=kk + 1, jdist=round(float(np.abs(np.array(c["qs"][kk + 1]) - qprev).max()), 3),
                                          pos_err=[float(v) for v in S.torch.as_tensor(getattr(rk, "position_error", [float("nan")])).reshape(-1)[:1].cpu().numpy()],
                                          rot_err=[float(v) for v in S.torch.as_tensor(getattr(rk, "rotation_error", [float("nan")])).reshape(-1)[:1].cpu().numpy()])
                    except Exception as e_:
                        tag["dbg"] = dict(err=repr(e_)[:80])
                    break
                chain.append(S._wps(rk)); prev = S._end_state(rk)
            detach()
            if req.get("prof"): print("[prof-insert] 후보 계획 %.2fs (구간 %d/%d, 성공 %s)" % (time.time() - _tcand, len(chain), len(wp) - 1, okc), flush=True)
            if not okc:
                tag["fail"] = "삽입 구간 계획 실패(%d/%d)" % (len(chain), len(wp) - 1); out["tried"].append(tag); continue
            w1 = S._wps(r1); w2 = np.vstack(chain); w3 = w2[::-1].copy()
            rt_perp = None
            if req.get("retreat_mode") == "perp":     # A: 막대에서 수직 방향(손 접근 반대)으로 먼저 빠진다. 삽입 역경로로 빠지면 손가락이 막대를 축 방향으로 끌어낸다
                try:
                    a_f = Rot.from_quat(wp[-1][1]).apply([0, 0, 1.0]); back = float(req.get("retreat_back", 0.10))
                    r3p = _pp(S._goal([float(v) for v in (np.array(wp[-1][0], float) - a_f * back)], [float(v) for v in wp[-1][1]]), prev)
                    if S._ok(r3p):
                        rt_perp = S._wps(r3p)
                except Exception:
                    rt_perp = None
            sane_ok, sane_why, sane_m = (S._traj_sane(dict(transit=w1, descend=w2, retreat=(rt_perp if rt_perp is not None else w3)), req) if hasattr(S, "_traj_sane") else (True, None, {}))    # 서버가 구버전이면(재시작 전) 검사 생략
            if not sane_ok:
                tag["fail"] = "궤적 불안정: " + sane_why; tag["sane"] = sane_m; out["tried"].append(tag); continue
            tr, T1 = S._resample(w1, dts[0], dt_i, tmin[0]); ds, T2 = S._resample(w2, dts[1], dt_i, tmin[1]); rt, T3 = S._resample(rt_perp if rt_perp is not None else w3, dts[2], dt_i, tmin[2])
            out['retreat_used'] = 'perp' if rt_perp is not None else 'rev'
            tip = c["res"]["tip_path"]
            out.update(ok=True, slot=[float(E[0]), float(E[1])], yaw_deg=c["roll"], tip_sign=c["sign"], tcp=[float(x) for x in tip[-1]], flange=[float(x) for x in wp[-1][0]], quat=[float(x) for x in wp[-1][1]],
                       approach=[float(x) for x in d], place_cfg=ds[-1], pre_cfg=tr[-1], transit=tr, descend=ds, retreat=rt, cost=c["cost"], planner="curobo", durations=dict(transit=T1, descend=T2, retreat=T3), slot_z=float(E[2]), traj_metrics=sane_m, tip_path=[[float(v) for v in q_] for q_ in tip])
            out["tried"].append(tag); break
    finally:
        detach()
    out["why"] = why; out["plan_sec"] = out["sec"] = round(time.time() - t0, 3)
    return out


def solve_insert(req, S):
    """손-홀더 여유 사다리(2026-10-04): 여유(rim_clear)가 크면 손이 테두리에 눌려 그리퍼가 안 열리는 실패(EXEC_FAIL)를 막지만 계획이 안 풀리는 배치가 있다
    (12mm: 4건 중 3건 NO_PLAN / 6mm: 0도 계열 해결, 45도 계열 NO_PLAN / 3mm: 45도 해결). 큰 여유부터 시도하고 계획이 안 되면 줄인다. 고정 요청(rim_clear 지정)은 사다리 없이 한 번만."""
    if "rim_clear" in req or os.environ.get("NERO_RIM_LADDER", "0") != "1":
        return _solve_insert_once(req, S)
    total = float(req.get("budget_s", 10.0)); out = None; t0 = time.time()
    for rc, frac in ((0.008, 0.4), (0.004, 0.35), (0.0, 1.0)):
        r2 = dict(req); r2["rim_clear"] = rc; r2["budget_s"] = max(2.0, min(total * frac, total - (time.time() - t0))) if rc > 0 else max(2.0, total - (time.time() - t0))
        out = _solve_insert_once(r2, S); out["rim_clear"] = rc
        if out.get("ok"):
            break
    out["plan_sec"] = out["sec"] = round(time.time() - t0, 3)
    return out
