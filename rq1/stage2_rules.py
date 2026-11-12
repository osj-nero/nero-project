"""규칙 기준선 87경우 Isaac 실행 (2026-11-12, 설계 = DESIGN_RULES87_20261112.md, 결과 보기 전 고정).
stage2_vlm.py 의 장면 세팅·로봇·인식·후보·판정·복귀를 그대로 쓰고, 조건·순서만 정책 고정값(멀티모달 LLM 호출 없음), 막히면 풀기 없이 포기.
사용: S2V_OUT=... NERO_NB_NORETRY=1 NERO_SCAN_SNAPSHOT=1 python3 stage2_rules.py blocked_v4/exec_vlm_cases87.json [S,F,C]"""
import os, sys, json, re, time, subprocess
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import stage2_vlm as SV
S, B, G, EM = SV.S, SV.B, SV.G, SV.EM
PR = json.load(open(H + "/prereg/RQ1_PREREG_v3.json"))
CMAP = {"U": "윗면이 위", "F": "바닥면이 위", "Lx+": "뒷면이 위", "Lx-": "앞면이 위", "Ey+": "왼면이 위", "Ey-": "오른면이 위"}      # exec_policies.py 와 같음
F_ORDER = ["윗면이 위", "바닥면이 위", "앞면이 위", "뒷면이 위", "왼면이 위", "오른면이 위"]
UPRIGHT = dict(hard=[dict(type="up", faces=["윗면"], tol="보통")], manner=["place"])


def cat_C(obj):
    for r in PR["category_table_C"]["rules"]:
        if any(k in obj for k in r["keyword"]): return [CMAP[o] for o in r["orient"]]
    return [CMAP[o] for o in PR["category_table_C"]["default"]["orient"]]


def policy(pol, it0):
    """-> (조건, 순서)"""
    if pol == "S": return UPRIGHT, ["윗면이 위"]
    if pol == "F": return dict(hard=[], manner=["place"]), F_ORDER
    o = cat_C(it0["obj"])
    return (dict(hard=[], manner=["place"]), o) if len(o) == 6 else (UPRIGHT, o)


def episode_rule(case, pol, n_run):
    sid, iid = case["scene"], case["item"]; sc = SV.SCENES[sid]; it0 = SV.ITEMS[iid]; asset = sc["asset"]; t_ep = time.time()
    tm = dict(robot=0.0, harness=0.0); hist = []; outcome = "abstain"
    spec, order = policy(pol, it0)
    D = f"{SV.OUT}/runs/{n_run[0]:03d}"; n_run[0] += 1; os.makedirs(D, exist_ok=True); pf = D + "/poses.json"
    t_h = time.time(); SV.setup_scene(sc, it0["texture"]); tm["harness"] += time.time() - t_h
    t_r = time.time(); p = SV.start_robot(D, sc); SV.CUR_P[0] = p
    try:
        fr = SV.wait_perc(D, asset, p=p)
        if fr is None: raise RuntimeError("인식 상자 파일 없음(로봇 인식 실패)")
        dyaw = fr["yaw"] - sc["yaw"]
        cands = B.server_candidates(spec, asset, fr["yaw"])
        pl = sorted([c["pose"] for c in cands if c["pose"] in order], key=order.index)
    except Exception as ex:
        SV.stop_robot(p); hist.append(dict(round=0, err=repr(ex)[:300], run=D))
        return dict(policy=pol, scene=sid, item=iid, outcome="error", relaxed=False, final_spec=spec, hist=hist, sec=round(time.time() - t_ep, 1), stage_sec=tm)
    h = dict(round=0, run=D, spec=spec, perc=fr, dyaw=round(dyaw, 1), pl=pl, t_list=round(time.time() - t_r, 1))
    json.dump(dict(place_spec=dict(hard=spec["hard"], asset=asset, yaw0=round(fr["yaw"], 2), order=pl)), open(pf + ".tmp", "w"), ensure_ascii=False); os.replace(pf + ".tmp", pf)
    nf = pf + ".fail1.json"; t_w = time.time()
    while p.poll() is None and not os.path.exists(nf) and time.time() - t_w < 900: time.sleep(0.2)
    log = open(D + "/run.log").read(); planned = "pick 과 place 모두 계획됨" in log
    h.update(planned=planned, n_c_fail=len(re.findall(r"C 실패\(", log)))
    if not planned:
        if p.poll() is None and os.path.exists(nf): json.dump(dict(abstain=True), open(pf + ".r1", "w"))      # 풀기 없음 -> 포기 알림
        elif p.poll() is None: SV.stop_robot(p); h["robot_timeout"] = True
        h["per_pose"] = SV.RC.per_pose_from_log(log)
    try: p.wait(timeout=900)
    except subprocess.TimeoutExpired: SV.stop_robot(p)
    tm["robot"] += time.time() - t_r; t_h = time.time()
    if planned: time.sleep(4)
    s = EM.final_state(); EM.shot(D + "/before_home.png"); h["home"] = SV.safe_home(s, asset, sc); h["state"] = s; tm["harness"] += time.time() - t_h
    if planned:
        log = open(D + "/run.log").read(); mp = re.findall(r"놓기 자세 선택 (\{.*\})", log); h["chosen"] = json.loads(mp[-1]) if mp else None
        h["own"] = SV.judge_state(s, spec, asset, dyaw); h["orig"] = SV.judge_state(s, G.gt_for(iid, spec), asset, 0.0)
        outcome = "executed_ok" if (h["own"] or {}).get("ok") else ("executed_bad" if s and s["on_platform"] else "dropped_or_offplatform")
    hist.append(h)
    return dict(policy=pol, scene=sid, item=iid, outcome=outcome, relaxed=False, final_spec=spec, hist=hist, sec=round(time.time() - t_ep, 1), stage_sec={k: round(v, 1) for k, v in tm.items()})


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))["cases"]; pols = (sys.argv[2] if len(sys.argv) > 2 else "S,F,C").split(",")
    lim = int(os.environ.get("RULES_LIMIT", "0"))
    fo_p = SV.OUT + "/episodes.jsonl"; done = {(r["policy"], r["scene"], r["item"]) for r in map(json.loads, open(fo_p))} if os.path.exists(fo_p) else set()
    fo = open(fo_p, "a"); n_run = [len(os.listdir(SV.OUT + "/runs"))]; k = 0
    for c in cases:
        for pol in pols:
            if (pol, c["scene"], c["item"]) in done: continue
            if lim and k >= lim: break
            gp_mb, _ = SV.gp_watch(); ep = episode_rule(c, pol, n_run); ep["gp_mb_before"] = gp_mb; k += 1
            fo.write(json.dumps(ep, ensure_ascii=False) + "\n"); fo.flush()
            lc = (ep["hist"][-1].get("chosen") or {}) if ep["hist"] else {}
            print(f"{pol} {c['scene']} {c['item']}: {ep['outcome']} {lc.get('name') or ''} {ep['sec']}s {ep['stage_sec']}", flush=True)
    S.clean_scene(); print("STAGE2_RULES_DONE", flush=True)
