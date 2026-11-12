"""CGN 후보 전부(집기 가능한 것)를 C 로 넘기기 (2026-11-12, 사용자 지시: 규칙 합성 말고 CGN 후보를 거르지 않고 모두 받기).
왜: Thor B(two_pick_plan_fast.py)가 집기 IK 로 거른 뒤 NERO_KEEP_TOPN=8 / NERO_SIEVE_WANT=12 / NERO_HIT_SPAN=20 에서 끊음 -> C 가 고를 후보 3~8개.
    예: smoke 상자 CGN 원시 83개 -> 집기 가능 12 -> 8개만 C 로. 상한을 풀면 집기 가능 29개.
    이 값들은 원본 서버(8012) 프로세스 환경에서만 읽혀 요청마다 못 바꾸고, 서버 재시작·~/grasp 수정은 금지.
방법: plan_upload 가 끝난 직후 Thor 가 쓴 원시 파일({oid}_grasps.json, _obstacles.npy, _target.npy)을 읽어 오고(읽기만),
      Thor 의 B 스크립트 '같은 파일 복사본'(md5 확인)을 PC 에서 HOME=<이 실행 폴더>/bhome 로(PC·Thor ~/grasp 안 건드림) 상한만 풀어 다시 돌린다.
      B 의 다른 거르기(집기 IK, 이웃 충돌, 접근 비접촉 등)는 그대로. 결과 = 원래 8개(순서 그대로) + 나머지 집기 가능 후보(뒤에).
사용: cl2, info = expand(oid, arm_q_str, cl, run_dir, ik_url)"""
import os, json, subprocess, shutil, time, hashlib

THOR = os.environ.get("NERO_C_ALL_CGN_HOST", "thor")
LIFT = {"NERO_KEEP_TOPN": "1000", "NERO_SIEVE_WANT": "1000", "NERO_HIT_SPAN": "1000", "NERO_SIEVE_BUDGET": "1000", "NERO_BAIL_EMPTY": "1000"}
_B_CACHE = {}


def _fetch(remote, local):
    with open(local, "wb") as f:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", THOR, f"cat {remote}"], stdout=f, stderr=subprocess.PIPE, timeout=30)
    return r.returncode == 0 and os.path.getsize(local) > 0


def _b_script(bdir):
    """Thor B 스크립트 복사본(md5 가 Thor 와 같을 때만 다시 받지 않음)."""
    m = subprocess.run(["ssh", "-o", "ConnectTimeout=5", THOR, "md5sum ~/grasp/two_pick_plan_fast.py"], capture_output=True, text=True, timeout=20).stdout.split()[:1]
    dst = os.path.join(bdir, "two_pick_plan_fast.py")
    if m and os.path.exists(dst) and hashlib.md5(open(dst, "rb").read()).hexdigest() == m[0]: return m[0]
    ok = _fetch("~/grasp/two_pick_plan_fast.py", dst)
    for extra in ("phold_v1.joblib", "place_reach_model.joblib"):
        p = os.path.join(bdir, extra)
        if not os.path.exists(p): _fetch(f"~/grasp/{extra}", p)
    return m[0] if ok and m else None


def _key(c):
    return tuple(round(float(x), 4) for x in list(c["flange"]) + list(c["quat"]))


def expand(oid, arm_q, cl, run_dir, ik_url="http://127.0.0.1:8020"):
    t0 = time.time(); info = dict(n_b=len(cl or []))
    home = os.path.join(run_dir, "bhome"); g = os.path.join(home, "grasp"); os.makedirs(g, exist_ok=True)
    try:
        md5 = _b_script(g); info["b_md5"] = md5
        if not md5: info["err"] = "B 스크립트 복사 실패"; return cl, info
        for suf in ("_grasps.json", "_obstacles.npy", "_target.npy", "_sem.json", "_exclude.json"):
            lp = os.path.join(g, oid + suf)
            if not _fetch(f"~/grasp/{oid}{suf}", lp) and os.path.exists(lp): os.remove(lp)
        if not os.path.exists(os.path.join(g, oid + "_grasps.json")): info["err"] = "원시 CGN 파일 없음"; return cl, info
        info["n_cgn_raw"] = len(json.load(open(os.path.join(g, oid + "_grasps.json")))["grasps"])
        pj = os.path.join(g, oid + "_pick.json")
        if os.path.exists(pj): os.remove(pj)
        env = dict(os.environ, HOME=home, NERO_ARM_Q=arm_q, NERO_CUROBO_IK_URL=ik_url, NERO_CUROBO_IK_CHUNK=os.environ.get("NERO_B_IK_CHUNK", "32"), **LIFT)
        r = subprocess.run(["python3", os.path.join(g, "two_pick_plan_fast.py"), oid], cwd=g, env=env, capture_output=True, text=True, timeout=180)
        open(os.path.join(run_dir, f"cgn_all_{oid}_{int(t0)}.log"), "w").write(r.stdout[-20000:] + r.stderr[-5000:])
        if not os.path.exists(pj): info["err"] = "B(상한 해제) 결과 없음"; return cl, info
        allc = json.load(open(pj)).get("candidates") or []
        have = {_key(c) for c in (cl or [])}; extra = [c for c in allc if _key(c) not in have]
        out = list(cl or []) + extra
        info.update(n_all=len(allc), n_out=len(out), sec=round(time.time() - t0, 2))
        json.dump(dict(info=info, candidates=out), open(os.path.join(run_dir, f"cgn_all_{oid}_{int(t0)}.json"), "w"))
        return out, info
    except Exception as e:
        info["err"] = repr(e)[:200]; return cl, info
