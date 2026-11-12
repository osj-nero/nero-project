"""place_rank.py -- (개발 사본, 2026-11-04) 파지 후보를 p(NO_PLAN) 낮은 순으로 정렬(필터 아님, 순서만). discover_object.py 가 NERO_PLACE_RANK=1 일 때만 import.
모델: 확정 H1|F2|gbm (exp_pro_ml_20261104/model_H1_F2_gbm.joblib), 특성 순서 features.json F2. 사후 필드(n_ik_ok, gt, verdict 등) 사용 금지.
sklearn 은 시스템 파이썬에 없어 venv 파이썬 상주 워커(place_rank_worker.py)로 추론. 어떤 실패든 후보 순서를 바꾸지 않고(폴백) 사유를 log 에 남긴다."""
import os as _nm_os; NERO_MODEL_DIR = _nm_os.environ.get("NERO_MODEL_DIR") or _nm_os.path.expanduser("~/nero_models")  # optional learned-ranker files (override: NERO_MODEL_DIR)
import os, sys, json, math, subprocess, threading
import numpy as np
D_TCP = 0.1358
DIR = os.environ.get("NERO_PLACE_RANK_DIR", "" + NERO_MODEL_DIR + "/exp_pro_ml_20261104")
VENV_PY = os.environ.get("NERO_PLACE_RANK_PY", "" + NERO_MODEL_DIR + "/exp_pro_ml_env_20261104/venv/bin/python")
_W = {"p": None}; _L = threading.Lock()
F2 = None
def _feats_order():
    global F2
    if F2 is None: F2 = json.load(open(DIR + "/features.json"))["F2"]
    return F2
def _worker():
    if _W["p"] is not None and _W["p"].poll() is None: return _W["p"]
    env = dict(os.environ, PYTHONNOUSERSITE="1"); env.pop("PYTHONPATH", None)
    p = subprocess.Popen([VENV_PY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "place_rank_worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    r = json.loads(p.stdout.readline() or "{}")
    if not r.get("ready"): raise RuntimeError("워커 준비 실패 %s" % r)
    _W["p"] = p; return p
def _nan(x): return float("nan") if x is None else float(x)
def cand_features(c, geom, dest):
    """c: CGN/B 후보(contact, approach, opening[, flange]); geom: obj(centre, dims, height, base_z, elong); dest: 도착지 항목(raw, xyz) 또는 None."""
    a = np.array(c["approach"], float); n = np.linalg.norm(a); a = a / n if n > 0 else a
    ang = math.degrees(math.acos(max(-1.0, min(1.0, -a[2])))) if n > 0 else float("nan")
    ct = np.array(c["contact"], float); oc = np.array(geom["centre"], float); d = geom["dims"]; h = geom["height"]
    fl = np.array(c["flange"], float) if c.get("flange") is not None else ct - D_TCP * a
    raw = dest["raw"] if dest else None; dx = dest["xyz"] if dest else None
    nan = float("nan")
    f = dict(approach_deg=ang, theta=ang, held_ratio=(ct[2] - geom["base_z"]) / h, opening=_nan(c.get("opening")), contact_dx=ct[0] - oc[0], contact_dy=ct[1] - oc[1], contact_dz=ct[2] - oc[2], flange_z=fl[2],
             obj_h=h, obj_d0=d[0], obj_d1=d[1], obj_wh_min=min(d[0], d[1]) / h, obj_wh_max=max(d[0], d[1]) / h, obj_elong=_nan(geom.get("elong")),
             dest_x_size=raw[0] if raw else nan, dest_y_size=raw[1] if raw else nan, dest_depth=raw[2] if raw else nan, dest_area=raw[0] * raw[1] if raw else nan, top_z=float(raw[2]) if raw else nan,
             foot_ratio=(d[0] * d[1]) / (raw[0] * raw[1]) if raw and raw[0] * raw[1] > 0 else nan,
             obj_cx=oc[0], obj_cy=oc[1], obj_radius=math.hypot(oc[0], oc[1]), dest_px=dx[0] if dx else nan, dest_py=dx[1] if dx else nan, dest_pz=dx[2] if dx else nan,
             obj_dest_dist_xy=math.hypot(oc[0] - dx[0], oc[1] - dx[1]) if dx else nan, obj_dest_dist_3d=float(np.linalg.norm(oc - np.array(dx[:3], float))) if dx else nan)
    return [float(f[k]) for k in _feats_order()]
_FM = {}
def nearest_prim(centre_xy):
    """기준 씬(NERO_PLACE_RANK_SCENE_REF 또는 NERO_TRIAL_SCENE_REF) 에서 xy 가 가장 가까운 gso_ prim 이름·거리(m). 환경변수가 없으면 name=None('unmatched')."""
    ref = os.environ.get("NERO_TRIAL_SCENE_REF") or os.environ.get("NERO_PLACE_RANK_SCENE_REF")
    if not ref: return dict(name=None, why="scene_ref 미지정")
    if "prims_all" not in _FM:
        _FM["prims_all"] = [(q["name"], q["pos"][0], q["pos"][1]) for q in json.load(open(ref))["prims"] if str(q.get("name", "")).startswith("gso_") and q.get("pos")]
    b = min(_FM["prims_all"], key=lambda t_: math.hypot(t_[1] - centre_xy[0], t_[2] - centre_xy[1]))
    d = math.hypot(b[1] - centre_xy[0], b[2] - centre_xy[1])
    return dict(name=b[0], dist_m=round(d, 4), matched=bool(d <= float(os.environ.get("NERO_PLACE_RANK_MATCH_M", "0.08"))))
def resolve_model(centre_xy, log=print, oid=""):
    """사례 물체별 OOF 모델 선택(NERO_PLACE_RANK_FOLDMAP 지정 시): 현재 물체 xy 와 가장 가까운 기준 씬(NERO_PLACE_RANK_SCENE_REF=scene_ref.json) prim 이름을
    매핑표(이름 접두 -> 제외 폴드)에서 찾아 'foldN', 없으면 None(= 전체 ABC 모델, 사례 아님). 지정 없으면 항상 None(워커 기본). 허용 거리 NERO_PLACE_RANK_MATCH_M(0.08 m)."""
    fm = os.environ.get("NERO_PLACE_RANK_FOLDMAP")
    if not fm: return None, None
    try:
        if "map" not in _FM:
            _FM["map"] = json.load(open(fm))["names"]
            ref = json.load(open(os.environ["NERO_PLACE_RANK_SCENE_REF"]))
            _FM["prims"] = [(q["name"], q["pos"][0], q["pos"][1]) for q in ref["prims"] if str(q.get("name", "")).startswith("gso_") and q.get("pos")]
        best = min(_FM["prims"], key=lambda t: math.hypot(t[1] - centre_xy[0], t[2] - centre_xy[1]))
        dist = math.hypot(best[1] - centre_xy[0], best[2] - centre_xy[1])
        if dist > float(os.environ.get("NERO_PLACE_RANK_MATCH_M", "0.08")):
            nm0 = best[0].split("_", 2)[2] if best[0].count("_") >= 2 else best[0]
            if any(nm0.startswith(k_) or k_.startswith(nm0) for k_ in _FM["map"]):      # 2026-11-04: 최근접 prim 이 사례 물체인데 허용 거리 초과 = 사례 물체가 밀렸을 가능성 -> 전체 ABC 모델로 보내지 않는다(OOF 보호)
                log(f"{oid}: [PLACE_RANK] 매칭 실패(최근접 {best[0]} {dist * 100:.0f}cm) → 정렬 생략(OOF 보호)"); return "SKIP", best[0]
            log(f"{oid}: [PLACE_RANK] 기준 씬 prim 매칭 실패(최근접 {best[0]} {dist * 100:.0f}cm, 사례 물체 아님) -> 전체 ABC 모델(시험 주장 대상 외)"); return None, None
        nm = best[0].split("_", 2)[2] if best[0].count("_") >= 2 else best[0]
        for key, f in _FM["map"].items():
            if nm.startswith(key) or key.startswith(nm): return "fold%d" % int(f), best[0]
        log(f"{oid}: [PLACE_RANK] {best[0]} 은(는) 사례 물체 아님 -> 전체 ABC 모델(시험 주장 대상 외)"); return None, best[0]
    except Exception as e:
        log(f"{oid}: [PLACE_RANK] 폴드 매핑 실패 {e!r} → 정렬 생략(OOF 보호: 사례 여부 확인 불가)"); return "SKIP", None
def rerank(cl, geom, dest, log=print, oid=""):
    """-> (정렬된 후보 리스트, 정보 dict). 실패 시 (원래 리스트, {'fallback': 사유})."""
    try:
        rows = [cand_features(c, geom, dest) for c in cl]
        mk, prim = resolve_model(geom["centre"][:2], log, oid)
        if mk == "SKIP":
            return cl, dict(fallback="OOF 매칭 실패 → 정렬 생략", skipped=True)
        with _L:
            p = _worker(); p.stdin.write(json.dumps({"rows": rows, "model": mk}) + "\n"); p.stdin.flush()
            r = json.loads(p.stdout.readline() or "{}")
        if "p" not in r: raise RuntimeError("추론 실패 %s" % r)
        ps = r["p"]; _W["tag"] = r.get("model") or _W.get("tag")
        if len(ps) != len(cl) or any(not math.isfinite(v) for v in ps): raise RuntimeError("출력 이상")
        order = sorted(range(len(cl)), key=lambda i: (ps[i], i))      # 안정 정렬: 같은 점수면 기존 순서
        out = [cl[i] for i in order]
        log(f"{oid}: [PLACE_RANK] 모델={r.get('model')} p(NO_PLAN) 정렬 {len(cl)}개: 기존 순 {[round(v, 2) for v in ps]} -> 새 순서 {order}")
        return out, dict(p=ps, order=order)
    except Exception as e:
        log(f"{oid}: [PLACE_RANK] 폴백(기존 순서 유지): {e!r}")
        return cl, dict(fallback=repr(e))
