"""CGN 상주 추론 서버 (Thor). 모델 1회 로드 -> POST /infer 로 npz 처리.
two_cgn.py 와 동일 로직 (robust circle_fit + depth-bias 보정 포함)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from axis_fit import fit_axis3, usable as axis_usable
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

R = os.path.expanduser("~/grasp/contact_graspnet_pytorch")
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "Pointnet_Pointnet2_pytorch"))
os.chdir(R)
import torch
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO

print("[cgn] loading model ...", flush=True)
cfg = config_utils.load_config(os.path.join(R, "checkpoints", "contact_graspnet"), batch_size=1)
EST = GraspEstimator(cfg)
CheckpointIO(checkpoint_dir=os.path.join(R, "checkpoints", "contact_graspnet", "checkpoints"),
             model=EST.model).load("model.pt")
EST.model.eval()
print("[cgn] ready", flush=True)


def _taubin(x, y):
    xm, ym = x.mean(), y.mean()
    xr, yr = x - xm, y - ym
    z = xr * xr + yr * yr
    Z = np.c_[z - z.mean(), xr, yr]
    _, _, V = np.linalg.svd(Z, full_matrices=False)
    A = V[-1]
    return xm - A[1] / (2 * A[0]), ym - A[2] / (2 * A[0])


def circle_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    cx, cy = _taubin(x, y)
    for _ in range(3):
        d = np.hypot(x - cx, y - cy); r = np.median(d)
        keep = np.abs(d - r) < np.percentile(np.abs(d - r), 75) + 1e-6
        if keep.sum() < 8:
            break
        cx, cy = _taubin(x[keep], y[keep]); x, y = x[keep], y[keep]
    return cx, cy


def R2q(Rm):
    t = np.trace(Rm)
    if t > 0:
        u = np.sqrt(t + 1) * 2; w = .25 * u
        x = (Rm[2, 1] - Rm[1, 2]) / u; y = (Rm[0, 2] - Rm[2, 0]) / u; z = (Rm[1, 0] - Rm[0, 1]) / u
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        u = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / u; x = .25 * u; y = (Rm[0, 1] + Rm[1, 0]) / u; z = (Rm[0, 2] + Rm[2, 0]) / u
    elif Rm[1, 1] > Rm[2, 2]:
        u = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / u; x = (Rm[0, 1] + Rm[1, 0]) / u; y = .25 * u; z = (Rm[1, 2] + Rm[2, 1]) / u
    else:
        u = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / u; x = (Rm[0, 2] + Rm[2, 0]) / u; y = (Rm[1, 2] + Rm[2, 1]) / u; z = .25 * u
    v = np.array([x, y, z, w]); return (v / np.linalg.norm(v)).tolist()


def infer(npz_path, out_dir, tilt_cyl=False):
    d = np.load(os.path.expanduser(npz_path), allow_pickle=True)
    out_dir = os.path.expanduser(out_dir)
    depth = d["depth_m"].astype(np.float32)
    k = d["K"].reshape(-1)
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
    seg = d["seg"].astype(np.int32)
    WT = np.array(d["W_T_cam"]) @ np.diag([1., -1., -1., 1.])

    targets = []
    for name, key in [("bottle", "bottle_id"), ("box", "box_id")]:
        if key in d.files and int(d[key][0]) > 0:
            targets.append((name, int(d[key][0])))
    full_seg = np.zeros_like(seg)
    for i, (_, sid) in enumerate(targets, start=1):
        full_seg[seg == sid] = i
    pc_full, pc_segs, _ = EST.extract_point_clouds(depth, K, segmap=full_seg, z_range=[0.2, 1.2])

    res = []
    for idx, (name, sid) in enumerate(targets, start=1):
        if idx not in pc_segs or len(np.asarray(pc_segs[idx])) < 50:
            res.append({name: "too few points"}); continue
        sp_cam = np.asarray(pc_segs[idx])
        sp = (WT[:3, :3] @ sp_cam.T).T + WT[:3, 3]
        zext = float(np.percentile(sp[:, 2], 97) - np.percentile(sp[:, 2], 3))
        xw = float(np.percentile(sp[:, 0], 95) - np.percentile(sp[:, 0], 5))
        yw = float(np.percentile(sp[:, 1], 95) - np.percentile(sp[:, 1], 5))
        aspect = zext / max(max(xw, yw), 1e-3)
        base_z, top_z = float(sp[:, 2].min()), float(sp[:, 2].max())

        with torch.no_grad():
            g, s, c, o = EST.predict_scene_grasps(
                pc_full, pc_segments={idx: sp_cam}, local_regions=True, filter_grasps=True)
        if idx not in g or np.asarray(g[idx]).size == 0:
            res.append({name: "no CGN grasps"}); continue
        G = np.asarray(g[idx]).reshape(-1, 4, 4)
        S = np.asarray(s[idx]).reshape(-1); O = np.asarray(o[idx]).reshape(-1)

        bbmin = [float(sp[:, j].min()) for j in range(3)]
        bbmax = [float(sp[:, j].max()) for j in range(3)]
        # 2026-09-23: ext_robust -- 손잡이 등 가늘게 튀어나온 돌출부가 min/max bbox
        # 가로폭을 부풀려서 tip_pen용 slenderness가 실제보다 약해지는 회귀 실측 확인
        # (컵 사례, 원통 fit 대비 ~10배 약함). "원통이면" 분기 없이 백분위수(2~98%)로
        # 이상치 점을 통계적으로 배제 -- 상자류는 min/max와 거의 동일, 손잡이류만
        # 자동 배제(카테고리 무관). two_pick_plan_fast.py._slenderness()가 우선 사용.
        ext_robust = [float(np.percentile(sp[:, j], 98) - np.percentile(sp[:, j], 2)) for j in range(3)]
        box_fit = dict(bbmin=bbmin, bbmax=bbmax,
                       centre=[0.5 * (bbmin[j] + bbmax[j]) for j in range(3)],
                       ext=[bbmax[j] - bbmin[j] for j in range(3)], ext_robust=ext_robust)
        # 원통 fit 은 항상 계산 (robust circle_fit 이 손잡이 outlier 제거).
        # 판정 기준(2026-09-14 개정): 기존 aspect(=z범위/XY폭, 수직축 가정)는 물체가
        # 기울면 뷰에 따라 출렁인다(실측: 같은 컵 32° 기울기에서 0.95~1.44 사이 요동,
        # 임계 1.4를 넘었다 못 넘었다 함). axis3(PCA 로 뽑은 실측 3D 축) 기준
        # aspect3 는 같은 컵에서 1.9~2.5 로 안정적. tilt_cyl=True 면 axis3 가 신뢰
        # 가능할 때(usable) 그걸로 판정하고, 아니면(축이 거의 수직/원형도 낮음/
        # tilt_cyl=False) 기존 aspect 로 폴백 — 회귀 없음.
        _a3 = fit_axis3(sp, cam_xyz=WT[:3, 3])
        _use_a3 = bool(tilt_cyl and _a3 and axis_usable(_a3))
        aspect_eff = float(_a3["aspect3"]) if _use_a3 else aspect
        cyl_auto = bool(aspect_eff > 1.4)
        if _a3:
            print(f"[{name}] axis3 {np.round(_a3['axis3'], 3).tolist()} "
                  f"tilt {_a3['tilt_deg']:.1f}° round {_a3['roundness']:.2f} "
                  f"aspect3 {_a3['aspect3']:.2f} usable={axis_usable(_a3)} "
                  f"-> cyl_auto 판정 기준={'aspect3' if _use_a3 else 'aspect(수직가정)'} "
                  f"({aspect_eff:.2f} > 1.4 = {cyl_auto})", flush=True)
        cx, cy = circle_fit(sp[:, 0], sp[:, 1])
        r = float(np.clip(np.median(np.hypot(sp[:, 0] - cx, sp[:, 1] - cy)) * 1.1, 0.015, 0.06))
        cam_xy = np.array([WT[0, 3], WT[1, 3]])
        v = np.array([cx, cy]) - cam_xy; nv = np.linalg.norm(v)
        if nv > 1e-3:
            cx, cy = (np.array([cx, cy]) + (v / nv) * r * 0.35).tolist()
        cyl = dict(axis=[float(cx), float(cy)], radius=r, base_z=base_z, top_z=top_z,
                   centre=[float(cx), float(cy), 0.5 * (base_z + top_z)])
        if _a3:                       # 기존 필드는 그대로 두고 추가만
            cyl.update(_a3)

        GD = 0.1034                     # CGN gripper_depth: 그리퍼 프레임 원점 -> 접촉 baseline
        grasps = []
        for i in range(len(G)):
            gw = WT @ G[i]
            b = gw[:3, 0]; a = gw[:3, 2]
            Rf = np.column_stack([-np.cross(a, b), b, a])
            # CGN grasp[:3,3] = contact + th/2·b - GD·a (그리퍼 원점).
            # +GD·a 하면 그립 중심(양 손끝 사이 baseline) = TCP 로 쓸 점.
            grip_c = gw[:3, 3] + GD * a
            grasps.append(dict(rank=i, score=float(S[i]), opening_m=float(O[i]),
                               position_xyz=[round(float(x), 4) for x in grip_c],
                               grasp_origin=[round(float(x), 4) for x in gw[:3, 3]],
                               quat_xyzw=[round(x, 5) for x in R2q(Rf)],
                               approach=[round(float(x), 3) for x in a],
                               closing=[round(float(x), 3) for x in b]))
        ow = d["cyl_world"].tolist() if (name == "bottle" and "cyl_world" in d.files) else \
            (d["box_world"].tolist() if (name == "box" and "box_world" in d.files) else [0, 0, 0])
        json.dump(dict(frame="base_link", object=name, obj_world=ow, aspect_w=aspect,
                       cyl_auto=cyl_auto, cyl_fit=cyl, box_fit=box_fit, grasps=grasps),
                  open(os.path.join(out_dir, f"{name}_grasps.json"), "w"), indent=1)
        res.append({name: f"{len(G)} grasps score {S.max():.3f} aspect {aspect:.2f}"
                          + (f" cyl axis ({cyl['axis'][0]:.3f},{cyl['axis'][1]:.3f})" if cyl else "")})
    return res


# 2026-09-29 이웃 물체 충돌 필터용 장애물 점군 (B단계 two_pick_plan_fast.py 가 읽음).
# 씬 전체 점군(pc_full)에서 테이블/노이즈/target 자신을 뺀 나머지를 world(base_link) 좌표로 저장.
# 장애물 판정에 쓰는 것은 "점유 복셀 중심"이라 후보별 충돌검사가 싸다.
NBR_VOX = 0.006          # 복셀 크기(m)
NBR_MIN_PTS = 4          # 복셀당 최소 점수 (혼합픽셀/비행점 노이즈 제거)
NBR_TABLE_Z = 0.015      # 이 높이 이하는 테이블로 간주
NBR_RADIUS = 0.30        # target 중심 xy 반경 (그 밖은 관련 없음)
NBR_SELF_DIST = 0.012    # target 점으로부터 이 거리 이내는 target 자신의 가장자리로 간주하고 제외


def _neighbor_obstacles(pc_full_cam, WT, target_world):
    from scipy.spatial import cKDTree
    P = (WT[:3, :3] @ np.asarray(pc_full_cam, dtype=np.float64).T).T + WT[:3, 3]
    ctr = target_world.mean(0)
    m = (P[:, 2] > NBR_TABLE_Z) & (P[:, 2] < 0.45) & (np.hypot(P[:, 0] - ctr[0], P[:, 1] - ctr[1]) < NBR_RADIUS)
    P = P[m]
    if len(P) == 0:
        return np.zeros((0, 3), np.float32)
    vi = np.floor(P / NBR_VOX).astype(np.int64)
    uniq, inv, cnt = np.unique(vi, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    keep = cnt >= NBR_MIN_PTS
    V = (uniq[keep].astype(np.float64) + 0.5) * NBR_VOX
    if len(V) == 0:
        return np.zeros((0, 3), np.float32)
    d, _ = cKDTree(target_world).query(V)
    V = V[d > NBR_SELF_DIST]
    return V.astype(np.float32)


# 2026-09-17 추가 (사용자 지시): infer_multi.py(다중물체) 를 매번 새 프로세스로 띄우면서
# 모델을 매번 새로 로드하고 있었음(cgn_venv 서브프로세스, ~8~15초 중 대부분이 모델
# 로드+CUDA 워밍업). 이 상시서버가 이미 로드해둔 EST 를 그대로 재사용하는 엔드포인트를
# 추가 -- 별도 프로세스/GPU 메모리 없이 같은 모델 인스턴스로 다중물체 처리.
# infer_multi.py 의 per-object 루프 로직을 그대로 옮김 (bottle/box 2슬롯 하드코딩 대신
# target_labels 임의 N개, capture_multi_npz.py 의 npz 스키마 기준).
def infer_multi(npz_path, out_dir, nbr_filter=True):
    d = np.load(os.path.expanduser(npz_path), allow_pickle=True)
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    depth = d["depth_m"].astype(np.float32)
    k = d["K"].reshape(-1)
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]])
    seg = d["seg"].astype(np.int32)
    WT = np.array(d["W_T_cam"]) @ np.diag([1., -1., -1., 1.])
    labels = [str(x) for x in d["target_labels"]]
    ids = [int(x) for x in d["target_ids"]]

    # 캡처/세그멘테이션/포인트클라우드 추출은 물체 전체에 대해 딱 1번 (기존 infer_multi.py 와 동일).
    pc_full, pc_segs, _ = EST.extract_point_clouds(depth, K, segmap=seg, z_range=[0.2, 1.2])

    res = []
    GD = 0.1034
    for label, sid in zip(labels, ids):
        if sid not in pc_segs or len(np.asarray(pc_segs[sid])) < 50:
            res.append({label: "too few points"}); continue
        sp_cam = np.asarray(pc_segs[sid])
        sp = (WT[:3, :3] @ sp_cam.T).T + WT[:3, 3]

        with torch.no_grad():
            g, s, c, o = EST.predict_scene_grasps(
                pc_full, pc_segments={sid: sp_cam}, local_regions=True, filter_grasps=True)
        if sid not in g or np.asarray(g[sid]).size == 0:
            res.append({label: "no CGN grasps"}); continue
        G = np.asarray(g[sid]).reshape(-1, 4, 4)
        S = np.asarray(s[sid]).reshape(-1); O = np.asarray(o[sid]).reshape(-1)

        grasps = []
        for i in range(len(G)):
            gw = WT @ G[i]
            b = gw[:3, 0]; a = gw[:3, 2]
            Rf = np.column_stack([-np.cross(a, b), b, a])
            grip_c = gw[:3, 3] + GD * a
            grasps.append(dict(rank=i, score=float(S[i]), opening_m=float(O[i]),
                               position_xyz=[round(float(x), 4) for x in grip_c],
                               grasp_origin=[round(float(x), 4) for x in gw[:3, 3]],
                               quat_xyzw=[round(x, 5) for x in R2q(Rf)],
                               approach=[round(float(x), 3) for x in a],
                               closing=[round(float(x), 3) for x in b]))
        bbmin = [float(sp[:, j].min()) for j in range(3)]
        bbmax = [float(sp[:, j].max()) for j in range(3)]
        # 2026-09-23: ext_robust -- 손잡이 등 가늘게 튀어나온 돌출부가 min/max bbox
        # 가로폭을 부풀려서 tip_pen용 slenderness가 실제보다 약해지는 회귀 실측 확인
        # (컵 사례, 원통 fit 대비 ~10배 약함). "원통이면" 분기 없이 백분위수(2~98%)로
        # 이상치 점을 통계적으로 배제 -- 상자류는 min/max와 거의 동일, 손잡이류만
        # 자동 배제(카테고리 무관). two_pick_plan_fast.py._slenderness()가 우선 사용.
        ext_robust = [float(np.percentile(sp[:, j], 98) - np.percentile(sp[:, j], 2)) for j in range(3)]
        box_fit = dict(bbmin=bbmin, bbmax=bbmax,
                       centre=[0.5 * (bbmin[j] + bbmax[j]) for j in range(3)],
                       ext=[bbmax[j] - bbmin[j] for j in range(3)], ext_robust=ext_robust)
        json.dump(dict(frame="base_link", object=label, n_points=len(sp_cam),
                       box_fit=box_fit, grasps=grasps),
                  open(os.path.join(out_dir, f"{label}_grasps.json"), "w"), indent=1)
        # 2026-09-29: 접근 중 그리퍼-타깃 비접촉 검사용 타깃 점군 (two_pick_plan_fast._sieve 가 사용)
        _tv = np.unique(np.floor(sp / 0.004).astype(np.int64), axis=0)
        np.save(os.path.join(out_dir, f"{label}_target.npy"), ((_tv + 0.5) * 0.004).astype(np.float32))
        _obs_msg = ""
        _obs_path = os.path.join(out_dir, f"{label}_obstacles.npy")
        if nbr_filter:
            _obs = _neighbor_obstacles(pc_full, WT, sp)
            np.save(_obs_path, _obs)
            _obs_msg = f", 이웃 장애물 복셀 {len(_obs)}"
        elif os.path.exists(_obs_path):
            os.remove(_obs_path)
        res.append({label: f"{len(G)} grasps, score max {S.max():.3f}, n_pts {len(sp_cam)}{_obs_msg}"})
    return res


app = FastAPI()


class Req(BaseModel):
    npz_path: str
    out_dir: str = "~/grasp"
    tilt_cyl: bool = False   # A/B: True 면 axis3(실측 축) 기준으로 cyl_auto 판정


@app.get("/health")
def health():
    return {"status": "ok", "model": "contact_graspnet", "device": str(next(EST.model.parameters()).device)}


@app.post("/infer")
def _infer(r: Req):
    import time
    t0 = time.time()
    out = infer(r.npz_path, r.out_dir, tilt_cyl=r.tilt_cyl)
    return {"ms": round((time.time() - t0) * 1000), "results": out}


class ReqMulti(BaseModel):
    npz_path: str
    out_dir: str = "~/grasp"
    nbr_filter: bool = True   # False 면 장애물 점군을 안 쓴다(ablation/baseline 용)


@app.post("/infer_multi")
def _infer_multi(r: ReqMulti):
    import time
    t0 = time.time()
    out = infer_multi(r.npz_path, r.out_dir, nbr_filter=r.nbr_filter)
    return {"ms": round((time.time() - t0) * 1000), "results": out}


# ── 2026-10-01 실시간 pick 상주 서비스: PC 에서 npz 바이트를 올리면 CGN -> B(two_pick_plan_fast.py, 프로세스 재시작 없이 exec) 를 한 번의 호출로 끝낸다.
#    ssh/rsync/파이썬 프로세스 시작 비용을 없앤다. B 스크립트는 그대로이고, 호출마다 argv/env 를 바꿔 같은 프로세스에서 실행한다.
import io, contextlib, threading
from fastapi import Request
_RT_LOCK = threading.Lock()
_B_CODE = {"mtime": 0.0, "code": None, "path": os.path.expanduser("~/grasp/two_pick_plan_fast.py")}


def _b_code():
    mt = os.path.getmtime(_B_CODE["path"])
    if _B_CODE["code"] is None or mt != _B_CODE["mtime"]:
        _B_CODE["code"] = compile(open(_B_CODE["path"]).read(), _B_CODE["path"], "exec"); _B_CODE["mtime"] = mt
    return _B_CODE["code"]


def run_b_inproc(name, arm_q, env):
    code = _b_code()
    keys = list(env) + ["NERO_ARM_Q"]
    saved = {k: os.environ.get(k) for k in keys}
    old_argv, cwd = sys.argv, os.getcwd()
    os.environ.update({k: str(v) for k, v in env.items()}); os.environ["NERO_ARM_Q"] = arm_q
    sys.argv = [_B_CODE["path"], name]; os.chdir(os.path.expanduser("~/grasp"))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, {"__name__": "__main__", "__file__": _B_CODE["path"]})
    except SystemExit:
        pass
    except Exception as e:
        buf.write("B 예외 %r\n" % (e,))
    finally:
        sys.argv = old_argv; os.chdir(cwd)
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return buf.getvalue()


@app.post("/plan_upload")
async def plan_upload(request: Request, name: str, arm_q: str, ik_url: str = "", chunk: int = 32, skip_b: int = 0):
    import time
    body = await request.body()
    path = os.path.expanduser(f"~/grasp/rt_{name}.npz")
    open(path, "wb").write(body)
    with _RT_LOCK:
        t0 = time.time()
        res = infer_multi(path, "~/grasp", nbr_filter=True)
        cgn_ms = round((time.time() - t0) * 1000)
        out = {"cgn_ms": cgn_ms, "cgn": res, "pick": None, "b_ms": None, "b_tail": ""}
        ok = any("grasps" in str(r) and "no CGN" not in str(r) for r in res)
        if ok and not skip_b:
            pj = os.path.expanduser(f"~/grasp/{name}_pick.json")
            for f in (pj, os.path.expanduser(f"~/grasp/{name}_step6_pick.json")):
                if os.path.exists(f): os.remove(f)
            t1 = time.time()
            env = {"NERO_CUROBO_IK_URL": ik_url, "NERO_CUROBO_IK_CHUNK": chunk} if ik_url else {}
            txt = run_b_inproc(name, arm_q, env)
            out["b_ms"] = round((time.time() - t1) * 1000); out["b_tail"] = "\n".join(txt.strip().splitlines()[-10:])
            if os.path.exists(pj):
                out["pick"] = json.load(open(pj))
    return out


@app.post("/store")
async def store(request: Request):
    d = await request.json()
    p = os.path.expanduser(f"~/grasp/{d['name']}_step6_pick{d.get('tag', '')}.json")
    json.dump(d["plan"], open(p, "w"))
    return {"ok": True, "path": p}


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("CGN_HOST", "0.0.0.0"), port=8010)
