"""바닥 기둥 4 + 판 뚜껑 Isaac 실행 (2026-11-12, 사용자 승인 'ㅇㅇ').
배치 = 계획만 시험(stage0_floor_all.py)에서 '전체 후보'로 8/8 이 나온 설정 중 cx=-0.42, s=0.09, θ=0 (판 시작 yaw 38.5, 막대 시작 yaw 117.3 = 수집 장면과 같음).
흐름: 막대를 (-0.30,0.30) 에 하나씩 생성 -> 로봇(rt_run, 시스템 그대로 + NERO_C_ALL_CGN=1)이 인식·집기·세워 놓기 -> 4개 끝나면 판 생성 -> 뚜껑(높이 0.20, 기둥 = 도착지).
자세 목록은 로봇의 인식 결과(poses.json.perc.json 의 yaw)로 만듦(정답 각도 안 씀). 기둥: 왼면이 위/오른면이 위, 뚜껑: 앞면이 위/뒷면이 위 + 정사각형 축에 맞는 yaw 4개.
뚜껑 도착지: discover_object 의 NERO_PLACE_DEST_BELOW=1(기본 꺼짐) -- 놓을 곳 아래 받침면 높이의 '인식된' 물체를 dest_ids 로.
복귀(실험 하네스, 채점 밖): 단계 사이 home_safe.py 에 지금까지 놓인 물체의 시뮬레이터 상자를 장애물로 줌(stage2 와 같은 방식).
결과 precise/floor_pillars_lid_<태그>/result.json"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import os, sys, json, time, math, signal, subprocess
import numpy as np
RQ = "" + NERO_ROOT + "/rq1"; sys.path.insert(0, RQ)
import stage2_rules as SR
SV = SR.SV; EM = SV.EM; S = SV.S; PF = SV.PF
import place_spec_core as C
from isaac_exec import run_code
TAG = sys.argv[1] if len(sys.argv) > 1 else "r1"
CX, CY, TH = float(os.environ.get("FPL_CX", "-0.42")), 0.0, 0.0; SS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.09      # 2026-11-12 r2: 간격을 인자로(r1 = 0.09)
BAR_YAW, PLATE_YAW = 117.3, 38.5; OX, OY = -0.30, 0.30; BAR_L = 0.20
OUT = f"{NERO_ROOT}/precise/floor_pillars_lid_{TAG}"; os.makedirs(OUT, exist_ok=True)
PL = [float(v) for v in os.environ.get("FPL_PLATE", "0.12,0.015,0.10").split(",")]      # 2026-11-12 r4(사용자: 판을 키우고 간격 15cm 이상): 판 크기(폭, 두께, 높이) 인자
DIMS = dict(bar=(0.20, 0.03, 0.03), plate=tuple(PL))
SPAWN = '''
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema
st=omni.usd.get_context().get_stage()
mp="/World/Looks/obj_blue"
if not st.GetPrimAtPath(mp).IsValid():
    m=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+"/s"); sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15,0.35,0.85)); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
m=UsdShade.Material(st.GetPrimAtPath(mp)); hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction")
c=UsdGeom.Cube.Define(st,"/World/{name}"); c.CreateSizeAttr(1.0); xf=UsdGeom.Xformable(c); xf.ClearXformOpOrder()
xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({x},{y},{z})); xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set({yaw}); xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({sx},{sy},{sz}))
pr=c.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.RigidBodyAPI.Apply(pr); UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.1)
PhysxSchema.PhysxRigidBodyAPI.Apply(pr).CreateSolverPositionIterationCountAttr(16)
UsdShade.MaterialBindingAPI.Apply(pr).Bind(m)
if hm.IsValid(): UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(hm), UsdShade.Tokens.weakerThanDescendants, "physics")
'''


def spawn(name, asset, yaw):
    sx, sy, sz = DIMS[asset]
    return run_code(SPAWN.format(name=name, x=OX, y=OY, z=sz / 2 + 0.002, yaw=float(yaw), sx=sx, sy=sy, sz=sz), timeout=40)["status"]


def states():
    run_code(EM.POSE_Q, timeout=20); d = json.load(open("/tmp/rq1_exec_pose.json")); out = {}
    for n, v in d.items():
        rows = np.array(v["rows"]); cols = np.stack([rows[i] / np.linalg.norm(rows[i]) for i in range(3)], axis=1); out[n] = dict(xyz=v["p"], R=cols.tolist())
    return out


def axis_tilt(s, i):
    """물체 로컬 축 i 가 연직에서 기운 각(도)."""
    a = np.array(s["R"])[:, i]; return round(math.degrees(math.acos(min(1.0, abs(a[2])))), 1)


def obs_boxes(st):
    """복귀 경로용(하네스): 놓인 물체를 실제 치수·방향 상자로."""
    ob = []
    for n, s in st.items():
        asset = "plate" if "plate" in n else "bar"; Rm = np.array(s["R"]); dm = DIMS[asset]; k = int(np.argmax(np.abs(Rm[2, :]))); i, j = [a for a in range(3) if a != k]
        ob.append(dict(id=n, xyz=s["xyz"], raw=[dm[i], dm[j], dm[k]], yaw=float(np.arctan2(Rm[1, i], Rm[0, i]))))
    return ob


def home(st):
    out = subprocess.run(["python3", os.path.join(RQ, "home_safe.py"), json.dumps(obs_boxes(st)), "0.005"], cwd=EM.PC, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
    return out[-1] if out else "FAIL 출력 없음"


def step(k, asset, place_xy, top_z, poses, yaws_fn, dest_below):
    D = f"{OUT}/step{k}_{asset}"; os.makedirs(D + "/creq", exist_ok=True)
    e = dict(EM.ENV, NERO_C_ALL_CGN="1", NERO_C_ALL_CGN_MAX="200", NERO_PLACE_TOPZ=str(top_z), NERO_PLACE_POSE_LIST_FILE=D + "/poses.json", NERO_C_REQ_DUMP=D + "/creq", NERO_MULTI_LOG=D)
    e.pop("NERO_PLACE_POSE_LIST", None)
    if dest_below: e["NERO_PLACE_DEST_BELOW"] = "1"
    if os.environ.get("FPL_AB", "0") == "1":      # 2026-11-12 r3(사용자 승인 A+B): 놓은 물체 기억 + 놓을 곳 정지 촬영
        e.update(NERO_PLACED_MEM=OUT + "/placed_mem.json", NERO_KEEP_PLACED="1", NERO_PLACE_LOOK="1")
    if os.environ.get("FPL_TWIST"): e["NERO_C_TWIST_MAX"] = os.environ["FPL_TWIST"]      # r6: 닫힘축-얇은축 어긋난 후보 제외
    t0 = time.time()
    p = subprocess.Popen(["./rt_run.sh", f"--target-at={OX},{OY}", f"--place-xy={place_xy[0]},{place_xy[1]}", "--vocab", EM.VOC, "--max-picks", "1", "--max-fail", "1",
                          "--place-clear", "0.015", "--out", D + "/run.json"], cwd=EM.PC, env=e, stdout=open(D + "/run.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    fr = SV.wait_perc(D, asset, timeout=300, p=p); t_perc = round(time.time() - t0, 1)
    if fr is not None:      # r6 고침: 인식 좌표의 로컬 x 를 실제 물체 형판(DIMS)으로 고름. S.GRID_DIMS 판 형판 12cm 로는 22cm 판에서 두께축이 폭으로 잡혀 '눕히기'가 '제자리 돌리기'가 됐음
        fr = PF.frame(json.load(open(D + "/poses.json.perc.json"))["obj"], asset, {asset: DIMS[asset]})
    if fr is not None:
        yws = yaws_fn(fr["yaw"])
        lst = [dict(name=n, rot=(None if C.world_rot(n, fr["yaw"]) is None else [float(v) for v in C.world_rot(n, fr["yaw"]).as_quat()]), yaws=yws, slots=[[place_xy[0], place_xy[1], top_z]]) for n in poses]
        json.dump(lst, open(D + "/poses.json.tmp", "w"), ensure_ascii=False); os.replace(D + "/poses.json.tmp", D + "/poses.json")
    try: p.wait(timeout=720)
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid, signal.SIGTERM); time.sleep(2); os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    sec = round(time.time() - t0, 1); time.sleep(5)
    log = open(D + "/run.log", errors="ignore").read()
    import re
    g = lambda pat: (re.search(pat, log).group(1) if re.search(pat, log) else None)
    st = states(); EM.shot(D + "/before_home.png")
    r = dict(k=k, asset=asset, place_xy=list(place_xy), top_z=top_z, perc_yaw=(fr or {}).get("yaw"), t_perc=t_perc, sec=sec,
             planned="pick 과 place 모두 계획됨" in log, verdict=g(r"place verdict=(\w+)"), c_sec=g(r"C 완료 ([\d.]+)s"), n_cand=g(r"\[cgn_all\] 후보 (\d+ -> \d+)개"),
             dest_below=g(r"놓을 곳 아래 받침 물체 (\d+)개"), c_fail=re.findall(r"C 실패\([^)]*\) (.{0,160})", log)[:2])
    return r, st


if __name__ == "__main__":
    t_all = time.time(); S.clean_scene(); EM.shot(OUT + "/t0_empty.png")
    t = math.radians(TH); Rr = lambda x, y: (CX + x * math.cos(t) - y * math.sin(t), CY + x * math.sin(t) + y * math.cos(t))
    corners = sorted([Rr(dx * SS / 2, dy * SS / 2) for dx in (-1, 1) for dy in (-1, 1)], key=lambda q: (q[0], q[1]))      # 먼 쪽(-x) 먼저
    res = []; st = {}
    for k, xy in enumerate(corners):
        nm = f"gso_{k}_bar"; print("생성", nm, spawn(nm, "bar", BAR_YAW), flush=True); time.sleep(5)
        r, st = step(k, "bar", xy, 0.0, ["왼면이 위", "오른면이 위"], lambda y: None, False)
        s = st.get(nm); r.update(prim=nm, final=s, pos_mm=(round(1000 * math.hypot(s["xyz"][0] - xy[0], s["xyz"][1] - xy[1]), 1) if s else None),
                                 z_mm=(round(1000 * (s["xyz"][2] - BAR_L / 2), 1) if s else None), tilt=(axis_tilt(s, 0) if s else None))
        r["others"] = {f"gso_{j}_bar": dict(tilt=axis_tilt(st[f"gso_{j}_bar"], 0), moved_mm=round(1000 * math.hypot(st[f"gso_{j}_bar"]["xyz"][0] - res[j]["final"]["xyz"][0], st[f"gso_{j}_bar"]["xyz"][1] - res[j]["final"]["xyz"][1]), 1)) for j in range(k) if f"gso_{j}_bar" in st and res[j].get("final")}      # r3: 앞서 세운 기둥이 이번 단계에서 밀렸나
        r["home"] = home(st); res.append(r); json.dump(res, open(OUT + "/result.json", "w"), ensure_ascii=False, indent=1)
        print(f"   앞 기둥 상태: {r['others']}", flush=True)
        print(f"기둥{k} 목표({xy[0]:.3f},{xy[1]:.3f}) 계획 {r['planned']} 판정 {r['verdict']} 위치오차 {r['pos_mm']}mm 높이오차 {r['z_mm']}mm 기울기 {r['tilt']}° | 인식 {r['t_perc']}s C {r['c_sec']}s 후보 {r['n_cand']} 전체 {r['sec']}s | 복귀 {r['home'][:60]}", flush=True)
    nm = "gso_4_plate"; print("생성", nm, spawn(nm, "plate", PLATE_YAW), flush=True); time.sleep(5)
    yfn = lambda py: sorted({int(round(((TH - py + 90 * kk + 180) % 360) - 180)) for kk in range(4)})
    r, st = step(4, "plate", (CX, CY), BAR_L, ["앞면이 위", "뒷면이 위"], yfn, True)
    s = st.get(nm)
    if s:
        Rm = np.array(s["R"]); c = np.array(s["xyz"]); inside = []
        for kk in range(4):
            q = st.get(f"gso_{kk}_bar")
            if q is None: inside.append(False); continue
            L = Rm.T @ (np.array(q["xyz"]) + np.array([0, 0, BAR_L / 2]) - c); inside.append(bool(abs(L[0]) <= PL[0] / 2 and abs(L[2]) <= PL[2] / 2))      # 기둥 윗면 중심이 판 면(12x10) 안
        r.update(pos_mm=round(1000 * math.hypot(c[0] - CX, c[1] - CY), 1), z_mm=round(1000 * (c[2] - (BAR_L + PL[1] / 2)), 1), tilt=axis_tilt(s, 1), covers=inside)
    r.update(prim=nm, final=s)
    pil = [dict(k=kk, tilt=axis_tilt(st[f"gso_{kk}_bar"], 0), z_mm=round(1000 * (st[f"gso_{kk}_bar"]["xyz"][2] - BAR_L / 2), 1)) for kk in range(4) if f"gso_{kk}_bar" in st]
    r["pillars_after_lid"] = pil; r["home"] = home(st); res.append(r)
    EM.shot(OUT + "/t1_final.png")
    ok_p = [x for x in res[:4] if x.get("tilt") is not None and x["tilt"] <= 5 and x["pos_mm"] <= 15]
    ok = len(ok_p) == 4 and r.get("covers") == [True] * 4 and (r.get("tilt") or 99) <= 5 and all(x["tilt"] <= 5 for x in pil)
    print(f"뚜껑 계획 {r['planned']} 판정 {r['verdict']} 중심오차 {r.get('pos_mm')}mm 높이오차 {r.get('z_mm')}mm 기울기 {r.get('tilt')}° 기둥 덮음 {r.get('covers')} | 받침 인식 {r['dest_below']} C {r['c_sec']}s 후보 {r['n_cand']} 전체 {r['sec']}s", flush=True)
    print(f"뚜껑 뒤 기둥 기울기 {[x['tilt'] for x in pil]}", flush=True)
    json.dump(dict(cfg=dict(cx=CX, cy=CY, s=SS, theta=TH), steps=res, ok=ok, wall_s=round(time.time() - t_all)), open(OUT + "/result.json", "w"), ensure_ascii=False, indent=1)
    print("끝 ok", ok, round(time.time() - t_all), "s", flush=True)
