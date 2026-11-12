"""VLM 실행 확인(2026-11-10, 논문 본체 경로 첫 실검증): 무작위 장면(scenes_rand) x VLM 문항(items_vlm)을 Isaac 에서 실제로 집고 놓는다.
한 경우의 흐름(카메라만 씀, 정답 각도·정답 사진 안 씀):
  Isaac: 무늬 직육면체(vlm_scene.spawn_tex, 정답 yaw 로 놓음) + 장면 장애물 블록(정적, 'Platform_blk' -> clean_scene 이 지움) -> 로봇 시작(rt_run.sh, 프레임 덤프 켬)
  로봇 인식이 끝나면(poses.json.perc.json) -> 인식 좌표(perc_frame) -> 이 실행의 스캔 프레임에서 원본·면 이름표 크롭(vlm_scene.frames_crop, perc)
  -> Gemma(사진 2장 + 깊이 모양 글 + cmd_v) 조건 -> 서버 후보(/candidates, 인식 yaw) -> Gemma 순서 -> poses.json(place_spec) -> 로봇이 계획·실행
  -> 계획 안 서면 자세별 실제 이유(relax_common, 2026-11-10 펼친 형식 수정본)로 Gemma 풀기/포기(최대 3라운드, 라운드마다 장면 다시 놓고 다시 인식)
판정: (1) Gemma 자기 조건 만족(인식 좌표) (2) 원래 뜻(GT, grade_table 의 사전 등록 기준, 정답 좌표) -- 실행 결과는 판단 결과와 따로 기록.
장애물 블록은 플래너에 dests 고정 장애물(바닥~블록 윗면 기둥, 받침대 위라 같은 공간)로 알림.
사용: python3 stage2_vlm.py [CASES.json]   (기본 blocked_v4/exec_vlm_cases.json)  결과: stage2_vlm/episodes.jsonl, stage2_vlm/runs/NNN/"""
import os, sys, json, re, time, subprocess, signal, threading, shutil, glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
os.environ.setdefault("BP_SERVER_CAND", "1")
import stage2_loop as S, vlm_scene as V, vlm_mode as VM, perc_frame as PF, relax_common as RC, blocked_plan_only as B, grade_table as G
from isaac_exec import run_code
EM = S.EM
EM.ENV["NERO_HELD_COVER"] = os.environ.get("NERO_HELD_COVER", "0"); EM.ENV["NERO_PLACE_NEIGHBOR_REJECT"] = os.environ.get("NERO_PLACE_NEIGHBOR_REJECT", "1")      # 2026-11-10 저녁: 외접구(bound)는 받침면 충돌로 되던 놓기를 막음(A/B) -> 끄고, 놓은 물체-이웃 정확 겹침 검사(서버 neighbor_reject, 회귀 59경우 0건)로 대신      # 2026-11-10: 든 물체를 외접구로(모서리 2~3cm 삐져나와 블록을 뚫고 놓던 문제, run 007/011). 사용자 승인
OUT = os.environ.get("S2V_OUT", os.path.join(H, "stage2_vlm")); os.makedirs(OUT + "/runs", exist_ok=True)
S.RANK_F = OUT + "/rank_cache.json"; S.RANK.clear()
ITEMS = {i["id"]: i for i in json.load(open(H + "/blocked_v4/items_vlm.json"))}
SCENES = {s["id"]: s for s in json.load(open(H + "/blocked_v4/scenes_rand.json"))}
VM.enable()
BLK = '''
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf
st=omni.usd.get_context().get_stage()
c=UsdGeom.Cube.Define(st,"/World/Platform_blk_{i}"); c.CreateSizeAttr(1.0)
xf=UsdGeom.Xformable(c); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(Gf.Vec3d({x},{y},{z})); xf.AddRotateZOp().Set({yaw}); xf.AddScaleOp().Set(Gf.Vec3f({sx},{sy},{sz}))
c.CreateDisplayColorAttr([(0.55,0.35,0.2)]); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
'''


# 2026-11-10 진단: 무늬 메시는 면마다 GeomSubset 에 보이는 재질을 묶는다. 일반 상자(v4)와 같은 경우에서 무늬 물체만 옮기는 중 2/2 떨어짐(일반 상자는 놓음) ->
# 면 묶음에도 고마찰 '물리' 재질을 직접 묶는다(PhysX 가 면 묶음 재질을 쓰는 경우 대비). 효과는 같은 경우 반복으로 확인(stage2_vlm_diag*).
SUBSET_FRIC = '''
import omni.usd
from pxr import UsdShade
st=omni.usd.get_context().get_stage(); hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction"); p=st.GetPrimAtPath("/World/gso_0_{name}")
if hm.IsValid() and p.IsValid():
    for q in p.GetChildren():
        UsdShade.MaterialBindingAPI.Apply(q).Bind(UsdShade.Material(hm), UsdShade.Tokens.strongerThanDescendants, "physics")
'''


VIS = '''
import omni.usd
from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt
st=omni.usd.get_context().get_stage(); par="/World/gso_0_{name}"
k=0.502
F=[("top",(0,0,1),(1,0,0),(0,1,0)),("bottom",(0,0,-1),(1,0,0),(0,-1,0)),("front",(0,-1,0),(1,0,0),(0,0,1)),("back",(0,1,0),(-1,0,0),(0,0,1)),("left",(-1,0,0),(0,-1,0),(0,0,1)),("right",(1,0,0),(0,1,0),(0,0,1))]
pts=[]; st_=[]; cnt=[]; idx=[]
for fi,(nm,n,u,v) in enumerate(F):
    c=[n[i]*k for i in range(3)]
    for su,sv in ((-1,-1),(1,-1),(1,1),(-1,1)):
        pts.append(Gf.Vec3f(*[c[i]+su*u[i]*k+sv*v[i]*k for i in range(3)])); st_.append(Gf.Vec2f((su+1)/2,(sv+1)/2))
    cnt.append(4); idx+=[4*fi+j for j in range(4)]
path=par+"/vis"; m=UsdGeom.Mesh.Define(st,path); m.CreatePointsAttr(Vt.Vec3fArray(pts)); m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx); m.CreateSubdivisionSchemeAttr("none")
UsdGeom.PrimvarsAPI(m).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray(st_))
for fi,(nm,n,u,v) in enumerate(F):
    mp="/World/Looks/tex_{texture}_"+nm
    if not st.GetPrimAtPath(mp).IsValid():
        mt=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+"/s"); sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(0.6)
        rd=UsdShade.Shader.Define(st,mp+"/st"); rd.CreateIdAttr("UsdPrimvarReader_float2"); rd.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st")
        tx=UsdShade.Shader.Define(st,mp+"/tex"); tx.CreateIdAttr("UsdUVTexture"); tx.CreateInput("file",Sdf.ValueTypeNames.Asset).Set("{tex}/{texture}/"+nm+".png")
        tx.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(),"result"); tx.CreateOutput("rgb",Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(),"rgb"); mt.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
    ss=UsdGeom.Subset.Define(st,path+"/f_"+nm); ss.CreateElementTypeAttr("face"); ss.CreateIndicesAttr([fi]); ss.CreateFamilyNameAttr("materialBind")
    UsdShade.MaterialBindingAPI.Apply(ss.GetPrim()).Bind(UsdShade.Material(st.GetPrimAtPath(mp)))
'''


def setup_scene(sc, texture):
    S.clean_scene()
    if os.environ.get("S2V_PLAIN") == "1": EM.spawn(sc["asset"], sc["yaw"])      # 진단 전용: 무늬 없는 기존 직육면체(v4 와 같은 물체)
    elif os.environ.get("S2V_TEX_MESH") == "1":      # 진단 전용: 무늬 메시 자체가 물리 물체(옮기는 중 3/4 떨어짐)
        V.spawn_tex(sc["asset"], sc["yaw"], texture)
        if os.environ.get("S2V_SUBSET_FRIC", "1") == "1": run_code(SUBSET_FRIC.format(name=sc["asset"]), timeout=30)
    else:      # 기본(2026-11-10): 물리 = v4 와 같은 직육면체(EM.spawn), 무늬 = 충돌 없는 보이기 전용 껍데기(자식, 0.2% 크게)
        EM.spawn(sc["asset"], sc["yaw"]); run_code(VIS.format(name=sc["asset"], texture=texture, tex=V.TEX), timeout=60)
    for i, o in enumerate(sc.get("obstacles") or []):
        run_code(BLK.format(i=i, x=o["xyz"][0], y=o["xyz"][1], z=o["xyz"][2], yaw=float(o.get("yaw", 0.0)), sx=o["raw"][0], sy=o["raw"][1], sz=o["raw"][2]), timeout=40)
    time.sleep(3)


def dests_file(sc, D):
    d = json.load(open(EM.DESTS))
    for o in sc.get("obstacles") or []:
        top = o["xyz"][2] + o["raw"][2] / 2
        d.append(dict(id=o["id"], label="block", kind="flat", xy=o["xyz"][:2], size=[o["raw"][0], o["raw"][1], top], yaw=float(o.get("yaw", 0.0))))
    f = D + "/dests.json"; json.dump(d, open(f, "w")); return f


def start_robot(D, sc):
    e = dict(EM.ENV); e["NERO_PLACE_POSE_LIST_FILE"] = D + "/poses.json"; e.pop("NERO_PLACE_POSE_LIST", None); e["NERO_MULTI_LOG"] = D
    os.makedirs(D + "/frames", exist_ok=True); e["NERO_FRAME_DUMP"] = D + "/frames"
    os.makedirs(D + "/creq", exist_ok=True); e["NERO_C_REQ_DUMP"] = D + "/creq"
    if SAME_PERC: e["NERO_REPLAN_LOOP"] = "1"; e["NERO_REPLAN_MAX"] = str(S.MAX_ROUND); e["NERO_PERC_EARLY"] = "1"; e["NERO_REPLAN_REUSE_CB"] = "1"      # 2026-11-11(S2V_SAME_PERC=1, 사용자 승인): 한 에피소드 안에서 로봇 프로세스를 유지하고 같은 인식으로 다시 계획      # 2026-11-10: 로봇이 보낸 C 요청 저장(다음 라운드 재계획에 재사용)
    return subprocess.Popen(["./rt_run.sh", f"--target-at={EM.OX},{EM.OY}", "--dests", dests_file(sc, D), "--dest-id", "platform", "--dest-kind", "flat", "--vocab", EM.VOC, "--max-picks", "1",
                             "--max-fail", "1", "--place-clear", "0.015", "--out", D + "/run.json"], cwd=EM.PC, env=e, stdout=open(D + "/run.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)


CUR_P = [None]


def _on_term(sig, frm):      # 2026-11-10: 러너를 멈추면 로봇 프로세스 묶음도 같이 멈춘다(전에는 남아서 팔이 계속 움직임)
    if CUR_P[0] is not None: stop_robot(CUR_P[0])
    sys.exit(1)


signal.signal(signal.SIGTERM, _on_term)


def stop_robot(p):
    try: os.killpg(p.pid, signal.SIGTERM); time.sleep(2); os.killpg(p.pid, signal.SIGKILL)
    except Exception: pass


def wait_perc(D, asset, timeout=420, p=None):
    f = D + "/poses.json.perc.json"; t0 = time.time()
    while not os.path.exists(f) and time.time() - t0 < timeout and (p is None or p.poll() is None): time.sleep(0.3)      # 로봇 프로세스가 먼저 죽으면 바로 끝냄(2026-11-10: 시작 직후 카메라 정보 KeyError 'K' 로 죽었는데 420s 를 기다리며 로봇이 멈춰 있었음)
    if not os.path.exists(f): return None
    time.sleep(0.3); return PF.frame(json.load(open(f))["obj"], asset, S.GRID_DIMS)


def vlm_item(it, asset, fr, imgs):
    VM.CUR["imgs"] = imgs; a_, b_, c_ = [round(100 * v, 1) for v in fr["dims"]]; fmt = lambda v: (str(int(v)) if float(v).is_integer() else str(v))
    return dict(it, obj=VM.OBJ_V.format(shape=VM.SHAPE[asset].format(a=fmt(a_), b=fmt(b_), c=fmt(c_))), cmd=it["cmd_v"], obj_text_orig=it["obj"])


def safe_home(s, asset, sc):
    """stage2_loop.safe_home 과 같고, 장면 장애물 블록도 피한다."""
    obs = [dict(id="platform", xyz=[EM.PX, EM.PY, EM.TOP / 2], raw=[0.24, 0.24, EM.TOP], yaw=0.0)] + [dict(id=o["id"], xyz=o["xyz"], raw=o["raw"], yaw=float(o.get("yaw", 0.0))) for o in sc.get("obstacles") or []]
    if s:
        Rm = np.array(s["R"], float); dm = list(S.GRID_DIMS[asset]); k = int(np.argmax(np.abs(Rm[2, :]))); i, j = [a for a in range(3) if a != k]
        obs.append(dict(id="placed", xyz=s["xyz"], raw=[dm[i], dm[j], dm[k]], yaw=float(np.arctan2(Rm[1, i], Rm[0, i]))))
    margins = [m.strip() for m in os.environ.get("S2V_HOME_MARGIN", "0.03,0.005").split(",") if m.strip()]      # 2026-11-12: 복귀 경로 여유(m). 여러 값이면 앞에서부터 시도(예 "0.03,0.005": 3cm 로 먼저, 시작 자세가 막히면 0.5cm). 기본 "0.03,0.005": 0.5cm 만 쓰면 손목 카메라가 작업대 모서리를 스쳤음(2026-11-12 촬영에서 확인), 3cm 로 2/2 접촉 없이 복귀
    tries = [(f"[여유 {m}] ", ob, m) for m in margins for ob in ([obs] + ([obs[:-1]] if s else []))]; parts = []
    tries = [(t + ("(놓인 물체 제외) " if ob is not obs else ""), ob, m) for t, ob, m in tries]
    for tag, ob, mg in tries:
        out = subprocess.run(["python3", os.path.join(H, "home_safe.py"), json.dumps(ob), mg], cwd=EM.PC, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
        m_ = out[-1] if out else "FAIL 출력 없음"; parts.append(tag + m_)
        if not m_.startswith("FAIL"): return " | ".join(parts)
    subprocess.run(["python3", "home_and_get_armq.py"], cwd=EM.PC, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, timeout=90)
    return " | ".join(parts) + " -> 기존 복귀"


def judge_state(s, spec, asset, dyaw):
    """놓인 상태 판정. spec 은 인식 좌표 면 이름 -> 인식 로컬축 = 정답 로컬축 * Rz(dyaw)."""
    if not (s and s["on_platform"]): return None
    R = Rot.from_matrix(np.array(s["R"])) * Rot.from_euler("z", dyaw, degrees=True)
    ok = S.check(R, spec, "place", pos=s["xyz"][:2], dims=S.GRID_DIMS[asset], center=S.CENTER, half=0.12); return dict(ok=bool(ok[0]), broken=ok[1])


REUSE_C = os.environ.get("S2V_REUSE_C", "0") == "1"
SAME_PERC = os.environ.get("S2V_SAME_PERC", "0") == "1"      # 2026-11-10(사용자 지시: 낭비 제거): 2라운드부터 앞 라운드 로봇의 C 요청(같은 인식·파지 후보)으로 먼저 계획만 해 보고, 서면 그때만 로봇을 다시 돌린다


def replan_prev(last_D, spec, asset, yaw0, names):
    """앞 라운드 로봇 실행의 C 요청들을 새 조건·순서로 8021 에 다시 보냄. -> (ok, per_pose 목록)"""
    pps = []
    for f in sorted(glob.glob(last_D + "/creq/*.json")):
        r = json.load(open(f)); pr = (r.get("place_check") or {}).get("req")
        if pr is None: continue
        pr["place_spec"] = dict(hard=spec.get("hard") or [], asset=asset, yaw0=round(float(yaw0), 2), order=list(names)); pr.pop("obj_rot_list", None); pr.pop("obj_rot", None)
        try: o = B.post(8021, "/plan", r)
        except Exception as ex: o = dict(ok=False, why=repr(ex)[:200])
        if o.get("ok"): return True, pps
        pps.append({k: dict(reasons=v[0], state=v[1]) for k, v in RC.per_pose(o.get("why")).items()})
    return False, pps


def episode(case, n_run):
    import rq1_decide3_prompt as P3
    sid, iid = case["scene"], case["item"]; sc = case.get("scene_def") or SCENES[sid]; it0 = ITEMS[iid]; asset = sc["asset"]; t_ep = time.time()
    hist = []; spec = why = it = None; imgs = None; outcome = "gave_up_rounds"; tm = dict(cond=0.0, rank=0.0, relax=0.0, robot=0.0, crop=0.0, harness=0.0)
    failed = {}; last_fr = None; last_D = None; pre = None      # 2026-11-10(사용자 승인): 이미 실패한 후보 (자세, yaw, 자리) -> 실패 이유. 새 조건의 후보가 모두 여기 있으면 로봇을 다시 돌리지 않는다
    key = lambda c: (c["pose"], tuple(c.get("yaws") or ()), tuple(c.get("xy") or ()))
    for rnd in range(S.MAX_ROUND):
        if rnd > 0 and last_fr is not None:
            cs_ = B.server_candidates(spec, asset, last_fr["yaw"])
            if cs_ and all(key(c) in failed for c in cs_):      # 같은 후보 = 같은 계획 요청(파지 표본만 다름) -> 다시 계획하지 않고 이전 이유로 바로 다시 묻는다
                tried = [dict(pose=c["pose"], where=None, why=failed[key(c)] + " (바뀐 조건의 후보가 이전 라운드에서 이미 실패한 후보와 같음)") for c in cs_]
                hist.append(dict(round=rnd, spec=spec, why=why, pl=[c["pose"] for c in cs_], skipped_same=True, tried_msgs=tried))
                t0 = time.time(); j, raw, sec = S.ask_relax(it, spec, why, tried); tm["relax"] += time.time() - t0
                if j.get("abstain"): hist.append(dict(round=rnd + 1, abstain=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
                new = S.norm_spec(j)
                if S.bad_faces(new): hist.append(dict(round=rnd + 1, invalid_spec=True, bad=S.bad_faces(new), spec_raw=j, why=j.get("why"), raw=raw)); outcome = "invalid_spec"; break      # 2026-11-11: 조건 언어에 없는 면 이름 -> 판단 실패로 기록(고치지 않음)
                if new == spec: hist.append(dict(round=rnd + 1, abstain=True, kept_same=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
                spec, why = new, j.get("why", ""); continue
        pre = None
        if rnd > 0 and REUSE_C and last_fr is not None and last_D and glob.glob(last_D + "/creq/*.json"):
            t0 = time.time(); cs_ = B.server_candidates(spec, asset, last_fr["yaw"]); ranked_, rk_ = S.rank(it, spec, why, cs_); tm["rank"] += time.time() - t0
            names = [c["pose"] for c in ranked_]
            if not names: hist.append(dict(round=rnd, spec=spec, why=why, pl=[], note="조건을 만족하는 자세 없음", replan_only=True)); outcome = "abstain_nopose"; break
            t0 = time.time(); ok_, pps = replan_prev(last_D, spec, asset, last_fr["yaw"], names); tm["replan"] = tm.get("replan", 0.0) + time.time() - t0
            if ok_: pre = names      # 계획이 섬 -> 로봇을 다시 돌려 실제 실행(순위는 재사용)
            else:
                tried = [dict(pose=n, where=None, why=RC.pose_msg(n, pps)) for n in names]
                hist.append(dict(round=rnd, spec=spec, why=why, pl=names, rank=rk_, replan_only=True, run=last_D, per_pose=pps, tried_msgs=tried, planned=False))
                for c_, t_ in zip(ranked_, tried): failed[key(c_)] = t_["why"]
                t0 = time.time(); j, raw, sec = S.ask_relax(it, spec, why, tried); tm["relax"] += time.time() - t0
                if j.get("abstain"): hist.append(dict(round=rnd + 1, abstain=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
                new = S.norm_spec(j)
                if S.bad_faces(new): hist.append(dict(round=rnd + 1, invalid_spec=True, bad=S.bad_faces(new), spec_raw=j, why=j.get("why"), raw=raw)); outcome = "invalid_spec"; break      # 2026-11-11: 조건 언어에 없는 면 이름 -> 판단 실패로 기록(고치지 않음)
                if new == spec: hist.append(dict(round=rnd + 1, abstain=True, kept_same=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
                spec, why = new, j.get("why", ""); continue
        D = f"{OUT}/runs/{n_run[0]:03d}"; n_run[0] += 1; os.makedirs(D, exist_ok=True); last_D = D
        t_h = time.time(); setup_scene(sc, it0["texture"]); tm["harness"] += time.time() - t_h; t_r = time.time(); p = start_robot(D, sc); CUR_P[0] = p; g = {}
        try:
            fr = wait_perc(D, asset, p=p)
            if fr is None and p.poll() is not None and "KeyError: 'K'" in open(D + "/run.log").read():      # 시작 직후 카메라 정보를 못 받고 죽음(실행 인프라) -> 같은 라운드 1회 다시(기록 남김)
                os.rename(D + "/run.log", D + "/run_crash_K.log"); time.sleep(5); setup_scene(sc, it0["texture"]); t_r = time.time(); p = start_robot(D, sc); fr = wait_perc(D, asset, p=p); g["restart_K"] = True
            if fr is None: raise RuntimeError("인식 상자 파일 없음(로봇 인식 실패)")
            g["perc"] = fr; g["t_perc"] = round(time.time() - t_r, 1); dyaw = fr["yaw"] - sc["yaw"]
            if rnd == 0:
                t0 = time.time(); imgs = [D + "/cam_raw.jpg", D + "/cam_lab.jpg"]
                if not V.frames_crop(asset, sc["yaw"], D + "/frames", imgs[0], imgs[1], perc=fr): raise RuntimeError("스캔 프레임에 물체가 온전히 보이는 프레임 없음")
                tm["crop"] += time.time() - t0
                it = vlm_item(it0, asset, fr, imgs)
                txt, rs, ct, sec, fin = P3.call(P3.prompt(it), 0); j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
                spec, why = S.norm_spec(j), j.get("why", ""); tm["cond"] += sec; g["cond_raw"] = txt
                if S.bad_faces(spec): raise S.BadSpec(f"조건 언어에 없는 면 이름 {S.bad_faces(spec)} | raw={txt[:400]}")
            else: VM.CUR["imgs"] = imgs
            cands = B.server_candidates(spec, asset, fr["yaw"]); last_fr = fr
            if pre is not None: ranked = sorted(cands, key=lambda c: pre.index(c["pose"]) if c["pose"] in pre else 99); rk = dict(reused=True)      # 재계획에서 이미 매긴 순위
            else: t0 = time.time(); ranked, rk = S.rank(it, spec, why, cands); tm["rank"] += time.time() - t0
            g["pl"] = [c["pose"] for c in ranked]; g["cands"] = ranked; g["rank"] = rk; g["t_list"] = round(time.time() - t_r, 1)
        except Exception as ex:
            stop_robot(p); hist.append(dict(round=rnd, err=repr(ex)[:300], run=D)); outcome = "invalid_spec" if isinstance(ex, S.BadSpec) else "error"; break
        h = dict(round=rnd, run=D, spec=spec, why=why, perc=fr, dyaw=round(dyaw, 1), pl=g["pl"], rank=g["rank"], t_perc=g["t_perc"], t_list=g["t_list"], cond_raw=g.get("cond_raw"))
        if not g["pl"]:
            stop_robot(p); h["note"] = "조건을 만족하는 자세 없음"; hist.append(h); outcome = "abstain_nopose"; break
        json.dump(dict(place_spec=dict(hard=spec["hard"], asset=asset, yaw0=round(fr["yaw"], 2), order=g["pl"])), open(D + "/poses.json.tmp", "w"), ensure_ascii=False); os.replace(D + "/poses.json.tmp", D + "/poses.json")
        try: p.wait(timeout=900)
        except subprocess.TimeoutExpired: stop_robot(p); h["robot_timeout"] = True
        tm["robot"] += time.time() - t_r; t_h = time.time()
        log = open(D + "/run.log").read(); planned = "pick 과 place 모두 계획됨" in log
        if planned: time.sleep(4)      # 2026-11-11: 놓은 뒤 정착 대기는 실제로 놓았을 때만(계획 실패 회차는 로봇이 안 움직임)
        s = EM.final_state(); EM.shot(D + "/before_home.png"); h["home"] = safe_home(s, asset, sc); tm["harness"] += time.time() - t_h
        mp = re.findall(r"놓기 자세 선택 (\{.*\})", log); h["chosen"] = json.loads(mp[-1]) if mp else None
        mw = re.search(r"자세 목록 파일 받음\(대기 ([0-9.]+)s\)", log); h["robot_waited_for_list"] = float(mw.group(1)) if mw else None
        h.update(planned=planned, state=s, n_c_fail=len(re.findall(r"C 실패\(", log)))
        if planned:
            h["own"] = judge_state(s, spec, asset, dyaw); h["orig"] = judge_state(s, G.gt_for(iid, spec), asset, 0.0)
            hist.append(h); outcome = "executed_ok" if (h["own"] or {}).get("ok") else ("executed_bad" if s and s["on_platform"] else "dropped_or_offplatform"); break
        pps = RC.per_pose_from_log(log); h["per_pose"] = pps
        tried = [dict(pose=n, where=None, why=RC.pose_msg(n, pps)) for n in g["pl"]]; h["tried_msgs"] = tried; hist.append(h)
        for c_, t_ in zip(g["cands"], tried): failed[key(c_)] = t_["why"]
        t0 = time.time(); j, raw, sec = S.ask_relax(it, spec, why, tried); tm["relax"] += time.time() - t0
        if j.get("abstain"): hist.append(dict(round=rnd + 1, abstain=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
        new = S.norm_spec(j)
        if S.bad_faces(new): hist.append(dict(round=rnd + 1, invalid_spec=True, bad=S.bad_faces(new), spec_raw=j, why=j.get("why"), raw=raw)); outcome = "invalid_spec"; break      # 2026-11-11: 조건 언어에 없는 면 이름 -> 판단 실패로 기록(고치지 않음)
        if new == spec: hist.append(dict(round=rnd + 1, abstain=True, kept_same=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
        spec, why = new, j.get("why", "")
    return dict(scene=sid, item=iid, outcome=outcome, relaxed=sum(1 for x in hist if "pl" in x) > 1, final_spec=spec, hist=hist, sec=round(time.time() - t_ep, 1), stage_sec={k: round(v, 1) for k, v in tm.items()})


def episode_sp(case, n_run):
    """2026-11-11(S2V_SAME_PERC=1, 사용자 승인): 실제 로봇처럼 한 에피소드 = 장면 세팅 1번 + 로봇 프로세스 1개 + 인식 1번.
    C 가 실패하면 로봇은 <목록>.fail<k>.json 을 쓰고 기다림 -> 여기서 Gemma 풂 -> (새 조건 후보 + 순서) <목록>.r<k> -> 로봇이 같은 프레임으로 다시 계획.
    질문(조건·순서·풂 프롬프트)과 라운드 수(S.MAX_ROUND)는 episode 와 같다. 이전 에피소드/실험의 결과는 쓰지 않는다."""
    import rq1_decide3_prompt as P3
    sid, iid = case["scene"], case["item"]; sc = case.get("scene_def") or SCENES[sid]; it0 = ITEMS[iid]; asset = sc["asset"]; t_ep = time.time()
    hist = []; outcome = "gave_up_rounds"; tm = dict(cond=0.0, rank=0.0, relax=0.0, robot=0.0, crop=0.0, harness=0.0)
    D = f"{OUT}/runs/{n_run[0]:03d}"; n_run[0] += 1; os.makedirs(D, exist_ok=True); pf = D + "/poses.json"
    t_h = time.time(); setup_scene(sc, it0["texture"]); tm["harness"] += time.time() - t_h
    t_r = time.time(); p = start_robot(D, sc); CUR_P[0] = p
    try:
        fr = wait_perc(D, asset, p=p)
        if fr is None: raise RuntimeError("인식 상자 파일 없음(로봇 인식 실패)")
        t_perc = round(time.time() - t_r, 1); dyaw = fr["yaw"] - sc["yaw"]
        t0 = time.time(); imgs = [D + "/cam_raw.jpg", D + "/cam_lab.jpg"]
        if not V.frames_crop(asset, sc["yaw"], D + "/frames", imgs[0], imgs[1], perc=fr): raise RuntimeError("스캔 프레임에 물체가 온전히 보이는 프레임 없음")
        tm["crop"] += time.time() - t0
        it = vlm_item(it0, asset, fr, imgs)
        txt, rs, ct, sec, fin = P3.call(P3.prompt(it), 0); j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        spec, why = S.norm_spec(j), j.get("why", ""); tm["cond"] += sec; cond_raw = txt
        if S.bad_faces(spec): raise S.BadSpec(f"조건 언어에 없는 면 이름 {S.bad_faces(spec)} | raw={txt[:400]}")
    except Exception as ex:
        stop_robot(p); hist.append(dict(round=0, err=repr(ex)[:300], run=D)); return dict(scene=sid, item=iid, outcome=("invalid_spec" if isinstance(ex, S.BadSpec) else "error"), relaxed=False, final_spec=None, hist=hist, sec=round(time.time() - t_ep, 1), stage_sec={k: round(v, 1) for k, v in tm.items()}, same_perc=True)
    log_off = 0; nfail = 0; h = None
    failed = {}; key = lambda c: (c["pose"], tuple(c.get("yaws") or ()), tuple(c.get("xy") or ()))      # episode 와 같은 '이미 실패한 후보 건너뛰기'(2026-11-10 사용자 승인) -- 판단 절차를 episode 와 같게
    for rnd in range(S.MAX_ROUND):
        cands = B.server_candidates(spec, asset, fr["yaw"])
        if rnd > 0 and cands and all(key(c) in failed for c in cands):      # 같은 후보 = 같은 계획 요청 -> 다시 계획하지 않고 이전 이유로 바로 다시 묻는다(episode 와 같은 문구)
            tried = [dict(pose=c["pose"], where=None, why=failed[key(c)] + " (바뀐 조건의 후보가 이전 라운드에서 이미 실패한 후보와 같음)") for c in cands]
            hist.append(dict(round=rnd, spec=spec, why=why, pl=[c["pose"] for c in cands], skipped_same=True, tried_msgs=tried, same_perc=True))
            t0 = time.time(); j, raw, sec = S.ask_relax(it, spec, why, tried); tm["relax"] += time.time() - t0
            if j.get("abstain"): hist.append(dict(round=rnd + 1, abstain=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
            new = S.norm_spec(j)
            if S.bad_faces(new): hist.append(dict(round=rnd + 1, invalid_spec=True, bad=S.bad_faces(new), spec_raw=j, why=j.get("why"), raw=raw)); outcome = "invalid_spec"; break      # 2026-11-11: 조건 언어에 없는 면 이름 -> 판단 실패로 기록(고치지 않음)
            if new == spec: hist.append(dict(round=rnd + 1, abstain=True, kept_same=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
            spec, why = new, j.get("why", ""); continue
        t0 = time.time(); ranked, rk = S.rank(it, spec, why, cands); tm["rank"] += time.time() - t0
        pl = [c["pose"] for c in ranked]
        h = dict(round=rnd, run=D, spec=spec, why=why, perc=fr, dyaw=round(dyaw, 1), pl=pl, rank=rk, t_perc=t_perc, t_list=round(time.time() - t_r, 1), cond_raw=cond_raw if rnd == 0 else None, same_perc=True)
        if not pl:
            h["note"] = "조건을 만족하는 자세 없음"; hist.append(h); h = None; outcome = "abstain_nopose"; break
        out_f = pf if nfail == 0 else f"{pf}.r{nfail}"
        json.dump(dict(place_spec=dict(hard=spec["hard"], asset=asset, yaw0=round(fr["yaw"], 2), order=pl)), open(out_f + ".tmp", "w"), ensure_ascii=False); os.replace(out_f + ".tmp", out_f)
        nf = f"{pf}.fail{nfail + 1}.json"; t_w = time.time()
        while p.poll() is None and not os.path.exists(nf) and time.time() - t_w < 900: time.sleep(0.2)
        log = open(D + "/run.log").read(); seg = log[log_off:]; log_off = len(log)
        planned = "pick 과 place 모두 계획됨" in seg
        mw = re.search(r"자세 목록 파일 받음\(대기 ([0-9.]+)s\)", seg) or re.search(r"\[replan\] 새 자세 목록 받음\(대기 ([0-9.]+)s\)", seg); h["robot_waited_for_list"] = float(mw.group(1)) if mw else None
        h.update(planned=planned, n_c_fail=len(re.findall(r"C 실패\(", seg)))
        if planned or p.poll() is not None or not os.path.exists(nf):
            if not planned and p.poll() is None: stop_robot(p); h["robot_timeout"] = True
            break
        nfail += 1
        pps = RC.per_pose_from_log(seg); h["per_pose"] = pps
        tried = [dict(pose=n, where=None, why=RC.pose_msg(n, pps)) for n in pl]; h["tried_msgs"] = tried; hist.append(h); h = None
        for c_, t_ in zip(ranked, tried): failed[key(c_)] = t_["why"]
        t0 = time.time(); j, raw, sec = S.ask_relax(it, spec, why, tried); tm["relax"] += time.time() - t0
        if j.get("abstain"): hist.append(dict(round=rnd + 1, abstain=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
        new = S.norm_spec(j)
        if S.bad_faces(new): hist.append(dict(round=rnd + 1, invalid_spec=True, bad=S.bad_faces(new), spec_raw=j, why=j.get("why"), raw=raw)); outcome = "invalid_spec"; break      # 2026-11-11: 조건 언어에 없는 면 이름 -> 판단 실패로 기록(고치지 않음)
        if new == spec: hist.append(dict(round=rnd + 1, abstain=True, kept_same=True, why=j.get("why"), raw=raw)); outcome = "abstain"; break
        spec, why = new, j.get("why", "")
    if p.poll() is None and not (h is not None and h.get("planned")):      # 로봇이 아직 새 목록을 기다리는 중 -> 포기를 알림(처음 목록 전이면 멈춤)
        if nfail == 0: stop_robot(p)
        else: json.dump(dict(abstain=True), open(f"{pf}.r{nfail}", "w"))
    try: p.wait(timeout=900)
    except subprocess.TimeoutExpired: stop_robot(p)
    _wl = [float(x) for x in re.findall(r"(?:자세 목록 파일 받음|\[replan\] 새 자세 목록 받음|\[replan\] 판단 = 포기) ?\(대기 ([0-9.]+)s\)", open(D + "/run.log").read())]
    tm["robot"] += time.time() - t_r - sum(_wl); tm["robot_wait_judge"] = sum(_wl)      # 로봇 시간 = 프로세스 시간 - 판단(Gemma) 기다린 시간(로봇 로그 기준)
    if h is not None and h not in hist: hist.append(h)
    t_h = time.time()
    if h is not None and h.get("planned"): time.sleep(4)
    s = EM.final_state(); EM.shot(D + "/before_home.png"); hh = safe_home(s, asset, sc); tm["harness"] += time.time() - t_h
    last = [x for x in hist if "pl" in x]
    if last: last[-1]["home"] = hh; last[-1]["state"] = s
    if h is not None and h.get("planned"):
        log = open(D + "/run.log").read(); mp = re.findall(r"놓기 자세 선택 (\{.*\})", log); h["chosen"] = json.loads(mp[-1]) if mp else None
        h["own"] = judge_state(s, spec, asset, dyaw); h["orig"] = judge_state(s, G.gt_for(iid, spec), asset, 0.0)
        outcome = "executed_ok" if (h["own"] or {}).get("ok") else ("executed_bad" if s and s["on_platform"] else "dropped_or_offplatform")
    return dict(scene=sid, item=iid, outcome=outcome, relaxed=sum(1 for x in hist if "pl" in x) > 1, final_spec=spec, hist=hist, sec=round(time.time() - t_ep, 1), stage_sec={k: round(v, 1) for k, v in tm.items()}, same_perc=True)


GP_RESTART = ("cd ~/nero/tools/phase_c_goalpose && tmux kill-session -t =curobo_gp19 2>/dev/null; sleep 3; tmux new-session -d -s curobo_gp19 "
              "'NERO_NB_FIRST=1 NERO_CUROBO_BUDGET_SCALE=3.0 ~/venvs/curobo_thor/bin/python curobo_c_server.py --port 8021 2>&1 | tee /tmp/curobo_gp19.log'; "
              "for i in $(seq 1 60); do curl -s -m3 http://127.0.0.1:8021/health >/dev/null && echo UP && break; sleep 5; done")


def gp_watch(limit_mb=int(os.environ.get("S2V_GP_MAX_MB", "4500"))):
    """2026-11-10: 계획 서버 복사본(8021)이 요청마다 GPU 메모리가 늘어(2.4 -> 6.2GB) Thor 메모리가 빠듯해지면 CGN 이 10배 느려짐(1.7 -> 16s, 측정).
    경우 사이에 8021 메모리를 보고 넘으면 새로 띄움(같은 코드·같은 설정, 결과에 영향 없음, 기록 남김)."""
    try:
        o = subprocess.run(["ssh", "thor", "p=$(ss -ltnp | grep ':8021 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | grep \"^$p,\""],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        mb = int(o.split(",")[1]) if o else -1
        if mb > limit_mb and os.environ.get("S2V_GP_RESTART", "0") == "1":      # 2026-11-11: Thor 에서 cuRobo 프로세스를 끄면(SIGINT 포함) GPU 메모리가 반환되지 않음(nvmap 누수, memcheck 로 측정) -> 기본은 기록만
            r = subprocess.run(["ssh", "thor", GP_RESTART], capture_output=True, text=True, timeout=400).stdout
            print(f"GP_RESTART 8021 {mb}MB > {limit_mb}MB -> {'UP' if 'UP' in r else 'FAIL'}", flush=True); return mb, "UP" in r
        return mb, None
    except Exception as e: print(f"GP_WATCH 실패 {e!r}", flush=True); return -1, None


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1] if len(sys.argv) > 1 else H + "/blocked_v4/exec_vlm_cases.json"))["cases"]
    fo_p = OUT + "/episodes.jsonl"; done = {(r["scene"], r["item"]) for r in map(json.loads, open(fo_p))} if os.path.exists(fo_p) else set()
    fo = open(fo_p, "a"); n_run = [len(os.listdir(OUT + "/runs"))]
    for c in cases:
        if (c["scene"], c["item"]) in done: continue
        gp_mb, gp_rs = gp_watch()
        ep = (episode_sp if SAME_PERC else episode)(c, n_run); ep["gp_mb_before"] = gp_mb; ep["gp_restarted"] = gp_rs; fo.write(json.dumps(ep, ensure_ascii=False) + "\n"); fo.flush()
        last = [x for x in ep["hist"] if "pl" in x]; lc = (last[-1].get("chosen") or {}) if last else {}
        print(f"{c['scene']} {c['item']}: {ep['outcome']}{' 풂' if ep['relaxed'] else ''} {lc.get('pose') or lc.get('name') or ''} {ep['sec']}s {ep['stage_sec']}", flush=True)
    S.clean_scene(); print("STAGE2_VLM_DONE", flush=True)
