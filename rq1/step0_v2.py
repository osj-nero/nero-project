"""RQ1 0단계 v2 (계획만, 로봇 실행 없음): 30cm 작업대(-0.46,0) + 물체 1개(-0.30,0.30, 작업대에서 15cm 이상) x 시작 yaw 0/60/120.
물체마다 rt_run.sh --no-exec 로 C 요청(덤프)을 저장한다. 2026-11-07: v1 판 장면은 작업대와 3cm 라 무효(PLAN_TRANSIT.md).
사용: python3 step0_v2.py [ONLY=name,...]  결과: step0_v2/<obj>_y<yaw>/dump/*.json"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import os, sys, json, subprocess, time
sys.path.insert(0, "" + NERO_ROOT + "/phase_c_L1")
from isaac_exec import run_code
R = "" + NERO_ROOT + "/rq1"; PC = "" + NERO_ROOT + "/phase_c_L1"
OX, OY = -0.30, 0.30
OBJS = [  # (이름, 종류, 크기 m (x,y,z) 또는 GSO 이름)
    ("plate", "cube", (0.12, 0.015, 0.10)),
    ("box", "cube", (0.07, 0.10, 0.15)),
    ("bar", "cube", (0.20, 0.03, 0.03)),
    ("block", "cube", (0.05, 0.05, 0.10)),      # t_wood_block 과 같은 크기
    ("eggbox", "cube", (0.15, 0.10, 0.07)),     # t_egg_box 와 같은 크기(10x15x7, 낮게 놓임)
    ("mug", "gso", "ACE_Coffee_Mug_Kristen_16_oz_cup"),
    ("bottle", "gso", "Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle"),
]
MUG = [o[2] for o in OBJS if o[0] == "mug"][0]
YAWS = [int(y) for y in os.environ.get('YAWS','0,60,120').split(',')]
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None

PLATFORM = '''
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
st=omni.usd.get_context().get_stage()
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("Platform_"): st.RemovePrim(p.GetPath())
mp="/World/Looks/plat_gray"
if not st.GetPrimAtPath(mp).IsValid():
    m=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+"/s"); sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6,0.6,0.62)); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
m=UsdShade.Material(st.GetPrimAtPath(mp)); hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction")
c=UsdGeom.Cube.Define(st,"/World/Platform_30cm"); c.CreateSizeAttr(1.0); xf=UsdGeom.Xformable(c); xf.ClearXformOpOrder()
xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(-0.46,0.0,0.15)); xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.24,0.24,0.30))
pr=c.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr); UsdShade.MaterialBindingAPI.Apply(pr).Bind(m)
if hm.IsValid(): UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(hm), UsdShade.Tokens.weakerThanDescendants, "physics")
'''

CUBE = '''
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema
st=omni.usd.get_context().get_stage()
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("gso_"): st.RemovePrim(p.GetPath())
mp="/World/Looks/obj_blue"
if not st.GetPrimAtPath(mp).IsValid():
    m=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+"/s"); sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15,0.35,0.85)); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
m=UsdShade.Material(st.GetPrimAtPath(mp)); hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction")
c=UsdGeom.Cube.Define(st,"/World/gso_0_{name}"); c.CreateSizeAttr(1.0); xf=UsdGeom.Xformable(c); xf.ClearXformOpOrder()
xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({x},{y},{z})); xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set({yaw}); xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({sx},{sy},{sz}))
pr=c.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.RigidBodyAPI.Apply(pr); UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.1)
PhysxSchema.PhysxRigidBodyAPI.Apply(pr).CreateSolverPositionIterationCountAttr(16)
UsdShade.MaterialBindingAPI.Apply(pr).Bind(m)
if hm.IsValid(): UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(hm), UsdShade.Tokens.weakerThanDescendants, "physics")
'''

ENV = dict(os.environ, ROS_LOCALHOST_ONLY="1", NERO_RT_IKURL="http://127.0.0.1:8020", NERO_RT_CUROBO="http://127.0.0.1:8021", NERO_RT_THOR="http://" + os.environ.get("NERO_THOR_IP", "127.0.0.1") + ":8012",
           NERO_PLACE_MODE="same", NERO_C_MAX_CANDS="40", NERO_RT_C_BUDGET="200", NERO_DEST_MASS="2.0", NERO_RT_PLACE_BUDGET="16.0", NERO_PLACE_AWARE="1")
for k in ("NERO_PLACE_TOPFIRST", "NERO_PLACE_TILT_FREE", "NERO_PLACE_YAWS", "NERO_PLACE_YAW0", "NERO_PLACE_TOPZ", "NERO_PLACE_WORLD_YAW", "NERO_PLACE_OBJ_ROT", "NERO_LOG_TRIAL"):
    ENV.pop(k, None)
VOC = subprocess.run(["python3", "-c", "import sort_tools as ST;print(','.join(ST.VOCAB))"], cwd=PC, capture_output=True, text=True).stdout.strip()
DESTS = f"{R}/dests_platform30.json"
json.dump([dict(id="platform", label="gray platform", kind="flat", xy=[-0.46, 0.0], size=[0.24, 0.24, 0.30], yaw=0.0)], open(DESTS, "w"))

# 원래 씬 물체(컵/병/책/TestBox)를 치우기 위해 GSO 배치를 한 번 거친다
subprocess.run(["python3", "gso_scene.py", "--placements", json.dumps([dict(name=MUG, x=OX, y=OY, yaw=0)]), "--objs", MUG, "--pose", "natural"], cwd=PC, capture_output=True)
print("작업대", run_code(PLATFORM, timeout=40)["status"], flush=True)
for nm, kind, spec in OBJS:
    if ONLY and nm not in ONLY: continue
    for yaw in YAWS:
        D = f"{R}/step0_v2/{nm}_y{yaw}"; os.makedirs(D + "/dump", exist_ok=True)
        if kind == "cube":
            sx, sy, sz = spec
            st = run_code(CUBE.format(name=nm, x=OX, y=OY, z=sz / 2 + 0.002, yaw=float(yaw), sx=sx, sy=sy, sz=sz), timeout=40)["status"]
        else:
            p = subprocess.run(["python3", "gso_scene.py", "--placements", json.dumps([dict(name=spec, x=OX, y=OY, yaw=yaw)]), "--objs", spec, "--pose", "natural"], cwd=PC, capture_output=True, text=True)
            st = "ok" if p.returncode == 0 else p.stderr[-200:]
        time.sleep(5)
        e = dict(ENV, NERO_C_REQ_DUMP=D + "/dump"); t0 = time.time()
        with open(D + "/run.log", "w") as lf:
            subprocess.run(["timeout", "400", "./rt_run.sh", "--no-exec", f"--target-at={OX},{OY}", "--dests", DESTS, "--dest-id", "platform", "--dest-kind", "flat",
                            "--vocab", VOC, "--max-picks", "1", "--place-clear", "0.015", "--out", D + "/run.json"], cwd=PC, env=e, stdout=lf, stderr=subprocess.STDOUT)
        log = open(D + "/run.log").read(); nd = len(os.listdir(D + "/dump"))
        ok = "pick 과 place 모두 계획됨" in log
        import re
        nc = re.findall(r"후보 (\d+)개", log)
        print(f"{nm} yaw{yaw}: 생성 {st} | 덤프 {nd}개 | CGN 후보 {nc[:2]} | 직립 높은작업대 계획 {'성공' if ok else '실패/없음'} | {time.time() - t0:.0f}s", flush=True)
run_code('''
import omni.usd
st=omni.usd.get_context().get_stage()
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("gso_") or p.GetName().startswith("Platform_"): st.RemovePrim(p.GetPath())
''', timeout=40)
print("끝 (작업대·물체 정리함)", flush=True)
