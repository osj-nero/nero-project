"""VLM 승격(2026-11-10): Isaac 에 무늬 직육면체(면마다 UV·텍스처) 물체를 지금과 같은 모양·질량·마찰로 놓는다 + 로봇 카메라 크롭 수집(--ground-only).
spawn_tex(asset, yaw, texture): exec_main.spawn 과 같은 받침대/위치/크기, 이름 gso_0_<asset> (final_state·clean_scene 그대로 동작)."""
import os, sys, glob, json, time, shutil, subprocess, re
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import exec_main as EM
from isaac_exec import run_code
TEX = H + "/vlm_textures"
MESH = '''
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema, Vt
st=omni.usd.get_context().get_stage()
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("gso_"): st.RemovePrim(p.GetPath())
hx,hy,hz={hx},{hy},{hz}
F=[("top",(0,0,1),(1,0,0),(0,1,0)),("bottom",(0,0,-1),(1,0,0),(0,-1,0)),("front",(0,-1,0),(1,0,0),(0,0,1)),("back",(0,1,0),(-1,0,0),(0,0,1)),("left",(-1,0,0),(0,-1,0),(0,0,1)),("right",(1,0,0),(0,1,0),(0,0,1))]
h=(hx,hy,hz); pts=[]; st_=[]; cnt=[]; idx=[]
for fi,(nm,n,u,v) in enumerate(F):
    hu=sum(abs(u[i])*h[i] for i in range(3)); hv=sum(abs(v[i])*h[i] for i in range(3)); c=[n[i]*h[i] for i in range(3)]
    for su,sv in ((-1,-1),(1,-1),(1,1),(-1,1)):
        pts.append(Gf.Vec3f(*[c[i]+su*u[i]*hu+sv*v[i]*hv for i in range(3)])); st_.append(Gf.Vec2f((su+1)/2,(sv+1)/2))
    cnt.append(4); idx+= [4*fi+k for k in range(4)]
path="/World/gso_0_{name}"; m=UsdGeom.Mesh.Define(st,path); m.CreatePointsAttr(Vt.Vec3fArray(pts)); m.CreateFaceVertexCountsAttr(cnt); m.CreateFaceVertexIndicesAttr(idx); m.CreateSubdivisionSchemeAttr("none")
pv=UsdGeom.PrimvarsAPI(m).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying); pv.Set(Vt.Vec2fArray(st_))
xf=UsdGeom.Xformable(m); xf.ClearXformOpOrder(); xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({x},{y},{z})); xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set({yaw})
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
pr=m.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.MeshCollisionAPI.Apply(pr).CreateApproximationAttr("convexHull"); UsdPhysics.RigidBodyAPI.Apply(pr); UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.1)
PhysxSchema.PhysxRigidBodyAPI.Apply(pr).CreateSolverPositionIterationCountAttr(16)
hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction")
if hm.IsValid(): UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(hm), UsdShade.Tokens.weakerThanDescendants, "physics")
'''


def spawn_tex(asset, yaw, texture):
    run_code(EM.PLATFORM_CODE, timeout=40)
    sx, sy, sz = EM.DIMS[asset]
    return run_code(MESH.format(name=asset, x=EM.OX, y=EM.OY, z=sz / 2 + 0.002, yaw=float(yaw), hx=sx / 2, hy=sy / 2, hz=sz / 2, texture=texture, tex=TEX), timeout=60)["status"]


def crop(asset, yaw, texture, out_jpg, log=None):
    """무늬 물체를 놓고 시스템의 훑기(--ground-only: 팔이 관측 자세에서 j1 만 돌림)로 로봇 카메라 크롭을 받아, 대상 위치에 가장 가까운 첫 크롭을 저장."""
    import stage2_loop as S
    S.clean_scene(); spawn_tex(asset, yaw, texture); time.sleep(4)
    shutil.rmtree("/tmp/gcrops", ignore_errors=True)
    e = dict(EM.ENV, ROS_LOCALHOST_ONLY="1")
    r = subprocess.run(["timeout", "240", "./rt_run.sh", "--ground-only", f"--target-at={EM.OX},{EM.OY}", "--dests", EM.DESTS, "--vocab", EM.VOC, "--out", "/tmp/vlm_ground.json"],
                       cwd=EM.PC, env=e, capture_output=True, text=True)
    if log: open(log, "w").write(r.stdout[-20000:] + r.stderr[-5000:])
    best = None
    for f in glob.glob("/tmp/gcrops/*.jpg"):
        m = re.search(r"a([+-][0-9.]+)_\d+_x(-?[0-9.]+)_y(-?[0-9.]+)\.jpg", os.path.basename(f))
        if not m: continue
        dd = (float(m.group(2)) - EM.OX) ** 2 + (float(m.group(3)) - EM.OY) ** 2
        if dd < 0.05 ** 2 and (best is None or abs(float(m.group(1))) < best[0]): best = (abs(float(m.group(1))), f)
    if best: shutil.copy(best[1], out_jpg)
    return bool(best)


FACE_DEF = {"top": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "bottom": ((0, 0, -1), (1, 0, 0), (0, -1, 0)), "front": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
            "back": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), "left": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)), "right": ((1, 0, 0), (0, 1, 0), (0, 0, 1))}
KO = {"top": "윗면", "bottom": "바닥면", "front": "앞면", "back": "뒷면", "left": "왼면", "right": "오른면"}


def frames_crop(asset, yaw, frame_dir, out_raw, out_lab, pad=0.25, perc=None):
    """프레임 덤프(rgb, W_T_cam, K)에서 물체(알려진 위치·yaw 의 직육면체)가 온전히 보이고 가장 크게 보이는 프레임 -> 원본 크롭 + 보이는 면 이름표 크롭.
    이름표 위치·보이는지 여부는 기하(물체 자세 + 카메라 자세)로 계산 -- 시스템이 실제로 아는 정보."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation as Rot
    if perc:      # 2026-11-10: 인식 결과(perc_frame.frame: 중심·yaw·치수) -- 플래너 회전과 같은 좌표. 정답값을 쓰지 않음
        h = np.array(perc["dims"]) / 2; c = np.array(perc["centre"]); R = Rot.from_euler("z", perc["yaw"], degrees=True)
    else:
        sx, sy, sz = EM.DIMS[asset]; h = np.array([sx, sy, sz]) / 2; c = np.array([EM.OX, EM.OY, sz / 2]); R = Rot.from_euler("z", yaw, degrees=True)
    corners = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]) * h; Cw = R.apply(corners) + c
    best = None
    for f in sorted(glob.glob(frame_dir + "/*.npz")):
        z = np.load(f); T = z["W_T_cam"]; fx, fy, cx, cy = z["K"]; img = z["rgb"]; Hh, Ww = img.shape[:2]
        Tc = np.linalg.inv(T); P = (Tc[:3, :3] @ Cw.T).T + Tc[:3, 3]; P = P * np.array([1, -1, -1])      # W_T_cam 은 OpenGL 식(-z 앞, y 위) -> 광학(z 앞, y 아래)
        if (P[:, 2] <= 0.05).any(): continue
        u = fx * P[:, 0] / P[:, 2] + cx; v = fy * P[:, 1] / P[:, 2] + cy
        if u.min() < 5 or v.min() < 5 or u.max() > Ww - 5 or v.max() > Hh - 5: continue
        area = (u.max() - u.min()) * (v.max() - v.min())
        if best is None or area > best[0]: best = (area, f, u, v)
    if best is None: return False
    z = np.load(best[1]); T = z["W_T_cam"]; fx, fy, cx, cy = z["K"]; img = Image.fromarray(z["rgb"]); u, v = best[2], best[3]
    w_, h_ = u.max() - u.min(), v.max() - v.min(); box = (int(max(0, u.min() - pad * w_)), int(max(0, v.min() - pad * h_)), int(min(img.width, u.max() + pad * w_)), int(min(img.height, v.max() + pad * h_)))
    img.crop(box).save(out_raw, quality=92)
    lab = img.copy(); dr = ImageDraw.Draw(lab); fk = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", max(12, int(min(w_, h_) * 0.09)))
    cam = T[:3, 3]; Tc = np.linalg.inv(T)
    for nm, (n, uu, vv) in FACE_DEF.items():
        n = np.array(n, float); fc = R.apply(n * h) + c; nw = R.apply(n); vis = float((cam - fc) @ nw / np.linalg.norm(cam - fc))
        if vis < 0.15: continue      # 거의 옆으로 보이거나 안 보이는 면
        p = (Tc[:3, :3] @ fc + Tc[:3, 3]) * np.array([1, -1, -1]); x, y = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
        s = KO[nm]; bb = dr.textbbox((0, 0), s, font=fk); tw, th = bb[2], bb[3]
        dr.rectangle([x - tw / 2 - 2, y - th / 2 - 2, x + tw / 2 + 2, y + th / 2 + 2], fill=(255, 255, 255)); dr.text((x - tw / 2, y - th / 2), s, font=fk, fill=(0, 0, 0))
    lab.crop(box).save(out_lab, quality=92)
    return os.path.basename(best[1])


def capture(asset, yaw, texture, out_raw, out_lab, log=None, perc=None, keep_frames=None):
    """무늬 물체를 놓고 시스템 훑기(--ground-only)를 프레임 덤프와 함께 돌려 원본·이름표 크롭을 만든다."""
    import stage2_loop as S
    S.clean_scene(); spawn_tex(asset, yaw, texture); time.sleep(4)
    fd = "/tmp/vlm_frames"; shutil.rmtree(fd, ignore_errors=True); os.makedirs(fd)
    e = dict(EM.ENV, ROS_LOCALHOST_ONLY="1", NERO_FRAME_DUMP=fd)
    r = subprocess.run(["timeout", "240", "./rt_run.sh", "--ground-only", f"--target-at={EM.OX},{EM.OY}", "--dests", EM.DESTS, "--vocab", EM.VOC, "--out", "/tmp/vlm_ground.json"],
                       cwd=EM.PC, env=e, capture_output=True, text=True)
    if log: open(log, "w").write(r.stdout[-20000:] + r.stderr[-5000:])
    if keep_frames: shutil.rmtree(keep_frames, ignore_errors=True); shutil.copytree(fd, keep_frames)      # 나중에 이름표를 다시 그릴 수 있게 프레임(rgb·카메라 자세·K) 보관
    return frames_crop(asset, yaw, fd, out_raw, out_lab, perc=perc)
