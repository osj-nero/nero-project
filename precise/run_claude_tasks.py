"""Claude(클라우드, 상한 기준)가 막대 구조 명세를 만들고 로봇이 실행하는 시험. 코드가 명세를 검사하고(겹침/허공/도달/순서) 위반 시 한 번씩 되돌려 다시 만들게 한다(최대 2회).
사용: python3 run_claude_tasks.py dolmen|jeong|jeong3|jeong6|triangle"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import precise_lib as L, json, time, os, sys, math, random, subprocess
import numpy as np
T=0.03; BL=0.20; CENTER=(-0.34,0.0)      # 2026-11-06: 井 위층 막대 중심이 반경 0.27 아래로 가던 문제로 구조 중심을 -0.34 로
TASKS={
 "dolmen":dict(n=3,cmd="고인돌 모양을 만들어라: 막대 2개를 서로 평행하게 놓고(두 막대 중심 간격 12cm), 그 두 막대 위에 나머지 막대 1개를 직각으로 가로질러 올린다(두 평행 막대의 한가운데)."),
 "jeong":dict(n=4,cmd="井(우물 정) 모양 2층 구조를 만들어라: 아래층에 막대 2개를 평행하게(중심 간격 12cm) 놓고, 그 위에 나머지 막대 2개를 직각으로 올려 정(井)자 모양이 되게 한다. 위층 막대는 아래층 막대의 끝에서 3cm 안쪽에 오게 한다."),
 "jeong3":dict(n=5,cmd="井(우물 정) 모양 2층 구조 위에 한 층을 더 쌓아 3층 구조를 만들어라: 아래층 막대 2개(평행, 중심 간격 12cm), 2층 막대 2개(아래층과 직각, 아래층 막대 끝에서 3cm 안쪽), 3층 막대 1개(아래층과 같은 방향, 구조 중심에 올림)."),
 "jeong6":dict(n=6,cmd="井(우물 정) 모양을 3층으로 쌓아라: 1층에 막대 2개를 평행하게(중심 간격 12cm) 놓고, 2층에 막대 2개를 1층과 직각으로 1층 막대 끝에서 3cm 안쪽에 올리고, 3층에 막대 2개를 1층과 같은 방향으로 2층 막대 끝에서 3cm 안쪽에 올린다. 각 층은 井자 모양이 되게 두 막대를 평행하게 둔다."),      # 2026-11-12: 막대 6개 3층 井자(공개 데모 영상용)
 "triangle":dict(n=3,flat=True,cmd="막대 3개로 한 변이 막대 중심선 기준 0.20m 정도인 정삼각형 테두리를 작업대 위에 만들어라. 막대끼리는 서로 겹치면 안 되고, 모서리에서 끝이 맞닿거나 약간(1cm 이내) 떨어져도 된다."),
}
task=sys.argv[1]; N=TASKS[task]["n"]; cmd=TASKS[task]["cmd"]
out=f"{NERO_ROOT}/precise/claude_{task}"; os.makedirs(out,exist_ok=True)
def spawn(poses):
    from isaac_exec import run_code
    items=[dict(name=f"gso_{i}_Bar_20x3x3",x=x,y=y,rz=a) for i,(x,y,a) in enumerate(poses)]
    code=f'''
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema
st=omni.usd.get_context().get_stage()
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("gso_"): st.RemovePrim(p.GetPath())
mp="/World/Looks/bar_blue"; m=UsdShade.Material(st.GetPrimAtPath(mp))
if not m:      # 2026-11-12: Isaac 을 새로 켜면 이 재질이 없어 Bind 에서 오류가 났다 -> 없으면 만든다
    m=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+"/s"); sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15,0.35,0.85)); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction")
for it in {json.dumps(items)}:
    c=UsdGeom.Cube.Define(st,"/World/"+it["name"]); c.CreateSizeAttr(1.0); xf=UsdGeom.Xformable(c); xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(it["x"],it["y"],{T}/2+0.002)); xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(float(it["rz"])); xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({BL},{T},{T}))
    pr=c.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.RigidBodyAPI.Apply(pr); UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.1)
    PhysxSchema.PhysxRigidBodyAPI.Apply(pr).CreateSolverPositionIterationCountAttr(16)
    UsdShade.MaterialBindingAPI.Apply(pr).Bind(m)
    if hm.IsValid(): UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(hm), UsdShade.Tokens.weakerThanDescendants, "physics")
'''
    return run_code(code,timeout=40)["status"]
def rect(x,y,yaw,L_=BL,W=T):
    c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw)); u=np.array([c,s]); v=np.array([-s,c]); p=np.array([x,y]); return [p+a*u*L_/2+b*v*W/2 for a,b in ((1,1),(1,-1),(-1,-1),(-1,1))]
def sat(r1,r2):
    for R in (r1,r2):
        for i in range(4):
            e=R[(i+1)%4]-R[i]; n=np.array([-e[1],e[0]]); n/=np.linalg.norm(n)
            a=[p@n for p in r1]; b=[p@n for p in r2]
            if max(a)<min(b)-1e-6 or max(b)<min(a)-1e-6: return False
    return True
def check(spec,ids):
    e=[]
    if TASKS[task].get("flat") and any(abs(s.get("z",0)-T/2)>0.003 for s in spec): e.append("평면 과제인데 작업대 위(z=0.015)가 아닌 막대가 있다: 모든 막대는 작업대에 직접 놓는다")
    if sorted(s.get("id") for s in spec)!=sorted(ids): return ["id 가 목록과 다르다(누락/중복/없는 id)"]
    for s in spec:
        r=math.hypot(s["x"],s["y"])
        if not(0.27<=r<=0.52): e.append(f"{s['id']} 중심 반경 {r:.2f}m 가 0.27~0.52 범위 밖")
        k=round(s["z"]/T-0.5)
        if abs(s["z"]-(k+0.5)*T)>0.003 or k<0: e.append(f"{s['id']} z={s['z']} 는 (층-0.5)*{T} 가 아니다")
    for i,a in enumerate(spec):
        for j,b in enumerate(spec):
            if j<=i: continue
            if abs(a["z"]-b["z"])<T-0.004 and sat(rect(a["x"],a["y"],a["yaw"]),rect(b["x"],b["y"],b["yaw"])): e.append(f"{a['id']} 와 {b['id']} 가 같은 높이에서 겹친다")
    for i,a in enumerate(spec):
        if a["z"]>T/2+0.003:
            sup=[j for j,b in enumerate(spec) if abs(b["z"]-(a["z"]-T))<0.004 and sat(rect(a["x"],a["y"],a["yaw"]),rect(b["x"],b["y"],b["yaw"]))]
            if not sup: e.append(f"{a['id']} 는 허공(아래층에 받치는 막대 없음)")
            elif any(j>i for j in sup) and not any(j<i for j in sup): e.append(f"{a['id']} 가 받침 막대보다 먼저 놓이도록 순서가 잘못됨")
    return e
def ask_claude(prompt):
    t0=time.time(); r=subprocess.run(["claude","-p","--tools","","--output-format","text"],input=prompt,capture_output=True,text=True,timeout=300,cwd="/tmp"); return r.stdout,round(time.time()-t0,1)
# ---- 장면 ----
rr=random.Random(7); ys=[0.20,-0.20,0.30,-0.30,0.40,-0.40][:N]
poses=[(-0.30,y,rr.uniform(-5,5)) for y in ys]      # 막대를 구조 양옆(y=±0.20~0.40)에 x 방향으로 눕혀 둔다
print("스폰",spawn(poses)); time.sleep(4)
g0=L.gt_poses(); obs=[dict(id=k,x=round(v["x"]+rr.gauss(0,0.01),3),y=round(v["y"]+rr.gauss(0,0.01),3),yaw=round((v["axis_x"]+rr.gauss(0,3))%180,1)) for k,v in sorted(g0.items())]
ids=[o["id"] for o in obs]; objs="\n".join(f"- {o['id']}: 중심 ({o['x']:.3f}, {o['y']:.3f}), 긴 축 {o['yaw']:.0f}도" for o in obs)
base=f"""로봇 작업대(높이 z=0이 작업대 윗면) 위에 같은 규격의 막대 {N}개가 눕혀 있다. 막대 한 개는 길이 0.20m, 폭과 두께가 모두 0.03m 이다. 위치는 막대 중심의 (x, y, z) 미터, yaw 는 막대 긴 축의 월드 각(도, +x 축 기준 반시계)이다. 작업대에 직접 놓인 막대의 중심 높이는 z = 0.015 이고 한 층 위는 0.045, 그 위는 0.075 이다.
현재 막대(로봇 베이스 좌표):
{objs}

명령: {cmd}
구조의 중심(xy)은 ({CENTER[0]}, {CENTER[1]}) 로 한다. 놓을 막대 중심은 로봇 베이스에서 반경 0.27~0.52m 안이어야 한다. 막대는 놓는 순서대로 적고(아래층 먼저), 모든 막대를 정확히 한 번씩 쓴다.
먼저 필요한 계산을 단계별로 간단히 쓰고, 마지막 줄에 JSON 하나만 쓴다: {{"steps":[{{"id":"<위 목록의 id 그대로>","x":<m>,"y":<m>,"z":<중심 높이 m>,"yaw":<긴 축 월드 각 도>}},...]}}"""
attempts=[]; prompt=base; spec=None
for k in range(3):
    txt,sec=ask_claude(prompt)
    try: spec=json.loads(txt[txt.rindex('{"steps"'):txt.rindex("}")+1])["steps"]
    except Exception as e: attempts.append(dict(k=k,sec=sec,err="파싱 실패")); spec=None; prompt=base+"\n\n이전 답이 JSON 형식이 아니었다. 마지막 줄에 JSON 하나만 쓴다."; continue
    errs=check(spec,ids); attempts.append(dict(k=k,sec=sec,errs=errs,spec=spec)); print(f"Claude 시도{k+1} {sec}s 위반 {errs or '없음'}",flush=True)
    if not errs: break
    prompt=base+"\n\n이전 답: "+json.dumps(spec,ensure_ascii=False)+"\n코드 검사에서 다음 위반이 나왔다: "+"; ".join(errs)+"\n위반을 고쳐서 다시 답하라(같은 형식)."
json.dump(dict(task=task,cmd=cmd,obs=obs,attempts=attempts),open(out+"/spec.json","w"),ensure_ascii=False,indent=1)
if not spec or check(spec,ids): print("검사 통과 명세 없음 -- 실행 안 함"); sys.exit(0)
print("명세",json.dumps(spec,ensure_ascii=False))
cur={o["id"]:(o["x"],o["y"]) for o in obs}; res=[]
for k,s in enumerate(spec):
    top_z=round(max(0.0,s["z"]-T/2),4); it=dict(id=s["id"],x=s["x"],y=s["y"],yaw=s["yaw"]%180)
    r=L.run_item(it,cur[s["id"]],f"{out}/step{k}_{s['id']}",top_z=top_z,extra_args=("--place-clear","0.015")); gg=L.gt_poses()[s["id"]]
    dy=abs(((gg["axis_x"]-s["yaw"]+90)%180)-90); r.update(id=s["id"],pos_mm=round(1000*math.hypot(gg["x"]-s["x"],gg["y"]-s["y"]),1),z_mm=round(1000*abs(gg["z"]-s["z"]),1),yaw_deg=round(dy,1),tilt=round(gg["tilt"],1)); res.append(r)
    print(f"{k} {s['id'][4]}번 목표({s['x']:.3f},{s['y']:.3f},z{s['z']:.3f},yaw{s['yaw']:.0f}) 판정 {r['verdict']} 위치 {r['pos_mm']}mm z {r['z_mm']}mm 방향 {r['yaw_deg']}° 기울기 {r['tilt']}° C {r['c_sec']}s",flush=True); time.sleep(2)
final=L.gt_poses(); ok=all(r["pos_mm"]<=15 and r["z_mm"]<=8 and r["yaw_deg"]<=6 and r["tilt"]<=5 for r in res)
print("구조 완성(모든 막대 위치15mm·높이8mm·방향6°·기울기5° 이내):",ok)
json.dump(dict(task=task,spec=spec,steps=res,final={k:dict(x=v["x"],y=v["y"],z=v["z"],axis_x=v["axis_x"],tilt=v["tilt"]) for k,v in final.items()},ok=ok),open(out+"/result.json","w"),ensure_ascii=False,indent=1); print("끝")
