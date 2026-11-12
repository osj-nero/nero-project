"""정밀 place task 실험 도구(2026-11-06): 장면 생성, 규칙/LLM 배치 명세 생성, 순차 실행, 채점. 명세 = [{"id","x","y","yaw"}] 순서대로, yaw 는 물체 긴 축의 월드 각(도, 0~180)."""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import json,math,os,subprocess,sys,time,random
import numpy as np
P="" + NERO_ROOT + "/phase_c_L1"; sys.path.insert(0,P)
DIMS=(0.07,0.10,0.15)
def spawn_boxes(poses):
    """poses: [(x,y,long_axis_deg)] -> Isaac 에 상자 N개(gso_k_Box)를 놓는다(모두 같은 7x10x15cm). 긴 축 각=박스 로컬 y 축 각 -> rotateZ = 긴축각-90."""
    from isaac_exec import run_code
    items=[dict(name=f"gso_{i}_Box_7x10x15",x=x,y=y,rz=la-90.0) for i,(x,y,la) in enumerate(poses)]
    code=f'''
import omni.usd, json
from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema
st=omni.usd.get_context().get_stage()
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("gso_"): st.RemovePrim(p.GetPath())
mp="/World/Looks/box_orange"; m=UsdShade.Material(st.GetPrimAtPath(mp))
if not m: 
    m=UsdShade.Material.Define(st,mp); sh=UsdShade.Shader.Define(st,mp+"/s"); sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9,0.45,0.1)); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface")
hm=st.GetPrimAtPath("/World/Materials/BoxHighFriction")
for it in {json.dumps(items)}:
    c=UsdGeom.Cube.Define(st,"/World/"+it["name"]); c.CreateSizeAttr(1.0); xf=UsdGeom.Xformable(c); xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(it["x"],it["y"],0.076)); xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(float(it["rz"])); xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({DIMS[0]},{DIMS[1]},{DIMS[2]}))
    pr=c.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr); UsdPhysics.RigidBodyAPI.Apply(pr); UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(0.1)
    PhysxSchema.PhysxRigidBodyAPI.Apply(pr).CreateSolverPositionIterationCountAttr(16)
    UsdShade.MaterialBindingAPI.Apply(pr).Bind(m)
    if hm.IsValid(): UsdShade.MaterialBindingAPI.Apply(pr).Bind(UsdShade.Material(hm), UsdShade.Tokens.weakerThanDescendants, "physics")
'''
    return run_code(code,timeout=40)["status"]
def gt_poses():
    from isaac_exec import run_code
    code=("import json, omni.usd\nfrom pxr import UsdGeom, Usd\nst=omni.usd.get_context().get_stage(); out={}\n"
"for p in st.GetPrimAtPath('/World').GetChildren():\n    n=p.GetName()\n    if n.startswith('gso_'):\n"
"        m=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()); t=m.ExtractTranslation(); r=m.ExtractRotationMatrix()\n"
"        out[n]=dict(p=[float(t[0]),float(t[1]),float(t[2])], z=[float(r[2][0]),float(r[2][1]),float(r[2][2])], x=[float(r[0][0]),float(r[0][1]),float(r[0][2])])\n"
"json.dump(out, open('/tmp/pp_poses.json','w'))\n")
    run_code(code,timeout=20); d=json.load(open("/tmp/pp_poses.json")); res={}
    for k,v in d.items():
        x=np.array(v["x"]);z=np.array(v["z"]);x/=np.linalg.norm(x);z/=np.linalg.norm(z);y=np.cross(z,x)
        res[k]=dict(x=v["p"][0],y=v["p"][1],z=v["p"][2],axis_x=math.degrees(math.atan2(x[1],x[0]))%180,long_axis=math.degrees(math.atan2(y[1],y[0]))%180,tilt=math.degrees(math.acos(max(-1,min(1,z[2])))))
    return res
def observed(poses,sigma_xy=0.01,sigma_deg=3.0,seed=0):
    """스캔 대용(시뮬레이터 정답 + 잡음; 한계로 명시): 인식이 주는 xy, 긴 축 각."""
    r=random.Random(seed); return [dict(id=k,x=round(v["x"]+r.gauss(0,sigma_xy),3),y=round(v["y"]+r.gauss(0,sigma_xy),3),yaw=round((v["long_axis"]+r.gauss(0,sigma_deg))%180,1)) for k,v in sorted(poses.items())]
def valid_spec(spec,ids):
    errs=[]
    if sorted(s.get("id") for s in spec)!=sorted(ids): errs.append("id 누락/중복")
    for s in spec:
        r=math.hypot(s["x"],s["y"])
        if not(0.27<=r<=0.52): errs.append(f"{s['id']} 반경 {r:.2f} 범위 밖")
    return errs
# ---- 규칙 기준선 ----
def rule_line(objs,x0=-0.27,gap=0.14,yaw=0.0):
    """상자 N개를 y 방향 일렬(x=x0), 간격 gap, 긴 축=yaw. 순서: 현재 y 가 작은 것부터."""
    n=len(objs); ys=[(i-(n-1)/2)*gap for i in range(n)]; objs=sorted(objs,key=lambda o:o["y"])
    return [dict(id=o["id"],x=x0,y=round(ys[i],3),yaw=yaw) for i,o in enumerate(objs)]
# ---- LLM 명세 ----
PROMPT="""너는 로봇 정리 플래너다. 로봇 베이스 좌표(m)에서 같은 크기(7x10x15cm 직육면체, 긴 쪽 10cm)의 상자 {n}개가 아래 위치에 놓여 있다. 긴 축 각은 +x 축 기준 반시계 도(0~180)이다.
{objs}

명령: "{cmd}"

제약: 놓을 위치는 로봇 베이스에서 반경 0.27~0.50m 안, 상자끼리 중심 간격 12cm 이상, 손이 이미 놓은 상자에 부딪히지 않게 먼 곳(로봇에서 먼 쪽)부터 놓는 순서로 적는다. 모든 상자를 정확히 한 번씩 쓴다.
출력은 JSON 하나만(설명 금지): {{"order":[{{"id":"gso_k_Box_7x10x15","x":<m>,"y":<m>,"yaw":<긴 축 월드 각 도>}},...]}}"""
def llm_spec(objs,cmd,max_tokens=500):
    txt="\n".join(f"- {o['id']}: 위치 ({o['x']:.3f}, {o['y']:.3f}), 긴 축 {o['yaw']:.0f}도" for o in objs)
    p=PROMPT.format(n=len(objs),objs=txt,cmd=cmd)
    req=dict(model="qwen3-vl",messages=[{"role":"user","content":p}],temperature=0,seed=0,max_tokens=max_tokens); t0=time.time()
    r=subprocess.run(["ssh","-o","ConnectTimeout=8","thor","curl -s -m 300 http://127.0.0.1:8005/v1/chat/completions -H 'Content-Type: application/json' -d @-"],input=json.dumps(req).encode(),capture_output=True,timeout=330)
    jj=json.loads(r.stdout.decode()); txt=jj["choices"][0]["message"]["content"]; secs=time.time()-t0
    try: spec=json.loads(txt[txt.index("{"):txt.rindex("}")+1])["order"]
    except Exception: spec=None
    return spec,dict(secs=round(secs,2),ptok=jj["usage"]["prompt_tokens"],ctok=jj["usage"]["completion_tokens"],raw=txt[:400])
# ---- 채점 ----
def score(final,spec,target_yaw=None):
    """final: gt_poses() 결과. 직선성(PCA 이탈 RMS), 간격(이웃 중심 거리) 평균/표준편차, 방향 오차(명세 yaw 대비, 180도 대칭), 기울기."""
    pts=np.array([[final[s["id"]]["x"],final[s["id"]]["y"]] for s in spec]); c=pts.mean(0); u,sv,vt=np.linalg.svd(pts-c); d=vt[0]; nrm=np.array([-d[1],d[0]])
    dev=(pts-c)@nrm; rms=float(np.sqrt((dev**2).mean())) if len(pts)>2 else 0.0
    order=np.argsort((pts-c)@d); sp=[float(np.linalg.norm(pts[order[i+1]]-pts[order[i]])) for i in range(len(pts)-1)]
    ye=[abs(((final[s["id"]]["long_axis"]-s["yaw"]+90)%180)-90) for s in spec]
    pe=[1000*math.hypot(final[s["id"]]["x"]-s["x"],final[s["id"]]["y"]-s["y"]) for s in spec]
    return dict(line_rms_mm=round(1000*rms,1),spacing_mean_mm=round(1000*float(np.mean(sp)),1) if sp else None,spacing_std_mm=round(1000*float(np.std(sp)),1) if sp else None,yaw_err_mean_deg=round(float(np.mean(ye)),2),yaw_err_max_deg=round(float(np.max(ye)),2),pos_err_mean_mm=round(float(np.mean(pe)),1),pos_err_max_mm=round(float(np.max(pe)),1),tilt_max_deg=round(max(final[s["id"]]["tilt"] for s in spec),1))
def spec_ideal_error(spec):
    """명세 자체의 정확도(실행과 분리): 점들의 직선 이탈, 간격 균등성."""
    pts=np.array([[s["x"],s["y"]] for s in spec]); c=pts.mean(0); u,sv,vt=np.linalg.svd(pts-c); d=vt[0]; nrm=np.array([-d[1],d[0]]); dev=(pts-c)@nrm
    order=np.argsort((pts-c)@d); sp=[float(np.linalg.norm(pts[order[i+1]]-pts[order[i]])) for i in range(len(pts)-1)]
    return dict(line_rms_mm=round(1000*float(np.sqrt((dev**2).mean())),1),spacing_std_mm=round(1000*float(np.std(sp)),1) if sp else None,min_spacing_mm=round(1000*min(sp),1) if sp else None)
# ---- 실행 ----
DESC_SIZE=[20,3,3]
def run_item(it,cur_xy,out,env_extra=None,top_z=0.0,extra_args=(),local=True):
    """local=True(기본): 겉모습(막대 크기)+힌트 위치로 국소 탐색(전체 사전 훑기 없음). 판정이 안 나오면 한 번 --target-at(전체 탐색)으로 재시도."""
    VOC=",".join(__import__("sort_tools").VOCAB)
    env=dict(os.environ,ROS_LOCALHOST_ONLY="1",NERO_RT_IKURL="http://127.0.0.1:8020",NERO_RT_THOR="http://" + os.environ.get("NERO_THOR_IP", "127.0.0.1") + ":8012",NERO_PLACE_MODE="same",NERO_C_MAX_CANDS="40",NERO_RT_C_BUDGET="120",NERO_DEST_MASS="2.0",NERO_RT_PLACE_BUDGET="16.0",NERO_PLACE_AWARE="1",NERO_PLACE_WORLD_YAW=str(it["yaw"]),NERO_PLACE_TOPZ=str(top_z),NERO_DESC_HINT_W="10",**(env_extra or {}))
    for k in ("NERO_PLACE_TOPFIRST","NERO_PLACE_TILT_FREE","NERO_PLACE_YAWS","NERO_PLACE_YAW0"): env.pop(k,None)
    import re
    def once(mode,o):
        os.makedirs(o,exist_ok=True); t0=time.time()
        sel=[f"--target-desc={json.dumps(dict(size_cm=DESC_SIZE))}",f"--hint-xy={cur_xy[0]},{cur_xy[1]}"] if mode=="local" else [f"--target-at={cur_xy[0]},{cur_xy[1]}"]
        try: subprocess.run(["./rt_run.sh",*sel,f"--place-xy={it['x']},{it['y']}","--vocab",VOC,"--max-picks","1","--out",o+"/run.json",*extra_args],cwd=P,env=env,stdout=open(o+"/run.log","w"),stderr=subprocess.STDOUT,timeout=420)
        except subprocess.TimeoutExpired: pass
        log=open(o+"/run.log").read(); mv=re.search(r"place verdict=(\w+)",log); mc=re.search(r"C 완료 ([\d.]+)s",log)
        return dict(verdict=(mv.group(1) if mv else None),c_sec=(float(mc.group(1)) if mc else None),wall_s=round(time.time()-t0),mode=mode)
    if local:
        r=once("local",out)
        if r["verdict"] is not None: return r
        r2=once("target_at",out+"_fallback"); r2["fallback_from_local"]=True; return r2
    return once("target_at",out)
# ---- 벽(붙여 놓기) 변형 ----
WALL_RULES="제약: 놓을 위치는 로봇 베이스에서 반경 0.27~0.50m 안. 상자는 서로 겹치면 안 되고(겹치면 실패), 명령대로 붙이되 오차를 고려해 인접 상자 사이 틈을 약 5mm로 둔다. 상자 크기는 폭 7cm, 길이 10cm이며 긴 축 각(yaw)에 따라 줄 방향으로 차지하는 길이가 달라진다(긴 축이 줄과 나란하면 10cm, 직각이면 7cm). 모든 상자를 정확히 한 번씩 쓴다. 이미 놓은 상자에 손이 부딪히지 않도록 놓는 순서를 정한다."
def llm_spec_rules(objs,cmd,rules,max_tokens=600):
    txt="\n".join(f"- {o['id']}: 위치 ({o['x']:.3f}, {o['y']:.3f}), 긴 축 {o['yaw']:.0f}도" for o in objs)
    p=f"""너는 로봇 정리 플래너다. 로봇 베이스 좌표(m)에 같은 크기(7x10x15cm 직육면체)의 상자 {len(objs)}개가 아래 위치에 놓여 있다. 긴 축 각은 +x 축 기준 반시계 도(0~180)이다. y 축 방향을 가로라고 부른다.
{txt}

명령: "{cmd}"

{rules}
출력은 JSON 하나만(설명 금지): {{"order":[{{"id":"gso_k_Box_7x10x15","x":<m>,"y":<m>,"yaw":<긴 축 월드 각 도>}},...]}}"""
    req=dict(model="qwen3-vl",messages=[{"role":"user","content":p}],temperature=0,seed=0,max_tokens=max_tokens); t0=time.time()
    r=subprocess.run(["ssh","-o","ConnectTimeout=8","thor","curl -s -m 300 http://127.0.0.1:8005/v1/chat/completions -H 'Content-Type: application/json' -d @-"],input=json.dumps(req).encode(),capture_output=True,timeout=330)
    jj=json.loads(r.stdout.decode()); t=jj["choices"][0]["message"]["content"]; secs=time.time()-t0
    try: spec=json.loads(t[t.index("{"):t.rindex("}")+1])["order"]
    except Exception: spec=None
    return spec,dict(secs=round(secs,2),ptok=jj["usage"]["prompt_tokens"],ctok=jj["usage"]["completion_tokens"],raw=t[:500],prompt=p)
def wall_gaps(items,key_xy=lambda s:(s["x"],s["y"]),key_yaw=lambda s:s["yaw"]):
    """줄 방향(PCA)을 따라 이웃 상자 사이 틈(mm, 음수=겹침). items: x,y,yaw(긴 축 각) 가진 dict 리스트."""
    pts=np.array([key_xy(s) for s in items]); c=pts.mean(0); u,sv,vt=np.linalg.svd(pts-c); d=vt[0]; ang=math.degrees(math.atan2(d[1],d[0]))
    proj=(pts-c)@d; o=np.argsort(proj); half=[]
    for s in items:
        dl=math.radians(key_yaw(s)-ang); half.append(0.05*abs(math.cos(dl))+0.035*abs(math.sin(dl)))
    return [round(1000*(proj[o[i+1]]-proj[o[i]]-half[o[i]]-half[o[i+1]]),1) for i in range(len(items)-1)]
def rule_wall(objs,x0=-0.27,yaw=90.0,gap=0.005):
    """규칙 벽: 긴 축을 줄(y)과 나란히(yaw=90), 중심 간격 = 10cm + gap."""
    n=len(objs); step=0.10+gap; ys=[(i-(n-1)/2)*step for i in range(n)]; objs=sorted(objs,key=lambda o:o["y"])
    return [dict(id=o["id"],x=x0,y=round(ys[i],3),yaw=yaw) for i,o in enumerate(objs)]
