#!/usr/bin/env python3
"""snap_pose.py <out.json> -- (시뮬 평가 하네스) 작업 영역 강체(로봇/테이블 제외)의 위치+자세 스냅샷 {prim: {"p":[x,y,z], "q":[w,x,y,z]}}.
place 정답 판정(안착 위치/세워짐)에 쓴다. snap_objects.py 와 같은 필터."""
import os, subprocess, sys, json
out = sys.argv[1]
code = '''
import omni.usd, json
from pxr import UsdGeom, UsdPhysics, Usd
st = omni.usd.get_context().get_stage()
res = dict()
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
for c in st.GetPrimAtPath("/World").GetChildren():
    nm = c.GetName()
    if nm in ("agx_arm", "ActionGraph", "Environment", "Materials") or (not nm.startswith("gso_") and ("Table" in nm or "Tape" in nm)) or nm.startswith("TestBox"):
        continue
    if not c.HasAPI(UsdPhysics.RigidBodyAPI):
        continue
    r = bbc.ComputeWorldBound(c).ComputeAlignedRange()
    ctr = (r.GetMin() + r.GetMax()) * 0.5
    q = UsdGeom.Xformable(c).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotationQuat()
    im = q.GetImaginary()
    res[nm] = {"p": [round(ctr[0], 4), round(ctr[1], 4), round(ctr[2], 4)], "q": [round(q.GetReal(), 5), round(im[0], 5), round(im[1], 5), round(im[2], 5)], "top": round(r.GetMax()[2], 4), "bot": round(r.GetMin()[2], 4)}
open("/tmp/_snap_pose_tmp.json", "w").write(json.dumps(res))
'''
try:
    os.remove("/tmp/_snap_pose_tmp.json")
except OSError:
    pass
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, "isaac_exec.py"), "--code", code, "--timeout", "15"], capture_output=True, text=True, timeout=30)
try:
    d = json.load(open("/tmp/_snap_pose_tmp.json")); json.dump(d, open(out, "w")); print(len(d))
except Exception:
    print(0)
