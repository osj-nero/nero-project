#!/usr/bin/env python3
"""make_dest_spec.py <out.json> [--bin PRIM] -- 현재 Isaac 장면의 목적지 에셋(보관함 gso_3_*, 홀 컵 gso_4_*, 받침 ins_plate 등)의 실제 위치/크기/방향을 읽어 목적지 입력 명세(JSON)로 만든다(2026-10-14).
목적지를 인식하지 않고 입력으로 주는 실험용: 설비 도면에서 읽은 값에 해당한다. 월드 정점으로 최소 면적 사각형(cv2.minAreaRect)을 구해 (xy, size, yaw)를 낸다."""
import os, sys, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from isaac_exec import run_code
out = sys.argv[1]
Q = '''
import json, omni.usd
from pxr import UsdGeom, Usd
st = omni.usd.get_context().get_stage(); res = {}
for q in st.GetPrimAtPath("/World").GetChildren():
    n = q.GetName()
    if not (n == "ins_plate" or "Basket" in n or "Storage_Bin" in n or "Saucer" in n or "Pencil_Holder" in n): continue
    pts = []
    for p in Usd.PrimRange(q):
        if p.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(p); V = m.GetPointsAttr().Get()
            if V is None: continue
            M = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for v in V[::max(1, len(V) // 3000)]:
                w = M.Transform(v); pts.append([float(w[0]), float(w[1]), float(w[2])])
    res[n] = pts
json.dump(res, open("/tmp/dest_pts.json", "w"))
'''
run_code(Q, timeout=60); D = json.load(open("/tmp/dest_pts.json")); spec = []
for n, P in D.items():
    P = np.array(P)
    if len(P) < 50: continue
    # 충돌체(보이지 않는 col_k)는 제외하고 싶지만 정점 구분이 어려워, 상위 99% 범위로 이상점 영향을 줄인다
    xy = P[:, :2].astype(np.float32); (cx, cy), (w, d), ang = cv2.minAreaRect(xy); zlo, zhi = float(np.percentile(P[:, 2], 0.5)), float(P[:, 2].max())
    yaw = np.deg2rad(ang)
    if "Basket" in n or "Storage_Bin" in n:
        nb = sum(1 for e0 in spec if e0["kind"] == "container"); col = "red" if "Red" in n else ("black" if "Black" in n else "blue")
        kind, did, label = "container", ("bin" if nb == 0 else "bin%d" % (nb + 1)), (col + (" woven basket" if "Basket" in n else " plastic storage bin"))
    elif "Pencil_Holder" in n: kind, did, label = "hole", "cup", "black cup"
    else: kind, did, label = "flat", "tray", "ceramic saucer"
    e = dict(id=did, label=label, kind=kind, xy=[round(float(cx), 4), round(float(cy), 4)], size=[round(float(w), 4), round(float(d), 4), round(zhi - zlo, 4)], yaw=round(float(yaw), 4), prim=n)
    if kind == "container": e.update(floor_z=0.012, rim_z=round(zhi, 4), wall=0.01)
    if kind == "hole": e.update(E=[round(float(cx), 4), round(float(cy), 4), round(zhi, 4)], d=[0, 0, -1], open=round(min(w, d) - 0.02, 3), outer=round(min(w, d), 3), depth=round(zhi - zlo, 3))
    spec.append(e)
json.dump(spec, open(out, "w"), indent=1, ensure_ascii=False); print(json.dumps(spec, ensure_ascii=False, indent=1))
