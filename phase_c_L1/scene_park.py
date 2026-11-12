#!/usr/bin/env python3
"""scene_park.py -- 목적지 후보가 될 수 있는 물체(홀더/막대/접시/연필통)를 작업 영역 밖(3m 바깥)으로 치워 목적지를 바구니 하나만 남긴다(2026-10-14, 단일 목적지 pick&place 시험용). 삭제하지 않고 옮기기만 함."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from isaac_exec import run_code
PARK = {"ins_holder": (3.0, 3.6), "ins_pole": (3.5, 3.2), "ins_plate": (3.2, 4.0), "gso_4_Pencil_Holder_Black_v3": (3.8, 3.6)}
if os.environ.get("NERO_KEEP_HOLDER", "0") == "1": PARK.pop("gso_4_Pencil_Holder_Black_v3", None)      # 2026-10-14: 검은 컵(연필통)을 홀 목적지로 쓰는 시험에서는 치우지 않는다
code = "import omni.usd\nfrom pxr import Gf\nst = omni.usd.get_context().get_stage()\n"
for nm, (x, y) in PARK.items():
    code += f'p = st.GetPrimAtPath("/World/{nm}")\nif p.IsValid():\n    t = p.GetAttribute("xformOp:translate"); v = t.Get(); t.Set(Gf.Vec3d({x}, {y}, float(v[2]) + 0.02))\n    for a in ("physics:velocity", "physics:angularVelocity"):\n        at = p.GetAttribute(a)\n        if at: at.Set(Gf.Vec3f(0, 0, 0))\n'
DELETE = ["TestBox", "TestBox1", "TestBox2", "ins_peg"]      # 2026-10-14 사용자 지시: Isaac 이 임의로 스폰한 박스 3개와 길쭉한 파란 막대(ins_peg)는 삭제한다(multi_scene_real3 이 매번 다시 만들므로 장면 준비 때마다 지움)
code += "for nm in " + repr(DELETE) + ":\n    p = st.GetPrimAtPath('/World/' + nm)\n    if p.IsValid(): st.RemovePrim(p.GetPath())\n"
r = run_code(code, timeout=30); print("목적지 후보 치움(밖으로 이동):", r.get("status"), list(PARK), "삭제:", DELETE)
