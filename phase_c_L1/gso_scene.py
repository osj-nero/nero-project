#!/usr/bin/env python3
"""gso_scene.py -- 비COCO GSO 물체(텍스처 USD)를 Isaac 씬에 배치한다(일반 물체 검출/파지 시험용).
  python3 gso_scene.py --seed 1 --objs NAME1,NAME2,NAME3 [--clear]   # --clear: 기존 gso_* 프림만 지우고 종료
- 기존 컵/병/책은 3m 밖으로 치운다(시험 중에는 gso 물체만 씬에 있음).
- 배치 영역 규칙은 randomize_general_objects.py 와 같다(반경 0.32~0.42, 옆 방위각 제외, x>0&y>0 금지, 테이블에서 15cm).
- 각 물체: 안정 자세(parts.pkl 의 stable_T[0]) 로 세워 놓고, RigidBody + 질량 100g + BoxHighFriction 마찰 + solver 16.
산출: /tmp/gso_gt_<seed>.json (물체별 프림/위치/크기)"""
import os, sys, json, math, pickle, argparse
sys.path.insert(0, os.path.expanduser("~/venvs/lib"))
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = os.environ.get("NERO_GSO_DIR") or os.path.expanduser("~/datasets/gso")      # Google Scanned Objects (USD 변환본) 위치
DEFAULTS = ["/World/OV_CreateBook_02", "/World/SM_Mug_B1", "/World/WhitePackerBottle_A04_PR_NVD_01", "/World/TestBox", "/World/TestBox1", "/World/TestBox2"]   # 씬에 원래 있던 물체 전부(이름 없는 탐지는 다 보므로 치운다)
R_MIN, R_MAX = 0.32, 0.42
EXCL = [(65, 115), (-115, -65)]
TABLE_C, TABLE_HALF, TABLE_CLEAR = (-0.330, 0.341), (0.195, 0.170), 0.15

_TEMPLATE = '''
import json, math
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf, Sdf
st = omni.usd.get_context().get_stage()
ITEMS = {items}
CLEAR = {clear}
for p in list(st.GetPrimAtPath("/World").GetChildren()):
    if p.GetName().startswith("gso_"):
        st.RemovePrim(p.GetPath())
if not CLEAR:
    for path in {defaults}:
        p = st.GetPrimAtPath(path)
        if p.IsValid():
            t = p.GetAttribute("xformOp:translate")
            t.Set(Gf.Vec3d(3.0 + 0.4 * {defaults}.index(path), 3.0, 0.3))
hm = st.GetPrimAtPath("/World/Materials/BoxHighFriction")
hmat = UsdShade.Material(hm) if hm.IsValid() else None
gt = {{}}
for it in ITEMS:
    path = "/World/" + it["prim"]
    p = st.DefinePrim(path, "Xform")
    p.GetReferences().AddReference(it["usd"])
    x = UsdGeom.Xformable(p); x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*it["pos"]))
    q = it["quat_wxyz"]
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(0.1)
    PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateSolverPositionIterationCountAttr(16)
    for c in Usd.PrimRange(p):
        if c.HasAPI(UsdPhysics.CollisionAPI) and hmat is not None:
            UsdShade.MaterialBindingAPI.Apply(c).Bind(hmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    gt[it["prim"]] = it["pos"]
json.dump(gt, open("/tmp/gso_gt_{seed}.json", "w"))
'''


NO_TABLE = False


def region_ok(x, y):
    r = math.hypot(x, y)
    if not (R_MIN <= r <= R_MAX + 0.03):
        return False
    th = math.degrees(math.atan2(y, x))
    if any(lo <= th <= hi for lo, hi in EXCL):
        return False
    if NO_TABLE:
        return True
    if x > 0 and y > 0:
        return False
    if abs(x - TABLE_C[0]) < TABLE_HALF[0] + TABLE_CLEAR and abs(y - TABLE_C[1]) < TABLE_HALF[1] + TABLE_CLEAR:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objs", default="")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--min-gap", type=float, default=0.12)
    ap.add_argument("--rmin", type=float, default=None); ap.add_argument("--rmax", type=float, default=None)
    ap.add_argument("--no-table", action="store_true", help="책상이 치워졌을 때: 책상 회피/사분면 금지 규칙을 끈다")
    ap.add_argument("--fixed", default="", help="x,y 로 첫 물체를 고정 위치에 놓는다(검출 시험용)")
    ap.add_argument("--placements", default="", help='JSON [{"name":..,"x":..,"y":..,"yaw":도}, ...] 로 위치/방향을 직접 지정(겹침 시험용)')
    ap.add_argument("--pose", default="natural", choices=["natural", "stable"], help="스폰 자세: natural=GSO 원본(메시) 자세 -- 서 있는 물체는 서 있다(기본), stable=parts.pkl 의 첫 안정 자세(가장 확률 높은 눕는 자세일 수 있다)")
    ap.add_argument("--yaw", type=float, default=None, help="도, 지정하면 무작위 요 대신 사용")
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    global R_MIN, R_MAX, NO_TABLE
    if a.rmin is not None: R_MIN = a.rmin
    if a.rmax is not None: R_MAX = a.rmax - 0.03       # region_ok 가 R_MAX+0.03 까지 허용하므로 사용자가 준 상한으로 맞춘다
    NO_TABLE = a.no_table
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from isaac_exec import run_code
    rng = np.random.default_rng(a.seed)
    items, placed = [], []
    names = [n for n in a.objs.split(",") if n] if not a.clear else []
    PL = json.loads(a.placements) if a.placements else None
    if PL: names = [d["name"] for d in PL]
    for k, nm in enumerate(names):
        rec = pickle.load(open(f"{ROOT}/models/{nm}/parts.pkl", "rb"))
        T = np.asarray(rec["stable_T"])[0]; Rs = T[:3, :3] if a.pose == "stable" else np.eye(3)      # 2026-10-01 기본 natural: 시계/판다/크레용은 stable_T[0] 이 눕는 자세였다
        pts = rec["pts"] @ Rs.T
        zoff = -float(pts[:, 2].min()) + 0.002
        rad = 0.5 * float(np.hypot(*(pts[:, :2].max(0) - pts[:, :2].min(0))))
        if PL:
            x, y = PL[k]["x"], PL[k]["y"]
            placed.append((x, y, rad))
            yaw = Rot.from_euler("z", math.radians(PL[k].get("yaw", 0.0))).as_matrix()
            q = Rot.from_matrix(yaw @ Rs).as_quat()
            items.append(dict(prim=f"gso_{k}_{nm[:24]}".replace("-", "_"), name=nm, usd=f"{ROOT}/models/{nm}/asset.usd", pos=[round(x, 4), round(y, 4), round(zoff, 4)],
                              quat_wxyz=[float(q[3]), float(q[0]), float(q[1]), float(q[2])], rad=rad, zoff=zoff))
            continue
        if a.fixed:
            x, y = [float(v) for v in a.fixed.split(",")]
            placed.append((x, y, rad))
            yaw = Rot.from_euler("z", math.radians(a.yaw) if a.yaw is not None else rng.uniform(0, 2 * math.pi)).as_matrix()
            q = Rot.from_matrix(yaw @ Rs).as_quat()
            items.append(dict(prim=f"gso_{k}_{nm[:24]}".replace("-", "_"), name=nm, usd=f"{ROOT}/models/{nm}/asset.usd", pos=[round(x, 4), round(y, 4), round(zoff, 4)],
                              quat_wxyz=[float(q[3]), float(q[0]), float(q[1]), float(q[2])]))
            continue
        for _ in range(500):
            r = rng.uniform(R_MIN, R_MAX); th = rng.uniform(-math.pi, math.pi)
            x, y = r * math.cos(th), r * math.sin(th)
            if region_ok(x, y) and all(math.hypot(x - px, y - py) >= rad + pr + a.min_gap for px, py, pr in placed):
                break
        else:
            print(f"[gso_scene] {nm}: 배치 실패(영역 부족)", file=sys.stderr); continue
        placed.append((x, y, rad))
        yaw = Rot.from_euler("z", rng.uniform(0, 2 * math.pi)).as_matrix()
        q = Rot.from_matrix(yaw @ Rs).as_quat()            # xyzw
        items.append(dict(prim=f"gso_{k}_{nm[:24]}".replace("-", "_"), name=nm, usd=f"{ROOT}/models/{nm}/asset.usd", pos=[round(x, 4), round(y, 4), round(zoff, 4)],
                          quat_wxyz=[float(q[3]), float(q[0]), float(q[1]), float(q[2])]))
    code = _TEMPLATE.format(items=repr(items), clear=repr(a.clear), defaults=repr(DEFAULTS), seed=a.seed)
    resp = run_code(code, a.host, a.port, timeout=60.0)
    if resp.get("status") == "error":
        print("[gso_scene] 실행 오류:", resp.get("message"), file=sys.stderr); sys.exit(1)
    print("[gso_scene] 배치 완료:", {it["prim"]: it["pos"][:2] for it in items})
    json.dump(items, open(f"/tmp/gso_items_{a.seed}.json", "w"))


if __name__ == "__main__":
    main()
