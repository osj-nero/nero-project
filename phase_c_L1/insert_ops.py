#!/usr/bin/env python3
"""insert_ops.py -- 긴 직육면체 삽입 과제: 시뮬 소품(막대+기울어진 사각 홀더) 생성 코드와 정답 판정. (Isaac 소켓으로 실행되는 코드는 문자열 템플릿)
좌표: 홀더 입구 중심 E(월드), 삽입 방향 d(바깥->안, 단위벡터). 홀더 로컬 z 축 = -d(홀 밖을 향함).
홀더 = 정적 충돌체 5개(벽 4 + 바닥) + 받침대(시각용 기둥). 막대 = 동역학 직육면체(기본 3x3x20cm, 질량 0.1kg, 마찰 높음).
단위 m. 이름: /World/ins_holder, /World/ins_peg  (gso_ 접두가 아니라 gso_scene 의 청소 대상이 아니다. clear_code() 로 지운다.)"""
import json, math
import numpy as np
from scipy.spatial.transform import Rotation as Rot

PEG_DIMS = (0.03, 0.03, 0.20)          # 로컬 z 가 장축
HOLE_INNER = 0.060                      # 홀 안쪽 한 변(m): 막대 3cm + 양쪽 여유 15mm
WALL = 0.010; TUBE_LEN = 0.14

_TEMPLATE = r"""
import json, math
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf, Sdf
st = omni.usd.get_context().get_stage()
P = {params}
for nm in ("ins_holder", "ins_peg"):
    pr = st.GetPrimAtPath("/World/" + nm)
    if pr.IsValid():
        st.RemovePrim(pr.GetPath())
def mat(path, rgb):
    m = UsdShade.Material.Define(st, path); sh = UsdShade.Shader.Define(st, path + "/s"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb)); m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return m
def box(path, size, pos, quat=None, color=None, dynamic=False):
    c = UsdGeom.Cube.Define(st, path); c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c); x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    if quat is not None: x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(quat[0], Gf.Vec3d(quat[1], quat[2], quat[3])))
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size))
    pr = c.GetPrim(); UsdPhysics.CollisionAPI.Apply(pr)
    if color is not None: UsdShade.MaterialBindingAPI.Apply(pr).Bind(color)
    return pr
orange = mat("/World/Looks/ins_orange", (0.95, 0.5, 0.1)); blue = mat("/World/Looks/ins_blue", (0.15, 0.4, 0.9))
fric = st.GetPrimAtPath("/World/Materials/BoxHighFriction")
MU = float(P.get("mu", 0.4))          # 막대/홀더 마찰. 0 이하면 기존 방식(막대에 BoxHighFriction 1.5, combine max)
def phys_mat(path, mu):
    m = UsdShade.Material.Define(st, path); pr_ = m.GetPrim(); api = UsdPhysics.MaterialAPI.Apply(pr_)
    api.CreateStaticFrictionAttr(mu); api.CreateDynamicFrictionAttr(mu); api.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(pr_).CreateFrictionCombineModeAttr("average")
    return m
real = phys_mat("/World/Materials/ins_real", MU) if MU > 0 else None
# 홀더: 부모 Xform(입구 중심 E, 로컬 z = u = -d)
hold = UsdGeom.Xform.Define(st, "/World/ins_holder"); hx = UsdGeom.Xformable(hold); hx.ClearXformOpOrder()
hx.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*P["E"]))
q = P["holder_quat"]; hx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))
h = P["inner"] / 2.0; w = P["wall"]; Lh = P["len"]
def sub(name, size, pos):
    pr_ = box("/World/ins_holder/" + name, size, pos, color=orange)
    if real is not None: UsdShade.MaterialBindingAPI.Apply(pr_).Bind(real, UsdShade.Tokens.weakerThanDescendants, "physics")
    return pr_
sub("wall_px", (w, P["inner"] + 2 * w, Lh), (h + w / 2, 0, -Lh / 2)); sub("wall_nx", (w, P["inner"] + 2 * w, Lh), (-h - w / 2, 0, -Lh / 2))
sub("wall_py", (P["inner"], w, Lh), (0, h + w / 2, -Lh / 2)); sub("wall_ny", (P["inner"], w, Lh), (0, -h - w / 2, -Lh / 2))
sub("bottom", (P["inner"] + 2 * w, P["inner"] + 2 * w, w), (0, 0, -Lh - w / 2))
# 받침대(시각용 기둥): 홀더 중간 높이까지 월드 수직 기둥. 삽입 경로(입구 쪽 연장)와 겹치지 않게 홀더 옆(축에 수직인 수평 방향)으로 비켜 세운다.
P_mid = [P["mid"][0] + P["pole_off"][0], P["mid"][1] + P["pole_off"][1]]
pole_h = P["mid"][2]
if pole_h > 0.03:
    box("/World/ins_pole", (0.04, 0.04, pole_h), (P_mid[0], P_mid[1], pole_h / 2.0), color=orange)
# 막대
pg = UsdGeom.Xform.Define(st, "/World/ins_peg"); px = UsdGeom.Xformable(pg); px.ClearXformOpOrder()
px.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*P["peg_pos"]))
pq = P["peg_quat"]; px.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(pq[0], Gf.Vec3d(pq[1], pq[2], pq[3])))
body = box("/World/ins_peg/body", P["peg_dims"], (0, 0, 0), color=blue)
pp = pg.GetPrim(); UsdPhysics.RigidBodyAPI.Apply(pp); UsdPhysics.MassAPI.Apply(pp).CreateMassAttr(0.1)
PhysxSchema.PhysxRigidBodyAPI.Apply(pp).CreateSolverPositionIterationCountAttr(16)
# 2026-10-03: 막대에 고마찰(1.5, combine max)을 주면 막대-벽 마찰이 1.5(마찰각 56도)가 돼 수직에서 34도 이상 기운 통에서 자기 잠금이 걸린다(기울기 30~45도 실패의 유력한 원인).
# 파지 마찰은 그리퍼 충돌체(GripperHighFriction 1.5, max)가 이미 담당하므로 막대는 현실적 마찰(펜-플라스틱 0.3~0.5)로 둔다.
if real is not None: UsdShade.MaterialBindingAPI.Apply(body).Bind(real, UsdShade.Tokens.weakerThanDescendants, "physics")
elif fric.IsValid(): UsdShade.MaterialBindingAPI.Apply(body).Bind(UsdShade.Material(fric), UsdShade.Tokens.weakerThanDescendants, "physics")
"""

CLEAR = r"""
import omni.usd
st = omni.usd.get_context().get_stage()
for nm in ("ins_holder", "ins_peg", "ins_pole"):
    pr = st.GetPrimAtPath("/World/" + nm)
    if pr.IsValid(): st.RemovePrim(pr.GetPath())
"""

def holder_quat(d):
    """로컬 z(홀 밖을 향함) = -d 가 되는 회전(xyzw 아님: wxyz 로 반환)"""
    u = -np.asarray(d, float) / np.linalg.norm(d); z = np.array([0, 0, 1.0])
    v = np.cross(z, u); s = np.linalg.norm(v); c = float(z @ u)
    R = Rot.identity() if s < 1e-9 and c > 0 else (Rot.from_rotvec([math.pi, 0, 0]) if s < 1e-9 else Rot.from_rotvec(v / s * math.atan2(s, c)))
    x, y, zq, w = R.as_quat(); return [float(w), float(x), float(y), float(zq)]

def build_params(E, d, peg_xy, peg_yaw_deg, dims=PEG_DIMS, inner=HOLE_INNER, wall=WALL, length=TUBE_LEN, stand=False):
    d = np.asarray(d, float) / np.linalg.norm(d); E = np.asarray(E, float)
    mid = E + d * (length / 2.0)                                           # 홀더 중간(월드)
    perp = np.cross(d, [0, 0, 1.0])
    perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else np.array([1.0, 0, 0])   # 수평이면서 축에 수직인 방향(수직 삽입이면 임의의 수평 방향)
    off = perp[:2] * (inner / 2 + wall + 0.035)
    R = Rot.from_euler("z", peg_yaw_deg, degrees=True) * (Rot.identity() if stand else Rot.from_euler("y", 90, degrees=True))   # 막대: 기본은 누워 있게(장축 수평), stand=True 면 똑바로 세움(장축 수직)
    x, y, z, w = R.as_quat()
    return dict(E=[float(v) for v in E], holder_quat=holder_quat(d), inner=inner, wall=wall, len=length, peg_dims=list(dims),
                peg_pos=[float(peg_xy[0]), float(peg_xy[1]), (dims[2] if stand else dims[0]) / 2 + 0.002], peg_quat=[float(w), float(x), float(y), float(z)],
                mid=[float(v) for v in mid], pole_off=[float(off[0]), float(off[1])], mu=float(__import__('os').environ.get('NERO_INS_MU', '0.4')))


def spawn(E, d, peg_xy, peg_yaw_deg, **kw):
    from isaac_exec import run_code
    P = build_params(E, d, peg_xy, peg_yaw_deg, **kw)
    return run_code(_TEMPLATE.format(params=repr(P)), timeout=60), P


def clear():
    from isaac_exec import run_code
    return run_code(CLEAR, timeout=30)

def judge_insert(snap_peg, E, d, dims=PEG_DIMS, depth_min=0.05, axis_tol_deg=12.0, lateral_tol=0.02, snap_prev=None):
    """포함 판정(2026-10-02, 사용자 제보로 교체): 막대 팁에서 입구 평면까지 막대 축이 홀 중심축에서 홀 안쪽 반폭(HOLE_INNER/2) 이내인 구간의 길이 = 들어간 길이.
    성공 = 들어간 길이 >= depth_min 이고(snap_prev 가 있으면) 정지해 있음(0.6초 간격 위치 변화 < 3mm, 회전 < 2도). 임계 숫자는 홀의 기하(반폭)와 요구 깊이뿐이다.
    기존 기준(중심 측면 오차 + 축 12도)은 *_old 로 남긴다. snap_peg/snap_prev: snap_pose 항목 {p:[x,y,z], q:[w,x,y,z]}."""
    d = np.asarray(d, float) / np.linalg.norm(d); E = np.asarray(E, float)
    w, x, y, z = snap_peg["q"]; R = Rot.from_quat([x, y, z, w]); ax = R.apply([0, 0, 1.0]); L = dims[2]
    c = np.asarray(snap_peg["p"], float)
    if float(ax @ d) < 0: ax = -ax
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ax @ d)))))
    tips = [c + ax * L / 2, c - ax * L / 2]; depth = [float((t - E) @ d) for t in tips]; deepest = max(depth); tp = tips[int(np.argmax(depth))]; tq = tips[int(np.argmin(depth))]
    lat = float(np.linalg.norm((c - E) - ((c - E) @ d) * d)); tip_lat = float(np.linalg.norm((tp - E) - ((tp - E) @ d) * d))
    old = bool(deepest >= depth_min and ang <= axis_tol_deg and lat <= lateral_tol + 0.0)
    gate = HOLE_INNER / 2.0; contained = 0.0
    for s_ in np.linspace(0.0, L, 81):                       # 깊은 팁에서 반대 끝 쪽으로 걸으며 홀 단면 안에 있는 구간을 센다
        p_ = tp + (tq - tp) * (s_ / L); dp = float((p_ - E) @ d)
        if dp < 0.0 or float(np.linalg.norm((p_ - E) - dp * d)) > gate: break
        contained = float(s_)
    settled = None
    if snap_prev is not None:
        wq = snap_prev["q"]; Rp = Rot.from_quat([wq[1], wq[2], wq[3], wq[0]])
        settled = bool(np.linalg.norm(c - np.asarray(snap_prev["p"], float)) < 0.003 and math.degrees((R.inv() * Rp).magnitude()) < 2.0)
    ok = bool(contained >= depth_min and (settled is None or settled))
    return dict(success=ok, contained_len_m=round(contained, 3), settled=settled, tip_depth_m=round(deepest, 3), axis_err_deg=round(ang, 1), lateral_m=round(lat, 3), tip_lat_m=round(tip_lat, 3),
                success_old=old, success_v2=bool(deepest >= depth_min and tip_lat <= gate))

if __name__ == "__main__":
    import sys
    # 오프라인 검산: 홀더 로컬 z 가 -d 로 가는지, 판정 함수 동작
    import insertion_geometry as G
    d = G.axis_from_angles(45, 90); wq = holder_quat(d); R = Rot.from_quat([wq[1], wq[2], wq[3], wq[0]]); zloc = R.apply([0, 0, 1.0])
    assert np.allclose(zloc, -d, atol=1e-6), (zloc, -d)
    E = np.array([0.35, 0.25, 0.20]); c = E + d * 0.07                 # 막대 중심이 홀 안 7cm
    q = Rot.align_vectors([d], [[0, 0, 1.0]])[0].as_quat(); sp = dict(p=list(c), q=[q[3], q[0], q[1], q[2]])
    print("판정(정상 삽입):", judge_insert(sp, E, d))
    sp2 = dict(p=list(E + np.array([0, 0, 0.12])), q=sp["q"]); print("판정(입구 밖):", judge_insert(sp2, E, d))
    print("검산 통과(홀더 축 방향)")
