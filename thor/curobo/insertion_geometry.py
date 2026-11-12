#!/usr/bin/env python3
"""insertion_geometry.py -- 긴 직육면체(막대)를 임의 각도의 홀더에 꽂는 6-DoF 그리퍼 자세 계산(numpy 만, 서버/시뮬 불필요).
핵심: 물체는 손에 단단히 붙어 있으므로(rigid in-hand) 집을 때의 손 자세 R0 와 물체 자세 R_obj 의 상대 관계가 고정이다.
  목표에서 물체가 R_obj' = R_delta · R_obj 이어야 하면 손은 R_hand' = R_delta · R0 이다.  (R_delta: 월드 회전)
  막대 끝(팁)이 홀 입구 E 의 축 d 위에 놓이게 하는 손(플랜지) 위치:  x_f' = X_tip' - R_delta · (x_tip - x_f)
R_delta = Rot(d, roll) · align(t0 -> d)   (t0: 집을 때 팁 방향(월드), d: 홀 축(삽입 방향), roll: 막대 축 둘레 자유 회전)
각도 표기: 홀 축의 '수직 기준 기울기' alpha(0 = 수직으로 아래로, 90 = 수평), 방위각 phi(월드 xy 에서 어느 쪽으로 기우나)."""
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def axis_from_angles(alpha_deg, phi_deg):
    """삽입 방향 d (바깥 -> 홀 안). alpha=0 이면 (0,0,-1) 수직 아래, alpha 가 커질수록 수평(phi 방향으로 전진)에 가까워진다."""
    a, p = math.radians(alpha_deg), math.radians(phi_deg)          # alpha 는 0~180: 90 초과면 위쪽으로 삽입
    return np.array([math.sin(a) * math.cos(p), math.sin(a) * math.sin(p), -math.cos(a)])


def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("삽입 방향 벡터가 0 이다")
    return v / n


def angles_from_axis(d):
    """3D 방향 d -> (alpha, phi) [도]. alpha = 수직 아래(0,0,-1) 기준 각(0=아래, 90=수평, 180=위), phi = 월드 xy 방위각."""
    d = unit(d)
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[2])))), math.degrees(math.atan2(d[1], d[0]))


def parse_insert_spec(spec):
    """'ex,ey,ez,dx,dy,dz'(방향 벡터, 자동 정규화) 또는 'ex,ey,ez,alpha_deg,phi_deg'(각도 표기) -> (E[3], d[3] 단위).
    d 는 홀 바깥 -> 안쪽 삽입 방향이며 임의의 3차원 벡터다(아래, 수평, 위 모두 가능)."""
    v = [float(x) for x in str(spec).split(",")]
    if len(v) == 6:
        return np.array(v[:3]), unit(v[3:6])
    if len(v) == 5:
        return np.array(v[:3]), axis_from_angles(v[3], v[4])
    raise ValueError("삽입 지정은 ex,ey,ez,dx,dy,dz 또는 ex,ey,ez,alpha,phi")


def peg_axis(dims, yaw):
    """누워 있는 직육면체의 장축 단위벡터(월드, 수평)와 (길이, 단면 두 변). dims=[w,d,h] OBB(월드 xy 기준, yaw 회전)."""
    w, d, h = dims
    if max(w, d) >= h:                                   # 누워 있음: 장축은 수평(OBB 긴 변)
        ax = np.array([math.cos(yaw), math.sin(yaw), 0.0]) if w >= d else np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return ax, max(w, d), (min(w, d), h)
    return np.array([0.0, 0.0, 1.0]), h, (w, d)          # 서 있음: 장축은 수직


def peg_frame(peg):
    """막대 장축/길이/단면. 점군 PCA 결과(peg["axis"], peg["length"], peg["cross"])가 있으면 그것을 쓰고(자세 일반), 없으면 OBB(dims, yaw)로 눕/섬 판정(구 방식)."""
    if peg.get("axis") is not None:
        return unit(peg["axis"]), float(peg["length"]), tuple(peg.get("cross", (0.03, 0.03)))
    return peg_axis(peg["dims"], peg["yaw"])


def align_rotation(a, b):
    """a 방향을 b 방향으로 보내는 최소 회전(Rot)."""
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-9:
        if c > 0:
            return Rot.identity()
        perp = np.cross(a, [1, 0, 0]); perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else np.array([0, 1, 0])
        return Rot.from_rotvec(math.pi * perp)
    return Rot.from_rotvec(v / s * math.atan2(s, c))


def insertion_waypoints(grasp, peg, E, d, tip_sign=+1, roll_deg=0.0, pre=0.08, depth=0.08, step=0.02):
    """집은 자세에서 막대 팁을 홀 입구 E(월드)에서 축 d 방향으로 넣는 손(플랜지) 자세 시퀀스.
    grasp: dict(flange[3], quat[xyzw])  peg: dict(center[3], dims[3], yaw)  반환: dict(R_delta, tip0, waypoints=[(flange_pos, quat_xyzw)], tip_path)"""
    d = np.asarray(d, float) / np.linalg.norm(d)
    ax, L, _ = peg_frame(peg); t0 = tip_sign * ax
    c0 = np.asarray(peg["center"], float); tip0 = c0 + t0 * (L / 2.0)
    R0 = Rot.from_quat(grasp["quat"]); xf0 = np.asarray(grasp["flange"], float)
    Rd = Rot.from_rotvec(d * math.radians(roll_deg)) * align_rotation(t0, d)
    n = int(round((pre + depth) / step)); wps = []; tips = []
    for k in range(n + 1):
        s = -pre + k * step                                        # 입구 기준 축 위 위치(음수=바깥, 양수=안쪽)
        X = np.asarray(E, float) + d * s
        xf = X - Rd.apply(tip0 - xf0)
        wps.append((xf, (Rd * R0).as_quat())); tips.append(X)
    return dict(R_delta=Rd, tip0=tip0, waypoints=wps, tip_path=np.array(tips), axis_hand=R0.inv().apply(t0))


def check(grasp, peg, E, d, **kw):
    """검산: 계산된 손 자세로 물체를 다시 배치했을 때 팁 위치/장축 방향이 목표와 맞는지(rigid 가정) -> (팁 오차 m, 축 각도 오차 deg)"""
    r = insertion_waypoints(grasp, peg, E, d, **kw); ax, L, _ = peg_frame(peg); t0 = kw.get("tip_sign", 1) * ax
    R0 = Rot.from_quat(grasp["quat"]); xf0 = np.asarray(grasp["flange"], float); tip_h = R0.inv().apply(r["tip0"] - xf0)            # 팁의 손 기준 좌표
    ax_h = R0.inv().apply(t0)
    worst_pos = 0.0; worst_ang = 0.0
    for (xf, q), X in zip(r["waypoints"], r["tip_path"]):
        R = Rot.from_quat(q); tip = xf + R.apply(tip_h); axis = R.apply(ax_h)
        worst_pos = max(worst_pos, float(np.linalg.norm(tip - X)))
        worst_ang = max(worst_ang, math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ (np.asarray(d, float) / np.linalg.norm(d))))))))
    return worst_pos, worst_ang


if __name__ == "__main__":
    # 시험: 누워 있는 3x3x20cm 막대를 윗면 중앙에서 위에서 집은 상태(손은 아래를 봄)
    grasp = dict(flange=[0.45, 0.0, 0.1358 + 0.03 + 0.015], quat=Rot.from_euler("xyz", [180, 0, 0], degrees=True).as_quat())
    peg = dict(center=[0.45, 0.0, 0.015], dims=[0.20, 0.03, 0.03], yaw=0.0)
    for alpha in (0, 30, 45, 60, 80, 90, 120, 150):
        for phi in (0, 90, 180):
            E = np.array([0.35, 0.25, 0.20]); d = axis_from_angles(alpha, phi)
            for sign in (+1, -1):
                ep, ea = check(grasp, peg, E, d, tip_sign=sign, roll_deg=20.0)
                assert ep < 1e-9 and ea < 1e-6, (alpha, phi, sign, ep, ea)
    print("검산 통과: 모든 (alpha, phi, 팁 방향)에서 팁 위치 오차 < 1e-9 m, 축 오차 < 1e-6 deg")
    r = insertion_waypoints(grasp, peg, E=[0.35, 0.25, 0.20], d=axis_from_angles(45, 90), tip_sign=+1)
    print("alpha=45, phi=90: 웨이포인트", len(r["waypoints"]), "개; 첫 손 위치", np.round(r["waypoints"][0][0], 3), " 마지막", np.round(r["waypoints"][-1][0], 3))
