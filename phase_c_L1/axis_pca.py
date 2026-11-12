#!/usr/bin/env python3
"""axis_pca.py -- 집기 전 물체 점군(base 좌표, 가려지지 않은 시점)의 PCA 로 장축 방향과 기울기를 구한다.
반환: axis(단위벡터, 부호 미정: z>=0 으로 통일), tilt_deg(수평면 기준 장축 기울기: 0=누움, 90=섬), elong(λ1/λ2, 클수록 막대형 -> 신뢰), n.
삽입 계산(insertion_geometry)에는 이 축을 t0 로 쓰고 tip_sign(+1/-1)은 후보로 스윕하므로 부호 모호성은 문제 없다."""
import math
import numpy as np

ELONG_MIN = 6.0      # axis_pca_threshold.py 합성 시험: 이 값 이상이면 축 오차 p90 6.5°, >10° 2.2% (장변:단변 약 3~4배 이상 막대형)


def pca_axis(P, trim=0.02):
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P) < 30:
        return dict(ok=False, why="점 부족")
    c = np.median(P, 0); d = np.linalg.norm(P - c, axis=1); P = P[d < np.percentile(d, 100 * (1 - trim))]
    C = np.cov((P - P.mean(0)).T); w, V = np.linalg.eigh(C); a = V[:, 2]
    if a[2] < 0:
        a = -a
    el = float(w[2] / max(w[1], 1e-12))
    if el < ELONG_MIN:
        return dict(ok=False, why=f"길쭉하지 않음(elong {el:.1f} < {ELONG_MIN}): 삽입 불가", elong=el)
    m = P.mean(0); t = (P - m) @ a; t0, t1 = float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))
    ext = lambda v: float(np.percentile((P - m) @ v, 99.5) - np.percentile((P - m) @ v, 0.5))
    return dict(ok=True, axis=a, tilt_deg=math.degrees(math.asin(min(1.0, abs(a[2])))), elong=float(w[2] / max(w[1], 1e-12)), n=len(P), center=m,
                length=t1 - t0, mid=m + a * (t0 + t1) / 2.0, cross=(ext(V[:, 1]), ext(V[:, 0])))


if __name__ == "__main__":      # 합성: 3x3x20 막대를 임의 기울기로 놓고 한쪽 시점에서 보이는 표면점만으로 PCA
    import sys; sys.path.insert(0, ".")
    from hole_estimator import _ray_obb
    from scipy.spatial.transform import Rotation as Rot
    rng = np.random.default_rng(0); L, W = 0.20, 0.03; rows = []
    for k in range(400):
        tilt = rng.choice([0, 15, 30, 45, 60, 75, 90]); yaw = rng.uniform(0, 360)
        ax = np.array([math.cos(math.radians(tilt)) * math.cos(math.radians(yaw)), math.cos(math.radians(tilt)) * math.sin(math.radians(yaw)), math.sin(math.radians(tilt))])
        R = Rot.align_vectors([ax], [[0, 0, 1.0]])[0] * Rot.from_rotvec([0, 0, rng.uniform(0, 6.28)]); Rm = R.as_matrix()
        ctr = np.array([rng.uniform(.3, .45), rng.uniform(-.15, .15), 0.10])
        cam = np.array([0.0, 0.0, rng.uniform(0.45, 0.7)]) + np.array([rng.uniform(.1, .35) * math.cos(math.radians(yaw)), rng.uniform(.1, .35) * math.sin(math.radians(yaw)), 0])
        fwd = ctr - cam; fwd /= np.linalg.norm(fwd); right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
        hw, hh = 160, 120; f = (hw / 2) / math.tan(math.radians(30)); ii, jj = np.meshgrid(np.arange(hw) - hw / 2, np.arange(hh) - hh / 2)
        dirs = (fwd + ii[..., None] / f * right - jj[..., None] / f * up).reshape(-1, 3); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        t = _ray_obb(cam, dirs, ctr, Rm, np.array([W / 2, W / 2, L / 2])); m = np.isfinite(t)
        if m.sum() < 30:
            continue
        P = cam + dirs[m] * t[m][:, None]; P = P + np.random.default_rng(k).normal(0, 0.002, P.shape)      # 깊이 노이즈 2mm
        e = pca_axis(P)
        err = math.degrees(math.acos(min(1, abs(float(e["axis"] @ ax))))); rows.append((tilt, err, e["tilt_deg"] - tilt, e["elong"], m.sum()))
    for tl in (0, 15, 30, 45, 60, 75, 90):
        X = [r for r in rows if r[0] == tl]
        if X:
            er = sorted(r[1] for r in X); print(f"기울기 {tl:2d}°: n={len(X):3d} 축 오차 중앙 {er[len(er)//2]:.1f}° p90 {er[int(.9*len(er))]:.1f}°  | 기울기 추정 편차 평균 {np.mean([r[2] for r in X]):+.1f}° | elong 중앙 {np.median([r[3] for r in X]):.1f}")
    er = sorted(r[1] for r in rows); print(f"전체 n={len(rows)} 축 오차 중앙 {er[len(er)//2]:.1f}° p90 {er[int(.9*len(er))]:.1f}° 최대 {er[-1]:.1f}°")
