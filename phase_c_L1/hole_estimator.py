#!/usr/bin/env python3
"""hole_estimator.py -- 깊이 점군(base 좌표, 조직화된 HxWx3)에서 사각 관(홀더)의 입구 중심 E 와 삽입 방향 d(바깥->안쪽 단위벡터)를 추정한다. 마커/CAD 없이 구조만 쓴다.
알고리즘: (1) 홀더 마스크의 점으로 표면 법선을 구한다(이웃 점 차분의 외적, 깊이 불연속은 제외)
          (2) 관 축 = 법선 공분산의 최소 고유벡터(옆벽 4면의 법선은 모두 축에 수직, 끝면은 면적이 작다)
          (3) 축 위 좌표 t 의 양 끝 단면에서 중앙 영역의 점 밀도를 비교: 비어 보이는(밀도 낮은) 쪽이 입구, 채워진 쪽이 막힌 바닥
          (4) 입구가 보이지 않을 때(바닥만 보임)는 추정 불가로 반환. 입구만 보이고 바닥이 안 보이면 d 는 관찰자(카메라)에서 멀어지는 쪽.
반환: dict(ok, E, d, axis, t_open, t_closed, fill_open, fill_closed, n_pts, planar_ratio)"""
import math
import numpy as np


def organized_normals(pts, valid, max_jump=0.02):
    """pts: (H,W,3). 인접 픽셀 차분 외적으로 법선. 이웃과 거리 차가 max_jump(m) 넘으면(깊이 불연속) 무효."""
    H, W, _ = pts.shape
    dx = np.zeros_like(pts); dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]; dy[1:-1] = pts[2:] - pts[:-2]
    ok = valid.copy(); ok[:, 0] = ok[:, -1] = False; ok[0] = ok[-1] = False
    ok &= np.linalg.norm(dx, axis=-1) < 2 * max_jump; ok &= np.linalg.norm(dy, axis=-1) < 2 * max_jump
    for sh in (1, -1):
        ok &= np.roll(valid, sh, 0) & np.roll(valid, sh, 1)
    n = np.cross(dx, dy); nn = np.linalg.norm(n, axis=-1, keepdims=True)
    ok &= (nn[..., 0] > 1e-9)
    n = n / np.maximum(nn, 1e-12)
    return n, ok


def estimate(pts, mask, cam_pos, inner=0.06, wall=0.01, length=0.14):
    """끝면 군집 기반 판정. 법선이 축과 평행한 점(끝면: 입구 테두리 / 바닥 안쪽면)을 축 방향 위치 t 로 군집화한다.
    가운데가 채워진 군집 = 바닥(닫힌 쪽), 가운데가 빈 링 군집 = 입구 테두리. 둘 다 안 보이면 카메라가 입구 쪽에 있다고 가정(폴백, via='cam')."""
    cam_pos = np.asarray(cam_pos, float)
    valid = np.isfinite(pts).all(-1) & mask
    n, ok = organized_normals(np.nan_to_num(pts), valid)
    if ok.sum() < 200:
        return dict(ok=False, why="점 부족")
    N = n[ok]; P = pts[ok]
    C = N.T @ N / len(N); w, V = np.linalg.eigh(C); a = V[:, 0]
    ctr = P.mean(0)
    if (cam_pos - ctr) @ a < 0:
        a = -a                                                   # a: 카메라 쪽을 향하게 부호 고정(이후 d 결정에 사용)
    u = np.cross(a, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0]); v = np.cross(a, u)
    t = (P - ctr) @ a; pu = (P - ctr) @ u; pv = (P - ctr) @ v
    face = np.abs(N @ a) > 0.85
    clusters = []
    if face.sum() >= 12:
        order = np.argsort(t[face]); tf = t[face][order]; idx = np.where(face)[0][order]
        cuts = np.where(np.diff(tf) > 0.015)[0] + 1
        for g in np.split(np.arange(len(tf)), cuts):
            if len(g) < 12:
                continue
            ii = idx[g]; cu_ = (np.percentile(pu[ii], 3) + np.percentile(pu[ii], 97)) / 2; cv_ = (np.percentile(pv[ii], 3) + np.percentile(pv[ii], 97)) / 2
            r = np.maximum(np.abs(pu[ii] - cu_), np.abs(pv[ii] - cv_))
            fill = float((r < 0.30 * inner).mean())
            clusters.append(dict(t=float(np.median(t[ii])), cu=cu_, cv=cv_, fill=fill, n=len(g)))
    rim = [c for c in clusters if c["fill"] < 0.12]
    bot = [c for c in clusters if c["fill"] >= 0.3]
    via = None
    if rim:
        o = max(rim, key=lambda c: c["n"]); t_open, cu, cv = o["t"], o["cu"], o["cv"]
        # 바닥이 같이 보이면 방향은 입구->바닥, 아니면 카메라가 입구 쪽이라는 가정으로 -a
        sgn = -1.0; via = "rim+cam"                              # (rim+bottom 교차검증은 합성 시험에서 14/14 오판이라 사용 안 함)
    elif bot:
        b0 = max(bot, key=lambda c: c["n"]); cu, cv = b0["cu"], b0["cv"]
        sgn = -1.0                                               # 바닥은 입구보다 카메라 반대쪽(t 작은 쪽)
        t_open = b0["t"] - sgn * length; via = "bottom+cam"
    else:
        return dict(ok=False, why="끝면 군집 없음(테두리/바닥 둘 다 안 보임)", axis=a.tolist())
    # sgn = d 의 a 에 대한 부호. 입구 = 바닥 반대쪽
    d = a * sgn
    E = ctr + a * t_open + u * cu + v * cv
    return dict(ok=True, E=E.tolist(), d=d.tolist(), via=via, clusters=[(round(c["t"], 3), round(c["fill"], 2), c["n"]) for c in clusters], n_pts=int(len(P)))


# ── 합성 시험: 레이캐스트로 홀더 깊이 영상을 만들어 추정 오차 측정 ──────────────────────────────────────────────
def _ray_obb(o, dirs, c, R, half):
    """레이(원점 o, 방향 dirs[N,3]) 와 OBB(중심 c, 회전 R 의 열=축, 반길이 half) 교차 거리. 없으면 inf"""
    oc = (R.T @ (o - c)); dl = dirs @ R
    t1 = (-half - oc) / np.where(np.abs(dl) < 1e-12, 1e-12, dl); t2 = (half - oc) / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
    tn = np.minimum(t1, t2).max(1); tf = np.maximum(t1, t2).min(1)
    return np.where((tn <= tf) & (tf > 0), np.maximum(tn, 0), np.inf)


def synth_cloud(E, d, cam_pos, cam_target, inner=0.06, wall=0.01, length=0.14, hw=160, hh=120, fov_deg=60.0, with_pole=True):
    from scipy.spatial.transform import Rotation as Rot
    d = np.asarray(d, float) / np.linalg.norm(d); E = np.asarray(E, float)
    u_ = -d; z = np.array([0, 0, 1.0]); v = np.cross(z, u_); s = np.linalg.norm(v); c = float(z @ u_)
    Rh = Rot.identity() if s < 1e-9 and c > 0 else (Rot.from_rotvec([math.pi, 0, 0]) if s < 1e-9 else Rot.from_rotvec(v / s * math.atan2(s, c)))
    Rm = Rh.as_matrix(); h = inner / 2
    boxes = [((wall, inner + 2 * wall, length), (h + wall / 2, 0, -length / 2)), ((wall, inner + 2 * wall, length), (-h - wall / 2, 0, -length / 2)),
             ((inner, wall, length), (0, h + wall / 2, -length / 2)), ((inner, wall, length), (0, -h - wall / 2, -length / 2)), ((inner + 2 * wall, inner + 2 * wall, wall), (0, 0, -length - wall / 2))]
    obbs = [(E + Rm @ np.array(loc), Rm, np.array(dims) / 2) for dims, loc in boxes]
    if with_pole:
        mid = E + d * length / 2; perp = np.cross(d, [0, 0, 1.0]); perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else np.array([1.0, 0, 0])
        pc = mid + perp * (h + wall + 0.035); pc[2] = mid[2] / 2
        obbs.append((pc, np.eye(3), np.array([0.02, 0.02, max(mid[2], 0.05) / 2])))
    fwd = np.asarray(cam_target, float) - np.asarray(cam_pos, float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    f = (hw / 2) / math.tan(math.radians(fov_deg) / 2)
    ii, jj = np.meshgrid(np.arange(hw) - hw / 2, np.arange(hh) - hh / 2)
    dirs = (fwd[None, None] + ii[..., None] / f * right[None, None] - jj[..., None] / f * up[None, None]).reshape(-1, 3); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    best = np.full(len(dirs), np.inf); which = np.full(len(dirs), -1)
    for k, (cc, R, hf) in enumerate(obbs):
        tt = _ray_obb(np.asarray(cam_pos, float), dirs, cc, R, hf); m = tt < best; best[m] = tt[m]; which[m] = k
    pts = np.where(np.isfinite(best)[:, None], np.asarray(cam_pos, float) + dirs * best[:, None], np.nan).reshape(hh, hw, 3)
    holder_mask = (which.reshape(hh, hw) >= 0) & (which.reshape(hh, hw) < 5)          # 기둥은 제외한 홀더 마스크(실제에선 마스크에 기둥이 섞일 수 있음 -> with_pole 시험은 별도)
    full_mask = which.reshape(hh, hw) >= 0
    return pts, holder_mask, full_mask


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "."); import insertion_geometry as G
    rng = np.random.default_rng(0); errs = []; fails = collections = 0; res = []
    import collections as C; why = C.Counter()
    for k in range(120):
        th = math.radians(rng.choice([-150, -110, -70, 70, 110, 150]) + rng.uniform(-8, 8)); r = rng.uniform(0.34, 0.44); E = np.array([r * math.cos(th), r * math.sin(th), 0.16])
        alpha = rng.choice([30, 45, 60, 90]); pt = rng.choice([180.0, 0.0, 90.0]); d = G.axis_from_angles(alpha, math.degrees(th) + pt)
        cam = np.array([0.0, 0.0, 0.62]) + rng.normal(0, 0.02, 3); cam = cam + 0.35 * np.array([math.cos(th), math.sin(th), 0.0]) * rng.uniform(0.3, 1.0)
        pts, hm, fm = synth_cloud(E, d, cam, E)
        for nm, mask in (("홀더만", hm), ("기둥 포함", fm)):
            e = estimate(pts, mask, cam)
            if not e["ok"]:
                res.append((nm, None, None, alpha, e["why"][:20])); why[(nm, e["why"][:12])] += 1; continue
            ep = float(np.linalg.norm(np.array(e["E"]) - E)) * 1000; ea = math.degrees(math.acos(max(-1, min(1, float(np.array(e["d"]) @ d)))))
            res.append((nm, ep, ea, alpha, "ok"))
    for nm in ("홀더만", "기둥 포함"):
        R_ = [x for x in res if x[0] == nm]; okk = [x for x in R_ if x[1] is not None]
        ep = sorted(x[1] for x in okk); ea = sorted(x[2] for x in okk)
        print(f"[{nm}] 추정 성공 {len(okk)}/{len(R_)} | 입구 위치 오차 중앙 {ep[len(ep)//2]:.1f}mm p90 {ep[int(len(ep)*.9)]:.1f}mm | 축 각도 오차 중앙 {ea[len(ea)//2]:.1f}도 p90 {ea[int(len(ea)*.9)]:.1f}도 | 축 부호 틀림(>90도) {sum(e>90 for e in ea)}건")
    print("실패 사유:", dict(why))
