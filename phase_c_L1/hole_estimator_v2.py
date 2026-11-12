#!/usr/bin/env python3
"""hole_estimator_v2.py -- 홀더 입구 E / 삽입 방향 d 추정 v2 (2026-10-06, 오프라인 개선 시도).
v1 약점: 법선 공분산의 최소 고유벡터를 축으로 삼아, 입구가 카메라를 정면으로 향하면(끝면 법선이 지배) 축을 90도 틀리게 잡는다.
v2: (1) 법선 군집에서 직교 3축(맨해튼 프레임)을 구한다 (2) 세 축 중 점군 길이(퍼짐)가 가장 큰 축을 관 축으로 (3) 끝면 군집의 속이 빈 링/채운 원판과 그 면적으로 입구 쪽 판정:
   링(입구 테두리) -> 그 자리가 입구, d=-a(a=카메라 쪽을 향한 축) / 채운 큰 원판(바깥 바닥) -> 입구는 카메라 반대쪽으로 length / 채운 작은 원판(안쪽 바닥) -> 입구는 카메라 쪽으로 length
estimate_pn(P, N, cam) 은 점·법선만 받는다(여러 시점을 합친 점군에도 사용). 반환 형식은 v1 과 같다."""
import math
import numpy as np
from scipy.spatial import ConvexHull
import hole_estimator as H


def _mode_dir(N, cands, cos_t):
    best = None
    for n in cands:
        c = int((np.abs(N @ n) > cos_t).sum())
        if best is None or c > best[0]: best = (c, n)
    n = best[1]
    for _ in range(4):                                         # 평균으로 다듬기(반대 방향 부호 맞춤)
        m = np.abs(N @ n) > cos_t; s = np.sign(N[m] @ n)[:, None]; v = (N[m] * s).sum(0)
        if np.linalg.norm(v) < 1e-9: break
        n = v / np.linalg.norm(v)
    return n, best[0]


def manhattan_axes(N, rng, n_cand=300):
    cos8 = math.cos(math.radians(8)); idx = rng.choice(len(N), min(n_cand, len(N)), replace=False)
    n1, c1 = _mode_dir(N, N[idx], cos8)
    rest = N[np.abs(N @ n1) < math.sin(math.radians(20))]
    if len(rest) < 30:                                          # 한 방향만 보임: 나머지는 임의 수직(점군 퍼짐 선택에서 걸러진다)
        u = np.cross(n1, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0]); return [(n1, c1), (u, 0), (np.cross(n1, u), 0)]
    Q = rest - np.outer(rest @ n1, n1); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    n2, c2 = _mode_dir(Q, Q[rng.choice(len(Q), min(n_cand, len(Q)), replace=False)], cos8); n2 = n2 - (n2 @ n1) * n1; n2 /= np.linalg.norm(n2)
    return [(n1, c1), (n2, c2), (np.cross(n1, n2), 0)]


def _hull_area(pu, pv):
    try: return float(ConvexHull(np.c_[pu, pv]).volume)
    except Exception: return 0.0


def estimate_pn(P, N, cam, inner=0.06, wall=0.01, length=0.14, seed=0):
    P = np.asarray(P, float); N = np.asarray(N, float); cam = np.asarray(cam, float)
    if len(P) < 200: return dict(ok=False, why="점 부족")
    rng = np.random.default_rng(seed); axes = manhattan_axes(N, rng); ctr = P.mean(0)
    ext = [float(np.percentile((P - ctr) @ a, 98) - np.percentile((P - ctr) @ a, 2)) for a, _ in axes]
    a = axes[int(np.argmax(ext))][0]
    if (cam - ctr) @ a < 0: a = -a                              # a: 카메라 쪽을 향함
    u = np.cross(a, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0]); v = np.cross(a, u)
    t = (P - ctr) @ a; pu = (P - ctr) @ u; pv = (P - ctr) @ v
    face = np.abs(N @ a) > 0.85; clusters = []
    if face.sum() >= 12:
        order = np.argsort(t[face]); tf = t[face][order]; idx = np.where(face)[0][order]; cuts = np.where(np.diff(tf) > 0.015)[0] + 1
        for g in np.split(np.arange(len(tf)), cuts):
            if len(g) < 12: continue
            ii = idx[g]; cu_ = (np.percentile(pu[ii], 3) + np.percentile(pu[ii], 97)) / 2; cv_ = (np.percentile(pv[ii], 3) + np.percentile(pv[ii], 97)) / 2
            r = np.maximum(np.abs(pu[ii] - cu_), np.abs(pv[ii] - cv_)); fill = float((r < 0.30 * inner).mean()); area = _hull_area(pu[ii], pv[ii])
            clusters.append(dict(t=float(np.median(t[ii])), cu=cu_, cv=cv_, fill=fill, n=len(g), area=area))
    rim = [c for c in clusters if c["fill"] < 0.12 and c["area"] > 0.8 * (inner + 2 * wall) ** 2 * 0.6]
    disk = [c for c in clusters if c["fill"] >= 0.3]
    if rim:
        o = max(rim, key=lambda c: c["n"]); t_open, cu, cv, sgn, via = o["t"], o["cu"], o["cv"], -1.0, "rim"
    elif disk:
        b = max(disk, key=lambda c: c["n"]); cu, cv = b["cu"], b["cv"]
        outer_a = (inner + 2 * wall) ** 2; inner_a = inner ** 2
        if abs(b["area"] - outer_a) < abs(b["area"] - inner_a): t_open, sgn, via = b["t"] - length, 1.0, "bottom_outer"       # 바깥 바닥이 보임: 입구는 카메라 반대쪽
        else: t_open, sgn, via = b["t"] + length, -1.0, "bottom_inner"                                                          # 안쪽 바닥이 보임: 입구는 카메라 쪽
    else:
        return dict(ok=False, why="끝면 군집 없음", axis=a.tolist())
    d = a * sgn; E = ctr + a * t_open + u * cu + v * cv
    out = dict(ok=True, E=E.tolist(), d=d.tolist(), via=via, n_pts=int(len(P)), ext=[round(e, 3) for e in ext])
    _ia = int(np.argmax(ext)); out["ext_axis"] = float(ext[_ia]); out["ext_cross"] = float(max(e for k_, e in enumerate(ext) if k_ != _ia))      # 2026-10-12: 축 방향 길이 / 단면 방향 최대 폭(통이면 단면 폭 = 링 바깥 한 변). 기존 동작은 안 바뀐다
    if via == "rim":      # 2026-10-08: 입구 링(끝면 군집)의 볼록 껍질 면적 -> 바깥 한 변 -> 개구부 한 변(벽 두께는 가정값). decide() 의 여유 판정에 쓴다
        out["rim_area"] = float(o["area"]); out["outer_est"] = float(math.sqrt(max(o["area"], 0.0))); out["open_est"] = float(max(0.0, out["outer_est"] - 2 * wall))
    return out


STEP = 3        # 법선 이웃 간격(픽셀). 1 이면 v1 과 같다. 클수록 깊이 노이즈에 강하지만 모서리가 번진다


def normals_step(pts, valid, step=STEP, max_jump=0.02):
    """±step 픽셀 차분 외적 법선. 깊이 불연속(이웃 간 거리 > 2*max_jump*step)과 경계는 제외."""
    if step <= 1: return H.organized_normals(pts, valid, max_jump)
    Hh, W, _ = pts.shape; dx = np.zeros_like(pts); dy = np.zeros_like(pts)
    dx[:, step:-step] = pts[:, 2 * step:] - pts[:, :-2 * step]; dy[step:-step] = pts[2 * step:] - pts[:-2 * step]
    ok = valid.copy(); ok[:, :step] = ok[:, -step:] = False; ok[:step] = ok[-step:] = False
    ok &= np.linalg.norm(dx, axis=-1) < 2 * max_jump * step; ok &= np.linalg.norm(dy, axis=-1) < 2 * max_jump * step
    for sh in (step, -step):
        ok &= np.roll(valid, sh, 0) & np.roll(valid, sh, 1)
    n = np.cross(dx, dy); nn = np.linalg.norm(n, axis=-1, keepdims=True); ok &= (nn[..., 0] > 1e-9); return n / np.maximum(nn, 1e-12), ok


def estimate_view(pts, mask, cam, step=STEP, **kw):
    valid = np.isfinite(pts).all(-1) & mask; n, ok = normals_step(np.nan_to_num(pts), valid, step)
    if ok.sum() < 200: return dict(ok=False, why="점 부족")
    return estimate_pn(pts[ok], n[ok], cam, **kw)


def view_pn(pts, mask, step=STEP):
    valid = np.isfinite(pts).all(-1) & mask; n, ok = normals_step(np.nan_to_num(pts), valid, step); return pts[ok], n[ok]


def pick_holder(dets, E_hint, tol=0.05, min_pts=3000):
    """홀더 마스크 고르기(2026-10-08). 이전 규칙은 마스크 중앙값 xy 가 E 에 가장 가까운 것이었는데, 기울어진 홀더는 관의 중심이 입구 E 에서 멀어(6~8cm) 막대 같은 작은 마스크와 구별이 흐렸다.
    입구는 홀더 표면에 있으므로 'E 에서 마스크 점까지 최소 거리'가 tol 이하인 마스크 중 점이 가장 많은 것을 고른다. 실측(5개 배치): 홀더 2.2~2.9cm / 막대 7~9cm.
    반환 (det | None, 최소거리 m). dets: DO.grab 결과 목록(각 항목에 P=(N,3))."""
    E = np.asarray(E_hint, float); best = None
    for d_ in dets:
        P = np.asarray(d_["P"], float)
        if len(P) < min_pts: continue
        m = float(np.min(np.linalg.norm(P - E, axis=1)))
        if m <= tol and (best is None or len(P) > len(best[0]["P"])): best = (d_, m)
    return best if best else (None, None)
