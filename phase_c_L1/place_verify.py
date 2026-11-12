#!/usr/bin/env python3
"""place_verify.py -- 놓은 직후 비전 검증(시뮬 정답 없음). 같은 관측 자세에서 목적지 영역의 높이 지도를 놓기 전(기준)과 후로 찍어 차이를 본다.
  heightmap(pts, zf, dest, cell) -> dict(grid(max z), x0, y0, cell)   : 목적지 중심 사각 영역(목적지 큰 변/2 + RING)의 셀별 최대 높이
  judge(base, now, dest) -> dict(verdict, area_in, area_out, ...)   : 증가(>RISE)한 셀 면적을 목적지 안/바깥 띠로 나눈다.
  verdict: 'in'(안에 들어감) / 'outside'(바깥 띠에 떨어짐) / 'unknown'(변화 없음: 안 보이거나 안 놓임)"""
import math
import numpy as np

CELL = 0.015; RING = 0.14; RISE = 0.015; WALL = 0.02; T_IN = 6.0; T_OUT = 8.0


def _rot(dx, dy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw); return dx * c - dy * s, dx * s + dy * c


def heightmap(pts, zf, dest, cell=CELL, ring=RING):
    cx, cy = dest["xy"]; half = 0.5 * max(dest["size"][0], dest["size"][1]) + ring
    n = int(math.ceil(2 * half / cell)); x0, y0 = cx - half, cy - half
    g = np.full((n, n), np.nan)
    v = np.isfinite(pts[..., :3]).all(-1) & (zf > 0.1) & (zf < 2.0)
    X, Y, Z = pts[..., 0][v], pts[..., 1][v], pts[..., 2][v]
    ix = np.floor((X - x0) / cell).astype(int); iy = np.floor((Y - y0) / cell).astype(int); ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (Z > -0.05) & (Z < 0.5)
    ix, iy, Z = ix[ok], iy[ok], Z[ok]; flat = np.full(n * n, -9.0); np.maximum.at(flat, ix * n + iy, Z); flat[flat < -8.0] = np.nan
    return dict(grid=flat.reshape(n, n), x0=x0, y0=y0, cell=cell, n=n, cover=float(np.isfinite(flat).mean()))


def judge(base, now, dest, floor_z=0.012):
    n = base["n"]; cell = base["cell"]; yaw = float(dest.get("yaw", 0.0)); w, d = dest["size"][0], dest["size"][1]; cx, cy = dest["xy"]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij"); X = base["x0"] + (ii + 0.5) * cell; Y = base["y0"] + (jj + 0.5) * cell
    lx, ly = _rot(X - cx, Y - cy, yaw); inside = (np.abs(lx) <= w / 2 - WALL) & (np.abs(ly) <= d / 2 - WALL); outside = (np.abs(lx) > w / 2 + WALL) | (np.abs(ly) > d / 2 + WALL)
    b = np.where(np.isfinite(base["grid"]), base["grid"], floor_z); c = np.where(np.isfinite(now["grid"]), now["grid"], np.nan)
    rise = (c - b > RISE) & np.isfinite(c)
    a_in = float((rise & inside).sum()) * cell * cell * 1e4; a_out = float((rise & outside).sum()) * cell * cell * 1e4
    mx_in = float(np.nanmax(np.where(rise & inside, c - b, np.nan))) if (rise & inside).any() else 0.0; mx_out = float(np.nanmax(np.where(rise & outside, c - b, np.nan))) if (rise & outside).any() else 0.0
    if a_in >= T_IN and a_out < T_OUT: v = "in"
    elif a_out >= T_OUT and a_in < T_IN: v = "outside"
    elif a_in >= T_IN and a_out >= T_OUT: v = "in" if a_in >= 2 * a_out else "outside"
    else: v = "unknown"
    return dict(verdict=v, area_in=round(a_in, 1), area_out=round(a_out, 1), max_in=round(mx_in, 3), max_out=round(mx_out, 3), cover=round(now["cover"], 2))


def stable_heightmap(capture_fn, dest, tries=3, gap_s=0.5, tol=0.008, frac=0.04):
    """capture_fn() -> heightmap(...). 0.5초 간격 두 장이 거의 같을 때만 채택한다(자세 도달 직후/이동 중 영상으로 기준을 잡으면 영역 전체가 어긋나 오탐이 난다). 끝내 안 같으면 마지막 것과 경고(stable=False)."""
    import time
    prev = capture_fn(); last_d = None
    for _ in range(tries):
        time.sleep(gap_s); cur = capture_fn()
        a, b = prev["grid"], cur["grid"]; both = np.isfinite(a) & np.isfinite(b)
        if both.sum() > 50:
            dd = np.abs(a - b)[both]; last_d = (float(dd.mean()), float((dd > 0.015).mean()))
            if last_d[0] <= tol and last_d[1] <= frac: cur["stable"] = True; cur["jitter"] = last_d; return cur
        prev = cur
    prev["stable"] = False; prev["jitter"] = last_d; return prev


if __name__ == "__main__":      # 합성 시험: 용기(0.28x0.20) 안에 물체 / 바깥 띠에 물체 / 변화 없음
    H, W = 240, 320; ys, xs = np.mgrid[0:H, 0:W]; X = (xs - W / 2) * 0.003 + 0.40; Y = (ys - H / 2) * 0.003 + 0.0
    dest = dict(xy=[0.40, 0.0], size=[0.28, 0.20, 0.13], yaw=0.0)
    def frame(obj=None):
        Z = np.full((H, W), 0.012); r = (np.abs(X - 0.40) <= 0.14) & (np.abs(Y) <= 0.10); Z[r] = 0.05      # 용기 안 바닥 임의 높이
        if obj: ox, oy = obj; Z[(np.abs(X - ox) < 0.03) & (np.abs(Y - oy) < 0.03)] = 0.10
        P = np.stack([X, Y, Z], -1); return P, np.full((H, W), 0.5)
    pb, zb = frame(); base = heightmap(pb, zb, dest)
    for name, obj in (("안", (0.40, 0.0)), ("바깥 띠", (0.40, 0.17)), ("없음", None)):
        p, z = frame(obj); print(name, judge(base, heightmap(p, z, dest), dest))
