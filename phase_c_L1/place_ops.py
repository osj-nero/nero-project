#!/usr/bin/env python3
"""place_ops.py -- place 보조 함수(슬롯 생성, 시뮬 정답 판정, 시행 기록). discover_object.py 가 쓴다."""
import os, json, math, os, time
import numpy as np


def support_z(pts, xy, radius, fallback):
    """슬롯 xy 주변(반경 radius) 도착지 점군의 높이 중앙값 = 물체가 실제로 닿을 면 높이. 오목한 접시는 가장자리가 아니라 바닥이 나온다."""
    if pts is None or len(pts) == 0:
        return float(fallback)
    P = np.asarray(pts, float); m = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) < radius
    if int(m.sum()) < 8:
        return float(fallback)
    return float(np.median(P[m, 2]))


def concavity(o, min_size=0.05, min_depth=0.02):
    """도착지 후보가 위가 열린 용기(바구니/접시/그릇)인지 점군으로 판별한다(문구와 무관한 형상 단서, 2026-10-07).
    물체 박스(OBB) 안쪽 중심부 점의 높이(중앙값 = 바닥)와 가장자리 띠 점의 높이(상위 분위 = 림)를 비교해 가장자리가 중심보다 min_depth 이상 높으면 오목.
    반환 {concave, depth, floor_z, rim_z} 또는 점이 모자라면 None. 보이는 점이 위에서 본 면뿐이므로 벽 바깥 높이가 아니라 림 높이를 잰다."""
    min_depth = float(os.environ.get("NERO_CONC_MIN_DEPTH", min_depth))      # 2026-10-28: 오목 판정 임계를 파라미터로(기본 0.02 = 기존 상수)
    pts = o.get("pts"); raw = o.get("raw")
    if pts is None or raw is None or len(pts) < 40 or min(raw[0], raw[1]) < min_size:
        o["concave_why"] = "점 %s개/최소변 %s" % (None if pts is None else len(pts), None if raw is None else round(min(raw[0], raw[1]), 3))      # 미측정 원인(2026-10-08)
        return None
    P = np.asarray(pts, float); cx, cy = o["xyz"][0], o["xyz"][1]; yaw = float(o.get("yaw", 0.0)); cs, sn = math.cos(-yaw), math.sin(-yaw)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy; lx = cs * dx - sn * dy; ly = sn * dx + cs * dy; w, d = raw[0], raw[1]
    inside = (np.abs(lx) <= w / 2 + 0.01) & (np.abs(ly) <= d / 2 + 0.01)
    core = inside & (np.abs(lx) < 0.25 * w) & (np.abs(ly) < 0.25 * d)
    ring = inside & ((np.abs(lx) > 0.36 * w) | (np.abs(ly) > 0.36 * d))
    if int(core.sum()) < 8 and int(ring.sum()) >= 100 and min(raw[0], raw[1]) >= 0.12:      # 2026-10-08: 가장자리 점은 많은데 중심 점이 하나도 없다 = 속이 빈 용기(바닥이 깊이/분할에서 빠짐). 바닥은 NERO_DEST_FLOOR_Z 로 가정한다
        rim_z = float(np.percentile(P[ring, 2], 90)); fl = float(os.environ.get("NERO_DEST_FLOOR_Z", "0.012"))
        if rim_z - fl >= min_depth:
            return dict(concave=True, depth=round(rim_z - fl, 4), floor_z=fl, rim_z=round(rim_z, 4), inferred=True)
    if int(core.sum()) < 8 or int(ring.sum()) < 8:
        o["concave_why"] = "중심점 %d/가장자리점 %d (각 8 미만, 전체 %d)" % (int(core.sum()), int(ring.sum()), len(P))
        return None
    floor_z = float(np.median(P[core, 2])); rim_z = float(np.percentile(P[ring, 2], 90)); depth = rim_z - floor_z
    return dict(concave=bool(depth >= min_depth), depth=round(depth, 4), floor_z=round(floor_z, 4), rim_z=round(rim_z, 4))


def describe_dest(o):
    """등록부 항목 -> place_decide.decide() 의 dest 형식(2026-10-08). 오목(속 빈 추정 포함)이면 receptacle(개구부 = 바깥 박스 - 벽 두께 x2), 아니면 surface.
    반환에 confidence(measured/inferred/unmeasured)와 근거(src), 오목 미측정 사유(why)를 같이 담는다. 바깥 박스는 일부만 보이면 작게 나오므로 open 은 하한 추정이다."""
    raw = o.get("raw"); c = o.get("concave"); h = o.get("hole")
    if h and h.get("ok"):      # 홀더(좁은 입구): hole_estimator_v2 가 입구 E / 축 d / 개구부를 추정했다
        dz = abs(float(h["d"][2])) if h.get("d") is not None else 1.0
        return dict(kind="receptacle", open=(float(h["open"][0]), float(h["open"][1])), depth=float(h.get("depth", 0.14)), tilt_deg=round(math.degrees(math.acos(max(0.0, min(1.0, dz)))), 1), confidence="measured", src="hole_estimator_v2", why=None, E=h["E"], d=h["d"])
    if c and c.get("concave") and raw is not None:
        wall = float(os.environ.get("NERO_DEST_WALL", "0.012")); _io = inner_open(o) if os.environ.get("NERO_CONTAINER_INNER", "0") == "1" else None
        return dict(kind="receptacle", open=(_io if _io is not None else (max(0.0, float(raw[0]) - 2 * wall), max(0.0, float(raw[1]) - 2 * wall))), depth=float(c["depth"]), tilt_deg=0.0,
                    confidence="inferred" if c.get("inferred") else "measured", src="concavity", why=None)
    return dict(kind="surface", open=None, depth=None, tilt_deg=0.0, confidence="measured" if c else "unmeasured", src="concavity", why=None if c else o.get("concave_why"))


def inner_open(o, band=0.03):
    """2026-10-14(NERO_CONTAINER_INNER=1): 바닥 근처(바닥 + band 이하) 점으로 용기 안쪽의 실제 빈 폭(로컬 x, y)을 잰다. 바구니는 천 라이너가 안쪽으로 처져 깔때기 모양이라
    바닥 근처 안쪽 폭이 바깥 박스 - 벽 두께 x2 보다 훨씬 좁다(실측: 바깥 25x30 에서 바닥 위 2.5~5.5cm 폭 12 x 19cm). 못 재면 None."""
    pts = o.get("pts"); raw = o.get("raw"); c = o.get("concave") or {}
    if pts is None or raw is None or len(pts) < 40 or not c.get("concave") or c.get("inferred"): return None
    P = np.asarray(pts, float); cx, cy = o["xyz"][0], o["xyz"][1]; yaw = float(o.get("yaw", 0.0)); cs, sn = math.cos(-yaw), math.sin(-yaw)
    dx, dy = P[:, 0] - cx, P[:, 1] - cy; lx = cs * dx - sn * dy; ly = sn * dx + cs * dy; w, d = raw[0], raw[1]
    sel = (np.abs(lx) <= w / 2) & (np.abs(ly) <= d / 2) & (P[:, 2] <= float(P[:, 2].min()) + band)      # 점군 최저 높이 + band(3cm): 실측 11.4x19.7cm (정답 12x19cm). 오목 판정의 floor_z(중앙값)는 쓰지 않는다 -- 그 기준은 18.7x21cm 로 틀렸다
    if int(sel.sum()) < 25: return None
    fx = float(np.percentile(lx[sel], 95) - np.percentile(lx[sel], 5)); fy = float(np.percentile(ly[sel], 95) - np.percentile(ly[sel], 5))
    return (max(0.06, min(fx, w - 0.02)), max(0.06, min(fy, d - 0.02)))


def container_walls(o, thick=0.02):
    """오목한 도착지의 벽 4면을 직육면체 장애물({id,xyz,raw,yaw})로 만든다: 박스(OBB) 가장자리 안쪽 thick 두께, 바닥(0)~림 높이. 바닥은 서버의 테이블이 이미 막는다."""
    c = o["concave"]; cx, cy = o["xyz"][0], o["xyz"][1]; w, d, _ = o["raw"]; yaw = float(o.get("yaw", 0.0)); cs, sn = math.cos(yaw), math.sin(yaw); h = max(float(c["rim_z"]), 0.02)
    thick = float(o.get("wall_thick", thick))      # 2026-10-14: 입력으로 준 목적지는 벽 두께도 준다(보관함 약 1cm). 기본 2cm 는 안쪽 폭을 4cm 가까이 좁게 본다
    out = []
    _io = inner_open(o) if os.environ.get("NERO_CONTAINER_INNER", "0") == "1" else None
    if _io is not None:      # 깔때기 안쪽: 바깥 가장자리에서 실제 안쪽 폭까지를 벽(직육면체)으로 본다 -- 계획이 손가락/쥔 물체가 들어갈 수 있는 폭만 쓰게 한다
        tx, ty = max(thick, (w - _io[0]) / 2), max(thick, (d - _io[1]) / 2)
        for nm, lx, ly, sx, sy in (("px", (w - tx) / 2, 0, tx, d), ("nx", -(w - tx) / 2, 0, tx, d), ("py", 0, (d - ty) / 2, w - 2 * tx, ty), ("ny", 0, -(d - ty) / 2, w - 2 * tx, ty)):
            out.append(dict(id="dest_wall_" + nm, xyz=[cx + cs * lx - sn * ly, cy + sn * lx + cs * ly, h / 2], raw=[sx, sy, h], yaw=yaw))
        return out
    for nm, lx, ly, sx, sy in (("px", (w - thick) / 2, 0, thick, d), ("nx", -(w - thick) / 2, 0, thick, d), ("py", 0, (d - thick) / 2, w - 2 * thick, thick), ("ny", 0, -(d - thick) / 2, w - 2 * thick, thick)):
        out.append(dict(id="dest_wall_" + nm, xyz=[cx + cs * lx - sn * ly, cy + sn * lx + cs * ly, h / 2], raw=[sx, sy, h], yaw=yaw))
    return out


def dest_slots(o, hold_xy, occupied=None):
    """도착지 물체(등록부 항목: xyz, raw=[w,d,top], yaw) 위의 슬롯 후보. 중심 먼저, 그다음 풋프린트 안쪽 격자. -> (slots, top_z)"""
    cx, cy = o["xyz"][0], o["xyz"][1]; w, d, top = o["raw"]; yaw = float(o.get("yaw", 0.0))
    _io = inner_open(o) if os.environ.get("NERO_CONTAINER_INNER", "0") == "1" else None
    if _io is not None: w, d = _io      # 슬롯은 측정한 안쪽 폭 안에서만 잡는다
    hx = max(0.0, w / 2 - hold_xy[0] / 2 - 0.01); hy = max(0.0, d / 2 - hold_xy[1] / 2 - 0.01)
    c, s = math.cos(yaw), math.sin(yaw); out = [[cx, cy]]
    _extra = ((0.75, 0.75), (-0.75, 0.75), (0.75, -0.75), (-0.75, -0.75), (0.75, 0), (-0.75, 0), (0, 0.75), (0, -0.75)) if occupied else ()
    for fx, fy in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (1.0, 0), (-1.0, 0), (0, 1.0), (0, -1.0)) + _extra:
        dx, dy = fx * hx, fy * hy
        if abs(dx) + abs(dy) < 0.004:
            continue
        out.append([cx + c * dx - s * dy, cy + s * dx + c * dy])
    rad = max(0.03, 0.5 * max(hold_xy[0], hold_xy[1]))
    if occupied:      # 2026-10-13(NERO_CONTAINER_SPREAD): 이미 놓인 물체(x, y, 반경)와 겹치는 칸은 뺀다. 전부 겹치면 원래 순서를 유지한다
        _free = [q for q in out if all(math.hypot(q[0] - ox, q[1] - oy) >= rad + orad + 0.01 for ox, oy, orad in occupied)]
        if _free: out = _free
    _c = o.get("concave") or {}; _fb = float(_c["floor_z"]) if _c.get("inferred") else top      # 속이 빈 용기로 추정한 경우 점군이 없으니 바닥 가정 높이를 쓴다
    out = [[x, y, support_z(o.get("pts"), (x, y), rad, _fb)] for x, y in out]     # 슬롯마다 면 높이를 추정해 세 번째 값으로 붙인다
    return out, float(top)


def free_slots(obstacles, rad, ref_xy, n=10, rmin=0.30, rmax=0.52, gap=0.05):
    """책상의 빈 자리 후보(로봇 기준 환형). 등록부 물체와 rad+그 반경+gap 이상 떨어진 점, ref_xy 에 가까운 순."""
    cands = []
    for r in np.arange(rmin, rmax + 1e-6, 0.04):
        for th in np.radians(np.arange(-180, 180, 12)):
            x, y = float(r * math.cos(th)), float(r * math.sin(th))
            ok = True
            for o in obstacles:
                orad = 0.5 * math.hypot(o["raw"][0], o["raw"][1])
                if math.hypot(x - o["xyz"][0], y - o["xyz"][1]) < rad + orad + gap:
                    ok = False; break
            if ok:
                cands.append([x, y])
    cands.sort(key=lambda p: math.hypot(p[0] - ref_xy[0], p[1] - ref_xy[1]))
    # 서로 6cm 이상 떨어진 것만(IK 배치가 같은 곳을 반복하지 않게)
    out = []
    for p in cands:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.06 for q in out):
            out.append(p)
        if len(out) >= n:
            break
    return out


def _R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_deg(qa, qb):
    """초기 자세 대비 기울어짐: 처음에 수직이던 물체 축이 지금 얼마나 누웠나(수직축 회전(yaw)은 무시)."""
    Ra, Rb = _R(np.array(qa, float)), _R(np.array(qb, float))
    v = Ra.T @ np.array([0.0, 0.0, 1.0])
    return float(math.degrees(math.acos(max(-1.0, min(1.0, (Rb @ v)[2])))))


BUMP_CM = float(os.environ.get("NERO_BUMP_CM", "2.0")); BUMP_TILT = float(os.environ.get("NERO_BUMP_TILT", "15"))


def judge(a, b, contact, slot, top_z, dest=None, H=0.1):
    """시뮬 정답: pick 전 스냅샷 a, place 후 스냅샷 b, 잡은 지점 contact -> dict(verdict, prim, dxy, dz, tilt)
    높이는 바닥(bbox 최저점)이 아니라 중심 기준: 기대 중심 = 면 높이 + H/2 (기울어진 물체의 bbox 는 실제보다 낮게 나온다)."""
    if not a or not b:
        return dict(verdict="NA")
    prim = min(a, key=lambda k: math.dist(a[k]["p"], contact))
    if prim not in b:
        return dict(verdict="NA", prim=prim)
    pa, pb = a[prim], b[prim]
    dxy = math.hypot(pb["p"][0] - slot[0], pb["p"][1] - slot[1]); dz = float(pb["p"][2] - (top_z + H / 2.0)); tilt = tilt_deg(pa["q"], pb["q"])
    on_dest = None; toppled_in = False
    if dest is not None:
        on_dest = math.hypot(pb["p"][0] - dest["xyz"][0], pb["p"][1] - dest["xyz"][1]) < 0.5 * max(dest["raw"][0], dest["raw"][1]) + 0.02
    in_region = (dxy < 0.08) or bool(on_dest)
    if dz > 0.08:
        v = "HELD_STILL"           # 손에 아직 있음(안 놓임)
    elif in_region:
        v = "PLACED" if tilt < 30.0 else "TOPPLED"
        # 2026-10-08(사용자 결정): 위가 열린 용기 안이면 기울어져 있어도 성공(PLACED). 쓰러짐 여부는 toppled_in 으로 남긴다. 접시/책상 같은 평면은 기존대로 TOPPLED 가 실패.
        if v == "TOPPLED" and on_dest and (dest or {}).get("concave", {}) and (dest.get("concave") or {}).get("concave") and os.environ.get("NERO_TOPPLED_IN_OK", "1") != "0":
            v = "PLACED"; toppled_in = True
    else:
        v = "MISSED"
    # 2026-10-16: 놓은 직후 다른 물체/용기가 밀렸는지(시뮬 정답 스냅샷 비교). 놓은 물체 자신은 제외, 수평 이동 BUMP_CM 초과 또는 기울기 BUMP_TILT 초과면 기록.
    bumped = []
    for k, pa_ in a.items():
        if k == prim or k not in b: continue
        d_ = math.hypot(b[k]["p"][0] - pa_["p"][0], b[k]["p"][1] - pa_["p"][1]); t_ = tilt_deg(pa_["q"], b[k]["q"])
        if d_ > BUMP_CM / 100.0 or t_ > BUMP_TILT: bumped.append(dict(prim=k, moved_cm=round(100 * d_, 1), tilt=round(t_, 0)))
    return dict(verdict=v, prim=prim, dxy=round(dxy, 3), dz=round(dz, 3), tilt=round(tilt, 1), on_dest=on_dest, toppled_in=toppled_in, bumped=bumped)


def log_place(rec, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
    except Exception as e:
        print("[place_ops] 기록 실패", e)
