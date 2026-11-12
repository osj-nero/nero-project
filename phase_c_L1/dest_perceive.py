#!/usr/bin/env python3
"""dest_perceive.py -- 목적지 인식 본체 1차(2026-10-28, NERO_DEST_PERCEIVE=1 일 때만 discover_object 가 호출; 기본 꺼짐). 이름·카테고리·색 분기 없음, 기하만.
입력: 등록부 엔티티 dict(world.objs: xyz, raw=[w,d,h], yaw, pts(N,3 base 좌표, 다운샘플)). 출력: 인식판 dests 목록(DEST_PERCEPTION_DESIGN.md §2.1 스키마, meas 필드).
유형 판정(DEST_PERCEPTION_DESIGN §1; 기존 _kind_of 구조 + d_f 파라미터):
  hole      : 원형 입구 링 증거 — 바깥 지름 NERO_HOLE_OUTER_MIN..MAX(기본 0.05~0.14, 기존 홀 밴드), 높이 >= HOLE_MIN_H(0.10), 종횡비>=0.8, 윗띠(최상단 2 cm) 점이 중심부에 비어 있음(반지름 < 0.55 r_out 인 점 비율 <= 0.1), 윗띠 점 >= 25
  container : 공동 깊이 d_c(concavity: rim-floor) >= d_f 이고 최소변 >= NERO_DEST_MIN_SIDE(기본 0.16, 기존 상수)
  flat      : d_c < d_f(또는 오목 미검출)이고 최소변 >= 0.12(기존 _kind_of 의 flat 하한) 
  d_f = NERO_DEST_DF (기본 0.05565 m = (tray d_c 19.4 + 가장 얕은 바구니 91.9)/2, PREREG 결정; 인식 모드에서만 적용, 기존 오목 임계 0.02 는 건드리지 않음)
측정 못 하는 값은 None + meas='unmeasured'. 정답으로 대체하지 않는다.
한계(미검증): 윗면만 보이는 점군이라 바닥이 안 보이면 floor 는 추정(concavity inferred), 벽 두께·원통 깊이는 미측정, 원통 입구는 링 윗띠 점의 최소 반지름 기반 추정."""
import os, math
import numpy as np
import place_ops

DF_DEFAULT = 0.05565


def _env(name, default): return float(os.environ.get(name, default))


def _hole(o, pts):
    """원형 입구 링 증거. -> dict(E, outer, open, n_top) 또는 None"""
    if pts is None or len(pts) < 60: return None
    w, d, h = o["raw"]; lo, hi = _env("NERO_HOLE_OUTER_MIN", 0.05), _env("NERO_HOLE_OUTER_MAX", 0.14)
    if not (lo <= min(w, d) and max(w, d) <= hi * 1.15) or min(w, d) / max(w, d) < 0.8 or h < _env("NERO_HOLE_MIN_H", 0.10): return None
    P = np.asarray(pts, float).reshape(-1, 3); zt = float(np.percentile(P[:, 2], 99)); top = P[P[:, 2] >= zt - 0.02]
    if len(top) < 25: return None
    c = np.array([o["xyz"][0], o["xyz"][1]]); r = np.hypot(top[:, 0] - c[0], top[:, 1] - c[1]); r_out = float(np.percentile(r, 95))
    if r_out < 0.5 * lo or float((r < 0.55 * r_out).mean()) > 0.10: return None
    r_in = float(np.percentile(r, 5)); return dict(E=[float(c[0]), float(c[1]), zt], outer=2 * r_out, open=2 * r_in, n_top=int(len(top)))


def _in_box(xy, q, pad):
    c = q["xy"]; cs, sn = math.cos(-q["yaw"]), math.sin(-q["yaw"]); dx, dy = xy[0] - c[0], xy[1] - c[1]
    return abs(cs * dx - sn * dy) <= q["size"][0] / 2 + pad and abs(sn * dx + cs * dy) <= q["size"][1] / 2 + pad


def classify_one(o):
    """엔티티 하나 -> dest dict(id 없음) 또는 None(목적지 후보 아님)"""
    if "raw" not in o or "xyz" not in o: return None
    refined = None
    if os.environ.get("NERO_PERCEIVE_REFINE", "0") == "1" and o.get("fobbs"):      # 2026-10-28(기본 0): 프레임별 사각형 군집 중앙값으로 중심·크기·yaw 재추정(트리밍·최신 프레임 박스·합집합 병합의 부풀림 제거); 유형 판정·concavity 도 이 기하로
        rf = refine_from_fobbs(o)
        if rf is not None: o = dict(o); o["xyz"] = [rf[0][0], rf[0][1], o["xyz"][2]]; o["raw"] = [rf[1][0], rf[1][1], o["raw"][2]]; o["yaw"] = rf[2]; refined = rf[3]
    w, d, h = [float(v) for v in o["raw"]]; pts = o.get("pts"); DF = _env("NERO_DEST_DF", DF_DEFAULT); ms = min(w, d)
    base = dict(xy=[round(float(o["xyz"][0]), 4), round(float(o["xyz"][1]), 4)], size=[round(w, 4), round(d, 4), round(h, 4)], yaw=round(float(o.get("yaw", 0.0)), 4))
    hl = _hole(o, pts)
    if hl is not None:
        return dict(kind="hole", **base, yaw_defined=False, E=[round(v, 4) for v in hl["E"]], d=[0, 0, -1], open=round(hl["open"], 4), outer=round(hl["outer"], 4), depth=None, d_c=None, src=["agn_mask", "ring_top_band"], n_pts=int(len(pts)),
                    meas=dict(xy="measured", size="measured", yaw="unmeasured", E="measured", d="assumed", open="inferred", outer="measured", depth="unmeasured"))
    c = place_ops.concavity(dict(pts=pts, raw=o["raw"], xyz=o["xyz"], yaw=o.get("yaw", 0.0)), min_size=0.0, min_depth=0.0) if pts is not None else None
    dc = None if c is None else float(c["depth"]); inferred = bool(c and c.get("inferred"))
    # 2026-11-04 NERO_DEST_USE_RIM_H=1(기본 꺼짐): 바닥 높이 측정 잡음으로 d_c 가 문턱 아래로 떨어진 깊은 용기(림·외곽 높이 >= 6 cm, 최소변 >= 16 cm)도 container 로 본다.
    _rim_ok = (os.environ.get("NERO_DEST_USE_RIM_H", "0") == "1" and c is not None and h >= 0.06 and float(c["rim_z"]) >= 0.06 and ms >= _env("NERO_DEST_MIN_SIDE", 0.16))
    if (dc is not None and dc >= DF and ms >= _env("NERO_DEST_MIN_SIDE", 0.16)) or _rim_ok: kind = "container"
    elif (dc is None or dc < DF) and ms >= 0.12: kind = "flat"
    else: return None
    e = dict(kind=kind, **base, yaw_defined=(kind == "container" and min(w, d) / max(w, d) < 0.9), d_c=(None if dc is None else round(dc, 4)), refined_frames=refined, src=["agn_mask", "concavity"] if c else ["agn_mask"], n_pts=(0 if pts is None else int(len(pts))))
    m = dict(xy="measured", size="measured", yaw=("measured" if e["yaw_defined"] else "unmeasured"))
    if c: e.update(rim_z=round(float(c["rim_z"]), 4), floor_z=round(float(c["floor_z"]), 4), wall=None); m.update(rim_z="measured", floor_z=("inferred" if inferred else "measured"), wall="unmeasured")
    else: e.update(rim_z=None, floor_z=None, wall=None); m.update(rim_z="unmeasured", floor_z="unmeasured", wall="unmeasured")
    e["meas"] = m; return e


def perceive(objs):
    """등록부(world.objs) -> 인식판 dests 목록(id 는 dest_N, 위치 순서 정렬: 정답 순서와 무관). 같은 위치(5 cm 이내) 중복은 점 수 많은 것 하나."""
    out = []
    for k, o in objs.items():
        if o.get("role") == "placed": continue
        e = classify_one(o)
        if e is not None: e["registry_id"] = k; out.append(e)
    out.sort(key=lambda e: -(e["size"][0] * e["size"][1]))      # 면적 큰 것부터: 더 큰 후보의 바닥 발자국(+2 cm) 안에 중심이 드는 작은 후보는 같은 물체의 조각으로 보고 버린다(마스크 조각남 방어, 이름 무관)
    keep = []
    for e in out:
        if any(math.hypot(e["xy"][0] - q["xy"][0], e["xy"][1] - q["xy"][1]) < 0.05 for q in keep): continue
        if any(_in_box(e["xy"], q, 0.02) for q in keep): e["dropped_as_fragment"] = True; continue
        keep.append(e)
    keep.sort(key=lambda e: (round(e["xy"][0], 1), round(e["xy"][1], 1)))
    for i, e in enumerate(keep, 1): e["id"] = f"dest_{i}"; e["label"] = ""
    return keep


def relook_candidates(objs, exclude=()):
    """부분 시야 의심 후보(기하만): 공동 깊이 d_c >= d_f 이면서 최소변이 용기 하한(NERO_DEST_MIN_SIDE)보다 작고 최대변 >= 0.10 인 엔티티 -> 등록부 id 목록(d_c 큰 순). 재관측 대상 선정용."""
    DF = _env("NERO_DEST_DF", DF_DEFAULT); out = []
    for k, o in objs.items():
        if k in exclude or o.get("role") == "placed" or "raw" not in o or o.get("pts") is None: continue
        w, d, h = [float(v) for v in o["raw"]]
        if min(w, d) >= _env("NERO_DEST_MIN_SIDE", 0.16) or max(w, d) < 0.10: continue
        c = place_ops.concavity(dict(pts=o["pts"], raw=o["raw"], xyz=o["xyz"], yaw=o.get("yaw", 0.0)), min_size=0.0, min_depth=0.0)
        if c is not None and float(c["depth"]) >= DF: out.append((float(c["depth"]), k))
    return [k for _, k in sorted(out, reverse=True)]


def refine_from_fobbs(o):
    """프레임별 사각형(o['fobbs'])으로 (중심 xy, 바깥 크기 w,d, yaw) 재추정. ① 점 수가 최대의 70% 이상인 프레임만(부분 시야 제외) ② 가장 점 많은 프레임의 yaw 에 90° 단위로 정렬한 크기를 5% 이내로 군집, 지지(프레임 수)가 가장 큰 군집(동률이면 점 수 최대) 선택(병합·조각 제외) ③ 군집 안의 중심·크기·yaw 중앙값.
    -> (xy, [w,d], yaw, n_used) 또는 None(프레임 없음)"""
    fb = list(o.get("fobbs") or [])
    if not fb: return None
    nmax = max(f["n"] for f in fb); pool = [f for f in fb if f["n"] >= 0.7 * nmax]; ref = max(pool, key=lambda f: f["n"]); ry = ref["yaw"]; rows = []
    for f in pool:
        k = min(range(4), key=lambda k: abs(((f["yaw"] + k * math.pi / 2) - ry + math.pi) % (2 * math.pi) - math.pi)); y = f["yaw"] + k * math.pi / 2; w, d = (f["wd"][0], f["wd"][1]) if k % 2 == 0 else (f["wd"][1], f["wd"][0])
        rows.append((f["c"], [w, d], ((y - ry + math.pi) % (2 * math.pi)) - math.pi, f["n"]))
    best = None
    for a in rows:
        cl = [r for r in rows if abs(r[1][0] - a[1][0]) <= 0.05 * a[1][0] and abs(r[1][1] - a[1][1]) <= 0.05 * a[1][1]]; sc = (len(cl), max(r[3] for r in cl))
        if best is None or sc > best[0]: best = (sc, cl)
    keep = best[1]; C = np.median(np.array([r[0] for r in keep]), axis=0); WD = np.median(np.array([r[1] for r in keep]), axis=0); ya = float(np.median([r[2] for r in keep])) + ry
    return [float(C[0]), float(C[1])], [float(WD[0]), float(WD[1])], ya, len(keep)
