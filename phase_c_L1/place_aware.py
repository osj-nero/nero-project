#!/usr/bin/env python3
"""place_aware.py -- place-aware 파지 후보 재정렬(개발 사본, 2026-11-03). discover_object.py 가 NERO_PAT=1 일 때만 import 한다(기본 0 = 모듈 미사용, 현행 동작 동일).
목적: 높이가 큰/긴 물체를 높이 쥐면 손 아래로 길게 매달려(예: 흰 상자 접촉 89% -> 약 14 cm) 용기 림 위에서 놓을 때 flange 가 너무 높아 팔 도달 영역 밖이 된다(place 후보 156개 중 IK 0개).
규칙(코드에 하드코딩된 물체 종류 없음, 인식 치수만 사용):
  대상 = 높이 H >= NERO_PAT_H(0.12 m) 또는 발자국 길이/단변 >= NERO_PAT_RATIO(2.0). 아니면 후보를 그대로 돌려준다.
  후보 접촉 높이 비율 r = (contact_z - base_z)/H. 순서 키 = (놓기 도달 0 으로 추정되는 후보는 뒤로, 띠 [NERO_PAT_LO 0.40, NERO_PAT_HI 0.60] 안 후보 먼저, 도달 점수 큰 순, 현행 순서). 삭제는 하지 않는다(서버 C 단계가 앞 max_cands 개만 보므로 순서가 곧 선택; 재정렬 결과가 같으면 현행 그대로).
  '놓기 도달 점수' = 드롭 시 flange 높이·목적지 xy 에서의 top-down IK 도달 지도(nl_control/holder_demo/reach_map2.npz)의 yaw 비율.
  flange 높이 추정: z_f = top + NERO_DROP_H(0.03) + place_clear + (contact_z - base_z) + D_TCP*|approach_z|  (curobo_c_server.py: tcp z = 슬롯면 + clear + HELD_H, flange = tcp - D_TCP*a; D_TCP=0.1358).
추정임: 도달 지도는 엄격한 top-down(yaw 4개) 지도이고 격자 밖은 가장자리로 클램프(낙관적)한다. 실제 IK 는 cuRobo /place 가 판정한다.
"""
import os, math
import numpy as np
D_TCP = 0.1358
_REACH = {"m": None}
def _f(k, d): return float(os.environ.get(k, str(d)))
def geom_from_points(P):
    """P: (N,3) 대상 점군(월드). -> dict(base_z, H, long, short, ratio)"""
    P = np.asarray(P, float); bz = float(max(0.0, min(0.03, P[:, 2].min()))); H = max(0.02, float(np.percentile(P[:, 2], 98)) - bz)
    xy = P[:, :2] - P[:, :2].mean(0); w, v = np.linalg.eigh(np.cov(xy.T)); pr = xy @ v; ext = sorted((float(pr[:, i].max() - pr[:, i].min()) for i in range(2)), reverse=True)
    return dict(base_z=bz, H=H, long=ext[0], short=max(ext[1], 1e-3), ratio=ext[0] / max(ext[1], 1e-3))
def geom_from_box(dims, center_z=None, base_z=None):
    """기록된 C 요청의 target(dims=[x,y,z], center z) 로 만드는 근사(점군이 없는 입력 재생용)."""
    d = sorted(dims[:2], reverse=True); bz = base_z if base_z is not None else max(0.0, float(center_z) - dims[2] / 2.0) if center_z is not None else 0.0
    return dict(base_z=bz, H=float(dims[2]), long=d[0], short=max(d[1], 1e-3), ratio=d[0] / max(d[1], 1e-3))
def is_target(g):
    h = _f("NERO_PAT_H", 0.12); r = _f("NERO_PAT_RATIO", 2.0)
    why = [x for x, c in (("높이 %.1fcm>=%.0f" % (100 * g["H"], 100 * h), g["H"] >= h), ("길이/단변 %.1f>=%.1f" % (g["ratio"], r), g["ratio"] >= r)) if c]
    return bool(why), why
def _load_reach():
    if _REACH["m"] is None:
        p = os.environ.get("NERO_PAT_REACH", os.path.expanduser("~/nero/analysis/code_pro_20261102/nl_control/holder_demo/reach_map2.npz"))
        try: _REACH["m"] = dict(np.load(p))
        except Exception: _REACH["m"] = {}
    return _REACH["m"]
LEVELS = (("z34", 0.34), ("z38", 0.38), ("z42", 0.42), ("z46", 0.46), ("z50", 0.50))
def reach_score(dest_xy, z):
    """0..1 = 그 xy 에서 flange 높이 z 로 top-down 도달 가능한 yaw 비율(4 yaw 중). 높이는 지도 수준 사이 선형 보간, 0.34 미만은 0.34 값(낙관), 0.50 초과는 0. -> (score, flag)"""
    m = _load_reach()
    if not m: return None, "도달 지도 없음"
    xs, ys = m["xs"], m["ys"]; i = int(np.argmin(abs(xs - dest_xy[0]))); j = int(np.argmin(abs(ys - dest_xy[1])))
    clamp = (dest_xy[0] < xs[0] - 0.02 or dest_xy[0] > xs[-1] + 0.02 or dest_xy[1] < ys[0] - 0.02 or dest_xy[1] > ys[-1] + 0.02)
    v = [float(m[k][i, j]) / 4.0 for k, _ in LEVELS]; zs = [zz for _, zz in LEVELS]
    if z <= zs[0]: s = v[0]; fl = "z<0.34 낙관"
    elif z >= zs[-1]: s = 0.0; fl = ""
    else: s = float(np.interp(z, zs, v)); fl = ""
    return s, (fl + (" 격자밖 클램프(낙관)" if clamp else "")).strip()
def flange_drop_z(c, base_z, top_z, drop_h, clear):
    a = np.array(c["approach"], float); a /= np.linalg.norm(a)
    return top_z + drop_h + clear + max(0.0, float(c["contact"][2]) - base_z) + D_TCP * abs(float(a[2]))
def rerank(cl, geom, dest, place_clear=0.006, log=None, oid="?"):
    """cl: 후보 목록(현행 순서), geom: geom_from_*, dest: dict(xy=(x,y), top_z=림/면 높이) 또는 None. -> (새 목록, info). 실패·비대상·정보 없음이면 cl 을 그대로 돌려준다."""
    info = dict(applied=False, reason="", target=False, rows=[])
    say = (lambda m: log(f"{oid}: [PAT] {m}")) if log else (lambda m: None)
    ok, why = is_target(geom); info["target"] = ok; info["target_why"] = why
    if not ok: info["reason"] = "비대상(H %.1fcm, 길이/단변 %.1f)" % (100 * geom["H"], geom["ratio"]); say("대상 아님 -- 현행 순위 유지 " + info["reason"]); return cl, info
    lo, hi = _f("NERO_PAT_LO", 0.40), _f("NERO_PAT_HI", 0.60); drop_h = _f("NERO_DROP_H", 0.03); H = geom["H"]; bz = geom["base_z"]
    rows = []
    for i, c in enumerate(cl):
        if c.get("contact") is None or c.get("approach") is None: rows.append(dict(i=i, r=None, inband=False, zf=None, reach=None, note="접촉/접근 없음")); continue
        r = (float(c["contact"][2]) - bz) / H; zf = flange_drop_z(c, bz, dest["top_z"], drop_h, place_clear) if dest else None
        sc, fl = (reach_score(dest["xy"], zf) if dest else (None, "목적지 정보 없음"))
        rows.append(dict(i=i, r=round(r, 3), inband=(lo <= r <= hi), zf=(round(zf, 3) if zf is not None else None), reach=(round(sc, 2) if sc is not None else None), note=fl))
    info["rows"] = rows; known = [x for x in rows if x["reach"] is not None]
    def key(x): return (1 if (x["reach"] is not None and x["reach"] <= 0.0) else 0, 0 if x["inband"] else 1, -(x["reach"] if x["reach"] is not None else 0.0), x["i"])      # 1) 놓기 도달 0 으로 추정되는 후보는 뒤로 2) 띠 안 3) 도달 점수 큰 순 4) 현행 순서(안정)
    order = [x["i"] for x in sorted(rows, key=key)]; nin = sum(1 for x in rows if x["inband"]); nzero = sum(1 for x in rows if x["reach"] is not None and x["reach"] <= 0.0)
    info.update(order=order, changed=(order != list(range(len(cl)))), applied=(order != list(range(len(cl)))))
    if not info["applied"]:
        info["reason"] = f"{'; '.join(why)} -> 재정렬 없음(띠 {lo:.2f}~{hi:.2f} 안 {nin}/{len(cl)}, 도달 0 추정 {nzero}) -- 현행 순위 유지"; say(info["reason"] + f" (비율 {[x['r'] for x in rows]})"); return cl, info
    info["reason"] = f"{'; '.join(why)} -> 띠 안 {nin}/{len(cl)}, 도달 0 추정 {nzero}개 뒤로"
    say(f"대상({'; '.join(why)}) 띠 {lo:.2f}~{hi:.2f}: 후보 {len(cl)}개 중 띠 안 {nin}, 도달 0 추정 {nzero} | 순서 {order} | 행(idx,비율,드롭 flange z,도달) {[(x['i'], x['r'], x['zf'], x['reach']) for x in rows]}")
    return [cl[i] for i in order], info
