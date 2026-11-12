#!/usr/bin/env python3
"""entity_registry.py -- 엔티티 등록부(World.objs)를 명령 사이에 이어 주는 저장/복원과 엔티티 표(2026-10-13, S1).
이 모듈은 discover_object.py 와 독립이다(아직 연결 안 함). 연결 후에도 NERO_REG_PERSIST=1 일 때만 쓴다.
- dump(objs, path, counter): 등록부를 JSON 으로 저장(점군은 최대 150점으로 줄임).
- load(path, max_age_s): 읽어서 모든 엔티티에 restored=True 를 붙여 돌려준다. 정지 프레임이 다시 확인하기 전에는 계획에 쓰지 않는다.
- table_text(objs): Claude 에게 줄 엔티티 표(텍스트). 라벨은 CLIP 추정이라 틀릴 수 있으므로 측정된 기하 사실(크기, 오목, 홀 입구)을 함께 적는다."""
import json, os, time
import numpy as np

KEEP = ("xyz", "raw", "yaw", "w", "label", "top", "s_t", "s_x", "s_p", "p_t", "role", "fails", "seen", "defer_gen", "tentative", "concave", "hole", "hole_note", "hole_neg", "feat", "feat_n")


def _plain(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, dict): return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_plain(x) for x in v]
    return v


def dump(objs, path, counter=0):
    out = {}
    for oid, o in objs.items():
        if "raw" not in o: continue
        e = {k: _plain(o[k]) for k in KEEP if k in o and o[k] is not None}
        if o.get("pts") is not None:
            P = np.asarray(o["pts"], float); step = max(1, len(P) // 150); e["pts"] = np.round(P[::step], 3).tolist()
        out[oid] = e
    tmp = path + ".tmp"; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(dict(t=time.time(), counter=int(counter), objs=out), open(tmp, "w"), ensure_ascii=False); os.replace(tmp, path)
    return len(out)


def load(path, max_age_s=900.0):
    """-> (objs, counter). 파일이 없거나 오래됐으면 ({}, 0)."""
    try: d = json.load(open(path))
    except Exception: return {}, 0
    if time.time() - float(d.get("t", 0)) > max_age_s: return {}, 0
    objs = {}
    for oid, e in (d.get("objs") or {}).items():
        o = dict(e); o["restored"] = True; o.setdefault("fails", 0); o.setdefault("defer_gen", -1); o.setdefault("seen", 0)
        if "pts" in o: o["pts"] = np.asarray(o["pts"], np.float32)
        objs[oid] = o
    return objs, int(d.get("counter", 0))


def _status(o):
    if o.get("restored"): return "미확인(이전 명령 기록)"
    if o.get("tentative"): return "임시"
    return "확정"


def table_text(objs):
    """엔티티 표. 크기는 cm, 위치는 로봇 기준 x,y(m)."""
    rows = []
    for oid, o in sorted(objs.items(), key=lambda kv: int(kv[0].split("_")[-1]) if kv[0].split("_")[-1].isdigit() else 0):
        if "raw" not in o or "xyz" not in o: continue
        r = o["raw"]; lab = (o.get("label") or "라벨 없음"); facts = []
        c = o.get("concave") or {}
        if c.get("concave"): facts.append(f"위가 열림 깊이 {100 * c.get('depth', 0):.0f}cm")
        if o.get("hole_note"): facts.append(str(o["hole_note"]))
        if o.get("role") in ("dest", "placed"): facts.append({"dest": "목적지 후보", "placed": "이미 놓은 물체"}[o["role"]])
        rows.append(f"{oid} | {lab} | {100 * r[0]:.0f}x{100 * r[1]:.0f}x{100 * r[2]:.0f}cm | ({o['xyz'][0]:+.2f},{o['xyz'][1]:+.2f}) | {_status(o)}" + (" | " + "; ".join(facts) if facts else ""))
    return "id | 라벨(CLIP 추정, 틀릴 수 있음) | 크기 | 위치 | 상태 | 측정된 사실\n" + "\n".join(rows)
