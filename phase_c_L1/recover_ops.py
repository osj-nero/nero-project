#!/usr/bin/env python3
"""recover_ops.py -- 국소 재시도(local re-pick)용 비전 판정: 놓은 직후(그리퍼가 빈 채 후퇴한 자세)의 프레임 하나로 물체 상태를 판정하고, 다시 집을 물체 마스크를 만든다.
스윕/옵저베이션 복귀 없이 현재 자세에서 한 번 보는 것이 목적이다(오버헤드 최소). 시뮬 정답 없이 깊이(base 좌표 점군)만 쓴다.
  assess(pts, zf, dest, slot, H_exp, floor_z) -> dict(status, mask, P, info)
    status: 'ok'(서 있게 목적지 위) / 'toppled'(목적지 위에 있으나 낮다=누웠다) / 'off_dest'(목적지 밖 테이블 위) / 'missing'(근처에 안 보임)
  목적지가 있으면: 목적지 풋프린트 안에서 면 높이(top_z) + margin 보다 높은 점만 물체로 본다(목적지와 한 덩어리로 합쳐 보이는 문제를 피한다)."""
import math
import numpy as np
import cv2


def _largest_cc(mask, min_px=60):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k) if stats[k, cv2.CC_STAT_AREA] >= min_px else None


def assess(pts, zf, dest, slot, H_exp, floor_z=0.012, margin=0.012, near=0.30):
    """pts: (H,W,3) base 좌표, zf: (H,W) 깊이(m). dest: dict(xyz[3], raw[3](w,d,top), yaw) 또는 None. slot: 놓은 위치 xy. H_exp: 예상 물체 높이(서 있을 때)."""
    valid = np.isfinite(pts).all(-1) & (zf > 0.1) & (zf < 2.0)
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    out = dict(status="missing", mask=None, P=None, info={})
    if dest is not None:
        cx, cy = dest["xyz"][0], dest["xyz"][1]; w, d, top = dest["raw"]; rad = 0.5 * math.hypot(w, d) + 0.02
        inside = valid & (np.hypot(X - cx, Y - cy) < rad)
        above = inside & (Z > top + margin)
        m = _largest_cc(above)
        if m is not None and np.hypot(np.median(X[m]) - slot[0], np.median(Y[m]) - slot[1]) < 0.5 * rad + 0.03:
            P = pts[m]; hmax = float(np.percentile(P[:, 2], 98) - top); ratio = hmax / max(H_exp, 0.02)
            out.update(mask=m, P=P, info=dict(h_above=round(hmax, 3), ratio=round(ratio, 2), n=int(m.sum())))
            out["status"] = "ok" if ratio >= 0.7 else ("toppled" if ratio <= 0.55 else "unknown")
            return out
    # 목적지 위에 없으면 테이블 위 근처에서 찾는다(목적지 풋프린트 밖)
    reg = valid & (Z > floor_z) & (np.hypot(X - slot[0], Y - slot[1]) < near)
    if dest is not None:
        cx, cy = dest["xyz"][0], dest["xyz"][1]; w, d, top = dest["raw"]; rad = 0.5 * math.hypot(w, d) + 0.03
        reg &= ~(np.hypot(X - cx, Y - cy) < rad)
    m = _largest_cc(reg)
    if m is not None:
        P = pts[m]; out.update(mask=m, P=P, status="off_dest", info=dict(n=int(m.sum()), xy=[round(float(np.median(P[:, 0])), 3), round(float(np.median(P[:, 1])), 3)]))
    return out


def make_det(a, obb_fn, margin_px=50):
    """assess 결과를 discover_object 의 grab() 검출 dict 형식으로: mask, w, P, box"""
    P = a["P"]; w = np.median(P, axis=0); w = np.array([w[0], w[1], 0.05])
    return dict(mask=a["mask"], w=w, P=P, box=obb_fn(P), margin=margin_px, conf=1.0, label="", s_t=None, s_x=None, s_p=None, probs={}, logits={})


if __name__ == "__main__":
    # 합성 시험: 접시(상단 높이 0.039, 반경 0.115) 위에 서 있는 물체 / 누운 물체 / 접시 밖 물체 / 없음
    H, W = 240, 320; ys, xs = np.mgrid[0:H, 0:W]
    X = (xs - W / 2) * 0.004 + 0.40; Y = (ys - H / 2) * 0.004 + 0.0; Z = np.zeros((H, W))
    dest = dict(xyz=[0.40, 0.0, 0.02], raw=[0.23, 0.23, 0.039], yaw=0.0)
    def scene(kind):
        Zs = Z.copy(); r = np.hypot(X - 0.40, Y - 0.0); Zs[r < 0.115] = 0.039           # 접시 윗면
        if kind == "standing": Zs[np.hypot(X - 0.40, Y) < 0.025] = 0.039 + 0.12
        if kind == "lying": Zs[(np.abs(X - 0.40) < 0.06) & (np.abs(Y) < 0.02)] = 0.039 + 0.03
        if kind == "off": Zs[(np.abs(X - 0.62) < 0.03) & (np.abs(Y - 0.0) < 0.03)] = 0.08
        return np.stack([X, Y, Zs], -1), np.full((H, W), 0.5)
    for kind, exp in (("standing", "ok"), ("lying", "toppled"), ("off", "off_dest"), ("none", "missing")):
        pts, zf = scene(kind); a = assess(pts, zf, dest, [0.40, 0.0], H_exp=0.12)
        print(f"{kind:9s} -> {a['status']:9s} {a['info']}", "OK" if a["status"] == exp else "<-- 기대 " + exp)
