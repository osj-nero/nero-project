#!/usr/bin/env python3
"""layout_c10.py -- C10(2026-10-29, 기본 꺼짐: 환경변수 NERO_LAYOUT_MINGAP=<mm> 설정 시에만 scene_sort.rand_layout 래퍼가 사용): 목적지 용기(레이아웃 키 순서 앞 4개) 각각의 스폰 표면 간격(xy 볼록껍질)이 다른 모든 물체와 min_gap 이상이 아니면 재추첨.
시도 0 = 원 rand_layout(seed) 그대로(만족하면 원 결과를 그대로 반환=바이트 동일). 재추첨 a>=1 은 결정적 파생 시드 (seed*1009+a)%3000000+1000000. 안착 후 간격은 여기서 못 잰다(별도 안착 후 검증 B).
메시 볼록껍질은 layout_c10_hulls.json(모델명 -> 3D 볼록껍질 꼭짓점)을 읽는다(pxr 불필요); 없는 모델이 있으면 검사하지 않고 거부(오류)한다."""
import os, sys, json, math
import numpy as np
from scipy.spatial import ConvexHull
HERE = os.path.dirname(os.path.abspath(__file__)); HULLS = os.path.join(HERE, "layout_c10_hulls.json")
def _poly(V, x, y, yawdeg):
    c, s = math.cos(math.radians(yawdeg)), math.sin(math.radians(yawdeg)); R = np.array([[c, s], [-s, c]]); P = np.asarray(V)[:, :2] @ R + np.array([x, y]); return P[ConvexHull(P).vertices]
def _gap(P, Q):
    def pe(P, Q):
        A = Q; B = np.roll(Q, -1, axis=0); AB = B - A; t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-18)[None], 0, 1)
        return np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=-1).min()
    def inn(P, Q):
        A = Q; B = np.roll(Q, -1, axis=0); cr = (B - A)[None, :, 0] * (P[:, None, 1] - A[None, :, 1]) - (B - A)[None, :, 1] * (P[:, None, 0] - A[None, :, 0]); return bool((np.all(cr >= 0, 1) | np.all(cr <= 0, 1)).any())
    return 0.0 if inn(P, Q) or inn(Q, P) else float(min(pe(P, Q), pe(Q, P)))
def min_container_gap_mm(L, A, n_cont=4):
    """L = rand_layout 결과(dict 키 순서: 용기 4개, 물체...). A = scene_sort.A(키 -> 모델명). 반환 용기 4종 각각의 최근접 이웃 간격 mm 목록."""
    H = json.load(open(HULLS)); ks = list(L); polys = []
    for k in ks:
        m = A.get(k, k)
        if m not in H: raise RuntimeError(f"C10: 메시 껍질 없음 {m}")
        polys.append(_poly(H[m], L[k][0], L[k][1], L[k][2]))
    return [1000 * min(_gap(polys[i], polys[j]) for j in range(len(ks)) if j != i) for i in range(n_cont)]
def derived_seed(seed, att):      # 2026-10-29: 파생 시드를 1,000,000~3,999,999 로 접어 봉인 범위(>=4,000,000) 밖에 둔다
    return (seed * 1009 + att) % 3000000 + 1000000
def rand_layout_c10(orig, A, min_mm, seed, *a, max_attempts=400, **k):
    for att in range(max_attempts):
        try: L = orig(seed if att == 0 else derived_seed(seed, att), *a, **k)
        except RuntimeError: continue
        if min(min_container_gap_mm(L, A)) >= min_mm:
            if att: print(f"[C10] 재추첨 {att}회", file=sys.stderr)
            return L
    raise RuntimeError(f"C10: {max_attempts}회 안에 간격 {min_mm} mm 만족 레이아웃 없음")
