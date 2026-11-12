"""agn_detect.py -- 이름 없는(class-agnostic) 물체 탐지 (2026-09-30).
깊이에서 바닥 위로 솟은 덩어리(이미지 공간 연결요소)를 구한 뒤, 각 덩어리를 월드 top-down 높이맵에서 다시 나눠 "큰 조각"만 물체로 본다.
(앞뒤로 붙은 물체는 이미지에서는 이어져 보이지만 월드에서는 틈이 있고, 한 물체가 갈라진 작은 파편은 버린다 -- gso_overlap_test3.py 에서 하이브리드가 정확 분리 83%)
multi_detect() 와 같은 반환 형식: [(mask HxW bool, bbox_norm [x1,y1,x2,y2], conf), ...]
사용 조건: geom = discover_boxes.capture_geom() 결과(zf=깊이, pts_base=base_link 좌표). 바닥이 base_link z≈0 이라고 가정한다(floor_z 위를 물체로 봄)."""
import os
import numpy as np
from scipy import ndimage

RES = 0.003           # top-down 격자 해상도(m)
MIN_CELLS = 40        # 큰 조각 기준(약 3.6cm^2): 이보다 작은 조각은 한 물체가 갈라진 파편으로 보고 버린다
ROBOT_R = 0.17        # 로봇 기저(xy 원점) 주변은 로봇 몸체로 보고 제외
GRIPPER_ROW = 0.72    # 화면 아래쪽(이 비율 아래) = 카메라에 붙은 그리퍼 손가락
R_MIN, R_MAX = 0.18, 0.85


def _hull_mask(cells, shape):
    """cells: (N,2) 격자 좌표 -> 볼록 껍질 안쪽 격자 불리언 마스크. 퇴화(점 3개 미만/일직선)면 None."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(cells, float)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    eq = hull.equations                                      # a*x + b*y + c <= 0 이 안쪽
    x0, y0 = np.floor(pts.min(0)).astype(int), np.floor(pts.min(0)).astype(int)
    lo = np.floor(pts.min(0)).astype(int); hi = np.ceil(pts.max(0)).astype(int)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    inside = np.all(eq[:, 0, None, None] * gx + eq[:, 1, None, None] * gy + eq[:, 2, None, None] <= 1e-6, axis=0)
    m = np.zeros(shape, bool)
    xs, ys = gx[inside], gy[inside]
    ok = (xs >= 0) & (xs < shape[0]) & (ys >= 0) & (ys < shape[1])
    m[xs[ok], ys[ok]] = True
    return m



def agnostic_masks(geom, floor_z=0.012, exclude_masks=(), max_count=8, top_z=0.35, verbose=False):
    H, W = geom["H"], geom["W"]
    zf = geom["zf"]; Pb = geom["pts_base"][..., :3]
    z = Pb[..., 2]
    valid = (zf > 0.1) & (zf < 2.0)
    m0 = valid & (z > floor_z) & (z < top_z)
    _fd = float(os.environ.get("NERO_FINGER_DEPTH", "0"))      # 2026-10-14(기본 꺼짐, NERO_LEAN 이 0.25 로 켬): 화면 아래 28% 를 통째로 지우지 않고, 카메라에 붙은 그리퍼 손가락만(깊이 _fd m 미만 = 손가락 8~12cm, 물체/바닥은 40cm 이상) 팽창해서 지운다 -- 로봇 가까이 앞에 놓인 물체(앞의 라쿤)가 아래쪽 영역에 걸려 등록되지 않던 문제
    if _fd > 0:
        m0 &= ~ndimage.binary_dilation((zf > 0) & (zf < _fd), np.ones((9, 9), bool))
    else:
        m0[int(GRIPPER_ROW * H):, :] = False
    rr = np.hypot(Pb[..., 0], Pb[..., 1]); m0 &= rr > ROBOT_R
    for em in exclude_masks:                                   # 테이블 등 이름 있는 대상(기존 검출기 마스크)은 물체 후보에서 뺀다
        if em is not None and em.any():
            m0 &= ~ndimage.binary_dilation(em, np.ones((15, 15), bool))
    m0 = ndimage.binary_opening(m0, np.ones((5, 5), bool))
    lab, nl = ndimage.label(m0)
    if os.environ.get("NERO_AGN_DEPTHCUT", "0") == "1":      # 2026-10-29(기본 꺼짐): 크기 상한(면적비/최대 변 0.40 m)에 걸려 어차피 통째로 폐기될 연결요소만, 이웃 픽셀 3D 거리 > J 인 경계를 끊어 다시 라벨링한다(맞닿은 두 용기의 림 높이 단차). 폐기되지 않던 연결요소는 건드리지 않는다.
        _J = float(os.environ.get("NERO_AGN_DEPTHCUT_J", "0.02")); _mf = float(os.environ.get("NERO_AGN_MAX_FRAC", "0.25")); _nx = nl + 1; _jx = _jy = None
        for q in range(1, nl + 1):
            comp = lab == q; area = int(comp.sum())
            if area < 300:
                continue
            _ys, _xs = np.where(comp); _Pq = Pb[_ys, _xs]; _ext = _Pq[:, :2].max(0) - _Pq[:, :2].min(0)
            if not (area > _mf * H * W or _ext.max() > 0.40):
                continue
            if _jx is None:
                _jx = np.zeros((H, W), bool); _jy = np.zeros((H, W), bool); _dx = np.linalg.norm(Pb[:, 1:] - Pb[:, :-1], axis=2) > _J; _dy = np.linalg.norm(Pb[1:] - Pb[:-1], axis=2) > _J
                _jx[:, :-1] |= _dx; _jx[:, 1:] |= _dx; _jy[:-1] |= _dy; _jy[1:] |= _dy
            comp2 = comp & ~ndimage.binary_dilation((_jx | _jy) & comp, np.ones((3, 3), bool)); sl, sn = ndimage.label(comp2); lab[comp] = 0
            for t in range(1, sn + 1):
                if int((sl == t).sum()) >= 300:
                    lab[sl == t] = _nx; _nx += 1
        nl = _nx - 1
    out = []
    for q in range(1, nl + 1):
        comp = lab == q
        area = int(comp.sum())
        _sb = os.environ.get("NERO_AGN_SPLIT_BIG", "0") == "1"      # 2026-10-28(기본 0): 큰 덩어리(화면 비율/폭 초과)를 버리지 않고 월드 재분할로 나눈 뒤 조각별로 크기·반경을 검사(이웃과 이어 붙은 용기 복구용)
        if area < 300 or (area > float(os.environ.get("NERO_AGN_MAX_FRAC", "0.25")) * H * W and not _sb):      # 2026-10-28: 상한 비율을 파라미터로(기본 0.25 = 기존; 가까이 놓인 큰 용기가 화면 25% 를 넘어 통째로 버려지는 경우용)
            continue
        ys, xs = np.where(comp); P = Pb[ys, xs]
        ext = P[:, :2].max(0) - P[:, :2].min(0)
        ctr = np.median(P[:, :2], axis=0); rc = float(np.hypot(*ctr))
        _isbig = _sb and (ext.max() > 0.40 or area > float(os.environ.get("NERO_AGN_MAX_FRAC", "0.25")) * H * W)
        if (ext.max() > 0.40 and not _sb) or ext.max() < 0.015 or (not _isbig and not (R_MIN <= rc <= R_MAX)):
            continue
        # 월드 top-down 에서 다시 나누기
        x0, y0 = P[:, 0].min(), P[:, 1].min()
        ix = ((P[:, 0] - x0) / RES).astype(int); iy = ((P[:, 1] - y0) / RES).astype(int)
        g = np.zeros((ix.max() + 1, iy.max() + 1), bool); g[ix, iy] = True
        g = ndimage.binary_opening(g, np.ones((3, 3), bool))
        gl, gn = ndimage.label(g)
        zg = np.zeros(g.shape, np.float32); np.maximum.at(zg, (ix, iy), P[:, 2].astype(np.float32))
        big = [t for t in range(1, gn + 1) if int((gl == t).sum()) >= MIN_CELLS]
        if not big:
            continue
        hmax = {t: float(zg[gl == t].max()) for t in big}
        par = {t: t for t in big}
        def _f(t):
            while par[t] != t:
                par[t] = par[par[t]]; t = par[t]
            return t
        for t in big:                                              # 조각 병합: (R1) 거의 붙은 조각(<6mm) (R2) 납작한 조각(<2.5cm)이 3cm 안에 있는 다른 조각 -- 줄자의 테이프 띠처럼 한 물체의 부속
            dist = ndimage.distance_transform_edt(gl != t) * RES
            for u in big:
                if u <= t:
                    continue
                dmin = float(dist[gl == u].min())
                if dmin < 0.006 or (dmin < 0.03 and min(hmax[t], hmax[u]) < 0.025):
                    par[_f(u)] = _f(t)
        # R3 (2026-09-30) 볼록 껍질이 서로 겹치는 조각은 한 물체의 부분으로 합친다(테이프 고리의 양쪽 호: 각 호의 껍질이 구멍 위에서 겹침). 겹침 면적 >= 약 3cm^2 이고
        # 작은 쪽 껍질의 10% 이상일 때만 -- 맞닿기만 한 별개 물체는 껍질이 겹치지 않아 합쳐지지 않는다. 합친 크기가 40cm 를 넘으면 합치지 않는다.
        if len(big) >= 2 and os.environ.get("NERO_AGN_HULL", "1") == "1":
            hm = {}
            for t in big:
                cs = np.argwhere(gl == t)
                hm[t] = _hull_mask(cs, gl.shape)
            for i_, t in enumerate(big):
                for u in big[i_ + 1:]:
                    if _f(t) == _f(u) or hm[t] is None or hm[u] is None:
                        continue
                    ov = int((hm[t] & hm[u]).sum()); small = min(int(hm[t].sum()), int(hm[u].sum()))
                    if ov >= 33 and ov >= float(os.environ.get("NERO_AGN_HULL_RATIO", "0.10")) * small:
                        cu = np.argwhere((gl == t) | (gl == u))
                        if (cu.max(0) - cu.min(0)).max() * RES <= 0.40:
                            par[_f(u)] = _f(t)
        if os.environ.get("NERO_AGN_FINE", "0") == "1" and len(big) >= 2:      # [S2 시제품] 미세 격자(FINE_RES) 연결성으로 같은 물체로 이어지는 조각 그룹만 추가 병합. 병합 안 된 그룹의 마스크는 기존과 동일.
            _fr = float(os.environ.get("NERO_AGN_FINE_RES", "0.002")); _ff = float(os.environ.get("NERO_AGN_FINE_FRAC", "0.30"))
            i2 = ((P[:, 0] - x0) / _fr).astype(int); j2 = ((P[:, 1] - y0) / _fr).astype(int)
            g2 = np.zeros((i2.max() + 1, j2.max() + 1), bool); g2[i2, j2] = True; g2 = ndimage.binary_opening(g2, np.ones((3, 3), bool)); l2, n2 = ndimage.label(g2)
            _gr = {}
            for t in big: _gr.setdefault(_f(t), []).append(t)
            _roots = list(_gr); _lab_of = {}
            for r_ in _roots:
                _sel = np.isin(gl[ix, iy], _gr[r_]); _l = l2[i2[_sel], j2[_sel]]; _l = _l[_l > 0]
                _lab_of[r_] = (np.bincount(_l, minlength=n2 + 1), int(_sel.sum()))
            for a_ in range(len(_roots)):
                for b_ in range(a_ + 1, len(_roots)):
                    ra, rb = _roots[a_], _roots[b_]
                    if _f(ra) == _f(rb): continue
                    ca, na = _lab_of[ra]; cb, nb_ = _lab_of[rb]; sm = ca + cb; sm[0] = 0; k = int(np.argmax(sm))
                    if k and ca[k] >= _ff * na and cb[k] >= _ff * nb_:
                        cu = np.argwhere(np.isin(gl, _gr[ra] + _gr[rb]))
                        if (cu.max(0) - cu.min(0)).max() * RES <= 0.40: par[_f(rb)] = _f(ra)
        if os.environ.get("NERO_AGN_RAW", "0") == "1":      # 병합 규칙 전부 끄고 날 조각을 그대로 돌려준다(SAM 판정 시험용)
            par = {t: t for t in big}
        groups = {}
        for t in big:
            groups.setdefault(_f(t), []).append(t)
        for members in groups.values():
            sel = np.isin(gl[ix, iy], members)
            mask = np.zeros((H, W), bool); mask[ys[sel], xs[sel]] = True
            if _isbig:      # 큰 덩어리에서 나온 조각은 개별 크기(<=0.40 m)·반경 범위를 만족해야 한다
                Pg = P[sel]; eg = Pg[:, :2].max(0) - Pg[:, :2].min(0); cg = np.median(Pg[:, :2], axis=0)
                if eg.max() > 0.40 or eg.max() < 0.015 or not (R_MIN <= float(np.hypot(*cg)) <= R_MAX): continue
            yy, xx = np.where(mask)
            out.append((mask, [xx.min() / W, yy.min() / H, xx.max() / W, yy.max() / H], 0.9, int(mask.sum())))
    if os.environ.get("NERO_AGN_ENCLOSED", "0") == "1" and len(out) >= 2:      # [2026-10-28 S5, 기본 꺼짐] 둘러싸인 바닥 높이 섬 병합(병합 전용): 마스크 B 의 점 >=ENC_FRAC 가 다른 마스크 A 의 xy 볼록 껍질 안, B 최대 높이 <= ENC_HMAX, A 최대 높이 > B 최대 높이이면 합침(용기 안쪽 바닥 원판이 전경 임계 경계에서 고리와 갈라지는 경우)
        from scipy.spatial import Delaunay
        _eh = float(os.environ.get("NERO_AGN_ENC_HMAX", "0.03")); _ef = float(os.environ.get("NERO_AGN_ENC_FRAC", "0.9")); _ps = []
        for _m in out:
            _yy, _xx = np.where(_m[0]); _sl = zf[_yy, _xx] > 0.1; _ps.append(Pb[_yy[_sl], _xx[_sl]])
        _par = list(range(len(out)))
        def _g(i):
            while _par[i] != i: i = _par[i]
            return i
        for _a in range(len(out)):
            if len(_ps[_a]) < 50: continue
            try: _tri = Delaunay(_ps[_a][::max(1, len(_ps[_a]) // 2000), :2])
            except Exception: continue
            for _b in range(len(out)):
                if _a == _b or len(_ps[_b]) < 20 or _g(_b) != _b: continue
                if _ps[_b][:, 2].max() > _eh or _ps[_a][:, 2].max() <= _ps[_b][:, 2].max(): continue
                if (_tri.find_simplex(_ps[_b][::max(1, len(_ps[_b]) // 500), :2]) >= 0).mean() >= _ef: _par[_b] = _g(_a)
        if any(_g(i) != i for i in range(len(out))):
            _gr = {}
            for i in range(len(out)): _gr.setdefault(_g(i), []).append(i)
            _new = []
            for _v in _gr.values():
                if len(_v) == 1: _new.append(out[_v[0]]); continue
                _mk = np.zeros_like(out[_v[0]][0])
                for _t in _v: _mk |= out[_t][0]
                _yy, _xx = np.where(_mk); _new.append((_mk, [_xx.min() / W, _yy.min() / H, _xx.max() / W, _yy.max() / H], 0.9, int(_mk.sum())))
            out = _new
    out.sort(key=lambda t: -t[3])
    if verbose:
        print(f"[agn] 덩어리 {nl} -> 물체 후보 {len(out)}")
    return [(m, b, c) for m, b, c, _ in out[:max_count]]
