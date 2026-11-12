"""기울어진 원통의 3D 축 추정 (2026-09-14, 2026-09-14 반경 보정 수정).

기존 cyl_fit 은 축이 수직이라고 가정하고 XY 평면에 투영해 circle_fit 했다
(axis=[cx,cy] 2D, base_z/top_z 는 z축 범위). 컵이 기울면
  · 합성 side 후보의 폐합축 b = cross([0,0,1], ah) 가 컵 축과 직교하지 않고
  · aspect = z범위 / XY폭 이 기울기에 따라 출렁인다
    (실측: 같은 컵 33° 기울임에서 1.386 / 0.953 / 1.411 로 요동, 축 정렬 시 2.52 고정)

여기서는 robust PCA 로 3D 축을 뽑아 axis3/centre3/span3/radius3/aspect3 를
**추가**한다. 기존 필드(axis/radius/base_z/top_z/centre)는 건드리지 않으므로
이 값을 안 읽는 쪽(exclude_aabb 의 on_axis 등)은 동작이 그대로다.

버그 수정 이력: 최초 구현은 반경을 "PCA 중심에서 각 점까지 거리의 중앙값"으로
구했는데, 카메라는 원통의 앞면 호(대략 절반)만 본다 — 그 호의 평균(PCA 중심)은
원의 진짜 중심이 아니라 카메라 쪽으로 쏠린다. 쏠린 중심에서 거리를 재면 정면
근처 점들이 가까워 보여 반경이 과소평가된다(실측: 진짜 반경 3.6cm를 2.6cm로
과소평가, aspect3 가 1.79 대신 2.52로 부풀려짐 — 2026-09-14 사용자 지적으로
발견). 레거시 2D `circle_fit`(cgn_server.py/two_cgn.py)은 이미 이 문제를 알고
있었다(Taubin 대수적 원피팅 + 카메라 반대방향 0.35r 보정) — 여기서는 축에
수직인 평면에 투영한 뒤 같은 방식을 그대로 적용해 3D로 일반화한다.
"""
import numpy as np


def _taubin2d(x, y):
    xm, ym = x.mean(), y.mean()
    xr, yr = x - xm, y - ym
    z = xr * xr + yr * yr
    Z = np.c_[z - z.mean(), xr, yr]
    _, _, V = np.linalg.svd(Z, full_matrices=False)
    A = V[-1]
    return xm - A[1] / (2 * A[0]), ym - A[2] / (2 * A[0])


def _circle_fit2d(x, y):
    """레거시 circle_fit 과 동일한 반복 강건 원피팅 (2D). cgn_server.py/two_cgn.py
    의 circle_fit() 과 알고리즘 동일 — 축에 수직인 평면에 적용하기 위해 여기 복제."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cx, cy = _taubin2d(x, y)
    for _ in range(3):
        d = np.hypot(x - cx, y - cy)
        r = np.median(d)
        keep = np.abs(d - r) < np.percentile(np.abs(d - r), 75) + 1e-6
        if keep.sum() < 8:
            break
        cx, cy = _taubin2d(x[keep], y[keep])
        x, y = x[keep], y[keep]
    return cx, cy


def fit_axis3(P, cam_xyz=None, trim=80.0, iters=3):
    """P: (N,3) world 좌표 세그먼트 포인트, cam_xyz: 카메라 world 위치(선택)
    -> 원통 축 dict 또는 None.

    1) 손잡이/노이즈가 방향추정을 끌어당기므로, 먼저 축 기준 반경 상위
       (100-trim)% 를 버리고 3회 재추정해 축 '방향' u 를 강건하게 뽑는다
       (머그 손잡이가 정확히 이 outlier 다).
    2) u 에 수직인 평면에 전체 점(트리밍 안 한 원본)을 투영하고, 그 평면에서
       레거시와 동일한 Taubin 원피팅 + 카메라 반대방향 보정으로 반경/중심을
       구한다 — 단순 PCA 중심+거리중앙값은 카메라가 보는 앞면 호에 편향되어
       반경을 과소평가한다(위 모듈 docstring 참고).
    """
    P = np.asarray(P, dtype=float)
    if len(P) < 50:
        return None

    def _pca(Q):
        c = Q.mean(0)
        _, S, V = np.linalg.svd(Q - c, full_matrices=False)
        return c, V, S

    # --- 축 방향 u: 반복 트리밍 PCA (기존과 동일, outlier 에 강건) ---
    c0, V, S = _pca(P)
    for _ in range(iters):
        u = V[0] / (np.linalg.norm(V[0]) + 1e-12)
        t = (P - c0) @ u
        rad = np.linalg.norm((P - c0) - np.outer(t, u), axis=1)
        keep = rad <= np.percentile(rad, trim)
        if keep.sum() < 50:
            break
        c0, V, S = _pca(P[keep])

    u = V[0] / (np.linalg.norm(V[0]) + 1e-12)
    if u[2] < 0:                       # 축은 항상 위쪽을 향하게
        u = -u
    s2, s3 = float(S[1]), float(S[2])
    roundness = float(min(s2, s3) / max(s2, s3)) if max(s2, s3) > 1e-9 else 0.0

    # --- 반경/중심: u 에 수직인 평면에 투영 후 2D 원피팅 (전체 점 사용) ---
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(ref @ u)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - float(ref @ u) * u; e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(u, e1);          e2 /= (np.linalg.norm(e2) + 1e-12)
    rel = P - c0
    x2, y2 = rel @ e1, rel @ e2
    cx2, cy2 = _circle_fit2d(x2, y2)
    # 반경은 "보정 전" 중심 기준 (레거시와 동일 순서: r 을 먼저 구하고 중심만 보정)
    radius = float(np.clip(np.median(np.hypot(x2 - cx2, y2 - cy2)) * 1.1, 0.015, 0.06))
    if cam_xyz is not None:
        cam_rel = np.asarray(cam_xyz, float) - c0
        camx, camy = float(cam_rel @ e1), float(cam_rel @ e2)
        v = np.array([cx2 - camx, cy2 - camy]); nv = np.linalg.norm(v)
        if nv > 1e-3:
            cx2, cy2 = (np.array([cx2, cy2]) + (v / nv) * radius * 0.35).tolist()
    centre3 = c0 + cx2 * e1 + cy2 * e2

    # --- span: 축 방향 스프레드. 평면 내 중심 이동으로는 안 변함(순수 이동), u
    # 방향 성분만 영향 -> centre3 기준으로 다시 계산해도 무방(투영값 자체 불변). ---
    t = (P - centre3) @ u
    span = float(np.percentile(t, 97) - np.percentile(t, 3))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(u[2]))))))

    return dict(axis3=[float(x) for x in u],
                centre3=[float(x) for x in centre3],
                span3=span, radius3=radius,
                aspect3=float(span / (2.0 * max(radius, 1e-3))),
                tilt_deg=tilt, roundness=roundness, n_fit=int(len(P)))


def usable(a3, min_tilt=10.0, min_round=0.6, min_aspect=1.2):
    """이 축 추정을 실제 후보 합성에 써도 되는지. 보수적으로 판단한다.

    · tilt 가 작으면 기존 수직 가정과 사실상 같으므로 굳이 바꾸지 않는다
      (기존 경로가 실측으로 검증돼 있음 — 불필요한 회귀 위험을 안 만든다).
    · roundness 가 낮으면 원통이 아니다 (판/박스) -> 축 의미 없음.
    · 너무 뭉툭하면 축 방향 자체가 불안정하다.
    """
    if not a3:
        return False
    return (a3["tilt_deg"] >= min_tilt
            and a3["roundness"] >= min_round
            and a3["aspect3"] >= min_aspect)


def side_frame(u, az_deg):
    """축 u 에 수직인 평면에서 방위각 az 의 접근방향 ah 와 폐합축 b 를 만든다.

    수직축(u=[0,0,1])을 넣으면 기존 식
        ah = [cos, sin, 0],  b = cross([0,0,1], ah)
    와 동일한 결과가 나온다 (일반화이지 대체가 아님).
    """
    u = np.asarray(u, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)
    # 기저는 반드시 "월드 +x 를 축에 수직인 평면에 투영한 것"을 e1 으로 잡는다.
    # 그래야 u=[0,0,1] 에서 e1=[1,0,0], e2=[0,1,0] 이 되어 az 가 기존 코드와
    # 같은 월드 방위각을 가리킨다 — cost 의 align 항과 _score_near(az) 조회가
    # 기존 의미 그대로 유지된다. (임의 기저를 쓰면 az 라벨이 회전해 점수가 어긋남.)
    ex = np.array([1.0, 0.0, 0.0])
    if abs(float(u @ ex)) > 0.95:          # 축이 거의 +x -> 대신 +y 로 기준
        ex = np.array([0.0, 1.0, 0.0])
    e1 = ex - float(ex @ u) * u; e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(u, e1);        e2 /= (np.linalg.norm(e2) + 1e-12)
    az = np.radians(az_deg)
    ah = np.cos(az) * e1 + np.sin(az) * e2      # 접근 = 축에 수직
    ah /= (np.linalg.norm(ah) + 1e-12)
    b = np.cross(u, ah)                          # 폐합축 = 축·접근 모두에 수직
    b /= (np.linalg.norm(b) + 1e-12)
    return ah, b
