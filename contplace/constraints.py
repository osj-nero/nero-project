"""연속 놓기 B단계: 안정 자세 생성 + 제약 명세 판정(사전 등록 v3 형식). 순수 numpy, 서버 불필요.
자세 R = 물체 로컬 -> 월드 회전(놓은 뒤). 면 법선은 로컬 축(face_axes)."""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import math, json, numpy as np
from scipy.spatial.transform import Rotation as Rot

FACES = {"윗면": [0, 0, 1], "바닥면": [0, 0, -1], "앞면": [0, -1, 0], "뒷면": [0, 1, 0], "왼면": [-1, 0, 0], "오른면": [1, 0, 0]}
UP = np.array([0.0, 0.0, 1.0])


def normal(R, face):
    return R.apply(np.array(FACES[face], float))


def ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))))


TOL = {"엄격": 10.0, "보통": 20.0, "느슨": 30.0}      # 2026-11-07 v3: LLM 은 각도 숫자 대신 단어를 고른다(대응은 사전 등록 고정)
REGIONS = ("가까운 쪽", "먼 쪽", "왼쪽", "오른쪽", "가운데")


def tol(c, key):
    v = c.get("tol", c.get(key))
    return TOL[v] if isinstance(v, str) else float(v)


def long_axis(dims):
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]][int(np.argmax(dims))]


def frame_dirs(center):
    """로봇 기준 방향: 앞뒤 = 로봇 베이스에서 목표면 중심으로 가는 수평 방향, 왼쪽 = 연직 x 앞뒤."""
    r = np.array([center[0], center[1], 0.0]); r /= (np.linalg.norm(r) or 1.0)
    return r, np.cross(UP, r)


def check(R, spec, manner="place", pos=None, dims=None, center=None, half=None):
    """spec = {"hard": [...], "manner": [...]} -> (ok, [어긴 조건]).
    up/vertical/not_down: 면 법선 조건(tol 은 엄격/보통/느슨 또는 도). yaw: 긴 축의 수평 방향이 로봇 기준 '앞뒤'/'좌우' 또는 deg(앞뒤 기준, 반시계)에서 tol 이내(축은 양방향 동일).
    region: 놓은 위치(pos)가 목표면(center, half=반폭) 의 지정 구역 안. pos/dims/center 가 없으면 해당 조건은 판정 보류(통과)."""
    bad = []
    for c in spec.get("hard", []):
        t = c["type"]
        if t == "up":
            if min(ang(normal(R, f), UP) for f in c["faces"]) > tol(c, "max_tilt_deg"): bad.append(c)
        elif t == "vertical":
            if abs(90.0 - ang(normal(R, c["face"]), UP)) > tol(c, "tol_deg"): bad.append(c)
        elif t == "not_down":
            if ang(normal(R, c["face"]), -UP) < float(c.get("min_from_down_deg", 90.0)): bad.append(c)
        elif t == "yaw":
            if dims is None or center is None: continue
            fwd, left = frame_dirs(center); v = R.apply(np.array(long_axis(dims), float)); v[2] = 0.0
            if np.linalg.norm(v) < 0.3: bad.append(c); continue        # 긴 축이 거의 연직이면 수평 방향이 정의되지 않는다
            d = c.get("dir", "앞뒤"); ref = math.radians(float(d)) if not isinstance(d, str) else (0.0 if d == "앞뒤" else math.pi / 2)
            a = math.atan2(float(np.dot(v, left)), float(np.dot(v, fwd))) - ref
            a = abs((math.degrees(a) + 90.0) % 180.0 - 90.0)      # 축은 방향 없음(180도 주기)
            if a > tol(c, "tol_deg"): bad.append(c)
        elif t == "region":
            if pos is None or center is None: continue
            fwd, left = frame_dirs(center); d = np.array(pos[:2] + [0.0]) - np.array(list(center[:2]) + [0.0])
            x, y = float(np.dot(d, fwd)), float(np.dot(d, left)); h = half or 0.12; w = c["where"]
            ok = {"가까운 쪽": x < 0, "먼 쪽": x > 0, "왼쪽": y > 0, "오른쪽": y < 0, "가운데": abs(x) < h / 3 and abs(y) < h / 3}[w]
            if not ok: bad.append(c)
        else:
            raise ValueError(t)
    if manner not in spec.get("manner", ["place"]): bad.append({"type": "manner", "got": manner})
    return (not bad), bad


def box_stable(yaws_deg=(0,)):
    """직육면체 안정 자세: 바닥에 닿을 면 6개 x yaw. 반환 [(이름, R)] -- R 은 '지금 자세' 기준 상대 회전 후 월드 yaw."""
    base = {"윗면이 위": Rot.identity(), "바닥면이 위": Rot.from_euler("x", 180, degrees=True),
            "뒷면이 위": Rot.from_euler("x", 90, degrees=True), "앞면이 위": Rot.from_euler("x", -90, degrees=True),
            "왼면이 위": Rot.from_euler("y", 90, degrees=True), "오른면이 위": Rot.from_euler("y", -90, degrees=True)}
    out = []
    for n, R0 in base.items():
        for y in yaws_deg:
            out.append((n, Rot.from_euler("z", y, degrees=True) * R0))
    return out


GRID_DIMS = {"plate": [0.12, 0.015, 0.10], "box": [0.07, 0.10, 0.15], "bar": [0.20, 0.03, 0.03], "block": [0.05, 0.05, 0.10], "eggbox": [0.15, 0.10, 0.07], "mug": [0.12, 0.09, 0.12], "bottle": [0.06, 0.06, 0.14], "ruler": [0.20, 0.006, 0.03]}
CENTER = [-0.46, 0.0]


def eval_grid(dims=(0.12, 0.015, 0.10)):
    """LLM 정확도 판정 격자: 안정 자세 6개 x 기울기 0/10/20/30도 x yaw 0~150(30도) x 위치 5곳 x 방식 2."""
    pts = []
    spots = {"c": [-0.46, 0.0], "near": [-0.40, 0.0], "far": [-0.52, 0.0], "L": [-0.46, -0.06], "R": [-0.46, 0.06]}
    for n, R in box_stable():
        for tilt in (0, 10, 20, 30):
            for ax in ("x", "y"):
                for yw in range(0, 180, 30):
                    Rt = Rot.from_euler("z", yw, degrees=True) * Rot.from_euler(ax, tilt, degrees=True) * R
                    for sk, p in spots.items():
                        for m in ("place", "topple"):
                            pts.append((f"{n}|{ax}{tilt}|y{yw}|{sk}|{m}", Rt, m, p))
    return pts


def admissible(spec, dims=(0.12, 0.015, 0.10)):
    return {k for k, R, m, p in eval_grid(dims) if check(R, spec, m, pos=p, dims=list(dims), center=CENTER)[0]}


def compare(pred, gt, dims=(0.12, 0.015, 0.10)):
    """허용 집합 비교: 위반 = gt 밖인데 허용, 놓침 = gt 안인데 금지."""
    P, G = admissible(pred, dims), admissible(gt, dims)
    tp = len(P & G)
    return dict(n_pred=len(P), n_gt=len(G), viol=len(P - G), miss=len(G - P), exact=P == G,
                prec=(tp / len(P) if P else None), rec=(tp / len(G) if G else None))


if __name__ == "__main__":      # 자체 점검: v3 정답이 v2 이산 정답과 같은 안정 자세를 허용하는지
    P = json.load(open("" + NERO_ROOT + "/rq1/prereg/RQ1_PREREG_v3_draft.json"))
    V2 = {it["id"]: it for it in json.load(open("" + NERO_ROOT + "/rq1/prereg/RQ1_PREREG_v2.json"))["items"]}
    MAP = {"U": "윗면이 위", "F": "바닥면이 위", "Lx+": "앞면이 위", "Lx-": "뒷면이 위", "Ey+": "왼면이 위", "Ey-": "오른면이 위"}
    st = dict(box_stable())
    bad = 0
    for it in P["items"]:
        if it["id"] not in V2: continue
        v3 = {n for n, R in st.items() if check(R, it["gt"])[0]}
        v2 = {MAP[o] for o in V2[it["id"]]["gt_orient"]}
        if v3 != v2: bad += 1; print("불일치", it["id"], "v2", sorted(v2), "v3", sorted(v3))
    print("v2->v3 정답 변환 점검: 불일치", bad)
