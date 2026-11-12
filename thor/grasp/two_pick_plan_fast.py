"""Stage B: rank CGN grasps (already in gripper_flange convention) + IK, per object.
Elongated -> CGN side grasp re-targeted to the fitted cylinder axis at mid-height.
Compact  -> CGN top-down grasp by score - w*theta^4 - w*d_com.
Writes <name>_pick.json (quat, approach, flange, pre, standoff, grasp_sol, pre_sol) for step6.

FAST IK (rank3 pattern): geometric prefilter (0 IK) -> cost sort -> 120ms IK on top-K only,
first standoff only, second standoff as fallback.
"""
import json, sys, os, time
import numpy as np
try:            # 2026-10-01 상주 서비스(ROS 없는 cgn_venv)에서도 돌도록 ROS 임포트를 선택적으로: cuRobo IK 만 쓸 땐 필요 없다(MoveIt 폴백 때만 필요)
    import rclpy
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.msg import RobotState
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from sensor_msgs.msg import JointState
    _HAVE_ROS = True
except Exception:
    _HAVE_ROS = False

NAMES = sys.argv[1:] or ["bottle", "box"]
D_TCP = 0.1358
# 2026-09-29d 사용자 지시: 파지를 1cm 더 깊게 -- 플랜지를 접근축 방향으로 DEEPEN 만큼 더 밀어 넣는다(contact = flange + D_TCP*a 는 실제 TCP 라 그만큼 깊은 접촉점이 기록되어 place 와도 일관).
DEEPEN = float(os.environ.get("NERO_GRASP_DEEPEN", "0.01"))
HOLD_W = float(os.environ.get("NERO_HOLD_W", "4.0"))     # place 를 고려한 잡는 높이 벌점 (0=끔)
HOLD_H0 = float(os.environ.get("NERO_HOLD_H0", "0.07"))
PLACE_W = float(os.environ.get("NERO_PLACE_W", "0.6"))          # place 도달성 모델 반영 가중치 (0=끔)
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14),
       (-2.75, 2.75), (-0.73, 0.95), (-1.5707963, 1.5707963)]


def min_margin(cfg):
    return float(min(min(v - lo, hi - v) for v, (lo, hi) in zip(cfg, LIM)))


def wrist_sing(cfg):
    """이 팔은 j6->0 에서 손목 특이점 (j5 ∥ j7). |j6| 이 작을수록 나쁨. 0~1 페널티."""
    return float(max(0.0, 0.20 - abs(cfg[5])))


# NERO_MIN_MARGIN_FLOOR (2026-09-17 추가, 사용자 실측): _rerank 의 min_margin 페널티는
# "부드러운" cost 감점이라(-0.8*max(0,0.35-mm)), 도달 가능한 후보 자체가 적으면(예: bowl
# 3개) 관절 한계에 거의 붙은(mm≈0, 심하면 음수 -- fik 가 약간의 여유로 통과시킨 경우) 해도
# 다른 후보가 없어서 그냥 채택돼버림. 실물에서 이런 해를 실행하면 그 관절이 한계 근처에서
# 컨트롤러가 못 붙잡고 "빙글 도는" 현상 실측(bowl, j1=2.71 vs 한계 2.70526). 여기서는
# 소프트 감점과 별개로 이 문턱 미만이면 후보 자체에서 하드 제외한다 -- 대안이 없으면
# "no candidate" 로 정직하게 실패 처리(엉뚱한 관절한계 근접 해를 실행하는 것보다 낫다).
MIN_MARGIN_FLOOR = float(os.environ.get("NERO_MIN_MARGIN_FLOOR", "0.05"))


# 2026-09-29 이웃 물체 충돌 필터 (씬 점군 기반). cgn_server.py 가 {name}_obstacles.npy 에
# target 을 뺀 씬 점유 복셀(world 좌표)을 저장하면, 후보별로 "그리퍼가 pre->grasp 로 들어오는
# 통로(sweep)"에 장애물 복셀이 몇 개 걸리는지 세서 HIT_MAX 이상이면 후보에서 제외한다.
# 그리퍼 형상은 nero_gripper_d435.urdf 메시 실측 (gripper_base 프레임: z=접근, y=닫힘축, TCP z=0.1358).
NBR_HIT_MAX = int(os.environ.get("NERO_NBR_HIT_MAX", "2"))
# 2026-09-29 접근 중 그리퍼-타깃 비접촉: 그리퍼는 완전 개방(손가락 한쪽 0.05)으로 pre->grasp 를
# 들어오므로, 그 스윕(손가락 판/손바닥/몸통)에 타깃 자신의 점이 걸리면 접근하다 물체를 친다
# (흰 병이 접근 중 밀려 쓰러짐, 육안 확인). CGN 의 opening 이 아니라 실제 개방폭으로 검사.
SELF_HIT_MAX = int(os.environ.get("NERO_SELF_HIT_MAX", "3"))
GRIP_OPEN_W = 0.10
# 2026-09-29 IK 호출 절감 (reach_log 28k건 분석): 쌍둥이(orig/flip)는 결과가 거의 같고(둘 다 성공 41%/둘 다
# 실패 54%/flip만 성공 5%), 공간 빈(위치 5cm x 접근방위 45도)의 86%가 전부 성공 또는 전부 실패다.
#  TWIN_LAZY: orig 가 (여유 기준까지 통과해) 성공한 자세의 flip 은 IK 를 안 부른다.
#  BIN_SKIP : 한 빈에서 성공 0 + 실패 N번이면 그 빈의 나머지 후보는 IK 를 안 부른다 (0 = 끔).
# 시뮬레이션: 호출 -45%, 도달 자세 손실 2.7%. NERO_TWIN_LAZY=0 NERO_BIN_SKIP=0 이면 기존 동작.
TWIN_LAZY = os.environ.get("NERO_TWIN_LAZY", "1") == "1"
BIN_SKIP = int(os.environ.get("NERO_BIN_SKIP", "2"))
DIVERSE = os.environ.get("NERO_DIVERSE", "1") == "1"


def _diversify(ks, n, waz=45.0):
    """rerank 순서를 유지하되 접근 방위(45도)별로 번갈아 뽑아 후보군이 한쪽 방향에 몰리지 않게 한다.
    전체 1위는 그대로 첫 후보. 이웃 물체가 한쪽 접근을 막을 때 반대편 후보가 C단계에 도달하게 한다
    (시드 100 병: 도달 자세는 많은데 상위 8개가 전부 머스타드 쪽이라 C 전멸)."""
    if len(ks) <= 1:
        return ks[:n]
    bins, order = {}, []
    for k in ks:
        b = int((np.degrees(np.arctan2(k["approach"][1], k["approach"][0])) + 180.0) // waz)
        if b not in bins:
            bins[b] = []; order.append(b)
        bins[b].append(k)
    out = []
    while len(out) < n and any(bins[b] for b in order):
        for b in order:
            if bins[b] and len(out) < n:
                out.append(bins[b].pop(0))
    return out

def _alloc(ks, n, quota_str):
    """_diversify 결과를 기본으로, 접근각 구간별 하한(quota_str="탑,기울기,옆")을 채운다. 부족한 구간의 후보(rerank 순서상 앞선 것)를 가져와
    목록 **뒤쪽**에 붙이고, 자리는 하한을 초과한 구간의 맨 뒤 후보부터 비운다(첫 후보는 유지). quota_str 이 비면 기존 동작 그대로."""
    base = _diversify(ks, n) if DIVERSE else ks[:n]
    try:
        q = [int(x) for x in str(quota_str).split(",") if x.strip() != ""]
    except Exception:
        return base
    if len(q) != 3 or sum(q) <= 0 or len(ks) <= 1:
        return base
    while sum(q) > n:
        q[q.index(max(q))] -= 1
    bo = lambda k: _theta_bin(float(k.get("approach_theta", np.degrees(np.arccos(np.clip(-np.array(k["approach"])[2], -1, 1))))))
    cnt = [0, 0, 0]
    for k in base:
        cnt[bo(k)] += 1
    pulled = []
    for b in (1, 2, 0):
        need = q[b] - cnt[b]
        if need > 0:
            take = [k for k in ks if bo(k) == b and not any(k is x for x in base) and not any(k is x for x in pulled)][:need]
            pulled += take; cnt[b] += len(take)
    excess = len(base) + len(pulled) - n
    i = len(base) - 1
    while excess > 0 and i >= 1:
        b = bo(base[i])
        if cnt[b] > q[b]:
            base.pop(i); cnt[b] -= 1; excess -= 1
        i -= 1
    return base + pulled[:max(0, n - len(base))]

NBR_MARGIN = float(os.environ.get("NERO_NBR_MARGIN", "0.005"))
_FINGER_Z0, _FINGER_T, _FINGER_HX = 0.0593, 0.0265, 0.028
_HEAD = (0.042, 0.082, 0.039, 0.065)     # (hx, hy, z0, z1) 넓은 손목 머리
_BODY = (0.040, 0.040, -0.05, 0.039)     # (hx, hy, z0, z1) 손목~플랜지 몸통


def _nbr_hits(obs, flange, Rf, opening, sweep):
    """obs: (N,3) world. 그리퍼(flange 프레임 Rf)가 -a 방향 sweep 만큼 뒤에서 전진해 들어올 때
    통로 안에 든 장애물 복셀 수."""
    L = (obs - np.asarray(flange)) @ np.asarray(Rf)        # 열 = 그리퍼 x,y,z
    x, y, z = L[:, 0], L[:, 1], L[:, 2]
    m = NBR_MARGIN
    h = 0.5 * float(opening)
    inside = np.zeros(len(L), bool)
    fz = (z >= _FINGER_Z0 - sweep - m) & (z <= D_TCP + m) & (np.abs(x) <= _FINGER_HX + m)
    inside |= fz & (np.abs(y) >= h - m) & (np.abs(y) <= h + _FINGER_T + m)
    hx, hy, z0, z1 = _HEAD
    inside |= (np.abs(x) <= hx + m) & (np.abs(y) <= hy + m) & (z >= z0 - sweep - m) & (z <= z1 + m)
    bx, by, bz0, bz1 = _BODY
    inside |= (np.abs(x) <= bx + m) & (np.abs(y) <= by + m) & (z >= bz0 - sweep - m) & (z <= bz1 + m)
    return int(inside.sum())


_SEED_DEFAULT = [0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491]
# A 수정: PC 팔의 실제 관절각을 seed 로 (run_pipeline 이 npz 의 arm_q 를 NERO_ARM_Q 로 넘김)
try:
    SEED = [float(x) for x in os.environ["NERO_ARM_Q"].split()]
    assert len(SEED) == 7
    print(f"[seed] PC arm_q: {[round(x,3) for x in SEED]}")
except Exception:
    SEED = _SEED_DEFAULT
TOPK = 24                       # IK budget after prefilter+sort
IK_NS = 200_000_000            # solver returns in <10 ms when a solution exists
# ablation 토글: unified cost 의 top-down 선호항 −W·(θ/90)⁴ (OK-Robot Part B).
# 기본 0 = 없음 (2026-09-10 제거). NERO_THETA4=0.4 로 원복해서 A/B 비교.
THETA4_W = float(os.environ.get("NERO_THETA4", "0") or "0")
if THETA4_W:
    print(f"[ablation] top-down 선호항 활성 W={THETA4_W}")
# 기울기 안정성 페널티 (2026-09-11). 하드코딩 "cyl이면 side"가 아니라, 실측된 cyl_fit 의
# span/radius(가늘고 긴 정도)로부터 tipping/lever-arm 리스크를 계산해 top-down 에 가까운
# (θ 작은) grasp 만 불리하게 함. A/B 검증(같은 컵, semantic OFF, N=5 vs N=5): OFF 0/5 성공
# (전부 top-down 계열 SLIP/NO_REACH) vs ON 4/5 성공(유일한 실패는 손잡이 타겟 미끄러짐 —
# tip_pen 범위 밖) -> 기본값 0.5 로 승격. NERO_TIP_W=0 으로 끄면 A/B 재비교 가능(ablation).
TIP_W = float(os.environ.get("NERO_TIP_W", "0.5") or "0.5")
HIT_SPAN = int(os.environ.get("NERO_HIT_SPAN", "20"))   # _sieve: 첫 IK 성공 이후 추가 탐색 폭

# ── 2026-11-04 접근각 구간(탑다운 ≤15° / 기울기 16~45° / 옆 >45°) 다양성 (기본 전부 꺼짐 = 출력 동일) ─────────────────────────────
#  NERO_BIN_MIN   : _sieve 끝에서 구간별 IK 성공이 이 값 미만이면 그 구간 후보를 NERO_BIN_BUDGET 개까지 추가 평가 (0=꺼짐)
#  NERO_KEEP_BIN  : 최종 keep 의 구간 하한 "탑,기울기,옆" (예 "4,2,2"; 빈 문자열=꺼짐). 하한을 채우는 후보는 목록 뒤쪽에 둔다
#  값은 환경변수, 없으면 ~/grasp/_bflags.json ({"NERO_BIN_MIN": 3, ...}) 에서 읽는다 (cgn_server /plan_upload 는 B 를 같은 프로세스에서 exec 하며
#  env 를 NERO_CUROBO_IK_* 만 넘기므로, 호출자가 ssh 로 이 파일을 써서 켠다. 파일이 없으면 꺼짐).
def _bflag(name, default=""):
    v = os.environ.get(name)
    if v not in (None, ""):
        return v
    try:
        return str(json.load(open(os.path.expanduser("~/grasp/_bflags.json"))).get(name, default))
    except Exception:
        return default
BIN_MIN = int(float(_bflag("NERO_BIN_MIN", "0") or "0"))
BIN_BUDGET = int(float(_bflag("NERO_BIN_BUDGET", "32") or "32"))
KEEP_BIN = _bflag("NERO_KEEP_BIN", "")
def _theta_bin(t):
    return 0 if t <= 15.0 else (1 if t <= 45.0 else 2)

# ── 후보2(태스크-적응형 cost 가중치, 2026-09-23): box 경로 cost의 d_com 계수를
# 자연어 태스크 지시에 따라 VLM(task_weight.py)이 이산 등급(low/normal/high)으로
# 조정 -- 논문 candidate2. 기본 OFF(NERO_TASK_WEIGHT != "1") -- 꺼져 있으면 배율
# 1.0 = 기존 동작과 100% 동일. task_weight.py가 안 돌았거나 실패했으면
# (_taskw.json 없음/파싱 실패) 마찬가지로 1.0 폴백 -- 이 기능이 어떤 이유로든
# 실패해도 기존 파이프라인이 절대 안 죽는다(이중 안전장치, task_weight.py 쪽도
# vLLM 다운 시 동일하게 1.0 폴백).
TASKW_MULT = 1.0
if os.environ.get("NERO_TASK_WEIGHT", "0") == "1":
    _taskw_path = os.path.expanduser(os.environ.get("NERO_TASKW_JSON", "~/grasp/_taskw.json"))
    try:
        TASKW_MULT = float(json.load(open(_taskw_path))["d_com_w_mult"])
        print(f"[taskw] d_com 가중치 배율 {TASKW_MULT} ({_taskw_path})")
    except Exception as e:
        print(f"[taskw] 로드 실패({e}) -- 배율 1.0(기본) 폴백")
        TASKW_MULT = 1.0

# ── reachability 로거 (2026-09-16, Track A 옵션1) ────────────────────────────
# _sieve 가 IK 호출 결과를 성공/실패 무관하게 전부 버리던 것을 기록으로 남긴다.
# 목적: "이 위치·이 접근방향이 이 로봇에서 도달 가능한가"를 예측하는 경량
# 회귀기를 학습할 데이터를 쌓는 것 — 매 파이프라인 실행(dry 포함)마다 공짜로
# 누적됨. 학습된 모델이 준비되기 전까진 이 로그는 순수 관측용이고 랭킹에
# 아무 영향도 안 준다(REACH_MODEL 이 None이면 기존 동작 그대로).
REACH_LOG = os.path.expanduser("~/grasp/reach_log.jsonl")
_reach_buf = []


def _log_reach(name, kind, tw, cost, score, flange, a, opening, ok, qsol=None):
    """IK 시도 1건을 버퍼에 적재 (디스크 쓰기는 런 종료 시 1회만 — 타이밍 안 건드림)."""
    x, y, z = float(flange[0]), float(flange[1]), float(flange[2])
    r = float(np.hypot(x, y))
    az = float(np.degrees(np.arctan2(y, x)))
    theta = float(np.degrees(np.arccos(np.clip(-float(a[2]), -1, 1))))
    rec = dict(source="pipeline", ts=os.environ.get("NERO_RUN_TS", ""), object=name, kind=kind, twin=tw,
               x=x, y=y, z=z, r=r, bearing_deg=az, theta_deg=theta,
               a_x=float(a[0]), a_y=float(a[1]), a_z=float(a[2]),
               cost=float(cost), score=float(score), opening=float(opening),
               ok=bool(ok))
    if ok and qsol is not None:
        rec.update(min_margin=min_margin(qsol), wrist_sing=wrist_sing(qsol),
                   j1_ik=float(qsol[0]), j6_ik=float(qsol[5]))
    _reach_buf.append(rec)


def _flush_reach_log():
    if not _reach_buf:
        return
    try:
        with open(REACH_LOG, "a") as f:
            for rec in _reach_buf:
                f.write(json.dumps(rec) + "\n")
        print(f"[reach-log] {len(_reach_buf)}건 -> {REACH_LOG}")
    except Exception as e:
        print(f"[reach-log] 기록 실패 (무시하고 계속): {e}")
    _reach_buf.clear()


# ── 학습된 reachability 모델 적용 (train_reach_model.py 산출물) ──────────────
# 기본은 비활성(모델 파일 없으면 REACH_W=0 과 동일 -> 정렬 완전히 그대로).
# NERO_REACH_MODEL 로 joblib 경로를 주면 로드해서 _sieve 정렬에 "가중치"로만
# 얹는다 — 절대 후보를 걸러내지(hard-filter) 않는다. 예측이 완전히 틀려도
# 최악의 경우 정렬 순서가 나빠질 뿐, 유일한 도달 가능 후보가 사라지는 사고는
# 구조적으로 안 생긴다 (사용자 지시, 2026-09-16).
REACH_W = float(os.environ.get("NERO_REACH_W", "1.0") or "1.0")
_REACH_MODEL = None
_reach_model_path = os.environ.get("NERO_REACH_MODEL", "")
if _reach_model_path and os.path.exists(_reach_model_path):
    try:
        import joblib
        _REACH_MODEL = joblib.load(_reach_model_path)
        print(f"[reach-model] {_reach_model_path} 로드 "
              f"(n_train={_REACH_MODEL.get('n_train')}, "
              f"held-out AUC={_REACH_MODEL.get('auc_holdout', 0):.3f})")
    except Exception as e:
        print(f"[reach-model] 로드 실패 (기존 cost 정렬로 폴백): {e}")
        _REACH_MODEL = None


def _reach_onehot(val, vocab):
    v = [0.0] * len(vocab)
    if val in vocab:
        v[vocab.index(val)] = 1.0
    return v


def _reach_scores_batch(name, cands):
    """cands: [(cost, flange, Rf, a, opening, kind, score, tw), ...]
    -> 같은 길이의 P(도달가능) 배열. 모델 없으면 전부 0 (기존 cost 그대로).
    특징 순서는 train_reach_model.py 의 featurize()/FEATURE_NAMES 와
    반드시 일치해야 한다 — 둘 중 하나만 고치면 조용히 어긋나니 주의."""
    if _REACH_MODEL is None or not cands:
        return np.zeros(len(cands))
    kv, tv, ov = (_REACH_MODEL["kind_vocab"], _REACH_MODEL["twin_vocab"],
                  _REACH_MODEL["obj_vocab"])
    X = []
    for cost, flange, Rf, a, opening, kind, score, tw in cands:
        x, y, z = float(flange[0]), float(flange[1]), float(flange[2])
        r = float(np.hypot(x, y))
        bearing = float(np.degrees(np.arctan2(y, x)))
        theta = float(np.degrees(np.arccos(np.clip(-float(a[2]), -1, 1))))
        X.append([x, y, z, r, bearing / 180.0, theta / 90.0,
                  float(a[0]), float(a[1]), float(a[2]),
                  float(cost), float(score), float(opening)]
                 + _reach_onehot(kind, kv) + _reach_onehot(tw, tv)
                 + _reach_onehot(name, ov))
    try:
        return _REACH_MODEL["model"].predict_proba(np.array(X))[:, 1]
    except Exception as e:
        print(f"[reach-model] 추론 실패 (기존 cost 정렬로 폴백): {e}")
        return np.zeros(len(cands))

if TIP_W:
    print(f"[ablation] 기울기 안정성 페널티 활성 W={TIP_W}")


def _slenderness(box_fit, cyl_fit=None):
    """'가늘고 긴 정도' 0~1. 뚱뚱함(ratio<1.5)=0, ratio>=4.0 에서 1로 포화.

    2026-09-23 이력: 처음엔 cyl_fit(원통 전용 반경 fit) 의존을 없애고 축정렬
    bbox(ext)만 쓰려 했는데("category-agnostic" 주장과 원통 특수분기 공존을 없애려는
    의도), 손잡이 같은 돌출부가 min/max bbox 가로폭을 부풀려서 slender가 실측
    ~10배 약해지는 회귀가 확인됨(컵 사례). 이후 백분위수(ext_robust) 트림을
    시도했으나 개선폭이 부족(~3배).

    **최종**: cgn_server.py의 `circle_fit()`(Taubin fit + 3회 반복 이상치 제거)이
    이미 **모든 물체에 라벨/형상 무관하게 무조건** 호출되고 있었고(카테고리 분기
    아님 -- 그냥 "이 물체의 단면에 가장 잘 맞는 원을 robust하게 찾는다"는 범용
    기하 추정), 그 결과(cyl_fit.radius/top_z/base_z)가 이미 서버 응답에 실려있었다.
    이름이 "cyl_fit"이라 오해했을 뿐, 실제로는 카테고리 가정이 없는 범용 로버스트
    추정치라 그대로 쓰는 게 맞다 -- 있으면 이걸 우선 쓰고(가장 정확), 없으면
    ext_robust(백분위 트림), 그것도 없으면 순수 bbox ext로 단계적 폴백한다."""
    if cyl_fit:
        span = cyl_fit["top_z"] - cyl_fit["base_z"]
        ratio = span / (2.0 * max(cyl_fit["radius"], 1e-3))
        return float(np.clip((ratio - 1.5) / 2.5, 0.0, 1.0))
    if not box_fit:
        return 0.0
    ext = box_fit.get("ext_robust", box_fit["ext"])
    footprint = 0.5 * (ext[0] + ext[1])
    ratio = ext[2] / max(footprint, 1e-3)
    return float(np.clip((ratio - 1.5) / 2.5, 0.0, 1.0))

# 2026-09-30 학습 기반 파지 성공 예측 p(hold) (NERO_PHOLD=1 일 때만). 규칙 항(tip/d_com/neck/hold) 대신 cost = CGN score + NERO_PHOLD_W * p_hold.
# 입력 = 플랜지 프레임 점유 격자(12x16x16, 대상 점군) + 중력방향 4. 학습: Isaac 헤드리스 grasp 시험 25k 샘플(GSO 448종), build_features.py 와 같은 격자 정의.
_PH_MODEL = None
PHOLD_W = float(os.environ.get("NERO_PHOLD_W", "3.0"))


def _phold_scores(grasps, tgt):
    global _PH_MODEL
    if os.environ.get("NERO_PHOLD", "0") != "1" or tgt is None:
        return {}
    try:
        import joblib
        if _PH_MODEL is None:
            _PH_MODEL = joblib.load(os.path.expanduser(os.environ.get("NERO_PHOLD_MODEL", "~/grasp/phold_v1.joblib")))
        GX, GY, GZ = 12, 16, 16
        XR, YR, ZR = (-0.05, 0.05), (-0.07, 0.07), (0.02, 0.16)
        feats, idx, agree = [], [], []
        for k, g in enumerate(grasps):
            if not (0.005 < g["opening_m"] <= 0.102):
                continue
            a = np.array(g["approach"], float)
            pos = np.array(g["position_xyz"], float)
            Rf = q2R(g["quat_xyzw"])
            fl = pos - (D_TCP - DEEPEN) * a
            L = (np.asarray(tgt, float) - fl) @ Rf
            ix = np.floor((L[:, 0] - XR[0]) / (XR[1] - XR[0]) * GX).astype(int)
            iy = np.floor((L[:, 1] - YR[0]) / (YR[1] - YR[0]) * GY).astype(int)
            iz = np.floor((L[:, 2] - ZR[0]) / (ZR[1] - ZR[0]) * GZ).astype(int)
            ok = (ix >= 0) & (ix < GX) & (iy >= 0) & (iy < GY) & (iz >= 0) & (iz < GZ)
            grid = np.zeros((GX, GY, GZ), np.uint8)
            grid[ix[ok], iy[ok], iz[ok]] = 1
            up = Rf.T @ np.array([0, 0, 1.0])
            feats.append(np.concatenate([grid.reshape(-1).astype(np.float32), np.r_[up, Rf[2, 2]].astype(np.float32)]))
            idx.append(k)
            agree.append(abs(float(Rf[:, 2] @ a)))
        if not feats:
            return {}
        pr = _PH_MODEL.predict_proba(np.array(feats))[:, 1]
        print("[phold] %d개 후보 예측: p 평균 %.2f 최대 %.2f, 접근축-플랜지z 일치도 %.2f" % (len(feats), pr.mean(), pr.max(), float(np.mean(agree))))
        return dict(zip(idx, map(float, pr)))
    except Exception as e:
        print("[phold] 실패(%s) -- 규칙 cost 로 폴백" % e)
        return {}


# coarse reachable workspace for the flange origin (base_link at world origin)
WS_R2 = (0.14, 0.78)          # xy radius bounds
WS_Z = (0.02, 0.90)

# 2026-10-01 cuRobo IK 를 쓰면 MoveIt 클라이언트 초기화(rclpy.init + 서비스 대기, 약 0.3~0.5s)가 필요 없다 -- 처음 fik 가 불릴 때(서버 폴백 시) 초기화한다.
n = None; ik = None
ss = None
if _HAVE_ROS:
    ss = RobotState(); ss.joint_state = JointState(name=ARM, position=SEED)
_ik_calls = [0]


def _moveit_init():
    global n, ik
    if ik is not None or not _HAVE_ROS:
        return
    rclpy.init()
    n = rclpy.create_node("two_pick")
    ik = n.create_client(GetPositionIK, "/compute_ik")
    ik.wait_for_service(timeout_sec=10)


if not os.environ.get("NERO_CUROBO_IK_URL", ""):
    _moveit_init()


# 2026-10-01 cuRobo 배치 IK 흡수: NERO_CUROBO_IK_URL(예: http://<PC>:8020) 이 있으면 _sieve 가 후보 전체의 IK 를 PC cuRobo 서버에 한 번에 묻는다
# (MoveIt pick_ik 순차 호출 대체). 서버는 충돌(테이블)/자기충돌을 고려한 해만 돌려준다. 실패하거나 서버가 죽으면 기존 fik 로 폴백한다.
CUROBO_IK_URL = os.environ.get("NERO_CUROBO_IK_URL", "")


def curobo_ik_batch(poses, seed=None):
    """poses: [(flange[3], quat_xyzw[4])...] -> [qsol(list7)|None ...]  (실패 시 None 리스트 전체가 아니라 예외 -> 호출부가 폴백)"""
    import urllib.request
    body = {"poses": [{"flange": [float(x) for x in p], "quat": [float(x) for x in q]} for p, q in poses]}
    if seed is not None:
        body["seed_q"] = [float(x) for x in seed]
    body["check_collision"] = os.environ.get("NERO_CUROBO_IK_COLLISION", "1") != "0"     # 튜닝/원인 분리용
    rq = urllib.request.Request(CUROBO_IK_URL + "/ik", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(rq, timeout=30).read())
    print("[curobo-ik] %d 후보 -> 도달 %d, 서버 %.2fs" % (len(poses), sum(1 for x in r["q"] if x is not None), r.get("sec", 0)))
    return r["q"]


def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def in_ws(p):
    r2 = p[0] * p[0] + p[1] * p[1]
    return (WS_R2[0] ** 2 < r2 < WS_R2[1] ** 2) and (WS_Z[0] < p[2] < WS_Z[1])


def fik(pos, quat, seed=None):
    _moveit_init()
    if ik is None:       # ROS 없음(상주 서비스) -- MoveIt IK 폴백 불가
        return None
    _ik_calls[0] += 1
    st = ss if seed is None else RobotState(joint_state=JointState(name=ARM, position=list(seed)))
    req = GetPositionIK.Request(); req.ik_request.group_name = "arm"; req.ik_request.robot_state = st
    ps = PoseStamped(); ps.header.frame_id = "base_link"
    ps.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    ps.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
    req.ik_request.pose_stamped = ps
    req.ik_request.timeout.sec = 0
    req.ik_request.timeout.nanosec = IK_NS
    # reachability filter only -- collision-free PATH is enforced downstream by step6's
    # STOMP + Cartesian (avoid_collisions=True) + planning-scene obstacle. With it True
    # here pick_ik converges to a self-colliding branch from the seed and never retries.
    req.ik_request.avoid_collisions = False
    f = ik.call_async(req); rclpy.spin_until_future_complete(n, f, timeout_sec=4); r = f.result()
    if r and r.error_code.val == 1:
        js = r.solution.joint_state
        return [round(js.position[js.name.index(j)], 4) for j in ARM]
    return None


def R2q(Rm):
    t = np.trace(Rm)
    if t > 0:
        u = np.sqrt(t + 1) * 2; w = .25 * u
        x = (Rm[2, 1] - Rm[1, 2]) / u; y = (Rm[0, 2] - Rm[2, 0]) / u; z = (Rm[1, 0] - Rm[0, 1]) / u
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        u = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / u; x = .25 * u; y = (Rm[0, 1] + Rm[1, 0]) / u; z = (Rm[0, 2] + Rm[2, 0]) / u
    elif Rm[1, 1] > Rm[2, 2]:
        u = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / u; x = (Rm[0, 1] + Rm[1, 0]) / u; y = .25 * u; z = (Rm[1, 2] + Rm[2, 1]) / u
    else:
        u = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / u; x = (Rm[0, 2] + Rm[2, 0]) / u; y = (Rm[1, 2] + Rm[2, 1]) / u; z = .25 * u
    v = np.array([x, y, z, w]); return (v / np.linalg.norm(v)).tolist()


for name in NAMES:
    t0 = time.time(); _ik_calls[0] = 0
    p = os.path.expanduser(f"~/grasp/{name}_grasps.json")
    if not os.path.exists(p):
        print(f"[{name}] no grasps file"); continue
    data = json.load(open(p))
    grasps = data["grasps"]
    _obs = None
    _obp = os.path.expanduser(f"~/grasp/{name}_obstacles.npy")
    if NBR_HIT_MAX > 0 and os.path.exists(_obp):
        try:
            _obs = np.load(_obp)
            if len(_obs) == 0:
                _obs = None
        except Exception:
            _obs = None
    print(f"[{name}] 이웃 충돌 필터: " + (f"ON (장애물 복셀 {len(_obs)}, hit_max {NBR_HIT_MAX})"
                                           if _obs is not None else "OFF (장애물 없음/미사용)"))
    _tgt = None
    _tgp = os.path.expanduser(f"~/grasp/{name}_target.npy")
    if os.path.exists(_tgp):
        try:
            _tgt = np.load(_tgp)
            if len(_tgt) == 0:
                _tgt = None
        except Exception:
            _tgt = None
    # 2026-09-29d 목/좁은 부위 잡기 페널티(형상 무관): 병을 위에서 뚜껑/목(폭이 몸통보다 좁은 곳)을 잡으면 lift 중 미끄러졌다
    # (nbr18~26 병 pick 16건: SLIPPED 6건 전부 접촉 높이 77~88%(목), HELD 8건은 몸통 35~68%). 컵 림은 몸통만큼 넓어 영향 없음.
    # 타깃 점군을 1cm 높이 구간으로 잘라 가로 최대 폭 프로파일을 만들고, 잡는 높이의 폭이 몸통(상위 폭)의 NECK_FRAC 미만이면 벌점.
    NECK_W = float(os.environ.get("NERO_NECK_W", "1.5"))
    NECK_FRAC = float(os.environ.get("NERO_NECK_FRAC", "0.92"))
    _prof = None
    if _tgt is not None and NECK_W > 0:
        try:
            _z0 = float(_tgt[:, 2].min())
            _nb = int((_tgt[:, 2].max() - _z0) / 0.01) + 1
            _zc, _wd = [], []
            for _k in range(_nb):
                _m = (_tgt[:, 2] >= _z0 + _k * 0.01) & (_tgt[:, 2] < _z0 + (_k + 1) * 0.01)
                if _m.sum() >= 8:
                    _q = _tgt[_m][:, :2]
                    _zc.append(_z0 + (_k + 0.5) * 0.01)
                    _wd.append(float(max(np.ptp(_q[:, 0]), np.ptp(_q[:, 1]))))
            if len(_wd) >= 4:
                _prof = (np.array(_zc), np.array(_wd), float(np.percentile(_wd, 85)))
                print(f"[{name}] 폭 프로파일: 몸통 폭 {_prof[2] * 100:.1f}cm, 최상단 폭 {_wd[-1] * 100:.1f}cm")
        except Exception:
            _prof = None

    def _neck_pen(zc):
        if _prof is None:
            return 0.0
        _i = np.argsort(np.abs(_prof[0] - zc))[:2]
        _loc = float(np.mean(_prof[1][_i]))
        return NECK_W * max(0.0, 1.0 - _loc / (NECK_FRAC * _prof[2]))

    # 2026-09-23: 원통 fit(cyl/cyl_fit) 및 그 기반 합성 side grasp 전부 제거함
    # (category-agnostic 헤드라인 주장과 "원통이면 특수 기하 분기" 코드가 공존하던
    # 모순 해소 -- CGN이 이미 일반 6-DoF 후보를 내므로 형상별 합성 자체가 불필요.
    # 남겨둔 흔적: 아래 cyl 변수는 항상 None -- 이 값에 의존하던 하위 분기(전부 삭제됨)
    # 없음, 남은 참조가 있다면 그건 버그이니 찾아서 지울 것).
    cyl = None

    # Task 1/2/3: VLM 시맨틱이 있으면 aspect_w 분기를 버리고 CGN raw grasp 단일 랭킹.
    sem = None
    _semp = os.path.expanduser(f"~/grasp/{name}_sem.json")
    if os.path.exists(_semp):
        try:
            sem = json.load(open(_semp))
        except Exception:
            sem = None
    _reg = {}
    _exp = os.path.expanduser(f"~/grasp/{name}_exclude.json")
    if os.path.exists(_exp):
        try:
            _reg = json.load(open(_exp))
        except Exception:
            _reg = {}
    _target_aabb = _reg.get("target_aabb")            # positive: 여기만 잡아라 (특정 파트일 때)
    _avoid_aabbs = _reg.get("avoid_aabbs", [])         # negative: 닿지 마라

    def _in_box(pt, b):
        return bool(np.all(np.abs(np.array(pt) - np.array(b[:3])) <= np.array(b[3:])))

    def _ok_region(pt):
        if _target_aabb is not None and not _in_box(pt, _target_aabb):
            return False
        for b in _avoid_aabbs:
            if _in_box(pt, b):
                return False
        return True

    def _in_excl(pt):                                 # 하위호환 (기존 호출부)
        return not _ok_region(pt)

    def _in_excl_span(center, jaw_axis, opening):
        """2026-09-11: 접촉점 하나 대신, 실제 그 grasp 의 opening/jaw축으로 핑거팁 두 점을
        계산해 각각 검사. 고정 마진으로 박스를 부풀리는 것보다 정확함 — 마진은 모든
        방위에 등방으로 걸려서 손잡이랑 무관한 방향에서 오는 후보까지 부당하게
        떨어뜨리지만(오염), 이건 그 grasp 고유의 실제 opening/방향만 반영해 그 방향으로
        안 벌어지는 후보는 그대로 통과시킴 — 다른 IK 해 오염 없음.
        회피(avoid)만 두 점 다 검사(핑거 어느 쪽이든 닿으면 탈락) — 타겟(target) 포함은
        중심점 하나만(좁은 target 에 넓은 opening 이 부당하게 걸리는 것 방지)."""
        half = float(opening) * 0.5
        p1 = center + half * jaw_axis
        p2 = center - half * jaw_axis
        if _target_aabb is not None and not _in_box(center, _target_aabb):
            return True
        for b in _avoid_aabbs:
            if _in_box(p1, b) or _in_box(p2, b):
                return True
        return False

    unified = sem is not None
    # (2026-09-23: "VLM shape=cylinder -> 원통 fit 강제 사용" 분기 제거 -- cyl 자체를
    # 없앴으므로. approach_pref는 side/top 합성이 아직 있던 시절의 게이트 신호였는데
    # 그 합성을 지웠으니 지금은 안 쓰임 -- sem_infer.py 스키마에는 남아있어도 무해함.)
    cand = []          # (cost, flange, Rf, approach, opening, kind, score) — CGN raw
    cand_side = []     # (2026-09-23: 원통 합성 제거로 항상 빈 리스트 -- 하위 코드 호환용으로 유지)
    cand_pinch = []    # 파트(손잡이) 튜브축 pinch 합성 — 게이트: reachable CGN < K 일 때만
    cand_top = []      # top-down 합성 — 최후 폴백
    npre = 0
    n_excl = 0
    _slender = _slenderness(data.get("box_fit"), data.get("cyl_fit"))   # 가늘고 긴 정도 0~1 (뚱뚱=0) — tip_pen 스케일용

    if unified:
        bf = data["box_fit"]
        c = np.array(bf["centre"]); bbmin = np.array(bf["bbmin"]); bbmax = np.array(bf["bbmax"])
        reg = sem.get("grasp_region", "body")
        if reg in ("top", "rim"):
            rc = np.array([c[0], c[1], bbmax[2] - 0.02])
        elif reg == "neck":
            rc = np.array([c[0], c[1], bbmin[2] + 0.75 * (bbmax[2] - bbmin[2])])
        else:                                  # body / center / side
            rc = c.copy()
        keep_up = bool(sem.get("keep_upright"))
        for g in grasps:
            a = np.array(g["approach"]); pos = np.array(g["position_xyz"]); w = g["opening_m"]
            if not (0.005 < w <= 0.102):
                continue
            flange = pos - (D_TCP - DEEPEN) * a
            if not in_ws(flange):
                npre += 1; continue
            Rf = q2R(g["quat_xyzw"])
            if _in_excl_span(pos, Rf[:, 1], w):   # Task 3: 핑거팁 두 점 다 검사(그리퍼 두께 반영)
                n_excl += 1; continue
            theta = np.degrees(np.arccos(np.clip(-a[2], -1, 1)))
            # d_reg = 높이(z) 차이만 (2026-09-11). rc.xy 는 항상 물체 축 근처라 xy 성분은
            # "표면 접촉이냐 축이냐"만 반영해 CGN raw(표면, ~반경만큼 큼) 를 side 합성(축,
            # ~0) 대비 부당하게 불리하게 만들던 원인 — 실제 구분해야 할 정보(top/rim/neck/
            # body 중 어디를 노렸나)는 z 뿐이라 xy 는 버려도 손실 없음.
            d_reg = float(abs(pos[2] - rc[2]))
            # keep_upright(내용물 쏟는 열린 용기)면 물체를 뒤집는 grasp(θ>110) 만 감점.
            up_pen = 0.4 * ((theta - 110) / 70.0) if (keep_up and theta > 110) else 0.0
            # top-down 선호항 (−W·(θ/90)⁴, OK-Robot Part B): 기본 W=0 (제거 — NERO 실패모드는
            # 반대로 탑다운이 컵/보울 림 slip). NERO_THETA4=0.4 로 원복해 A/B 비교 (ablation).
            th_pen = THETA4_W * (theta / 90.0) ** 4 if THETA4_W else 0.0
            # tipping 페널티: 물체가 가늘고 길수록(slender) top-down(θ 작음)에 불리하게.
            # side(θ=90)면 0, top-down(θ=0)이면 최대 TIP_W*slender.
            tip_pen = TIP_W * _slender * (1.0 - theta / 90.0) ** 2 if TIP_W else 0.0
            cost = g["score"] - 1.0 * d_reg - th_pen - tip_pen - up_pen
            cand.append((cost, flange, Rf, a, min(w + 0.02, 0.10), "cgn", g["score"]))
        # (2026-09-23: 원통 회전대칭 합성 side 블록 전체 제거 -- CGN이 이미 일반 6-DoF
        # 후보를 내므로 형상별 합성 grasp이 불필요. cand_side는 항상 빈 채로 남는다.)
        tgt_str = f"target#{_reg.get('target_id')}" if _target_aabb is not None else "body"
        print(f"[{name}] unified: {tgt_str} keep_upright={keep_up} avoid={_reg.get('avoid_ids')}  "
              f"CGN {len(grasps)} -> {len(cand)} cgn cand + {len(cand_side)} side(게이트)  "
              f"(ws drop {npre}, region drop {n_excl})")
        # target 이 특정 파트(손잡이) -> 파트 튜브축 기준 pinch 합성 (cand_pinch, 게이트).
        # reachable CGN < K 면 폴백 (원통 side 게이트와 같은 패턴).
        if _target_aabb is not None:
            tc = np.array(_target_aabb[:3]); the = np.array(_target_aabb[3:])
            tax = np.array(_reg.get("target_axis", [0.0, 0.0, 1.0]))
            tax = tax / (np.linalg.norm(tax) + 1e-9)
            perp_ext = sorted(the[i] for i in range(3) if abs(tax[i]) < 0.7)
            tube_d = 2 * (perp_ext[0] if perp_ext else min(the))
            opn = float(np.clip(tube_d + 0.022, 0.02, 0.09))
            rxy = tc[:2] / max(np.linalg.norm(tc[:2]), 1e-6)
            base_deg = np.degrees(np.arctan2(rxy[1], rxy[0]))
            L = float(max(the)) * 0.5
            tlong = tax / (np.linalg.norm(tax) + 1e-9)
            for along in (0.0, L, -L):
                gc = tc + along * tlong
                for az_off in (0, 20, -20, 40, -40, 60, -60):
                    aa = np.radians(base_deg + az_off)
                    ah = np.array([np.cos(aa), np.sin(aa), 0.0])
                    fl = gc - (D_TCP - DEEPEN) * ah
                    if not in_ws(fl):
                        continue
                    bb = np.cross(tax, ah)                 # ⊥ 튜브축, ⊥ approach
                    if np.linalg.norm(bb) < 1e-3:
                        bb = np.cross([0, 0, 1.0], ah)
                    bb /= np.linalg.norm(bb)
                    Rf = np.column_stack([-np.cross(ah, bb), bb, ah])
                    cand_pinch.append((-0.4 - abs(az_off) * 1e-3 - abs(along) * 1e-2,
                                       fl, Rf, ah, opn, "cgn", 0.15))
            print(f"[{name}] unified: target 파트 pinch 합성 {len(cand_pinch)}개(게이트) "
                  f"(튜브축 {np.round(tax,2)}, 지름 {tube_d*100:.1f}cm, opening {opn*100:.1f}cm)")
        # top-down 합성 (cand_top): 항상 만들어두고 keep 가 0 일 때만 씀.
        gz = float(bbmax[2] - 0.02); gc = np.array([c[0], c[1], gz]); ad = np.array([0, 0, -1.0])
        _topop = float(min(min(bbmax[0] - bbmin[0], bbmax[1] - bbmin[1]) + 0.02, 0.10))
        for yaw in (45, 30, 60, 90, 20, 75, 15, 0):
            b = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
            Rf = np.column_stack([-np.cross(ad, b), b, ad])
            fl = gc - (D_TCP - DEEPEN) * ad
            if in_ws(fl):
                cand_top.append((-1.0 - abs(yaw - 45) * 1e-4, fl, Rf, ad, _topop, "top", 0.0))

    # (2026-09-23: "cyl and not unified" 원통 전용 합성 분기 전체 제거 -- cyl이 항상
    # None이라 어차피 죽은 코드였음. 이 분기가 하던 일(시맨틱 없을 때도 공정한 랭킹) 중
    # 실제로 유효했던 부분(tip_pen)은 아래 box_fit 분기에 합쳐 넣었다 -- 시맨틱 유무와
    # 무관하게 모든 물체에 기울기 안정성 페널티가 이제 동일하게 적용됨(category-agnostic).)

    # compact object(모든 형상 공용): CGN raw grasp을 (score - th_pen - tip_pen - w*d_com)로
    # 랭킹하고, bbox 기반 top-down 합성을 폴백으로 추가. step6 STOMP+Cartesian이 순서대로 검증.
    # 2026-09-29d place 를 고려한 파지 선택(학습판): 논문(Grasp-Dependent Feasibility / Execution-Aware Ranking)의 핵심 --
    # "파지가 정하는 물체 대비 손 자세"가 place 가능성을 좌우한다. place 도달성 모델(place_reach_model.joblib)로 각 파지 후보가
    # 테이블 슬롯(NERO_TABLE_RECT) x yaw 어디에든 놓일 확률의 최대값을 구해 비용에 더한다. 테이블 정보/모델이 없으면 0(기존 동작).
    _PPL = {}
    _place_model = None
    _trect = None
    if PLACE_W > 0 and os.environ.get("NERO_TABLE_RECT") and data.get("box_fit") and not unified:
        try:
            import joblib
            from scipy.spatial.transform import Rotation as _Rot
            _place_model = joblib.load(os.path.expanduser("~/grasp/place_reach_model.joblib"))["model"]
            _trect = json.loads(os.environ["NERO_TABLE_RECT"])
        except Exception as _e:
            print(f"[{name}] place 모델 로드 실패({_e}) -- 반영 안 함")
            _place_model = None

    def _frame_from_b(A):
        ref = np.where((np.abs(A[:, 0]) < 0.9)[:, None], np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        yv = np.cross(A, ref); yv /= np.linalg.norm(yv, axis=1, keepdims=True)
        xv = np.cross(yv, A)
        return np.stack([xv, yv, A], axis=2)          # (N,3,3) 열 = x,y,z

    def _place_pmax(gl):
        """gl: [(index, pos, Rf)] -> {index: 최대 통과 확률}"""
        bf_ = data["box_fit"]
        base_z = max(0.0, min(0.03, float(bf_["bbmin"][2]))); ctr = np.array(bf_["centre"], float)
        H_ = float(bf_["bbmax"][2]) - float(bf_["bbmin"][2])
        cx, cy = _trect["center"]; ha, hb = _trect["half"]; yw = _trect["yaw"]; topz = float(_trect["top_z"])
        EC = 0.06; st = 0.06
        us = np.arange(-(max(ha - EC, 0.0)), max(ha - EC, 0.0) + 1e-9, st) if ha > EC else np.array([0.0])
        vs = np.arange(-(max(hb - EC, 0.0)), max(hb - EC, 0.0) + 1e-9, st) if hb > EC else np.array([0.0])
        slots = np.array([[cx + u * np.cos(yw) - v * np.sin(yw), cy + u * np.sin(yw) + v * np.cos(yw)] for u in us for v in vs])
        yaws = np.radians(np.arange(0, 360, 45))
        idx = [g_[0] for g_ in gl]
        P = np.array([g_[1] for g_ in gl], float); R0m = np.array([g_[2] for g_ in gl], float)   # (N,3),(N,3,3)
        A0 = R0m[:, :, 2]
        contact = P + DEEPEN * A0
        O = contact - np.stack([np.full(len(P), ctr[0]), np.full(len(P), ctr[1]), np.full(len(P), base_z + H_ / 2.0)], axis=1)
        held = contact[:, 2] - base_z
        theta0 = np.degrees(np.arccos(np.clip(-A0[:, 2], -1, 1)))
        # plumb(접근 45도 미만): 손을 수직으로 세움 -- a -> -z 정렬 회전
        ax = np.cross(A0, np.array([0, 0, -1.0])); an = np.linalg.norm(ax, axis=1)
        ang = np.arctan2(an, -A0[:, 2])
        rv = np.where(an[:, None] > 1e-9, ax / np.maximum(an, 1e-9)[:, None] * ang[:, None], 0.0)
        QA = _Rot.from_rotvec(rv)
        plumb = theta0 < 45.0
        Rbase = np.where(plumb[:, None, None], (QA * _Rot.from_matrix(R0m)).as_matrix(), R0m)
        rows = []
        N = len(gl)
        for yv_ in yaws:
            cz, sz = np.cos(yv_), np.sin(yv_)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
            Rq = np.einsum("ij,njk->nik", Rz, Rbase)
            a_ = Rq[:, :, 2]
            Fm = _frame_from_b(a_)
            rel = np.einsum("nji,njk->nik", Fm, Rq)                       # F^T Rq
            roll = _Rot.from_matrix(rel).as_rotvec()[:, 2]
            ox = O[:, 0] * cz - O[:, 1] * sz; oy = O[:, 0] * sz + O[:, 1] * cz
            for sl in slots:
                tcp = np.stack([sl[0] + ox, sl[1] + oy, topz + 0.006 + held], axis=1)
                fl = tcp - D_TCP * a_
                b_ = np.arctan2(tcp[:, 1], tcp[:, 0]); ph = np.arctan2(a_[:, 1], a_[:, 0])
                th = np.degrees(np.arccos(np.clip(-a_[:, 2], -1, 1)))
                rows.append(np.column_stack([tcp, np.hypot(tcp[:, 0], tcp[:, 1]), np.sin(b_), np.cos(b_), a_, th,
                                             np.sin(ph), np.cos(ph), np.sin(roll), np.cos(roll), fl, np.hypot(fl[:, 0], fl[:, 1])]))
        Xall = np.concatenate(rows, axis=0)                                # (S*Y*N, 18)
        pr = _place_model.predict_proba(Xall)[:, 1].reshape(-1, N)         # (S*Y, N)
        pm = pr.max(axis=0)
        print(f"[{name}] place 모델 반영: 후보 {N}개 x 슬롯 {len(slots)} x yaw {len(yaws)}, 놓을 확률 최대값 분포 "
              f"min {pm.min():.2f} 중앙 {np.median(pm):.2f} max {pm.max():.2f}")
        return dict(zip(idx, pm.tolist()))

    if _place_model is not None:
        try:
            _gl = [(k, np.array(g_["position_xyz"], float), q2R(g_["quat_xyzw"])) for k, g_ in enumerate(grasps)
                   if 0.005 < g_["opening_m"] <= 0.102]
            _PPL = _place_pmax(_gl) if _gl else {}
        except Exception as _e:
            print(f"[{name}] place 모델 계산 실패({_e}) -- 반영 안 함")
            _PPL = {}

    if not unified and data.get("box_fit"):
        bf = data["box_fit"]; c = np.array(bf["centre"]); ext = np.array(bf["ext"])
        centroid = np.array(bf["centre"])
        opening = float(min(min(ext[0], ext[1]) + 0.02, 0.10))
        cand = []
        _PHD = _phold_scores(grasps, _tgt)
        for _gi, g in enumerate(grasps):
            a = np.array(g["approach"]); pos = np.array(g["position_xyz"])
            w = g["opening_m"]
            if not (0.005 < w <= 0.102):
                continue
            Rf = q2R(g["quat_xyzw"])
            flange = pos - (D_TCP - DEEPEN) * a                        # CGN grasp -> flange
            if not in_ws(flange):
                npre += 1; continue
            theta = np.degrees(np.arccos(np.clip(-a[2], -1, 1)))   # 0 = straight down
            d_com = float(np.linalg.norm(pos - centroid))
            # 레거시(비-unified, sem 없을 때) 경로도 top-down 선호항 제거 — unified 경로와
            # 동일 기준(2026-09-11). NERO_THETA4 로 A/B 가능(unified 과 같은 토글 공유).
            th_pen2 = THETA4_W * (theta / 90.0) ** 4 if THETA4_W else 0.0
            # 2026-09-23: tip_pen을 이 분기에도 적용(이전엔 unified/cyl 분기에만 있었음) --
            # _slender는 위에서 이미 box_fit 기반으로 계산해둔 것(형상 무관, category-agnostic).
            tip_pen2 = TIP_W * _slender * (1.0 - theta / 90.0) ** 2 if TIP_W else 0.0
            # 2026-09-29d place 를 고려한 잡는 높이 벌점: 물체를 바닥 기준 높이 h 에서 잡으면 place 때 플랜지가 테이블+h+13.6cm 까지 올라가야 하고,
        # h>~0.09(책 0.092~0.105, 병 0.093~0.107)에선 수직 손 place 의 IK 가 전멸했다(오프라인 재현: 책 슬롯 6곳 전부 통과 0, h=0.05~0.06 인 컵/병은 통과).
            _hold = float(pos[2]) - float(bf["bbmin"][2])
            cost = g["score"] - th_pen2 - tip_pen2 - (1.2 * TASKW_MULT) * d_com - _neck_pen(float(pos[2])) - (0.0 if _PPL else HOLD_W * max(0.0, _hold - HOLD_H0)) + (PLACE_W * _PPL.get(_gi, 0.0) if _PPL else 0.0)
            if _PHD:                                   # 학습 점수로 규칙 항 전체를 대체(A/B 용): cost = CGN score + W * p_hold
                cost = g["score"] + PHOLD_W * _PHD.get(_gi, 0.0)
            cand.append((cost, flange, Rf, a, min(w + 0.02, 0.10), "cgn", g["score"]))
        # fallback: synthesized top-down from bbox (yaw sweep), ranked below real grasps
        gz = float(bf["bbmax"][2] - 0.02)
        if _prof is not None:
            _body = _prof[0][_prof[1] >= NECK_FRAC * _prof[2]]
            if len(_body):
                gz = float(min(gz, _body.max() - 0.025))      # 몸통이 시작되는 높이 안쪽에서 잡는다 (목/뚜껑 회피)
        gc = np.array([c[0], c[1], gz]); ad = np.array([0, 0, -1.0])
        for yaw in (45, 30, 60, 90, 20, 75, 15, 0):
            b = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
            Rf = np.column_stack([-np.cross(ad, b), b, ad])
            flange = gc - (D_TCP - DEEPEN) * ad
            if not in_ws(flange):
                continue
            cand.append((-1.0 - abs(yaw - 45) * 1e-4, flange, Rf, ad, opening, "top", 0.0))

    # 180° twin + FAST IK 체를 함수로. cand tuple 리스트 -> reachable keep dict 리스트.
    # (경로충돌 + Cartesian 타당성은 downstream step6 가 STOMP+Cartesian 으로 검증)
    FLIP = np.diag([-1.0, -1.0, 1.0])
    _twin = os.environ.get("NERO_TWIN", "both")     # both | orig | flip (검증용)

    def _sieve(cand_in, budget=int(os.environ.get("NERO_SIEVE_BUDGET", "220")),
               want=int(os.environ.get("NERO_SIEVE_WANT", "12")),
               bail_empty=int(os.environ.get("NERO_BAIL_EMPTY", "220")),
               hit_span=None):
        # (2026-09-16 개정) budget/bail_empty 를 48/20 -> 220/220 으로 상향.
        # 이유: 예전엔 "막히면 예산을 늘려 재탐색"하는 2단 에스컬레이션이었는데,
        # 사용자 지적대로 그건 "숨은 해를 뒤늦게 캐내는" 방식이라 실패 사례에서만
        # 비용을 더 쓰는 구조였음. 원하는 그림은 "처음부터 풀을 넓게 열어두고,
        # cost+reach모델 정렬이 잘 되면 적은 시도(앞 순위)만으로 바로 IK 성공을
        # 만나는 것" -- 즉 속도는 예산이 아니라 랭킹 품질이 좌우해야 함.
        # 실측: hit0(첫 성공 인덱스)이 대부분 0~15 근방이라, 넓은 예산 자체는
        # 흔한 케이스의 평균 시간을 거의 안 늘림(hit_span/want 로 성공 즉시 조기
        # 종료) -- 예산을 넓혀도 비용은 "정말 다 막힌" 드문 경우에만 커짐.
        if hit_span is None:
            hit_span = HIT_SPAN
        if _tgt is not None and cand_in and SELF_HIT_MAX > 0:
            _kept = []
            _sk = {"side": 0.10, "top": 0.05, "cgn": 0.07}
            for _c in cand_in:
                if _nbr_hits(_tgt, _c[1], _c[2], GRIP_OPEN_W, _sk.get(_c[5], 0.06)) < SELF_HIT_MAX:
                    _kept.append(_c)
            print(f"[{name}] 접근 중 타깃 접촉 필터: {len(cand_in)} -> {len(_kept)} (그리퍼가 물체를 치는 {len(cand_in) - len(_kept)}개 제외)")
            cand_in = _kept
        if _obs is not None and cand_in:
            _kept = []
            _sk = {"side": 0.10, "top": 0.05, "cgn": 0.07}
            for _c in cand_in:
                _n = _nbr_hits(_obs, _c[1], _c[2], _c[4], _sk.get(_c[5], 0.06))
                if _n < NBR_HIT_MAX:
                    _kept.append(_c)
            print(f"[{name}] 이웃 충돌 필터: {len(cand_in)} -> {len(_kept)} (통로에 장애물 걸린 {len(cand_in) - len(_kept)}개 제외)")
            cand_in = _kept
        # bail_empty: 한 건도 못 푼 채 이 인덱스에 닿으면 중단 (실패 IK ~70ms 씩
        # 다 태우면 낭비 — CGN 전부 불도달 시 빨리 side 폴백으로).
        # HIT_SPAN: 첫 성공 이후로는 그 지점 기준 상대 구간만 더 본다.
        _o = [] if _twin == "flip" else [c + ("orig",) for c in cand_in]
        _f = [] if _twin == "orig" else \
             [(c[0] - 1e-4, c[1], c[2] @ FLIP, c[3], c[4], c[5], c[6], "flip") for c in cand_in]
        _all = _o + _f
        _rs = _reach_scores_batch(name, _all)   # 모델 없으면 전부 0 -> 순수 cost 정렬과 동일
        cc = [t for t, _ in sorted(zip(_all, _rs), key=lambda z: -(z[0][0] + REACH_W * z[1]))]
        # cuRobo IK 는 순위 상위부터 청크(기본 32개)씩 지연 평가한다: 랭킹이 좋으면 첫 청크 하나로 끝나고(보통 0.1~0.2s), 못 풀면 다음 청크를 이어서 본다.
        # 루프의 조기 종료(hit_span/want/bail_empty)가 그대로 평가 범위를 정한다 -- 후보 전체를 풀지 않는다.
        _ck = {}; _ck_state = {"on": bool(CUROBO_IK_URL and cc)}; _CH = int(os.environ.get("NERO_CUROBO_IK_CHUNK", "32"))

        def _ck_need(i):
            if not _ck_state["on"] or i in _ck:
                return
            lo = (i // _CH) * _CH; sub = cc[lo:min(lo + _CH, budget)]
            try:
                res = curobo_ik_batch([(c[1], R2q(c[2])) for c in sub], seed=SEED)
                for k, r in enumerate(res):
                    _ck[lo + k] = r
            except Exception as _e:
                print(f"[{name}] [curobo-ik] 실패({_e!r}) -- 이후 MoveIt IK 로 폴백")
                _ck_state["on"] = False
        ks = []
        hit0 = None                     # 첫 IK 성공 인덱스 (None = 아직 없음)
        _seen = set()                   # 2026-11-04: 평가(또는 건너뜀) 한 cc 인덱스 -- _fill_bins 용
        _bin_fail, _bin_ok, _orig_ok = {}, {}, {}
        _n_skip_bin = _n_skip_twin = 0
        _calls0 = _ik_calls[0]
        for _si, (cost, flange, Rf, a, opening, kind, score, tw) in enumerate(cc[:budget]):
            # 탐색창은 "첫 성공 시점 기준" 상대 구간. 아직 하나도 못 풀었으면
            # bail_empty 까지 계속 찾고(성공이 cost 하위에 몰려 있어도 놓치지 않음),
            # 한 번 풀린 뒤엔 거기서 HIT_SPAN 개만 더 보고 끊는다(랭킹 품질은
            # 그 구간에서 이미 충분 — 예산을 성공 구간에만 쓴다).
            if hit0 is None:
                if _si >= bail_empty:
                    break
            elif _si - hit0 >= hit_span:
                break
            _seen.add(_si)
            _pk = (round(float(flange[0]), 3), round(float(flange[1]), 3), round(float(flange[2]), 3),
                   round(float(a[0]), 2), round(float(a[1]), 2))
            _azd = float(np.degrees(np.arctan2(a[1], a[0])))
            _bk = (int(round(float(flange[0]) / 0.05)), int(round(float(flange[1]) / 0.05)),
                   int(round(float(flange[2]) / 0.05)), int((_azd + 180.0) // 45.0))
            if TWIN_LAZY and tw == "flip" and _orig_ok.get(_pk) and not _ck_state["on"]:     # cuRobo IK 는 싸서 MoveIt 호출 절약용 휴리스틱을 끈다
                _n_skip_twin += 1
                continue
            if BIN_SKIP and not _ck_state["on"] and _bin_ok.get(_bk, 0) == 0 and _bin_fail.get(_bk, 0) >= BIN_SKIP:
                _n_skip_bin += 1
                continue
            q = R2q(Rf)
            contact = np.array(flange) + D_TCP * np.array(a)
            j1s = np.arctan2(contact[1], contact[0]) + np.pi
            j1s = (j1s + np.pi) % (2 * np.pi) - np.pi
            seed_face = [float(np.clip(j1s, -2.9, 2.9)), -0.5, 0.0, 1.4, 0.0, 0.6, 1.5]
            _ck_need(_si)
            if _ck_state["on"] and _si in _ck:
                qsol = _ck[_si]
            else:
                qsol = fik(flange, q)
                # 재시드: 아직 한 건도 못 풀었으면 끝까지, 풀린 뒤엔 기존대로 상위 24 까지만
                if qsol is None and (hit0 is None or _si < 24):
                    qsol = fik(flange, q, seed=seed_face)
            _log_reach(name, kind, tw, cost, score, flange, a, opening,
                       ok=qsol is not None, qsol=qsol)
            if qsol is None:
                _bin_fail[_bk] = _bin_fail.get(_bk, 0) + 1
                continue
            mm = min_margin(qsol)
            if mm < MIN_MARGIN_FLOOR:
                _bin_fail[_bk] = _bin_fail.get(_bk, 0) + 1
                # 관절 한계 근접/초과(예: j1 2.71 vs 한계 2.70526) -- 실물 실행 시 "빙글
                # 도는" 현상 실측(2026-09-17). IK 실패와 동일하게 취급해 후보에서 제외.
                continue
            _bin_ok[_bk] = _bin_ok.get(_bk, 0) + 1
            if tw == "orig":
                _orig_ok[_pk] = True
            if hit0 is None:
                hit0 = _si
                print(f"[{name}] [sieve] 첫 IK 성공 #{_si}/{len(cc)} (kind={kind}) "
                      f"-> 여기서 +{hit_span} 까지 탐색")
            so = {"side": 0.10, "top": 0.05, "cgn": 0.07}.get(kind, 0.06)
            theta = float(np.degrees(np.arccos(np.clip(-np.array(a)[2], -1, 1))))   # 0=down 90=horiz
            cfg_dev = float(sum(abs(qsol[j] - SEED[j]) for j in range(4)))
            ws = wrist_sing(qsol)
            ks.append(dict(object=name, kind=kind, twin=tw, quat=q,
                           approach=[float(x) for x in a],
                           flange=[float(x) for x in flange],
                           pre=[float(x) for x in (np.array(flange) - so * a)],
                           standoff=so,
                           contact=[float(x) for x in (np.array(flange) + D_TCP * a)],
                           opening=opening, radius=None,  # 2026-09-23: 원통 fit 제거로 항상 None
                           grip_close=float(max(0.010, (opening - 0.020) / 2 - 0.003)),
                           score=float(score), cost=float(cost), approach_theta=theta,
                           j1_ik=float(qsol[0]), cfg_dev=cfg_dev, min_margin=mm,
                           j6_ik=float(qsol[5]), wrist_sing=ws,
                           ik_cfg=[float(x) for x in qsol]))
            if len(ks) >= want:
                break
        print(f"[{name}] [sieve] IK 호출 {_ik_calls[0] - _calls0}회 "
              f"(건너뜀: 빈 {_n_skip_bin}, 쌍둥이 {_n_skip_twin}), 도달 {len(ks)}개")
        if BIN_MIN > 0 and cc:
            _fill_bins(ks, cc, _seen, budget, bool(_ck_state["on"]))
        return ks

    def _fill_bins(ks, cc, seen, budget, use_curobo):
        """2026-11-04(NERO_BIN_MIN>0 일 때만): 접근각 구간별 IK 성공이 BIN_MIN 미만이면 그 구간의 미평가 후보를 cc(비용+reach 정렬) 순서대로
        최대 BIN_BUDGET 개 평가해 ks 에 추가한다. 기존 루프 결과는 건드리지 않는다(추가만). 평가는 cuRobo IK 배치 1회/구간(없으면 MoveIt IK)."""
        cnt = [0, 0, 0]
        for k in ks:
            cnt[_theta_bin(k["approach_theta"])] += 1
        before = list(cnt); t_f = time.time(); n_eval = 0
        for b in (1, 2, 0):
            if cnt[b] >= BIN_MIN:
                continue
            pool = [i for i in range(min(len(cc), budget)) if i not in seen
                    and _theta_bin(float(np.degrees(np.arccos(np.clip(-np.array(cc[i][3])[2], -1, 1))))) == b][:BIN_BUDGET]
            if not pool:
                continue
            res = None
            if use_curobo:
                try:
                    res = curobo_ik_batch([(cc[i][1], R2q(cc[i][2])) for i in pool], seed=SEED)
                except Exception as _e:
                    print(f"[{name}] [bin-fill] cuRobo IK 실패({_e!r}) -- MoveIt IK 로 폴백")
                    res = None
            for j, i in enumerate(pool):
                cost, flange, Rf, a, opening, kind, score, tw = cc[i]
                q = R2q(Rf); n_eval += 1
                if res is not None:
                    qsol = res[j]
                else:
                    contact = np.array(flange) + D_TCP * np.array(a)
                    j1s = (np.arctan2(contact[1], contact[0]) + np.pi + np.pi) % (2 * np.pi) - np.pi
                    qsol = fik(flange, q) or fik(flange, q, seed=[float(np.clip(j1s, -2.9, 2.9)), -0.5, 0.0, 1.4, 0.0, 0.6, 1.5])
                _log_reach(name, kind, tw, cost, score, flange, a, opening, ok=qsol is not None, qsol=qsol)
                if qsol is None:
                    continue
                mm = min_margin(qsol)
                if mm < MIN_MARGIN_FLOOR:
                    continue
                so = {"side": 0.10, "top": 0.05, "cgn": 0.07}.get(kind, 0.06)
                theta = float(np.degrees(np.arccos(np.clip(-np.array(a)[2], -1, 1))))
                ks.append(dict(object=name, kind=kind, twin=tw, quat=q, approach=[float(x) for x in a], flange=[float(x) for x in flange],
                               pre=[float(x) for x in (np.array(flange) - so * a)], standoff=so,
                               contact=[float(x) for x in (np.array(flange) + D_TCP * np.array(a))],
                               opening=opening, radius=None, grip_close=float(max(0.010, (opening - 0.020) / 2 - 0.003)),
                               score=float(score), cost=float(cost), approach_theta=theta, j1_ik=float(qsol[0]),
                               cfg_dev=float(sum(abs(qsol[jj] - SEED[jj]) for jj in range(4))), min_margin=mm, j6_ik=float(qsol[5]),
                               wrist_sing=wrist_sing(qsol), ik_cfg=[float(x) for x in qsol], bin_fill=True))
                cnt[b] += 1
                if cnt[b] >= BIN_MIN:
                    break
        print(f"[{name}] [bin-fill] 구간별 도달(탑/기울기/옆) {before} -> {cnt}, 추가 평가 {n_eval}개, {time.time() - t_f:.2f}s")

    def _rerank(ks):
        # cost 유지 − SEED 편차 − 관절여유(0.35 미만 감점) − 손목특이점(j6→0 감점).
        # (2026-09-17: cfg_dev 가중치 상향 재검증 -- 1차 시도는 nero_pick.sh 화이트
        # 리스트 누락으로 무효였고, 화이트리스트 고친 뒤 재검증함. 컵은 j1_travel이
        # 4배 줄었지만(1.79->0.44) 가위/병은 이미 최적후보라 j1_travel 안 바뀌었고
        # C단계(STOMP) 시간은 물체마다 제각각(개선/악화) -- j1_travel과 STOMP 시간
        # 상관관계 가설은 298개 샘플 상관분석(r=0.07~0.17)과 일치하게 기각. 원복.)
        ks.sort(key=lambda d: -(d["cost"] - 0.25 * d["cfg_dev"]
                                - 0.8 * max(0.0, 0.35 - d["min_margin"])
                                - 2.0 * d["wrist_sing"]))
        return ks

    if not cand and not cand_side and not cand_pinch and not cand_top:
        print(f"[{name}] no candidates"); continue

    GATE_K = int(os.environ.get("NERO_SIDE_GATE_K", "3"))
    # ablation: 합성 후보(측면/pinch/탑다운) 전부 끄고 CGN raw 만으로 랭킹 (A/B 대조군)
    CGN_ONLY = os.environ.get("NERO_CGN_ONLY", "") not in ("", "0")
    # 1차: CGN raw 만 IK 체
    keep = _sieve(cand)
    n_cgn_reach = len(keep)
    if CGN_ONLY:
        print(f"[{name}] [ablation] NERO_CGN_ONLY -> 합성 후보(side/pinch/top) 전부 스킵, "
              f"CGN raw reachable {n_cgn_reach}개만 사용")
    # (2026-09-23: 원통 side 합성 게이트 제거 -- cand_side가 이제 항상 비어 있어
    # 죽은 코드였음. approach_pref도 이 게이트 말고는 안 쓰여서 함께 제거.)
    if not CGN_ONLY and unified and cand_pinch:
        if n_cgn_reach < GATE_K:
            add = _sieve(cand_pinch)
            keep += add
            print(f"[{name}] reachable CGN {n_cgn_reach} < {GATE_K} "
                  f"-> 파트 튜브축 pinch 합성 {len(add)}개 추가")
        else:
            print(f"[{name}] reachable CGN {n_cgn_reach} >= {GATE_K} -> pinch 합성 스킵 (CGN 신뢰)")
    # 3차: 그래도 0 -> top-down 합성 최후 폴백
    if not CGN_ONLY and not keep and cand_top:
        keep += _sieve(cand_top)
        print(f"[{name}] reachable 0 -> top-down 합성 폴백 ({len(keep)}개)")

    # 시각화/검사용으로 상한을 늘릴 수 있게 env화 (2026-09-17). 기본값 8은
    # 그대로 -- step6 모션계획에 넘기는 실제 후보수는 안 바뀜, 늘리려면
    # NERO_KEEP_TOPN 만 지정.
    _topn = int(os.environ.get("NERO_KEEP_TOPN", "8"))
    keep = _rerank(keep)
    keep = _alloc(keep, _topn, KEEP_BIN)      # 2026-11-04: KEEP_BIN 비면 기존(_diversify/slice)과 동일
    if KEEP_BIN or BIN_MIN > 0:
        _bc = [0, 0, 0]
        for _k in keep:
            _bc[_theta_bin(_k["approach_theta"])] += 1
        print(f"[{name}] [bin] 최종 keep 구간(탑/기울기/옆) {_bc}")
    dt = time.time() - t0
    if keep:
        n_o = sum(k["twin"] == "orig" for k in keep); n_f = len(keep) - n_o
        k0 = keep[0]
        print("[%s] rerank #0: kind=%s twin=%s min_margin=%.2f j6=%.2f cfg_dev=%.2f  "
              "(%d orig + %d flip)"
              % (name, k0["kind"], k0["twin"], k0["min_margin"], k0["j6_ik"], k0["cfg_dev"], n_o, n_f))
        out = dict(k0); out["candidates"] = keep
        out["n_cgn_raw"] = len(grasps); out["n_cand"] = len(cand) + len(cand_side) + len(cand_pinch)
        out["n_reach"] = len(keep); out["n_reach_cgn"] = n_cgn_reach
        out["n_excluded"] = n_excl; out["unified_rank"] = bool(unified)
        json.dump(out, open(os.path.expanduser(f"~/grasp/{name}_pick.json"), "w"), indent=1)
        print(f"[{name}] {len(keep)} reachable, primary approach {np.round(k0['approach'], 2)} "
              f"contact {np.round(k0['contact'], 3)} kind={k0['kind']} standoff {k0['standoff']} "
              f"-> {name}_pick.json")
    else:
        print(f"[{name}] no reachable grasp")
    print(f"[{name}] {_ik_calls[0]} IK calls, {dt:.2f} s total")
    _flush_reach_log()   # 이번 물체에서 모은 IK 시도 전부(성공+실패) 디스크에 1회 기록

if ik is not None:
    rclpy.shutdown()
