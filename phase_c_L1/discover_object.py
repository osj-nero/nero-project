#!/usr/bin/env python3
"""discover_object.py -- 실시간 pick 루프 (cuRobo 판, 2026-10-01 새로 작성).

동작: 관측(observation) 자세에서 시작 -> joint1 을 돌리며 훑는다 -> 화면에 물체가 걸리면 그 물체를 화면 중앙으로 정렬(j1 재조준 + 1회 촬영)
      -> 곧바로 CGN -> B -> C(cuRobo) -> pick. 같은 화면에 같이 걸린 다른 물체는 "장애물"로만 등록한다(CGN/B/C 없음, 박스 장애물).
      pick 이 끝나면 팔이 관측 자세로 돌아오고, 직전에 물체를 찾은 각도부터 다시 훑는다. 한 번 훑는 동안 더 집을 물체가 없으면 종료.

구성(모두 상주 서비스, ssh/rsync/파이썬 프로세스 시작 없음):
  - 검출: PC 에서 직접(agn_detect.agnostic_masks: 바닥 위 덩어리, 이름 없음)
  - CGN + B: Thor cgn_server.py 의 /plan_upload 한 번(npz 바이트 업로드 -> CGN -> B 를 같은 프로세스에서 실행; B 의 IK 는 PC cuRobo /ik)
  - C: PC cuRobo 서버 /plan (후보 goalset, 이웃 장애물은 이 프로그램이 직접 만든 박스)
  - 실행/판정: 기존 pick_one.sh (chain 모드로 홈 복귀 없이 현재 자세에서 곧바로 실행, 시뮬 정답 판정/물체 치움 포함)
사용: python3 discover_object.py [--sweep a,b,c] [--j7 1.325] [--max-picks N]   (환경: NERO_* 는 pick_one.sh 가 그대로 쓴다)"""
import os as _nd_os; NERO_RUNS_DIR = _nd_os.environ.get("NERO_RUNS_DIR") or _nd_os.path.expanduser("~/nero_runs")  # run logs (override: NERO_RUNS_DIR)
import llm_decide
from thor_net import thor_ip
import llm_recover
import llm_advisor
import argparse, threading, traceback, io, json, math, os, subprocess, sys, time, urllib.parse, urllib.request
import numpy as np
import cv2
import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discover_boxes import Discoverer, bearing_to_j1, J1_LIMIT      # noqa: E402  (노드/카메라/이동 재사용)
from agn_detect import agnostic_masks                                 # noqa: E402
import grounding                                                      # noqa: E402  (접지: 이름 없는 마스크 + Thor CLIP 크롭 채점)
import place_verify                                                  # noqa: E402  (놓은 직후 비전 검증, GT 없음)
import recover_ops                                                    # noqa: E402  (놓은 뒤 비전 판정/국소 재시도)
import insert_ops                                                     # noqa: E402  (삽입 과제 판정)
import place_ops                                                      # noqa: E402  (place 슬롯/정답 판정/시행 기록)
import entity_registry as REG                                       # noqa: E402  (2026-10-13: 엔티티 등록부 저장/복원, NERO_REG_PERSIST=1)

HERE = os.path.dirname(os.path.abspath(__file__))
PLACE_LOG = os.environ.get("NERO_PLACE_LOG", "" + NERO_RUNS_DIR + "/place_log.jsonl")     # place 시행 기록(학습 데이터): 한 줄 = 한 번의 놓기


STATE_PATH = os.path.expanduser(os.environ.get("NERO_RT_STATE", "~/grasp/rt_state.json"))
STATE = dict(stage="시작", target=None, angle=0.0, objs={}, picked=[], stats=dict(picks_ok=0, attempts=0), timing={}, plan=None, log=[], t0=time.time(), insert=None)


def flush_state():
    """UI(visualize_rt_curobo.py)가 읽는 상태 파일. 임시 파일에 쓰고 이름을 바꿔 읽는 쪽이 반쯤 쓴 파일을 보지 않게 한다."""
    try:
        STATE["t"] = time.time()
        tmp = STATE_PATH + ".tmp"
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        json.dump(STATE, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass


def set_state(**kw):
    STATE.update(kw); flush_state()


# ── 색(2026-10-13, NERO_COLOR_MATCH=1, 기본 꺼짐) ── 마스크 안 픽셀의 중앙값 색을 직접 재서 명령의 색 단어와 비교한다(CLIP 이 못 읽을 때의 보조 근거). 이름 하드코딩이 아니라 일반 색 범주(명도/색상각)이다.
_COLOR_WORDS = {"black": "black", "dark": "black", "white": "white", "gray": "gray", "grey": "gray", "red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue", "purple": "purple", "brown": "brown", "pink": "pink"}


def color_name(hsv):
    """OpenCV HSV(H 0~179, S/V 0~255) -> 색 범주. 어두우면 검정, 채도가 낮으면 흰/회색, 아니면 색상각 구간."""
    h, sat, v = [float(x) for x in hsv]
    if v < 70: return "black"
    if sat < 45: return "white" if v > 190 else "gray"
    if h < 8 or h >= 170: return "red" if v >= 110 else "brown"
    if h < 21: return "orange" if v >= 130 else "brown"
    if h < 35: return "yellow"
    if h < 85: return "green"
    if h < 135: return "blue"
    return "purple"


def query_colors(phrase):
    return {_COLOR_WORDS[w] for w in str(phrase or "").lower().replace("|", " ").split() if w in _COLOR_WORDS}


def color_bonus(o, qcols, amount=None):
    """질의에 색 단어가 있고 엔티티 색을 쟀으면 일치 +amount, 불일치 -amount(색 모르면 0). amount 기본 NERO_COLOR_BONUS=1.5: 문구/형태 신호(정답 대 오답 보통 2~6점)를 색이 뒤집지 못하게 작게 둔다."""
    if amount is None: amount = float(os.environ.get("NERO_COLOR_BONUS", "1.5"))
    if not qcols or not o.get("hsv"): return 0.0
    return amount if color_name(o["hsv"]) in qcols else -amount


_T_LOG0 = time.time()


def log(msg):
    print(f"[rt] +{time.time() - _T_LOG0:5.1f}s {msg}", file=sys.stderr, flush=True)      # 2026-10-13: 경과 시간 표시(단계별 시간 계측용, 동작 영향 없음)
    STATE["log"] = (STATE.get("log", []) + [time.strftime("%H:%M:%S ") + str(msg)])[-16:]
    flush_state()


_HOOK = llm_recover.enabled()      # 2026-10-22(기본 꺼짐, NERO_LLM_RECOVER_HOOK=1): 체인 계획 실패 직후 LLM 복구 판단 훅(llm_recover.py). 기존 NERO_LLM_RECOVER(llm_advisor)와 무관
_NPL = os.environ.get("NERO_NOPLAN_LOG", "0") == "1"      # 2026-10-22(기본 꺼짐): 계획 실패 원인 계측용 추가 줄. 켜면 '[noplan-log] ' 로 시작하는 새 줄만 늘고 기존 줄 형식은 그대로
_C1 = os.environ.get("NERO_C1_FASTFAIL", "0") == "1"; _B1 = os.environ.get("NERO_B1_SKIP_RESHOT", "0") == "1"; _FIN = os.environ.get("NERO_FINAL_RETRY_ONCE", "0") == "1"      # 2026-10-30(기본 꺼짐; 로봇 동작 변경 후보, PC 프로세스가 읽음): C1=첫 패스 3시점 모두 파지 후보 없음이면 두 번째 패스 생략, B1=국소 탐색 확정 직후 같은 방위 본 루프 재촬영 생략, FINAL=C1 포기 항목은 끝 재시도에서 C1 끔
_ATT_CAP = int(os.environ.get("NERO_ITEM_ATTEMPT_CAP", "0") or 0)      # 2026-10-30(기본 0=꺼짐; 로봇 동작 변경 후보, PC 프로세스가 읽음): 항목별 연속 실패 패스 상한 -- 한 항목이 파지 후보 없음/계획 실패/못 찾음으로 N 패스 실패하면 홈 재시도·목적지 후보 교체 없이 끝 재시도 큐로 보내고(끝에서 1 패스만 재시도) 다음 항목으로 간다
def _att_capped(it_): return _ATT_CAP > 0 and it_.get("_fails", 0) >= (1 if it_.get("_tried") else _ATT_CAP)
_TRC = os.environ.get("NERO_C_TRACE", "0") == "1"      # 2026-10-23(기본 꺼짐): C 요청에 시도 식별(trace)을 넣고 '[c-trace]' 줄을 남긴다(덤프-시도 대응용). 요청의 추가 키는 서버가 무시한다
_RJL = os.environ.get("NERO_C_REJECT_LOG", "0") == "1"      # 2026-10-23(기본 꺼짐, 서버 재기동 필요): 요청에 reject_log 를 실어 서버가 cand_reject(후보별 거부 단계 코드)를 응답에 넣게 하고 '[c-reject]' 줄을 남긴다
_NPZK = os.environ.get("NERO_B_NPZ_KEEP", "")      # 2026-10-23(기본 꺼짐): CGN+B 로 보내는 점군 npz 를 이 폴더에 <id>_<ms>.npz 로 보존한다(Thor 의 rt_<name>.npz 는 덮어써짐)
def nlog(msg):
    if _NPL: log("[noplan-log] " + msg)


def _c_why_class(why):
    """C 실패 사유 목록 분류(2026-10-22): ik_only = 모든 후보가 연속 IK 없음(장애물/시작 자세와 무관한 거부), place, plan(계획 실패 포함), other, none."""
    w = [str(x) for x in (why or [])]
    if not w: return "none"
    if all("연속 IK 없음" in x for x in w): return "ik_only"
    if any("place 불가" in x for x in w): return "place"
    if any("계획 실패" in x for x in w): return "plan"
    return "other"


def bump(world, oid, key="fails"):
    """등록부에서 이미 지워진 물체(시야 안인데 안 보여 삭제됨)에도 안전하게 실패를 센다."""
    o = world.objs.get(oid)
    if o is not None:
        o[key] += 1


def _scan_snap(v_):
    """2026-11-11(NERO_SCAN_SNAPSHOT=1, 기본 꺼짐 -> 그대로 v_): fast_scan 작업 스레드가 공유 got 에 프레임을 넣고 grab 을 부르는 동안 카메라 콜백이 got 의 rgb/depth 를
    새 영상으로 덮어써, 마스크(깊이)와 게이트·CLIP(컬러)이 다른 시각 영상을 보던 문제(스윕 프레임 게이트 전멸). grab 시작 시점에 got 을 복사해 그 프레임 하나로만 처리한다.
    (fast_scan.py 는 읽기 전용이라 여기서 감쌈. 복사 직전 몇 마이크로초 사이 덮어쓰기는 남을 수 있음)"""
    if os.environ.get("NERO_SCAN_SNAPSHOT", "0") != "1": return v_
    import types as _ty, discover_boxes as _DB
    w_ = _ty.SimpleNamespace(got=dict(v_.got)); w_.fresh_frame = lambda *a_, **k_: True; w_.capture_geom = lambda: _DB.Discoverer.capture_geom(w_)
    return w_


def _replan_wait(oid, k):
    """2026-11-11(NERO_REPLAN_LOOP): C 실패를 판단 쪽에 알리고(<목록파일>.fail<k>.json) 새 자세 목록(<목록파일>.r<k>)을 기다린다. 포기/시간초과면 False."""
    pf = os.environ.get("NERO_PLACE_POSE_LIST_FILE")
    if not pf: return False
    t0 = time.time()
    try: json.dump(dict(k=k, oid=oid, t=t0), open(f"{pf}.fail{k}.json", "w"))
    except Exception as e_: log(f"{oid}: [replan] 실패 알림 쓰기 실패 {e_!r}"); return False
    log(f"{oid}: [replan] C 실패 {k} -- 판단 쪽에 새 자세 목록 요청(같은 인식으로 다시 계획)")
    fn = f"{pf}.r{k}"
    while not os.path.exists(fn) and time.time() - t0 < float(os.environ.get("NERO_REPLAN_WAIT", "300")): time.sleep(0.2)
    if not os.path.exists(fn): log(f"{oid}: [replan] 새 목록 시간 초과"); return False
    time.sleep(0.2); txt = open(fn).read()
    try: j = json.loads(txt)
    except Exception as e_: log(f"{oid}: [replan] 목록 읽기 실패 {e_!r}"); return False
    if not j or (isinstance(j, dict) and j.get("abstain")): log(f"{oid}: [replan] 판단 = 포기 (대기 {time.time() - t0:.1f}s)"); return False
    os.environ["NERO_PLACE_POSE_LIST"] = txt; log(f"{oid}: [replan] 새 자세 목록 받음(대기 {time.time() - t0:.1f}s) -- 같은 인식으로 다시 계획"); return True


def post_json(url, obj, timeout=90):
    rq = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(rq, timeout=timeout).read())


# ── 검출 / 박스 ────────────────────────────────────────────────────────────
def obb_box(P):
    """점군(N,3, base 좌표)에서 장애물 박스: 최소면적 회전 사각형(OBB, AABB 보다 15% 넘게 작을 때) 또는 AABB. step6_scene.py 와 같은 규칙."""
    xy = P[:, :2].astype(np.float32)
    lo, hi = np.percentile(xy, 1.0, axis=0), np.percentile(xy, 99.0, axis=0)
    Q = xy[np.all((xy >= lo) & (xy <= hi), axis=1)]
    if len(Q) < 20:
        Q = xy
    ax = np.maximum(Q.max(0) - Q.min(0), 0.03)
    zt = float(np.percentile(P[:, 2], 98))
    cx, cy = [float(v) for v in (Q.max(0) + Q.min(0)) / 2.0]; yaw = 0.0; raw = [float(ax[0]), float(ax[1]), zt]
    try:
        (ocx, ocy), (ow, oh), ang = cv2.minAreaRect(Q)
        ow, oh = max(float(ow), 0.03), max(float(oh), 0.03)
        if ow * oh < 0.85 * ax[0] * ax[1]:
            cx, cy, yaw, raw = float(ocx), float(ocy), float(np.radians(ang)), [ow, oh, zt]
    except Exception:
        pass
    return dict(xyz=[cx, cy, zt / 2.0], raw=raw, yaw=yaw)


def aabb_box(P, margin=0.01):
    lo, hi = P.min(0), P.max(0)
    d = np.maximum(hi - lo, 0.03) + margin
    return dict(xyz=[float(x) for x in (lo + hi) / 2.0], dims=[float(x) for x in d])


VLM_FUT = {}      # oid -> [Future]  (NERO_VLM_LABEL=1: 크롭을 비동기로 VLM 에 보내고 결과는 스캔 끝 출력 때 모은다 -- 추론 시간이 스캔 뒤에 숨는다)


def _vlm_crop(rgb, mask, pad=0.15):
    ys, xs = np.where(mask)
    if len(xs) < 40: return None
    H, W = rgb.shape[:2]; x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1; pw, ph = pad * (x1 - x0), pad * (y1 - y0)
    cr = rgb[int(max(0, y0 - ph)):int(min(H, y1 + ph)), int(max(0, x0 - pw)):int(min(W, x1 + pw)), :3]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(cr.astype(np.uint8))[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


# ── 예열 실행기: C 결과가 나오자마자 출발하도록 exec_pick/exec_place 를 미리 띄워 둔다(초기화 약 2~4s: bash 환경, rclpy 임포트, 액션 서버 대기) ──
_WARM = []


def warm_start(script, oid, extra_env=None):
    go = f"/tmp/warm_go_{script[:-3]}_{oid}"
    try:
        os.remove(go)
    except OSError:
        pass
    try:
        os.remove(f"/tmp/{oid}.warm_done")
    except OSError:
        pass
    env = dict(os.environ, NERO_EXEC_GO_FILE=go, **(extra_env or {}))
    lf = open(f"/tmp/warm_{script[:-3]}_{oid}.log", "w")
    args = [sys.executable, "-u", os.path.join(HERE, script)] + ([oid] if script == "exec_place.py" else [])
    w = dict(p=subprocess.Popen(args, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT), go=go, script=script, oid=oid, used=False, log=lf.name)
    _WARM.append(w)
    return w


def warm_go(w):
    w["used"] = True; open(w["go"], "w").close()


def warm_cleanup(only_unused=True):
    """신호를 안 준(쓰지 않은) 예열 프로세스를 정리한다."""
    for w in list(_WARM):
        if (not only_unused or not w["used"]) and w["p"].poll() is None:
            try:
                w["p"].terminate()
            except Exception:
                pass
        if w["p"].poll() is not None:
            _WARM.remove(w)


def keep_exec_log(w, oid, tag):
    """실행기(exec_pick/exec_place) 출력과 궤적 기록을 시행 폴더에 남긴다(NERO_MULTI_LOG). 동작에는 영향 없음 -- 예열 로그가 물체별로 덮어써져 -4 원인 추적이 안 됐다."""
    d = os.environ.get("NERO_MULTI_LOG")
    if not d:
        return
    try:
        with open(os.path.join(d, "exec_all.log"), "a") as f:
            f.write(f"\n===== {time.strftime('%H:%M:%S')} {tag} {oid} =====\n" + open(w["log"], errors="replace").read())
        if tag == "pick":
            import shutil; os.makedirs(os.path.join(d, "exec_trace"), exist_ok=True)
            shutil.copy("/tmp/exec_trace_last.json", os.path.join(d, "exec_trace", f"{time.strftime('%H%M%S')}_{oid}.json"))
    except Exception:
        pass


def warm_finish_pick(w, oid):
    """exec_pick 예열 프로세스가 끝나면 exec_multi.sh 가 하던 결과 복사를 하고 pick_one.sh 에 완료 신호를 준다(별도 스레드)."""
    def _run():
        w["p"].wait()
        keep_exec_log(w, oid, "pick")
        for src, dst in (("/tmp/exec_pick_res.json", f"/tmp/{oid}.exec_res.json"), ("/tmp/exec_trace_last.json", f"/tmp/{oid}.exec_trace.json")):
            try:
                import shutil; shutil.copy(src, dst)
            except Exception:
                pass
        open(f"/tmp/{oid}.warm_done", "w").close()
    threading.Thread(target=_run, daemon=True).start()


def ent_event(msg):
    """2026-10-13(기본 꺼짐, NERO_RVIZ_ENT=1): 엔티티 등록 과정 이벤트를 RViz 카드용 상태에 남긴다(최근 6개). 동작에는 영향 없음."""
    if os.environ.get("NERO_RVIZ_ENT") != "1": return
    STATE["ent_view"] = True; STATE["ent_events"] = (STATE.get("ent_events", []) + [time.strftime("%H:%M:%S ") + msg])[-6:]; flush_state()


class World:
    """지금까지 본 물체들(집기 전). 같은 xy(7cm 이내)면 같은 물체로 보고 갱신한다. 이웃 충돌 장애물의 원천."""

    def __init__(self):
        self.objs = {}; self._k = 0

    def new_id(self):
        self._k += 1
        return f"obj_{self._k}"

    def match(self, xy, tol=0.07):
        for _oid, _o in self.objs.items():      # 2026-10-14: 입력으로 준 목적지는 발자국(+3cm) 안의 검출을 흡수한다 -- 큰 용기의 중심이 부분 시야로 흔들려도 복제 엔티티가 안 생긴다
            if _o.get("given") and bool(self._in_box(np.array(xy[:2], float), dict(xyz=_o["xyz"], raw=_o["raw"], yaw=_o["yaw"]), 0.03)[0]): return _oid
        best, bd = None, tol
        for oid, o in self.objs.items():
            if o.get("role") == "placed" and os.environ.get("NERO_KEEP_PLACED", "0") == "1": continue      # 2026-10-13: 놓은 물체 기록은 새 검출을 흡수하지 않는다(용기를 복제해 NO_PLAN 을 만들었다)
            d = float(np.hypot(o["xyz"][0] - xy[0], o["xyz"][1] - xy[1]))
            if d < bd:
                best, bd = oid, d
        return best

    @staticmethod
    def _in_box(xy, box, pad):
        """xy 점(들)(N,2)이 box(OBB, xyz/raw/yaw) 바닥 사각형(+pad) 안에 드는지(N 불리언)."""
        c = np.array(box["xyz"][:2], float); cs, sn = math.cos(-box["yaw"]), math.sin(-box["yaw"]); dp = np.atleast_2d(xy) - c
        lx = cs * dp[:, 0] - sn * dp[:, 1]; ly = sn * dp[:, 0] + cs * dp[:, 1]
        return (np.abs(lx) < box["raw"][0] / 2 + pad) & (np.abs(ly) < box["raw"][1] / 2 + pad)

    def match_tentative(self, d):
        """정지 프레임 검출을 임시 엔티티에 맞춘다(스윕 위치는 4~7cm 어긋나므로 허용 거리 NERO_ENT_MERGE_TOL, 높이 차 제한 없음 -- 스윕이 본 높이도 못 믿는다)."""
        tol = float(os.environ.get("NERO_ENT_MERGE_TOL", "0.12")); best, bd = None, tol
        for oid, o in self.objs.items():
            if not o.get("tentative") or "w" not in o or (o.get("role") == "placed" and os.environ.get("NERO_KEEP_PLACED", "0") == "1"): continue
            dist = float(np.hypot(o["w"][0] - d["w"][0], o["w"][1] - d["w"][1]))
            if dist < bd: best, bd = oid, dist
        return best

    def match_gated(self, d):
        """2026-10-13(회귀 비교 통과로 기본 켜짐, NERO_ENT_MERGE=0 으로 끔): 연속 스윕에서 온 검출(게이트에 걸린 프레임)은 위치가 4~7cm 어긋나서(실측: 라쿤 첫 정지 프레임과 스윕 프레임의 중심 거리 6.7~7.1cm) 7cm 기준에 걸려 같은 물체가 두 엔티티가 된다.
        게이트된 검출만 허용 거리를 NERO_ENT_MERGE_TOL(기본 0.12m)로 넓히되, 윗면 높이 차가 NERO_ENT_MERGE_DH(기본 3cm) 이내인 기존 엔티티일 때만 합친다(접시 위 장난감처럼 높이가 다른 이웃은 합치지 않음)."""
        if not d.get("gated"): return None
        tol = float(os.environ.get("NERO_ENT_MERGE_TOL", "0.12")); dh = float(os.environ.get("NERO_ENT_MERGE_DH", "0.03")); best, bd = None, tol
        for oid, o in self.objs.items():
            if "raw" not in o or "w" not in o or (o.get("role") == "placed" and os.environ.get("NERO_KEEP_PLACED", "0") == "1"): continue
            if abs(o["raw"][2] - d["box"]["raw"][2]) > dh: continue
            dist = float(np.hypot(o["w"][0] - d["w"][0], o["w"][1] - d["w"][1]))
            if os.environ.get("NERO_ENT_MERGE_FOOT", "0") == "1":      # 2026-10-14(기본 꺼짐): 중심 거리 대신 발자국 기준 -- 검출 중심이 기존 엔티티 발자국(반변 최댓값 + NERO_ENT_FOOT_PAD)안일 때만 합친다. 7~8cm 장난감 이웃(중심 거리 8.6cm, 반변 2~4cm)은 합쳐지지 않고, 연필통처럼 큰 물체의 어긋난 부분 검출(6cm)은 합쳐진다
                _hx = max(o["raw"][0], o["raw"][1], d["box"]["raw"][0], d["box"]["raw"][1]) / 2.0 + float(os.environ.get("NERO_ENT_FOOT_PAD", "0.02"))
                if dist > _hx: continue
            if dist < bd: best, bd = oid, dist
        if best is not None and os.environ.get("NERO_ENT_TRACE"): log(f"[ent] 게이트 검출 w=({d['w'][0]:+.3f},{d['w'][1]:+.3f}) 박스 {[round(100 * x_, 1) for x_ in d['box']['raw']]}cm 를 {best} 에 합침(거리 {bd:.3f}m; {best} w=({self.objs[best]['w'][0]:+.3f},{self.objs[best]['w'][1]:+.3f}) 박스 {[round(100 * x_, 1) for x_ in self.objs[best]['raw']]}cm)")
        if best is not None: ent_event(f"스윕 검출을 {best} 에 합침 (거리 {100 * bd:.1f}cm)")
        return best

    def match_overlap(self, d):
        """중심 거리로 못 찾은 검출을 겹침으로 다시 찾는다(2026-10-07). 접시처럼 큰 물체는 화면 경계에 잘린 프레임에서 점 중앙값이 실제 중심에서 7cm 넘게 벗어나 같은 물체가 두 ID 로 등록됐다.
        (a) 잘린 검출의 점 60% 이상이 기존 물체 바닥 사각형 안 / (b) 기존 물체 중심이 새 검출의 박스 안, 그리고 높이(윗면)가 3cm 이내로 같을 때만 같은 물체로 본다 -- 접시 위에 놓인 더 높은 물체는 합치지 않는다."""
        nb = d["box"]; P = d["P"]; best, bs = None, 0.0
        for oid, o in self.objs.items():
            if "raw" not in o or abs(o["raw"][2] - nb["raw"][2]) > 0.03 or (o.get("role") == "placed" and os.environ.get("NERO_KEEP_PLACED", "0") == "1"): continue
            ob = dict(xyz=o["xyz"], raw=o["raw"], yaw=o["yaw"]); score = 0.0
            if d["margin"] < 8: score = float(self._in_box(P[:, :2], ob, 0.03).mean())
            if score < 0.6 and self._in_box(np.array(o["xyz"][:2]), nb, 0.02)[0] and o["raw"][0] * o["raw"][1] < nb["raw"][0] * nb["raw"][1]: score = 0.6
            if score >= 0.6 and score > bs: best, bs = oid, score
        return best

    def update(self, dets, cur_j1):
        ids = []
        for d in dets:
            _stat = os.environ.get("NERO_ENT_STATIONARY", "0") == "1"      # 2026-10-13(기본 꺼짐): 연속 스윕(게이트) 검출은 기하를 못 믿는다 -> 기존 엔티티는 갱신하지 않고, 새 물체는 임시(tentative)로만 등록해 정지 프레임이 확정하게 한다
            oid = self.match(d["w"][:2])
            if oid is None: oid = self.match_overlap(d)
            if oid is None and (_stat or os.environ.get("NERO_ENT_MERGE", "1") == "1"): oid = self.match_gated(d)
            if oid is None and _stat and not d.get("gated"): oid = self.match_tentative(d)
            if oid is not None and _stat and d.get("gated"):
                self.objs[oid]["seen"] += 1; self.objs[oid]["blk"] = self.objs[oid].get("blk", 0) + 1; ids.append(oid); continue
            if oid is None:
                oid = self.new_id(); self.objs[oid] = dict(fails=0, defer_gen=-1, seen=0)
                if _stat and (d.get("gated") or d["margin"] < 8): self.objs[oid]["tentative"] = True
                ent_event(f"{oid} 새 엔티티 ({'임시: ' + ('스윕 프레임' if d.get('gated') else '가장자리에 걸려 일부만 보임') if self.objs[oid].get('tentative') else '정지 프레임'})")      # 게이트에 걸렸거나 화면 가장자리에 걸려 일부만 보인 첫 관측도 임시(온전한 정지 프레임이 확정)
                if os.environ.get("NERO_ENT_TRACE"): log(f"[ent] 새 엔티티 {oid} j1={cur_j1:+.2f} w=({d['w'][0]:+.3f},{d['w'][1]:+.3f}) 박스 {[round(100 * v, 1) for v in d['box']['raw']]}cm margin={d['margin']} 게이트={bool(d.get('gated'))} 잘림={bool(d.get('cut'))} 점 {len(d['P'])}개 | 기존과의 거리(박스중심/w): " + ", ".join(f"{k_}:{np.hypot(o_['xyz'][0] - d['w'][0], o_['xyz'][1] - d['w'][1]):.3f}/{np.hypot(o_['w'][0] - d['w'][0], o_['w'][1] - d['w'][1]):.3f}" for k_, o_ in self.objs.items() if k_ != oid and "xyz" in o_ and "w" in o_ and np.hypot(o_["w"][0] - d["w"][0], o_["w"][1] - d["w"][1]) < 0.15))
            o = self.objs[oid]; o.pop("stale", None); o.pop("restored", None)      # 이전 명령에서 복원한 엔티티는 다시 보이면 확인된 것
            if o.get("frozen") and os.environ.get("NERO_DEST_FREEZE", "0") == "1":      # 2026-10-14(기본 꺼짐): 초반 조사로 위치/형상/홀 추정을 끝낸 목적지는 이후 프레임(스윕, 재채점)으로 덮지 않는다 -- 놓기 직전 호버 보정만 예외
                o["seen"] = o.get("seen", 0) + 1; o["miss"] = 0; ids.append(oid); continue
            if _stat and not d.get("gated") and d["margin"] >= 8 and o.get("tentative"):      # 정지 프레임이 임시 엔티티를 확정: 스윕에서 얻은 기하를 버리고 이 프레임으로 새로 채운다
                o.pop("tentative", None); o.pop("raw", None); o.pop("pts", None)
                if os.environ.get("NERO_ENT_TRACE"): log(f"[ent] {oid} 정지 프레임이 확정(스윕 기하 폐기)")
                ent_event(f"{oid} 정지 프레임이 확정 (스윕 기하 폐기, 스윕 갱신 {o.get('blk', 0)}회 차단했었음)")
            _pre = (list(o["raw"]), list(o["xyz"][:2]), len(o["pts"]) if o.get("pts") is not None else 0) if "raw" in o else None      # 2026-10-13: 갱신 전후 추적용(NERO_ENT_TRACE, 로그만)
            if d["margin"] >= 8 or "raw" not in o:          # 온전히 보인 프레임은 그대로 덮어쓴다(CGN 에 쓴 프레임과 같은 최신 정보)
                o.update(xyz=d["box"]["xyz"], raw=d["box"]["raw"], yaw=d["box"]["yaw"])
            else:
                # 화면 경계에 잘려 보인 프레임: 기존 박스를 버리지도, 잘린 작은 박스로 덮어쓰지도 않고 합친다(치수는 큰 쪽, 위치/방향은 면적이 큰 쪽).
                ob, nb = o, d["box"]
                # 2026-10-07: 잘린 시야는 모두 실제 물체의 일부라서 두 박스의 합집합(기존 박스 방향 기준 경계 사각형)으로 합친다. 이전(큰 쪽 위치 + 치수 최대)은 시야보다 큰 용기(바구니 30cm)에서 중심이 8~9cm 어긋났다.
                _y = float(ob["yaw"]); _c, _s = math.cos(-_y), math.sin(-_y); _pts = []
                for _b in (ob, nb):
                    _hx, _hy, _by = _b["raw"][0] / 2, _b["raw"][1] / 2, float(_b["yaw"]); _cb, _sb = math.cos(_by), math.sin(_by)
                    for _sx, _sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        _wx = _b["xyz"][0] + _cb * _sx * _hx - _sb * _sy * _hy; _wy = _b["xyz"][1] + _sb * _sx * _hx + _cb * _sy * _hy
                        _pts.append((_c * (_wx - ob["xyz"][0]) - _s * (_wy - ob["xyz"][1]), _s * (_wx - ob["xyz"][0]) + _c * (_wy - ob["xyz"][1])))
                _lx0, _lx1 = min(p_[0] for p_ in _pts), max(p_[0] for p_ in _pts); _ly0, _ly1 = min(p_[1] for p_ in _pts), max(p_[1] for p_ in _pts)
                _mx, _my = (_lx0 + _lx1) / 2, (_ly0 + _ly1) / 2; _cy_, _sy_ = math.cos(_y), math.sin(_y)
                _skipu = os.environ.get("NERO_ENT_UNION_GUARD", "0") == "1" and (_lx1 - _lx0) * (_ly1 - _ly0) > float(os.environ.get("NERO_ENT_UNION_MAXRATIO", "1.5")) * max(ob["raw"][0] * ob["raw"][1], nb["raw"][0] * nb["raw"][1])      # 2026-10-28(기본 0): 합집합 면적이 기존/새 박스 면적의 1.5배를 넘으면(이웃과 합쳐진 부분 프레임) 기하를 합치지 않고 기존 값을 유지
                if not _skipu:
                    ob["xyz"] = [ob["xyz"][0] + _cy_ * _mx - _sy_ * _my, ob["xyz"][1] + _sy_ * _mx + _cy_ * _my, ob["xyz"][2]]
                    ob["raw"] = [_lx1 - _lx0, _ly1 - _ly0, max(ob["raw"][2], nb["raw"][2])]
            if _pre is not None and os.environ.get("NERO_ENT_TRACE"):
                _a0 = _pre[0][0] * _pre[0][1]; _a1 = o["raw"][0] * o["raw"][1]
                log(f"[upd] {oid} j1={cur_j1:+.2f} margin={d['margin']} 게이트={bool(d.get('gated'))} 잘림={bool(d.get('cut'))} 점 {len(d['P'])}개 | 박스 {[round(100 * v, 1) for v in _pre[0]]} -> {[round(100 * v, 1) for v in o['raw']]} 바닥면적 {100 * 100 * _a0:.0f}->{100 * 100 * _a1:.0f}cm2 ({'축소' if _a1 < 0.8 * _a0 else ('확대' if _a1 > 1.25 * _a0 else '유지')}) 중심이동 {100 * math.hypot(o['xyz'][0] - _pre[1][0], o['xyz'][1] - _pre[1][1]):.1f}cm")
            if d.get("feat") is not None and not d.get("gated"):      # 임베딩은 마스크와 컬러가 정렬된 프레임에서만 방향 평균으로 누적(최대 20프레임)
                _n0 = min(int(o.get("feat_n", 0)), 19); _f = np.asarray(d["feat"], float); _s = (np.asarray(o["feat"], float) * _n0 if (o.get("feat") is not None and _n0) else 0.0) + _f
                o["feat"] = (_s / (np.linalg.norm(_s) + 1e-9)).tolist(); o["feat_n"] = _n0 + 1
            if d.get("_crop") is not None and len(VLM_FUT.get(oid, [])) < 2:      # 2026-10-19: VLM 라벨(비동기) -- 엔티티당 서로 다른 프레임 2회
                _fu = grounding.vlm_submit(d.pop("_crop"))
                if _fu is not None: VLM_FUT.setdefault(oid, []).append(_fu)
            if d.get("hsv") and not d.get("gated"):      # 색은 마스크와 컬러가 정렬된 프레임에서만 쌓는다(최근 7개 중앙값)
                _hh = (o.get("hsv_hist") or []) + [d["hsv"]]; o["hsv_hist"] = _hh[-7:]; o["hsv"] = [int(x) for x in np.median(np.array(o["hsv_hist"]), axis=0)]
            o["w"] = [float(x) for x in d["w"]]; o["seen"] += 1
            if d.get("fobb") is not None: o.setdefault("fobbs", []).append(dict(d["fobb"], gated=bool(d.get("gated"))))      # 2026-10-28(NERO_PERCEIVE_FRAMEOBB)
            if d["margin"] >= 8 or "pts" not in o:         # 도착지 면 높이 추정용 점군(다운샘플). 온전히 보인 프레임으로 갱신
                Pd = d["P"]; o["pts"] = Pd[:: max(1, len(Pd) // 700)].astype(np.float32)
            if d.get("label"):
                if os.environ.get("NERO_LABEL_BEST", "0") == "1":      # 2026-10-13(기본 꺼짐): 라벨은 마지막 프레임이 아니라 가장 믿을 만한 프레임(게이트 안 걸림 > 온전히 보임 > 최고 점수)의 것을 유지한다
                    _lq = (not d.get("gated"), d["margin"] >= 8, float((d.get("top") or [("", -9.0)])[0][1]))
                    if _lq >= o.get("_lq", (False, False, -99.0)): o["label"] = d["label"]; o["top"] = d.get("top", []); o["_lq"] = _lq
                else:
                    o["label"] = d["label"]; o["top"] = d.get("top", [])
                    if os.environ.get("NERO_RVIZ_ENT") == "1" and not d.get("gated"):      # 2026-10-13: RViz 표시용 라벨은 최근 6개 정지 프레임의 다수결(실시간 갱신은 유지하되 프레임마다 뒤바뀌지 않게)
                        _lh = (o.get("lab_hist") or []) + [d["label"]]; o["lab_hist"] = _lh[-6:]
                        o["label_disp"] = max(set(o["lab_hist"]), key=lambda x_: (o["lab_hist"].count(x_), len(o["lab_hist"]) - 1 - o["lab_hist"][::-1].index(x_)))
            if d["margin"] >= 8 or "s_t" not in o:       # 접지 점수도 온전히 보인 프레임만 반영(잘린 크롭은 점수가 흔들린다)
                for _k in ("s_t", "s_x", "s_p", "p_t"):
                    # 2026-10-01 최신 값으로 덮지 않고 최고값을 유지한다: 같은 물체도 시점에 따라 점수가 크게 흔들려(7.7 -> 음수) 사전 훑기에서 찾은 대상이 본 스윕에서 탈락하는 일이 있었다.
                    _new, _old = d.get(_k), o.get(_k)
                    o[_k] = _new if _old is None else (_old if _new is None else max(_old, _new))
            ids.append(oid)
        # 현재 시야(bearing 이 j1 근처)에 있어야 하는데 안 보이는 물체는 사라진 것(치워졌거나 옮겨짐)으로 본다
        _skipmiss = os.environ.get("NERO_MISS_SKIP_GATED", "1") == "1" and (not dets or all(d.get("gated") for d in dets))      # 2026-10-13(기본 꺼짐): 위치가 어긋나는 스윕(게이트) 프레임이나 검출이 하나도 없는 프레임에서는 '안 보임'으로 세지 않는다(실측: 스윕 중 홀더/연필통/기둥이 지워져 도착지가 사라짐)
        for oid in list(self.objs):
            if oid in ids:
                self.objs[oid]["miss"] = 0
                continue
            if self.objs[oid].get("given"): continue      # 입력으로 준 목적지는 안 보여도 지우지 않는다
            if _skipmiss: continue
            if os.environ.get("NERO_KEEP_PLACED", "0") == "1" and self.objs[oid].get("role") == "placed": continue      # 2026-10-13(기본 꺼짐): 이미 놓은 물체는 용기 안에 가려져 안 보여도 지우지 않는다(실측: 복원된 placed 가 첫 스윕 중 '안 보임'으로 지워져 놓을 칸 분산이 아무것도 못 피했음)
            o = self.objs[oid]; x, y = o["w"][0], o["w"][1]
            # 2026-10-07: (1) 큰 물체(30cm 바구니)는 중심 방위가 시야 근처여도 화면에 다 안 들어오므로, 각도 반폭만큼 "시야 안" 조건을 좁힌다. (2) 한 번 안 보인 것으로 지우지 않고 연속 miss 회수로 지운다(도착지는 더 늦게).
            _half = math.atan2(0.5 * max(o["raw"][0], o["raw"][1]), max(math.hypot(x, y), 0.2)) if "raw" in o else 0.0
            if abs(bearing_to_j1(x, y) - cur_j1) < max(0.05, 0.35 - _half):
                o["miss"] = o.get("miss", 0) + 1
                need = int(os.environ.get("NERO_MISS_DEST", "3")) if o.get("role") == "dest" else int(os.environ.get("NERO_MISS_N", "1"))
                if o["miss"] >= need:
                    log(f"{oid}: 시야 안인데 {o['miss']}번 연속 안 보임 -- 장애물에서 제외")
                    del self.objs[oid]
                else:
                    log(f"{oid}: 시야 안인데 안 보임({o['miss']}/{need}) -- 유지")
        return ids

    def obstacles(self, exclude):
        # 2026-10-08: 집기에 실패한 물체(팔이 건드려 쓰러졌을 수 있다)는 다시 보이기 전까지 옛 위치를 장애물로 남기지 않는다(stale). 쓰러진 종의 옛 위치가 링 접근을 막았다.
        _conf = os.environ.get("NERO_OBS_CONFIRMED", "0") == "1"      # 2026-10-14(기본 꺼짐, NERO_LEAN 이 켬): 정지 프레임에서 한 번도 확인되지 않은 가상 물체(라벨/점수/색 없음 = 스윕의 조각 검출)는 계획 장애물에서 뺀다 -- 요시 시행에서 (None, None) 엔티티 3개와 바구니 복제가 C 를 막았다
        _ob = [dict(id="other_" + oid, xyz=o["xyz"], raw=o["raw"], yaw=o["yaw"]) for oid, o in self.objs.items() if oid != exclude and "raw" in o and not o.get("stale")
                and not (_conf and not o.get("role") and o.get("label") is None and o.get("s_t") is None and o.get("s_p") is None and not o.get("hsv_hist") and not o.get("clearing"))]
        if os.environ.get("NERO_SHELF_CELL", "0") == "1":      # 2026-11-04(기본 꺼짐): 선반 칸 상자 제거 + 판 box 추가(shelf_cell.py)
            import shelf_cell as _SC; _ob = _SC.apply_obstacles(self.objs, _ob)
        return _ob


# ── 한 프레임 ───────────────────────────────────────────────────────────
GROUND = dict(queries=[], excludes=[], place_queries=[], vocab=[], delta=1.5, floor=1.0, refs=dict(t=None, x=None))   # 접지 설정(main 이 채운다)


def grab(n, floor_z, target=None, full_probs=False):
    """한 프레임: 이름 없는 마스크 -> 물체별 {위치, 박스, 접지 점수}. 질의/제외 문구가 있으면 CLIP 으로 크롭마다 점수를 구한다.
    점수 s = 로짓(문구) - 로짓("an object")  (같은 크롭 안에서 뺀 값 -- 크롭마다 다른 기준선을 제거). 판정은 장면 안 상대비교(eligible)로 한다."""
    if not n.fresh_frame():
        return None
    geom = n.capture_geom(); H, W = geom["H"], geom["W"]
    if os.environ.get("NERO_FRAME_DUMP"):      # 2026-10-28(기본 꺼짐): 스캔 프레임(RGB·깊이·자세·K)을 저장(H2 마스크 홀드아웃 수집용; 저장 직후 h2_hook.py 가 규칙으로 분류, 개발 세션은 격리분을 열람하지 않음)
        try:
            grab._cnt = getattr(grab, "_cnt", 0) + 1; _fx = geom["fx"]; np.savez_compressed(os.path.join(os.environ["NERO_FRAME_DUMP"], f"f_{os.environ.get('NERO_FRAME_TAG', 'x_0')}_{grab._cnt:03d}.npz"),
                rgb=np.ascontiguousarray(n.got["rgb"][..., :3].astype(np.uint8)), depth=geom["zf"].astype(np.float32), W_T_cam=geom["W_T_cam"], K=np.array([geom["fx"], geom["fy"], geom["cx"], geom["cy"]]), stamps=np.array([n.got.get("depth_stamp", np.nan), n.got.get("rgb_stamp", np.nan), n.got.get("tf_stamp", np.nan), n.got.get("wall_capture", np.nan)]), arm_q=np.array(n.got.get("arm_q", [np.nan] * 7)))
        except Exception as _e: log(f"프레임 덤프 실패 {_e!r}")
    dets = []
    for m, bb, conf in agnostic_masks(geom, floor_z=floor_z, max_count=12):
        sel = m & (geom["zf"] > 0.1) & (geom["zf"] < 2.0) & (geom["pts_base"][..., 2] > floor_z)
        if int(sel.sum()) < 50:
            continue
        P = geom["pts_base"][sel][:, :3]
        w = np.median(P, axis=0); w = np.array([w[0], w[1], 0.05])
        ys, xs = np.where(m)
        margin = int(min(xs.min(), ys.min(), W - 1 - xs.max(), int(0.72 * H) - ys.max()))
        _fobb = None
        if os.environ.get("NERO_PERCEIVE_FRAMEOBB", "0") == "1":      # 2026-10-28(기본 꺼짐): 프레임별 트리밍 없는 최소면적 사각형(중심·크기·yaw) -- 등록부가 최신 프레임 박스(1~99 분위 트리밍)로 덮어써 크기가 작고 흔들리는 것을 보완(인식 모드가 여러 프레임 중앙값으로 쓴다)
            try: (_cx, _cy), (_fw, _fd), _fa = cv2.minAreaRect(P[:, :2].astype(np.float32)); _fobb = dict(c=[float(_cx), float(_cy)], wd=[float(_fw), float(_fd)], yaw=float(math.radians(_fa)), n=int(len(P)), margin=int(margin), gated=False)
            except Exception: _fobb = None
        dets.append(dict(fobb=_fobb, mask=m, w=w, P=P, box=obb_box(P), margin=margin, conf=float(conf), label="", s_t=None, s_x=None, s_p=None, probs={}, logits={}, cut=bool(xs.min() <= 2 or xs.max() >= W - 3 or ys.min() <= 2)))
    if dets and int(os.environ.get("NERO_FRAME_MULTI", "1")) > 1 and os.environ.get("NERO_PERCEIVE_FRAMEOBB", "0") == "1":      # 2026-10-29(기본 1 = 기존): 같은 스톱에서 추가 프레임 K-1장(대기 없이 새 프레임 도착만 기다림)의 프레임별 사각형을 구해 검출마다 중앙값으로 합친다(중심·크기·yaw). 합칠 대응은 같은 프레임 안 사각형 중심 거리 <3 cm 이고 크기 비 15% 이내인 것만
        try:
            _K = int(os.environ["NERO_FRAME_MULTI"]); _sv = dict(n.got); _lists = [[d["fobb"]] if d.get("fobb") is not None else [] for d in dets]
            for _k in range(_K - 1):
                if not n.fresh_frame(_nosleep=True): continue
                _g2 = n.capture_geom()
                for _m2, _bb2, _c2 in agnostic_masks(_g2, floor_z=floor_z, max_count=12):
                    _s2 = _m2 & (_g2["zf"] > 0.1) & (_g2["zf"] < 2.0) & (_g2["pts_base"][..., 2] > floor_z)
                    if int(_s2.sum()) < 50: continue
                    _P2 = _g2["pts_base"][_s2][:, :3]; (_cx2, _cy2), (_fw2, _fd2), _fa2 = cv2.minAreaRect(_P2[:, :2].astype(np.float32)); _f2 = dict(c=[float(_cx2), float(_cy2)], wd=[float(_fw2), float(_fd2)], yaw=float(math.radians(_fa2)), n=int(len(_P2)))
                    _best = None
                    for _i2, _d2 in enumerate(dets):
                        _f1 = _d2.get("fobb")
                        if _f1 is None: continue
                        _dd = math.hypot(_f1["c"][0] - _f2["c"][0], _f1["c"][1] - _f2["c"][1]); _sz = sorted(_f1["wd"]); _sz2 = sorted(_f2["wd"])
                        if _dd < 0.03 and abs(_sz[0] - _sz2[0]) <= 0.15 * _sz[0] and abs(_sz[1] - _sz2[1]) <= 0.15 * _sz[1] and (_best is None or _dd < _best[0]): _best = (_dd, _i2)
                    if _best is not None: _lists[_best[1]].append(_f2)
            for _d2, _fl in zip(dets, _lists):
                if len(_fl) < 2: continue
                _ry = _fl[0]["yaw"]; _rows = []
                for _f in _fl:
                    _kk = min(range(4), key=lambda k: abs(((_f["yaw"] + k * math.pi / 2) - _ry + math.pi) % (2 * math.pi) - math.pi)); _yy = _f["yaw"] + _kk * math.pi / 2
                    _rows.append(((_f["wd"][0], _f["wd"][1]) if _kk % 2 == 0 else (_f["wd"][1], _f["wd"][0]), ((_yy - _ry + math.pi) % (2 * math.pi)) - math.pi, _f["c"], _f["n"]))
                _d2["fobb"] = dict(c=[float(np.median([r[2][0] for r in _rows])), float(np.median([r[2][1] for r in _rows]))], wd=[float(np.median([r[0][0] for r in _rows])), float(np.median([r[0][1] for r in _rows]))], yaw=float(np.median([r[1] for r in _rows])) + _ry, n=int(np.median([r[3] for r in _rows])), margin=int(_d2["fobb"]["margin"]), gated=False, nf=len(_fl))
            n.got.update(_sv)      # 추가 프레임이 got(rgb/depth/tf)를 덮었으므로 첫 프레임 값으로 되돌림(이후 색·CLIP 채점은 첫 프레임 기준)
        except Exception as _e:
            log(f"다중 프레임 합치기 실패 {_e!r} -- 첫 프레임만 사용")
            try: n.got.update(_sv)
            except Exception: pass
    if dets and os.environ.get("NERO_COLOR_MATCH", "1") == "1":      # 마스크 안쪽(3픽셀 침식) 픽셀의 중앙값 색 -> HSV
        try:
            _rgb = np.ascontiguousarray(n.got["rgb"][..., :3].astype(np.uint8))
            for _d in dets:
                _m = _d["mask"]; _e = cv2.erode(_m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                _px = _rgb[_e if int(_e.sum()) >= 30 else _m]
                if len(_px) >= 30: _d["hsv"] = [int(x) for x in cv2.cvtColor(np.median(_px, axis=0).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]]
        except Exception as _e: log(f"색 측정 실패 {_e!r}")
    qs, xq, pq = GROUND["queries"], GROUND["excludes"], GROUND["place_queries"]
    if dets and os.environ.get("NERO_FRAME_GATE", "1") == "1":      # 2026-10-12(2026-10-13 기본 켜짐, =0 으로 끔): 깊이 마스크와 컬러 영상의 전경이 안 겹치는 프레임(연속 스윕 중 크롭 어긋남)은 CLIP 채점에서 뺀다
        try:
            _lab = cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(n.got["rgb"][..., :3].astype(np.uint8))[..., ::-1], cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 3)
            _fg = (np.linalg.norm(_lab - np.median(_lab.reshape(-1, 3), axis=0), axis=2) > 28); _U = np.zeros((H, W), bool)
            for _d in dets: _U |= _d["mask"]
            _cut = int(0.72 * H); _fg[_cut:] = False; _U[_cut:] = False
            _ov = float((_U & _fg).sum() / max(int(_U.sum()), 1))
            if _ov < float(os.environ.get("NERO_FRAME_GATE_MIN", "0.75")):
                log(f"  프레임 게이트: 마스크-컬러 겹침 {_ov:.2f} < {os.environ.get('NERO_FRAME_GATE_MIN', '0.75')} -- 이 프레임은 채점 생략(마스크 {len(dets)}개)")
                for _d in dets: _d["gated"] = True
                return geom, dets
        except Exception as _e:
            log(f"프레임 게이트 계산 실패 {_e!r} -- 게이트 없이 진행")
    if (qs or xq or pq or full_probs or GROUND.get("label")) and dets:
        try:
            if os.environ.get("NERO_GROUND_DUMP"): grounding.ground_masks.depth_dump = geom["zf"]      # 분석용(2026-10-13 기본 켜짐, =0 으로 끔): 어긋남 추정을 위해 깊이 영상도 저장
            _bigf = float(os.environ.get("NERO_SCORE_BIG_CUT_FRAC", "0.04")) if os.environ.get("NERO_SCORE_BIG_CUT", "1") == "1" else None      # 2026-10-13(2026-10-13 기본 켜짐, =0 으로 끔): 화면보다 큰 물체(바구니 등)는 항상 가장자리에 걸려 채점에서 빠져 라벨이 비었다 -> 마스크가 화면의 4% 이상이면 잘려도 채점한다(작은 조각만 계속 뺀다)
            _gd = [d for d in dets if not (os.environ.get("NERO_SKIP_PARTIAL", "1") == "1" and d.get("cut") and not (_bigf is not None and float(d["mask"].mean()) >= _bigf))]      # 2026-10-12(2026-10-13 기본 켜짐, =0 으로 끔): 화면 좌/우/위 가장자리에 걸려 잘린 검출은 CLIP 채점에서 뺀다(조각이 blue bar 등으로 읽힘). 아래쪽 컷 행 접촉은 크롭 확장으로 처리
            if len(_gd) < len(dets): log(f"  잘린 검출 {len(dets) - len(_gd)}개는 채점 생략")
            if not _gd: return geom, dets
            pr = grounding.ground_masks(n.got["rgb"], [d["mask"] for d in _gd], qs + xq + pq, GROUND["vocab"])
            lgs = getattr(grounding.ground_masks, "last_logits", []) or [{}] * len(_gd)
            _fts = getattr(grounding.ground_masks, "last_feats", None)
            for _ii, (d, p, lg) in enumerate(zip(_gd, pr, lgs)):
                if _fts and _ii < len(_fts): d["feat"] = _fts[_ii]      # 2026-10-14(NERO_EMBED=1): 크롭 CLIP 이미지 임베딩
                if os.environ.get("NERO_VLM_LABEL", "0") == "1" and not d.get("gated") and not d.get("cut") and d.get("margin", 0) >= 8: d["_crop"] = _vlm_crop(n.got["rgb"], d["mask"])      # 2026-10-19: VLM 라벨용 크롭(온전히 보인 프레임만)
                ref = lg.get("an object", 0.0)
                d["s_t"] = max((lg[q] - ref for q in qs if q in lg), default=None)
                d["s_x"] = max((lg[q] - ref for q in xq if q in lg), default=None)
                d["s_p"] = max((lg[q] - ref for q in pq if q in lg), default=None)
                d["p_t"] = (min(1.0, sum(p.get(q, 0.0) for q in qs)) if qs else None)      # 어휘 전체 대비 대상 문구 확률(동의어 합산): 연속 스윕의 정지 판정용(장면 무관 기준)
                names = sorted((k for k in p if k != "an object" and not k.startswith(("a table", "the floor", "a wall", "a robot", "a shadow", "background", "a black"))), key=lambda k: -lg.get(k, -99))
                d["top"] = [(k.replace("a ", "", 1), round(lg.get(k, 0.0) - ref, 1)) for k in names[:3]]
                d["label"] = names[0].replace("a ", "", 1) if names else ""
                if full_probs:
                    d["probs"] = p; d["logits"] = lg
        except Exception as e:
            log(f"접지 실패 {e!r} -- 이 프레임은 점수 없음")
    return geom, dets


def embed_rescore(world, G):
    """2026-10-14(NERO_SWEEP_SKIP=1): 이전 명령에서 복원한 엔티티의 저장 임베딩을 새 질의 문구와 비교해 점수(s_t/s_x/s_p/p_t)와 라벨을 다시 매긴다(카메라 없이).
    서버 /ground 와 같은 식: 로짓 = 100 * (이미지 임베딩 . 텍스트 임베딩), s = 로짓(문구) - 로짓('an object'), p_t = 질의 문구 softmax 합. -> 점수 낸 엔티티 수."""
    ents = {k: o for k, o in world.objs.items() if o.get("restored") and o.get("feat") is not None and o.get("role") != "placed"}
    if len(ents) < 3: return 0
    qs, xq, pq = G["queries"], G["excludes"], G["place_queries"]
    phr = list(dict.fromkeys(list(qs) + list(xq) + list(pq) + list(G.get("vocab") or []) + grounding.GENERIC)); T = grounding.text_feats(phr); ri = phr.index("an object"); n = 0
    for k, o in ents.items():
        lg = 100.0 * (T @ np.asarray(o["feat"], np.float32)); pr_ = np.exp(lg - lg.max()); pr_ /= pr_.sum(); ref = float(lg[ri])
        _sc = lambda L: (max(float(lg[phr.index(q)]) - ref for q in L) if L else None)
        o["s_t"], o["s_x"], o["s_p"] = _sc(qs), _sc(xq), _sc(pq)
        o["p_t"] = min(1.0, float(sum(pr_[phr.index(q)] for q in qs))) if qs else None
        names = sorted((p_ for p_ in phr if p_ != "an object" and not p_.startswith(("a table", "the floor", "a wall", "a robot", "a shadow", "background", "a black"))), key=lambda p_: -float(lg[phr.index(p_)]))[:3]
        o["top"] = [(p_.replace("a ", "", 1), round(float(lg[phr.index(p_)]) - ref, 1)) for p_ in names]; o["label"] = names[0].replace("a ", "", 1) if names else ""; o["embed_ok"] = True; n += 1
    return n


GRIPPER_OPEN = float(os.environ.get("NERO_GRIPPER_OPEN", "0.10"))      # 그리퍼 최대 개구(m): 치울 수 있는 크기/용기 최소 크기를 여기서 유도(2026-10-14)


_WORLD = [None]


def _desc_score(o, d):
    """겉모습 점수(낮을수록 비슷): 정렬한 크기 차(cm) + 색 불일치 + 라벨 1순위 일치. 위치는 보지 않는다."""
    sc = 0.0
    if d.get("size_cm") and "raw" in o: sc += 0.06 * sum(abs(x - 100 * y) for x, y in zip(sorted(d["size_cm"]), sorted(float(v) for v in o["raw"])))
    if d.get("color") and o.get("hsv") and color_name(o["hsv"]) != d["color"]: sc += 0.6
    t1 = lambda t: (t[0] if isinstance(t[0], str) else str(t[0])) if t else None
    # (라벨 1순위 일치 가산은 뺐다: CLIP 라벨은 프레임마다 바뀌어 같은 물체의 정체성 기준으로 못 쓴다 -- 2026-10-14 S4 시험에서 라쿤 지정이 독수리로 넘어갔다)
    return sc


def _desc_best(d):
    """대상 후보 선택. 기준 임베딩(d["feat"])이 있으면 CLIP 외형 임베딩 코사인으로(점수 = 1-cos), 없으면 크기/색 점수로. 반환 (점수, id) -- 1위와 2위가 충분히 구별되지 않으면 (9.9, id)(아직 모름)."""
    W = _WORLD[0]
    if W is None: return None
    ok = [(k, o) for k, o in W.objs.items() if "raw" in o and o.get("role") not in ("dest", "placed") and not o.get("given")]
    if d.get("feat"):
        f0 = np.asarray(d["feat"], float); f0 /= (np.linalg.norm(f0) + 1e-9); c = []
        for k, o in ok:
            if o.get("feat") is None: continue      # 아직 정지 프레임으로 임베딩을 못 얻은 엔티티는 후보가 아니다(재채점이 채운다)
            f = np.asarray(o["feat"], float); c.append((1.0 - float(f0 @ (f / (np.linalg.norm(f) + 1e-9))), k))
        # 2026-10-16: 위치 일관성 -- 임베딩만 보면 시점이 다른 같은 물체(렌즈 0.23)가 다른 물체(설탕통 0.13, 힌트에서 22cm)보다 멀게 나와 엉뚱한 물체를 집었다(E134 첫 항목).
        # 순위 점수 = 임베딩 거리 + 힌트(조사 때 위치)에서 8cm 넘게 떨어진 만큼의 벌점(NERO_EMB_POS_W / 30cm). 힌트 8cm 안이고 위치 벌점 반영 1위가 충분히 앞서면 임베딩 거리 NERO_EMB_THR_POS(0.30)까지 확정한다.
        _h = GROUND.get("hint"); _pw = float(os.environ.get("NERO_EMB_POS_W", "0.6")); _dh = {}
        if _h is not None and _pw > 0:
            for _i, (_v, _k) in enumerate(c):
                _o = W.objs[_k]; _p = _o.get("w") or _o.get("xyz")
                if _p is not None: _dh[_k] = math.hypot(_p[0] - _h[0], _p[1] - _h[1])
            c = [(_v + _pw * max(0.0, _dh.get(_k, 0.0) - 0.08) / 0.30, _k, _v) for _v, _k in c]
        else:
            c = [(_v, _k, _v) for _v, _k in c]
        c.sort(); _raw = {k_: r_ for _, k_, r_ in c}; _sig = tuple((k, round(v, 2)) for v, k, _r in c[:3])
        _near_ok = bool(c) and _dh.get(c[0][1], 9.9) <= 0.08 and _raw[c[0][1]] <= float(os.environ.get("NERO_EMB_THR_POS", "0.30")) and (len(c) == 1 or c[1][0] - c[0][0] >= float(os.environ.get("NERO_EMB_MARGIN", "0.05")))
        c = [(v, k) for v, k, _r in c]
        if getattr(_desc_best, "_last", None) != _sig: _desc_best._last = _sig; log(f"  [embed-id] 기준 임베딩과 코사인 거리 상위: {[(k, v) for k, v in _sig]} (후보 {len(c)}개, 기준 <= {os.environ.get('NERO_EMB_THR', '0.15')}, 1·2위 차 >= {os.environ.get('NERO_EMB_MARGIN', '0.05')})")
        mg = float(os.environ.get("NERO_EMB_MARGIN", "0.05")); thr = float(os.environ.get("NERO_EMB_THR", "0.15"))
        if not c: return None
        if _near_ok: return (c[0][0], c[0][1])      # 힌트 위치에 있고 임베딩도 충분히 가깝다 -> 확정
        if c[0][0] > thr or (len(c) > 1 and c[1][0] - c[0][0] < mg):
            # 2026-10-15: 보조 판정 -- 조사 때의 참조 크기와 같은 물체 하나만 임베딩 거리 <= NERO_EMB_THR2(0.22) 로 남으면 확정한다(시점이 달라 거리가 0.15~0.2 로 벌어져 키 23cm 손전등을 못 찾고 정리가 멈췄다). 위치는 쓰지 않는다
            _sz = d.get("size_cm"); _t2 = float(os.environ.get("NERO_EMB_THR2", "0.22")); _tol = float(os.environ.get("NERO_EMB_SIZE_TOL", "0.035"))
            if _sz and len(_sz) == 3:
                _rf = sorted(float(v) / 100.0 for v in _sz); _both = [(v, k) for v, k in c if v <= _t2 and max(abs(a_ - b_) for a_, b_ in zip(sorted(float(x) for x in W.objs[k]["raw"]), _rf)) <= _tol]
                if len(_both) == 1: log(f"  [embed-id] 보조 판정: {_both[0][1]} 임베딩 {_both[0][0]:.2f} + 크기 일치(허용 {100 * _tol:.1f}cm) -> 확정"); return _both[0]
            return (9.9, c[0][1])
        return c[0]
    _hw = float(os.environ.get("NERO_DESC_HINT_W", "0") or 0); _hh = GROUND.get("hint")      # 2026-11-06(기본 0 = 이전과 동일): 겉모습(크기/색)만으로 구별 안 되는 같은 물체들(막대 여러 개)에서 힌트 위치와의 거리(m)를 가중해 고른다 -- 이미 놓은 막대를 다시 집던 문제
    c = sorted((_desc_score(o, d) + ((_hw * math.hypot((o.get("w") or o["xyz"])[0] - _hh[0], (o.get("w") or o["xyz"])[1] - _hh[1])) if (_hw > 0 and _hh is not None) else 0.0), k) for k, o in ok)
    if not c: return None
    if len(c) > 1 and c[1][0] - c[0][0] < float(os.environ.get("NERO_DESC_MARGIN", "0.15")): return (9.9, c[0][1])      # 1위와 2위가 구별되지 않으면 아직 모르는 것으로 본다
    return c[0]


_VLM_LAB = {}      # oid -> 확정된 VLM 라벨(NERO_VLM_MATCH_FIRST)


def _vlm_first(o, ok):      # 2026-11-03(NERO_VLM_MATCH_FIRST=1 일 때만 호출, 기본 0=현행 불변): VLM 라벨이 대상 문구 명사와 일치하는 물체가 있으면 그 물체들만 대상, 없으면 CLIP 판정(ok) 폴백. NERO_VLM_LABEL=1 이 같이 켜져 있어야 라벨이 생긴다
    try:
        W = _WORLD[0].objs; words = {str(q).strip().split()[-1].lower() for q in GROUND["queries"] if str(q).strip()}
        def _hit(oo):
            k = next((k_ for k_, v_ in W.items() if v_ is oo), None)
            if k is None: return False
            if k not in _VLM_LAB:
                _l = grounding.vlm_resolve(VLM_FUT.get(k), wait=0.0)
                if _l: _VLM_LAB[k] = _l
            return any(w in _VLM_LAB.get(k, "") for w in words)
        if any(_hit(v_) for v_ in W.values() if v_.get("role") not in ("dest", "placed")): return _hit(o)
    except Exception: pass
    return ok


def eligible(o):
    """등록부 물체 o 가 이번 명령의 대상인가. 질의가 있으면 질의 점수가 장면 최고에서 delta 이내(그리고 floor 이상), 제외 문구가 있으면 그 문구에 해당하는 물체는 뺀다."""
    G = GROUND
    if o.get("forced"):
        return True
    if o.get("clearing"):
        return True              # (위치 지정 모드에서도 치우는 물체는 대상이다)
    if G.get("desc"):      # 겉모습 지정 모드: 겉모습이 가장 가까운 하나(점수 0.9 미만)만 대상
        _b = _desc_best(G["desc"])
        return bool(_b and _b[0] < float(os.environ.get("NERO_DESC_MAX", "0.9")) and _WORLD[0].objs.get(_b[1]) is o)      # 같은 물체의 크기 오차는 0.1 안쪽, 다른 물체(독수리 대 라쿤)는 0.27 이상. 아직 못 본 대상 대신 본 다른 물체를 집지 않도록 엄격하게
    if G.get("at") is not None:      # 위치 지정 모드: 점수와 무관하게 지정한 위치 근처의 물체만 대상이다
        return bool(o.get("role") not in ("dest", "placed", "obstacle") and not o.get("given") and "xyz" in o and math.hypot(o["xyz"][0] - G["at"][0], o["xyz"][1] - G["at"][1]) < G.get("at_tol", 0.10))
    if o.get("restored") and not o.get("embed_ok"):
        return False             # 2026-10-13: 이전 명령에서 복원만 되고 이번에 아직 못 본 엔티티는 집지 않는다(유령 방지). 임베딩으로 점수를 낸 것(embed_ok)은 후보가 되고 다시 보일 때 확인된다
    if o.get("role") in ("dest", "placed"):
        return False             # 도착지 물체와 이미 놓은 물체는 다시 집지 않는다
    if not (G["queries"] or G["excludes"]):
        return True
    ok = True
    if G["queries"]:
        st, rt = o.get("s_t"), G["refs"]["t"]
        ok = st is not None and rt is not None and st >= max(G["floor"], rt - G["delta"])
        if os.environ.get("NERO_VLM_MATCH_FIRST", "0") == "1": ok = _vlm_first(o, ok)
    if ok and G["excludes"]:
        sx, rx = o.get("s_x"), G["refs"]["x"]
        if sx is not None and rx is not None and sx >= max(G["floor"], rx - G["delta"]):
            ok = False
    return ok


def npz_bytes(geom, mask, label, w):
    seg = np.zeros(mask.shape, np.int32); seg[mask] = 1
    buf = io.BytesIO()
    np.savez(buf, depth_m=geom["zf"].astype(np.float32), seg=seg, K=np.array([geom["fx"], geom["fy"], geom["cx"], geom["cy"]], dtype=np.float64),
             W_T_cam=geom["W_T_cam"].astype(np.float64), target_labels=np.array([label]), target_ids=np.array([1], dtype=np.int32),
             target_worlds=np.array([w], dtype=np.float64), target_is_bottle=np.array([False]))
    return buf.getvalue()


def arm_q_str(n):
    return " ".join("%.5f" % x for x in n.got["arm_q"])


def wait_observation(n, timeout=30.0):
    ref = n._observation_pose(); target = [ref[f"joint{i}"] for i in range(1, 8)]
    t0 = time.time()
    while time.time() - t0 < timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
        q = n.got.get("arm_q")
        if q and max(abs(a - b) for a, b in zip(q, target)) < 0.05:
            return True
    return False


def main():
    if os.environ.get("NERO_LEAN", "0") == "1":      # 2026-10-14(기본 꺼짐): 불필요한 재탐색/재채점을 걷어낸 모드 -- 처음 스윕에 마스크가 없던 방위는 다시 찾지 않고(이송 경로 안 본 각도 확인 생략), 정지 재채점은 3회, 목적지 되돌아보기는 1회로 제한. C 계획 실패(막힘 없음)는 시점을 바꿔 재시도하거나 다시 훑지 않고 그 물체를 포기한다
        for _k, _v in (("NERO_COVER_PATH", "0"), ("NERO_RESTILL_MAX", "3"), ("NERO_DEST_RELOOK", "1"), ("NERO_CLEAR_MAX", "3"), ("NERO_RT_PLACE_BUDGET", "16.0"), ("NERO_TARGET_COLOR", "1"), ("NERO_HOLE_MEASURED", "1"), ("NERO_OPEN_RECOVER", "1"), ("NERO_OBS_CONFIRMED", "1"), ("NERO_CONTAINER_INNER", "1")): os.environ.setdefault(_k, _v)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="-2.6,-1.73,-0.87,0.0,0.87,1.73,2.6")
    ap.add_argument("--j7", type=float, default=float(os.environ.get("NERO_RT_J7", "1.325")))
    ap.add_argument("--floor-z", type=float, default=0.012)
    ap.add_argument("--max-picks", type=int, default=30)
    ap.add_argument("--max-fail", type=int, default=2)
    ap.add_argument("--margin-px", type=int, default=20)
    ap.add_argument("--thor", default=os.environ.get("NERO_RT_THOR", f"http://{thor_ip()}:8010"))
    ap.add_argument("--curobo", default=os.environ.get("NERO_RT_CUROBO", "http://localhost:8020"))
    ap.add_argument("--ik-url", default=os.environ.get("NERO_RT_IKURL", "http://127.0.0.1:8020"))
    ap.add_argument("--pick-script", default=os.path.join(HERE, "pick_one.sh"))
    ap.add_argument("--target", default="", help="자연어 대상(예: \"red mug\", \"flashlight\"): open-vocab 검출로 접지해 그 물체만 집는다. 비우면 전부(청소)")
    ap.add_argument("--exclude", default="", help="제외 문구(예: \"cup\"): 이 문구에 해당하는 물체만 빼고 전부 집는다(쉼표/| 로 여러 개)")
    ap.add_argument("--delta", type=float, default=1.5, help="대상 판정: 질의 점수가 장면 최고 점수에서 이 값 이내인 물체를 대상(같은 종류 여러 개)")
    ap.add_argument("--floor", type=float, default=1.0, help="대상 판정: 질의 점수 절대 하한(문구와 전혀 무관한 물체 차단)")
    ap.add_argument("--scan-only", action="store_true", help="사전 훑기만 하고(팔은 관측 자세로 복귀) 물체별 라벨 후보/점수를 --out 에 저장한 뒤 종료")
    ap.add_argument("--plan-file", default="", help="2026-10-14: 한 프로세스에서 여러 물체를 이어서 처리한다. JSON [{id, desc{size_cm,color,top,feat}, hint:[x,y], dest, kind}, ...] (첫 항목은 --target-desc/--hint-xy/--dest-id 와 같아야 한다). 실패하면 홈 복귀/재시작 없이 그 자리에서 다음 항목으로 넘어간다")
    ap.add_argument("--hint-xy", default="", help="2026-10-14: 대상이 대략 있을 위치 x,y(m). 있으면 360도 스윕 대신 그 방위각 주변만 정지 촬영한다(못 찾으면 전체로 번지지 않고 끝낸다: 판단은 LLM 이)")
    ap.add_argument("--target-desc", default="", help="2026-10-14: 대상을 겉모습(JSON {size_cm, color, top})으로 지정한다. 위치는 쓰지 않는다 -- 등록부 물체 중 크기(정렬)/색/라벨이 가장 비슷한 하나가 대상. 목적지만 위치 입력이고 집을 물체는 모두 겉모습 기준")
    ap.add_argument("--target-at", default="", help="2026-10-14: 대상을 문구가 아니라 위치 x,y(m)로 지정한다(등록부에서 이 위치 10cm 안의 물체가 대상). 다물체 실행기가 LLM 계획의 물체를 정확히 집게 하는 용도")
    ap.add_argument("--dests", default="", help="2026-10-14: 목적지 명세 JSON 파일([{id,label,kind(container|flat|hole),xy,size,yaw,floor_z,rim_z,E,d,open,outer,depth}]) -- 목적지는 인식하지 않고 입력값으로 준다(위치/형상/입구가 정해진 설비처럼 취급)")
    ap.add_argument("--dest-id", default="", help="--dests 명세 중 이번 물체를 놓을 목적지 id")
    ap.add_argument("--place-on", default="", help="pick 한 물체를 이 문구에 해당하는 물체 위/안(예: \"box|bin\")에 놓는다(접지로 도착지를 찾는다)")
    ap.add_argument("--place-free", action="store_true", help="pick 한 물체를 책상의 빈 자리에 놓는다")
    ap.add_argument("--place-xy", default="", help="x,y 좌표(m)에 놓는다(시험용)")
    ap.add_argument("--place-clear", type=float, default=float(os.environ.get("NERO_PLACE_CLEAR", "0.006")), help="놓을 때 물체 바닥-면 간격(m)")
    ap.add_argument("--no-place-gt", action="store_true", default=os.environ.get("NERO_NO_GT", "0") == "1", help="시뮬 정답 판정(스냅샷) 생략(NERO_NO_GT=1 이면 기본)")
    ap.add_argument("--retry-views", type=int, default=int(os.environ.get("NERO_RT_RETRY_VIEWS", "2")), help="후보 없음/C 실패 시 j1 을 틀어 즉시 다시 시도하는 횟수(0=끔)")
    ap.add_argument("--no-warm", dest="warm", action="store_false", help="예열 실행기 끄기(기존: C 완료 후 pick_one.sh 가 exec 프로세스를 새로 띄움)")
    ap.add_argument("--recover", action="store_true", default=os.environ.get("NERO_RT_RECOVER", "0") == "1", help="놓은 직후 현재 자세에서 상태를 보고 쓰러짐/이탈이면 스윕 없이 바로 다시 집어 놓는다(국소 재시도)")
    ap.add_argument("--max-recover", type=int, default=1)
    ap.add_argument("--dest-kind", default="", help="2026-10-12(2026-10-13 기본 켜짐, =0 으로 끔): 목적지의 기하 종류를 언어 쪽(Claude)이 정해 주면 hole(좁은 입구 홀) / container(위가 열린 큰 용기) / flat(평면) 중 그 종류의 후보만 목적지로 쓴다. 자동 모드에서만 적용. 비우면 기존 동작")
    ap.add_argument("--insert-at", default="", help="긴 직육면체 삽입: ex,ey,ez,dx,dy,dz (홀 입구 중심 m + 삽입 방향 3D 벡터, 자동 정규화; 아래/수평/위 모두 가능) 또는 ex,ey,ez,alpha_deg,phi_deg. 주면 place 대신 /insert 로 계획한다")
    ap.add_argument("--insert-depth", type=float, default=0.05, help="삽입 성공 판정: 막대 끝이 입구 평면 아래로 이 깊이 이상(m)")
    ap.add_argument("--no-preview", action="store_true", help="질의가 있어도 사전 훑기(장면 최고 점수 확보) 생략 -- 비권장")
    ap.add_argument("--vocab", default="", help="경쟁 어휘(쉼표 구분 영문 문구). 비우면 기본 어휘. 대상 문구의 동의어를 넣지 말 것")
    ap.add_argument("--ground-only", action="store_true", help="로봇은 훑기만 하고 물체별 접지 확률을 --out 에 저장한 뒤 종료(접지 정확도 시험)")
    ap.add_argument("--no-exec", action="store_true", help="계획까지만(팔 실행/판정 없음) -- 지연 측정용")
    ap.add_argument("--out", default="/tmp/discover_object.json")
    a = ap.parse_args()
    angles = [float(x) for x in a.sweep.split(",")]
    AUTO = os.environ.get("NERO_AUTO_MODE", "0") == "1" and not a.insert_at      # 2026-10-08: 목적지 기술자로 place/insert 를 자동 결정(--insert-at 정답 주입이 있으면 끈다). 기본 꺼짐
    GROUND["queries"] = grounding.split_queries(a.target) if a.target else []
    GROUND["excludes"] = grounding.split_queries(a.exclude.replace(",", "|")) if a.exclude else []
    GROUND["vocab"] = [grounding.norm_phrase(v) for v in a.vocab.split(",") if v.strip()] or list(grounding.DEFAULT_VOCAB)
    GROUND["delta"] = a.delta; GROUND["floor"] = a.floor; GROUND["label"] = True
    GROUND["place_queries"] = grounding.split_queries(a.place_on) if a.place_on else []
    if a.target_desc: GROUND["desc"] = json.loads(a.target_desc); GROUND["queries"] = GROUND["queries"] or ["an object"]; GROUND["hint"] = ([float(v) for v in a.hint_xy.split(",")] if a.hint_xy else None)
    if a.target_at: GROUND["at"] = tuple(float(v) for v in a.target_at.split(",")); GROUND["queries"] = GROUND["queries"] or ["an object"]
    placing = bool(a.place_on or a.place_free or a.place_xy or a.insert_at or a.dest_id)       # 삽입도 place 경로(대기 실행기/사전 계획)를 탄다
    STATE["command"] = dict(target=a.target, exclude=a.exclude, place=(a.place_on or ("free" if a.place_free else (a.place_xy or ("insert" if a.insert_at else None)))))

    rclpy.init(); n = Discoverer()
    for _ in range(30):
        rclpy.spin_once(n, timeout_sec=0.1)
    n.home(); wait_observation(n, 10)
    world = World(); _WORLD[0] = world; events = []; picks_ok = 0; gen = 0; t_run = time.time()
    n.jmove_obstacles = (lambda: world.obstacles(None)); n.jmove_url = a.curobo      # 2026-10-15: j1 큰 회전은 cuRobo 충돌 회피 이동(용기를 침)
    if a.dests:      # 목적지 입력화: 인식 없이 등록부에 고정 엔티티로 넣는다. 선택된 id 는 role=dest, 나머지는 장애물(고정)
        os.environ.setdefault("NERO_DEST_FREEZE", "1")
        for _d in json.load(open(a.dests)):
            _oid = world.new_id(); _w, _dd, _h = [float(v) for v in _d.get("size", [0.2, 0.2, 0.1])]; _x, _y = [float(v) for v in _d["xy"]]; _kind = _d.get("kind", "container")
            _o = dict(fails=0, defer_gen=-1, seen=1, given=True, frozen=True, given_id=_d["id"], label=_d.get("label", _d["id"]), label_disp=_d.get("label", _d["id"]), xyz=[_x, _y, _h / 2], raw=[_w, _dd, _h], yaw=float(_d.get("yaw", 0.0)), w=[_x, _y, _h / 2], s_p=10.0, s_p_adj=10.0, s_t=-9.0, s_x=-9.0,
                      role=("dest" if _d["id"] == a.dest_id else "obstacle"))
            if _kind == "container": _o["wall_thick"] = float(_d["wall"]) if _d.get("wall") is not None else 0.02      # 2026-10-28: wall=null(인식 모드 미측정)이면 기존 기본값 0.02 로 읽는다(값이 있거나 키가 없으면 기존과 동일)
            if _kind == "shelf_cell" and os.environ.get("NERO_SHELF_CELL", "0") == "1":      # 2026-11-04(기본 꺼짐): 선반 칸 — 칸 부피는 장애물 아님, 판은 plates 로 장애물 추가(shelf_cell.py), 슬롯 높이 = 칸 바닥 z
                _fz0 = float(_d.get("floor_z", 0.014)); _o["shelf_cell"] = True; _o["plates"] = _d.get("plates", []); _o["raw"] = [_w, _dd, _fz0]; _o["xyz"] = [_x, _y, _fz0 / 2]; _o["w"] = [_x, _y, _fz0 / 2]
            _o["dest_meas"] = _d.get("meas")      # 2026-10-28: 필드별 측정 여부(인식 모드 스키마; 없으면 None)
            if _kind == "container":
                _rim_ = float(_d.get("rim_z", _h)); _fz_ = float(_d.get("floor_z", 0.012))
                if os.environ.get("NERO_DEST_USE_RIM_H", "0") == "1" and _rim_ - _fz_ < _rim_ - float(os.environ.get("NERO_DEST_FLOOR_Z", "0.012")):      # 2026-11-04(기본 꺼짐): 바닥 측정이 비정상으로 높으면 깊이 = max(측정 깊이, 림 − 0.012), 바닥은 그에 맞춰 낮춤(일관성)
                    _fz_ = float(os.environ.get("NERO_DEST_FLOOR_Z", "0.012"))
                _o["concave"] = dict(concave=True, depth=round(_rim_ - _fz_, 4), floor_z=_fz_, rim_z=_rim_, inferred=True)
            elif _kind == "hole": _o["hole"] = dict(ok=True, E=[float(v) for v in _d["E"]], d=[float(v) for v in _d.get("d", [0, 0, -1])], open=(float(_d["open"]), float(_d["open"])), outer=float(_d["outer"] if _d.get("outer") is not None else _d["open"] + 0.02), depth=float(_d["depth"] if _d.get("depth") is not None else 0.14), views=0)
            world.objs[_oid] = _o; log(f"[dest 입력] {_oid}={_d['id']} {_kind} xy=({_x:+.3f},{_y:+.3f}) 박스 {[round(100 * v, 1) for v in (_w, _dd, _h)]}cm" + (" <- 이번 목적지" if _d["id"] == a.dest_id else " (장애물)"))
    CLEAR = dict(used=0, done=0)      # 2026-10-14: 막은 물체 치우기 횟수
    _CMD_SET = set()      # 2026-10-14: 원래 명령 대상 엔티티(치우는 중인 물체 뒤에 가려진 흐름에서도 판다가 '치울 물체'로 오인되지 않게 한다)
    def _cmd_target(oid):
        seen = set()
        while oid in world.objs and world.objs[oid].get("clearing") and oid not in seen:
            seen.add(oid); oid = world.objs[oid].get("clear_for", oid)
        return oid
    def _clearable(b):
        if b in _CMD_SET: return False
        if PL["items"] and os.environ.get("NERO_CHAIN_CLEAR", "0") != "1": return False      # 2026-10-15: 체인(플랜 파일) 모드는 무엇을 먼저 치울지 LLM 플래너가 순서로 정한다 -- 실행기가 이웃을 임의로 치우다 대상<->이웃을 번갈아 시도하며 끝나지 않던 문제
        o_ = world.objs.get(b)
        if o_ is None or "raw" not in o_ or o_.get("role") in ("dest", "placed") or o_["fails"] >= a.max_fail or o_.get("restored"): return False
        if o_.get("hole") or ((o_.get("concave") or {}).get("concave") and (os.environ.get("NERO_LEAN", "0") != "1" or min(o_["raw"][0], o_["raw"][1]) >= float(os.environ.get("NERO_DEST_MIN_SIDE", "0.16")))): return False      # 홀더/용기 같은 고정물은 치우지 않는다. 2026-10-14(NERO_LEAN): 오목 판정은 용기 크기(최소변 16cm 이상)일 때만 고정물로 본다 -- 10x8cm 라쿤이 깊이 3cm 오목으로 오탐돼 앞에서 막는 물체가 치울 수 없는 것으로 표시됐다
        return min(o_["raw"][0], o_["raw"][1]) <= float(os.environ.get("NERO_CLEAR_MAXSIDE", str(GRIPPER_OPEN - 0.015))) and o_["raw"][2] <= 0.20      # 2026-10-14: 상수 0.09 -> 그리퍼 최대 개구(0.10)-여유 0.015      # 그리퍼에 잡히는 크기만
    def _infer_blockers(tid):      # 2026-10-14(NERO_LEAN=1): 서버가 막힘을 못 짚어도(`막힘 없음`) 등록부의 깊이 마스크 기반 위치/크기로 막은 물체를 추정한다. 대상 발자국 가장자리에서 NERO_NEAR_GAP(기본 0.06m: 손가락/팔이 닿는 범위) 이내 이웃을 간격 순으로, 대상과 가까운 순
        T = world.objs.get(tid)
        if T is None or "raw" not in T or "xyz" not in T: return []
        gap_max = float(os.environ.get("NERO_NEAR_GAP", "0.06")); out = []
        for k_, o_ in world.objs.items():
            if k_ == tid or "raw" not in o_ or "xyz" not in o_ or o_.get("role") in ("dest", "placed") or o_.get("clearing"): continue
            gap = math.hypot(o_["xyz"][0] - T["xyz"][0], o_["xyz"][1] - T["xyz"][1]) - 0.5 * (max(T["raw"][0], T["raw"][1]) + max(o_["raw"][0], o_["raw"][1]))
            if gap <= gap_max: out.append((gap, k_))
        out.sort()
        return [k_ for _, k_ in out][:3]
    def _clear_decision(oid, why):      # 2026-10-14(NERO_LEAN=1): B 단계에서 도달 가능한 파지 후보가 0개일 때(이웃 장애물 복셀이 파지를 전부 막음)에도 C 실패와 같은 방식으로 막은 물체를 정한다(규칙: 깊이 마스크 기준 가장자리 6cm 이내 이웃, LLM 모드: 대상 15cm 이내 치울 수 있는 물체를 Claude 가 고름). 치우기로 정하면 True
        if oid not in world.objs or "xyz" not in world.objs[oid] or CLEAR["used"] >= int(os.environ.get("NERO_CLEAR_MAX", "2")): return False
        _CMD_SET.add(_cmd_target(oid))
        blk = [b for b in _infer_blockers(oid) if b not in _CMD_SET]; src = "infer"
        if not blk and llm_decide.enabled():
            blk = [k_ for k_ in world.objs if k_ != oid and "xyz" in world.objs[k_] and world.objs[k_].get("role") not in ("dest", "placed") and _clearable(k_) and math.hypot(world.objs[k_]["xyz"][0] - world.objs[oid]["xyz"][0], world.objs[k_]["xyz"][1] - world.objs[oid]["xyz"][1]) < 0.15]; src = "near"
        _cl = [b for b in blk if _clearable(b)]
        if not _cl: return False
        _bt = min(_cl, key=lambda b: math.hypot(world.objs[b]["xyz"][0] - world.objs[oid]["xyz"][0], world.objs[b]["xyz"][1] - world.objs[oid]["xyz"][1]))
        if llm_decide.enabled():
            _near = [k_ for k_ in world.objs if k_ not in _cl and k_ != oid and "xyz" in world.objs[k_] and _clearable(k_) and math.hypot(world.objs[k_]["xyz"][0] - world.objs[oid]["xyz"][0], world.objs[k_]["xyz"][1] - world.objs[oid]["xyz"][1]) < 0.15]
            _opts = [dict(id=k_, text=f"{world.objs[k_].get('label_disp') or world.objs[k_].get('label')} 를 먼저 집어 빈 자리에 치운 뒤 대상을 다시 시도" + (({"infer": " (깊이 마스크 기준 대상 가장자리 6cm 이내)", "near": " (대상 15cm 이내)"}.get(src, "")) if k_ in _cl else " (대상 근처)")) for k_ in _cl + _near] + [dict(id="defer", text="아무것도 치우지 않고 이 대상은 보류(다른 물체 먼저, 없으면 종료)")]
            _ch, _src, _why = llm_decide.ask("clear_blocker", f"{oid} 의 파지 후보가 모두 이웃 때문에 막혔다({why}). 무엇을 치울까?", dict(table=_llm_table(oid, oid, blk), 치운횟수=f"{CLEAR['used']}/{os.environ.get('NERO_CLEAR_MAX', '2')}", 명령=f"대상 {a.target if hasattr(a, 'target') else ''} -> 목적지 {a.place_on}"), _opts, _bt, log=log)
            if _ch == "defer": return False
            _bt = _ch
        world.objs[_bt]["clearing"] = True; world.objs[_bt]["clear_for"] = oid; CLEAR["used"] += 1; world.objs[oid]["defer_gen"] = gen
        log(f"{oid}: 파지 후보 0개 -- 막은 물체 {_bt}({world.objs[_bt].get('label')}) 를 먼저 치운다({CLEAR['used']}/{os.environ.get('NERO_CLEAR_MAX', '2')}) -- 후보 {_cl}")
        return True
    def _llm_table(focus=None, tgt=None, blk=()):      # 2026-10-14(NERO_LLM_DECIDE=1): Claude 에게 줄 엔티티 표. 이미지 대신 라벨 후보, 색, 크기, 위치, 대상과의 관계(거리/앞뒤), 치울 수 있는지를 텍스트로
        if focus: _CMD_SET.add(_cmd_target(focus))
        rows = []; T = world.objs.get(tgt) if tgt else None
        for k_, o_ in sorted(world.objs.items(), key=lambda kv: int(kv[0].split("_")[-1]) if kv[0].split("_")[-1].isdigit() else 0):
            if "raw" not in o_ or "xyz" not in o_: continue
            r_ = o_["raw"]; x_, y_ = o_["xyz"][0], o_["xyz"][1]; f_ = []
            lab_ = o_.get("label_disp") or o_.get("label") or "라벨 없음"
            try: col_ = color_name(o_["hsv"]) if o_.get("hsv") else "색 미측정"
            except Exception: col_ = "색 미측정"
            if T is not None and k_ != tgt:
                tx_, ty_ = T["xyz"][0], T["xyz"][1]; dd_ = math.hypot(x_ - tx_, y_ - ty_)
                cs_ = (x_ * tx_ + y_ * ty_) / max(1e-6, math.hypot(x_, y_) * math.hypot(tx_, ty_))
                rel_ = f"대상에서 {100 * dd_:.0f}cm" + (", 로봇과 대상 사이(앞)" if cs_ > 0.82 and math.hypot(x_, y_) < math.hypot(tx_, ty_) else (", 대상 뒤쪽" if cs_ > 0.82 else ""))
                f_.append(rel_)
            if k_ == tgt: f_.append("대상" + (" (지금 막혀서 먼저 치우는 중인 물체)" if o_.get("clearing") else ""))
            if k_ in _CMD_SET: f_.append("원래 명령 대상(CLIP 라벨은 틀릴 수 있음) -- 치우면 안 됨" + (f", 명령 문구 점수 s_t={o_['s_t']:.1f}" if o_.get("s_t") is not None else ""))
            if k_ in blk: f_.append("C단계가 막힘으로 보고")
            if o_.get("role") == "dest": f_.append("목적지 후보")
            if o_.get("role") == "placed": f_.append("이미 놓은 물체(건드리지 말 것)")
            if (o_.get("concave") or {}).get("concave"): f_.append(f"위가 열린 용기(깊이 {100 * o_['concave'].get('depth', 0):.0f}cm) -- 고정물, 치울 수 없음")
            if o_.get("hole"):
                _hh = o_["hole"]; f_.append("홀(입구 있음" + (f": 입구 {100 * _hh['open'][0]:.1f}cm, 링 {100 * _hh['outer']:.1f}cm, 깊이 {100 * _hh.get('depth', 0):.0f}cm" if _hh.get("ok") and _hh.get("open") else "") + ") -- 고정물")
            if o_.get("frozen"): f_.append("초반 조사 1회로 위치/형상 확정(고정)")
            if o_.get("fails"): f_.append(f"실패 {o_['fails']}회")
            f_.append("치울 수 있는 크기" if _clearable(k_) else "치울 수 없음(크기/고정물/실패)")
            rows.append(f"{k_} | {lab_} | {col_} | {100 * r_[0]:.0f}x{100 * r_[1]:.0f}x{100 * r_[2]:.0f}cm | ({x_:+.2f},{y_:+.2f}) | " + "; ".join(f_))
        return "id | 라벨(CLIP, 틀릴 수 있음) | 색 | 크기 | 위치(로봇 기준 m) | 사실\n" + "\n".join(rows)
    REG_ON = os.environ.get("NERO_REG_PERSIST", "0") == "1"; REG_PATH = os.path.expanduser(os.environ.get("NERO_REG_PATH", "~/grasp/rt_registry.json"))      # 2026-10-13(기본 꺼짐): 엔티티 등록부를 명령 사이에 이어 쓴다
    def reg_save(tag):
        if REG_ON:
            try: log(f"[reg] 등록부 저장({tag}): 엔티티 {REG.dump(world.objs, REG_PATH, world._k)}개")
            except Exception as _e: log(f"[reg] 저장 실패 {_e!r}")
    if REG_ON:
        _ro, _rc = REG.load(REG_PATH, max_age_s=float(os.environ.get("NERO_REG_MAX_AGE", "1800")))
        for _k_, _v_ in _ro.items(): _v_["reg_xyz"] = [float(x) for x in _v_["xyz"]]      # 2026-10-13: 복원 당시 위치(확인된 도착지 재사용 판단용)
        if _ro: world.objs.update(_ro); world._k = max(world._k, _rc); log(f"[reg] 이전 명령의 엔티티 {len(_ro)}개 복원(아직 못 본 상태, id 이어서 사용 k={_rc}): {sorted(_ro)}")
        else: log("[reg] 복원할 이전 등록부 없음")
    _PM = os.environ.get("NERO_PLACED_MEM")      # 2026-11-12(기본 꺼짐, 사용자 승인 A): 앞 단계(다른 로봇 프로세스)에서 이 시스템이 놓은 물체 기억 -- 놓기 계획의 물체 목표 자세+인식 치수로 만든 상자. 가려져 안 보여도 충돌체로 남는다(NERO_KEEP_PLACED=1 과 함께)
    if _PM and os.path.exists(_PM):
        try:
            for _e in json.load(open(_PM)):
                _mid = "mem_" + str(_e["id"]); world.objs[_mid] = dict(fails=99, defer_gen=-1, seen=1, role="placed", xyz=list(_e["xyz"]), raw=list(_e["raw"]), yaw=float(_e["yaw"]), w=[_e["xyz"][0], _e["xyz"][1], 0.05], label=_e.get("label"), mem=True)
            log(f"[placed-mem] 앞 단계에서 놓은 물체 {len(json.load(open(_PM)))}개 등록: " + ", ".join(f"{k_}({o_['xyz'][0]:+.3f},{o_['xyz'][1]:+.3f}) {[round(v, 3) for v in o_['raw']]}" for k_, o_ in world.objs.items() if o_.get("mem")))
        except Exception as _e: log(f"[placed-mem] 불러오기 실패 {_e!r}")
    _ek = ("NERO_DEST_MIN_SIDE", "NERO_DEST_SMALL_PEN", "NERO_DEST_UNMEAS_PEN", "NERO_DEST_CONC_BONUS", "NERO_DEST_CONT_BYPASS", "NERO_RT_DEFER_EXHAUSTED", "NERO_TOPPLED_IN_OK", "NERO_STALE_ON_FAIL", "NERO_MISS_N", "NERO_PLACED_VANISH")
    log("설정값(미설정=기본): " + " ".join(f"{k_.replace('NERO_', '')}={os.environ.get(k_, '-')}" for k_ in _ek))      # 2026-10-08: 실험 로그에 실제 스위치 값을 남긴다(화이트리스트 사고 재발 방지)
    (_C1 or _B1 or _FIN) and log(f"[flags] C1_FASTFAIL={int(_C1)} B1_SKIP_RESHOT={int(_B1)} FINAL_RETRY_ONCE={int(_FIN)} C1_OFF_FINAL={os.environ.get('NERO_C1_OFF_FINAL', '-')} (기본 모두 0)")
    log(f"시작: 관측 자세, 스윕 {angles}, j7={a.j7}" + (f", 대상 {GROUND['queries']}" if GROUND["queries"] else ", 대상 없음(청소)"))
    if a.ground_only:
        GROUND["queries"] = list(GROUND["vocab"]); rec = []
        for ang in angles:
            n.move_j1(ang, j7_abs=a.j7)
            r = grab(n, a.floor_z, a.target, full_probs=True)
            if r is None:
                continue
            os.makedirs("/tmp/gcrops", exist_ok=True)
            for ii, d in enumerate(r[1]):
                ys_, xs_ = np.where(d["mask"]); rgb_ = n.got["rgb"]
                cv2.imwrite(f"/tmp/gcrops/a{ang:+.2f}_{ii}_x{d['w'][0]:.2f}_y{d['w'][1]:.2f}.jpg", np.ascontiguousarray(rgb_[max(0, ys_.min() - 12):ys_.max() + 12, max(0, xs_.min() - 12):xs_.max() + 12, :3][..., ::-1]))
                rec.append(dict(ang=ang, w=[float(x) for x in d["w"]], margin=d["margin"], probs=d["probs"], logits=d.get("logits", {})))
        n.home(); json.dump(dict(dets=rec), open(a.out, "w"), indent=1)
        log(f"접지 전용 종료: 검출 {len(rec)}개 -> {a.out}"); rclpy.shutdown(); return

    def order_from(a0):
        up = sorted(x for x in angles if x >= a0 - 0.30); dn = sorted((x for x in angles if x < a0 - 0.30), reverse=True)
        return up + dn

    order = order_from(angles[0]); attempts = 0; clean = {}
    if os.environ.get("NERO_PLACE_LOOK", "0") == "1" and a.place_xy:      # 2026-11-12(기본 꺼짐, 사용자 승인 B): 집기 전에 놓을 곳 방향을 정지 촬영해 등록부에 넣는다(훑기 프레임은 흔들림 게이트로 많이 버려지고, 놓을 곳 주변 물체가 빠지면 충돌체가 없다)
        try:
            _pxy = [float(v) for v in a.place_xy.split(",")]; _j = float(np.clip(bearing_to_j1(_pxy[0], _pxy[1]), -J1_LIMIT + 0.05, J1_LIMIT - 0.05)); _t0l = time.time(); _n0 = len(world.objs)
            n.move_j1(_j, j7_abs=a.j7); _rr = grab(n, a.floor_z)
            if _rr is not None: world.update(_rr[1], float(n.got["arm_q"][0]))
            log(f"[place-look] 놓을 곳 ({_pxy[0]:+.3f},{_pxy[1]:+.3f}) 방향 j1={_j:+.2f} 정지 촬영 {time.time() - _t0l:.1f}s: 검출 {0 if _rr is None else len(_rr[1])}개, 등록부 {_n0} -> {len(world.objs)}")
        except Exception as _e: log(f"[place-look] 실패 {_e!r}")

    def pub(target=None):
        """등록부를 UI 상태로 내보낸다(물체별 역할: target/obstacle/failed/deferred)."""
        objs = {}
        for oid2, o2 in world.objs.items():
            if "raw" not in o2:
                continue
            st2 = "dest" if o2.get("role") == "dest" else ("placed" if o2.get("role") == "placed" else "target" if oid2 == target else ("failed" if o2["fails"] >= a.max_fail else ("deferred" if o2["defer_gen"] == gen else "obstacle")))
            objs[oid2] = dict(xyz=o2["xyz"], raw=o2["raw"], yaw=o2["yaw"], fails=o2["fails"], status=st2, label=((o2.get("label_disp") or o2.get("label", "")) if os.environ.get("NERO_RVIZ_ENT") == "1" else o2.get("label", "")), s=o2.get("s_t"), match=bool(eligible(o2)))
            if os.environ.get("NERO_RVIZ_ENT") == "1": objs[oid2].update(dadj=o2.get("s_p_adj") if o2.get("role") == "dest" else None)
        if a.insert_at:      # 2026-10-08: 삽입 정보를 계획 전부터 보여 준다 -- 홀더 마스크 점, 홀더 점군 PCA 축(입구 축 추정), 막대 PCA 축, 입력된 삽입 방향. 계획 쪽 키(plan/tip_path/fail/verdict)는 보존
            try:
                import insertion_geometry as _G, axis_pca as _AP
                E_, dd_ = _G.parse_insert_spec(a.insert_at); al_, _ = _G.angles_from_axis(dd_); ins_ = dict(STATE.get("insert") or {})
                ins_.update(E=[float(v) for v in E_], d=[float(v) for v in dd_], alpha=round(float(al_), 1), inner=insert_ops.HOLE_INNER, wall=insert_ops.WALL, len=insert_ops.TUBE_LEN)
                _cands = sorted(((math.hypot(o3["xyz"][0] - E_[0], o3["xyz"][1] - E_[1]), k3) for k3, o3 in world.objs.items() if "raw" in o3 and o3.get("pts") is not None and k3 != target and o3.get("role") != "placed"), key=lambda t: t[0])
                if _cands and _cands[0][0] < 0.15:
                    _Ph = np.asarray(world.objs[_cands[0][1]]["pts"], float).reshape(-1, 3); _sel = np.linspace(0, len(_Ph) - 1, min(len(_Ph), 300)).astype(int); ins_["hole_pts"] = _Ph[_sel].round(4).tolist()
                    _hp = _AP.pca_axis(_Ph)
                    if _hp.get("ok"): ins_["hole_pca"] = dict(ok=True, axis=[float(v) for v in _hp["axis"]], center=[float(v) for v in _Ph.mean(0)], length=float(_hp.get("length", 0.0)), id=_cands[0][1])
                    elif not (ins_.get("hole_pca") or {}).get("ok"): ins_["hole_pca"] = dict(ok=False, why=_hp.get("why", ""))      # 직전에 잘 나온 값이 있으면 유지(깜빡임 방지)
                if target in world.objs and world.objs[target].get("pts") is not None and not (ins_.get("peg") or {}).get("ok"):
                    _pp = _AP.pca_axis(world.objs[target]["pts"])
                    if _pp.get("ok"): ins_["peg"] = dict(ok=True, axis=[float(v) for v in _pp["axis"]], center=[float(v) for v in _pp["mid"]], length=float(_pp["length"]))
                STATE["insert"] = ins_
            except Exception as e_:
                log(f"UI 삽입 사전 정보 실패 {e_!r}")
        STATE["objs"] = objs; STATE["stats"] = dict(picks_ok=picks_ok, attempts=attempts); flush_state()

    def ee_path(plan):
        """계획 웨이포인트의 그리퍼(플랜지) 3D 경로: cuRobo 서버 /fk"""
        qs = []
        for key in ("transit", "advance", "lift"):
            qs += plan.get(key, [])
        try:
            return post_json(a.curobo + "/fk", dict(q=qs), timeout=10)["pos"]
        except Exception:
            return []
    if (GROUND["queries"] or GROUND["excludes"] or GROUND["place_queries"] or a.scan_only) and not a.no_preview:
        # 사전 훑기: 질의 점수의 장면 최고값(기준)을 먼저 확보한다 -- 대상 판정이 절대 점수가 아니라 장면 안 상대비교라서 필요하다.
        set_state(stage="사전 훑기")
        FOCUS = os.environ.get("NERO_FOCUS_SCAN", "1") == "1"; early = False
        def _refs_now():
            for key, f in (("t", "s_t"), ("x", "s_x"), ("p", "s_p")):
                vals = [o[f] for o in world.objs.values() if o.get(f) is not None and o.get("role") not in ("dest", "placed")]
                GROUND["refs"][key] = max(vals) if vals else None
        def _needed_found():
            _refs_now(); tg_ = [oid for oid, o in world.objs.items() if eligible(o)]
            if not tg_ or not GROUND["queries"]:
                return False
            if GROUND["place_queries"]:
                rp_ = GROUND["refs"]["p"]
                _msd = float(os.environ.get("NERO_DEST_MIN_SIDE", "0.16"))      # 2026-10-08: 화면에 일부만 보인 도착지(작은 박스)는 아직 다 못 본 것 -- 충분한 크기로 보일 때까지 탐색을 계속한다
                if rp_ is None or not any(o.get("s_p") is not None and o["s_p"] >= max(GROUND["floor"], rp_ - 0.5) and min(o["raw"][0], o["raw"][1]) >= _msd for o in world.objs.values() if o.get("raw") is not None):
                    return False
            return True
        def _first_dir():
            ev = os.environ.get("NERO_SCAN_FIRST_DIR")
            if ev: return float(ev)
            j_ = float(n.got["arm_q"][0]) if "arm_q" in n.got else 0.0
            return -1.0 if abs(-J1_LIMIT - j_) <= abs(J1_LIMIT - j_) else 1.0          # 현재 각도에서 가까운 관절 한계 쪽부터
        def _conf_ok():
            """조기 종료 신뢰 확인(2026-10-07, 기본 켜짐 0.9; NERO_SCAN_PGATE=0 으로 끔): 대상 후보 중 어휘 전체 대비 확률이 NERO_SCAN_PGATE 이상인 것이 있어야 멈춘다. 다물체 장면에서 장면 상대 판정만으로는 방해물에서 멈춘다(48장면 중 7건)."""
            g_ = float(os.environ.get("NERO_SCAN_PGATE", "0.9"))
            if GROUND.get("desc"): return True      # 겉모습 지정에서는 p_t 가 없다
            return g_ <= 0 or any((o.get("p_t") if o.get("p_t") is not None else 0.0) >= g_ for o in world.objs.values() if eligible(o))
        _seen_override = None; _angs = (sorted(angles, key=lambda x: abs(x)) if FOCUS else angles)
        _sk = 0
        if REG_ON and os.environ.get("NERO_SWEEP_SKIP", "0") == "1" and (GROUND["queries"] or GROUND["place_queries"]):      # 2026-10-14(기본 꺼짐): 저장 임베딩으로 대상/도착지를 고르고 전체 스윕을 건너뛴다. 대상이 안 골라지면 기존 스윕으로 되돌아간다
            try:
                _t_e = time.time(); _sk = embed_rescore(world, GROUND)
                if _sk:
                    _refs_now()
                    _el = sorted([(o_.get("s_t") if o_.get("s_t") is not None else -99.0, k_, o_) for k_, o_ in world.objs.items() if eligible(o_)], key=lambda t_: -t_[0])
                    _dc_ = sorted([(o_.get("s_p"), k_) for k_, o_ in world.objs.items() if o_.get("s_p") is not None and not eligible(o_)], key=lambda t_: -t_[0])
                    log(f"[sweep-skip] 임베딩 점수: 대상 후보 {[(k_, round(v_, 1), None if o_.get('p_t') is None else round(o_['p_t'], 2)) for v_, k_, o_ in _el[:3]]} | 도착지 후보 {[(k_, round(v_, 1)) for v_, k_ in _dc_[:3]]} | floor {GROUND['floor']} delta {GROUND['delta']}")
                    _top_ok = bool(_el) and (_el[0][2].get("p_t") is not None and _el[0][2]["p_t"] >= float(os.environ.get("NERO_SKIP_PT", "0.5"))) and (len(_el) < 2 or _el[0][0] - _el[1][0] >= float(os.environ.get("NERO_SKIP_MARGIN", "1.0"))) and _el[0][0] >= GROUND["floor"]      # 대상 1위가 확률 0.5 이상, 2위보다 점수 1.0 이상 높고, floor 이상
                    if not (_top_ok and (not GROUND["place_queries"] or bool(_dc_))): log(f"[sweep-skip] 임베딩 점수로 대상/도착지가 확정되지 않음 -> 전체 스윕으로 진행"); _sk = 0
                    else: log(f"[sweep-skip] 저장 임베딩 {_sk}개로 {time.time() - _t_e:.1f}s 에 새 질의 점수 계산 -> 전체 스윕 생략, 대상/도착지 방위각만 재확인")
            except Exception as _e: log(f"[sweep-skip] 실패 {_e!r} -> 전체 스윕"); _sk = 0
        if not _sk and REG_ON and os.environ.get("NERO_DESC_SKIP", "0") == "1" and GROUND.get("desc"):      # 2026-10-14(기본 꺼짐): 겉모습 지정 + 이전 조사에서 복원한 등록부에 겉모습이 맞는 엔티티가 있으면 전체 스윕을 건너뛰고 그 방위각만 확인한다(못 찾으면 기존 스윕으로 되돌아간다)
            _b_ = _desc_best(GROUND["desc"])
            if _b_ and _b_[0] < 0.6 and world.objs[_b_[1]].get("restored"):
                _sk = 1; _refs_now(); log(f"[desc-skip] 등록부 {_b_[1]} 가 겉모습과 맞음(점수 {_b_[0]:.2f}) -> 전체 스윕 생략, 저장 위치 방위각만 확인")
        if _sk: _angs = []; early = True; ang = 0.0; _seen_override = []
        if not _sk and a.hint_xy and GROUND.get("desc"):      # 국소 탐색: 힌트 방위각 + 양옆 두 곳만(찾으면 바로 멈춘다). 360도 스윕과 폴백을 하지 않는다
            _hx_, _hy_ = (float(v) for v in a.hint_xy.split(","))
            _hb = float(np.clip(bearing_to_j1(_hx_, _hy_), -J1_LIMIT + 0.05, J1_LIMIT - 0.05))
            _angs = [_hb] + [float(np.clip(_hb + dlt, -J1_LIMIT + 0.05, J1_LIMIT - 0.05)) for dlt in (-0.4, 0.4)]; early = True; FOCUS = True
            log(f"[local-scan] 힌트 ({_hx_:+.2f},{_hy_:+.2f}) 방위각 j1={_hb:+.2f} 주변만 촬영(전체 스윕 생략)")
        if FOCUS and os.environ.get("NERO_SCAN_FAST", "1") == "1" and not (a.hint_xy and GROUND.get("desc")):        # 연속 스윕 2단계(2026-10-07 기본 켜짐; NERO_SCAN_FAST=0 으로 끔): 정면 한 장을 먼저 보고, 없으면 연속 스윕 + 발견 즉시 감속 정지
            _angs = _angs[:1]                                              # 정면(가장 가까운 각도)만 정지-촬영
        for ang in _angs:      # 필요한 부분만: 정면(j1=0)에서 가까운 각도부터
            n.move_j1(ang, j7_abs=a.j7)
            r = grab(n, a.floor_z)
            if r is None:
                continue
            world.update(r[1], float(n.got["arm_q"][0])); pub()
            if FOCUS and _needed_found() and _conf_ok():
                early = True; log(f"필요한 물체를 찾음 -- 남은 각도 훑기 생략(j1={ang:+.2f}에서 중단)"); break
        if FOCUS and os.environ.get("NERO_SCAN_FAST", "1") == "1" and not early:
            try:
                import fast_scan
                _ok, _jst, _nf, _dur = fast_scan.get(n).search(a.j7, lambda v_: grab(_scan_snap(v_), a.floor_z), lambda dets, j1: (world.update(dets, j1), pub()), lambda: (_refs_now(), _needed_found() and any((o.get('p_t') if o.get('p_t') is not None else 0.0) >= float(os.environ.get('NERO_SCAN_PSTOP', '0.9')) for o in world.objs.values() if eligible(o)))[1], first_dir=_first_dir(), log=log)
                if _ok:
                    early = True; ang = _jst; log(f"필요한 물체를 찾음 -- 연속 스윕에서 발견(j1={_jst:+.2f}에서 정지)")
                _seen_override = [x for x in angles if min(0.0, _jst) - 0.4 <= x <= max(0.0, _jst) + 0.4]
            except Exception as e_:
                log(f"연속 스윕 탐색 실패 -> 기존 정지-촬영으로 대체: {e_!r}")
                for ang in sorted(angles, key=lambda x: abs(x))[1:]:
                    n.move_j1(ang, j7_abs=a.j7); r = grab(n, a.floor_z)
                    if r is None: continue
                    world.update(r[1], float(n.got["arm_q"][0])); pub()
                    if _needed_found() and _conf_ok(): early = True; log(f"필요한 물체를 찾음 -- 남은 각도 훑기 생략(j1={ang:+.2f}에서 중단)"); break
        _STAT = os.environ.get("NERO_ENT_STATIONARY", "0") == "1"
        if (os.environ.get("NERO_RESTILL", "1") == "1" and (GROUND["queries"] or GROUND["excludes"] or GROUND["place_queries"])) or (_STAT and any(o_.get("tentative") for o_ in world.objs.values())):      # 2026-10-12(2026-10-13 기본 켜짐, =0 으로 끔): 점수가 없는 물체를 정지 프레임에서 다시 채점(스윕 프레임은 게이트가 걸러 점수가 없을 수 있다)
            _need = lambda o: "raw" in o and o.get("role") != "placed" and not o.get("frozen") and (not _STAT or o.get("rs_n", 0) < 2) and (all(o.get(k_) is None for k_ in ("s_t", "s_x", "s_p")) or (_STAT and bool(o.get("tentative"))))
            _DESC = GROUND.get("desc")
            if _DESC:      # 겉모습(임베딩) 지정: 점수가 아니라 임베딩이 없는 물체만, 기준 크기와 비슷한 것부터 정지 촬영한다(크기가 전혀 다른 물체는 건너뜀)
                _dsz = sorted(_DESC.get("size_cm") or [])
                _dd = lambda o: (sum(abs(x - 100 * y) for x, y in zip(_dsz, sorted(float(v) for v in o["raw"]))) if _dsz and o.get("raw") is not None else 0.0)
                _need = lambda o: "raw" in o and o.get("role") not in ("dest", "placed") and not o.get("given") and not o.get("frozen") and o.get("feat") is None and _dd(o) < 14.0
            for _rs in range(int(os.environ.get("NERO_RESTILL_MAX", "8"))):
                def _dest_plaus(o_):      # 2026-10-14(NERO_RESTILL_KIND_ORDER=1, 기본 꺼짐): 목적지 종류에 맞는 크기/높이의 물체를 먼저 다시 본다(왼쪽 끝부터 방위순으로 돌면 도착지가 마지막에야 채점됐다: 재채점 3~4회, 20~25초)
                    r_ = o_.get("raw"); dk_ = (a.dest_kind or "").strip().lower()
                    if os.environ.get("NERO_RESTILL_KIND_ORDER", "0") != "1" or r_ is None or dk_ not in ("hole", "container", "flat") or not GROUND["place_queries"]: return 1
                    ms_ = min(r_[0], r_[1])
                    ok_ = (ms_ >= GRIPPER_OPEN) if dk_ == "container" else ((ms_ >= 0.10 and r_[2] < 0.05) if dk_ == "flat" else (0.05 <= ms_ <= 0.14 and r_[2] >= 0.10))
                    return 0 if ok_ else 1
                _FOC = os.environ.get("NERO_RESTILL_FOCUS", "0") == "1"      # 2026-10-14(기본 꺼짐): 관심 물체만 다시 본다 -- (1) 현재 팔 방위에서 가까운 순(방위순으로 끝에서 끝까지 돌던 이동 제거) (2) 대상이 이미 채점됐으면 도착지 종류에 맞을 만한 물체만 (나머지는 건너뜀)
                try: _cj = float(n.got["arm_q"][0])
                except Exception: _cj = 0.0
                _un = sorted((o for o in world.objs.values() if _need(o)), key=((lambda o: _dd(o)) if _DESC else ((lambda o: (_dest_plaus(o), abs(bearing_to_j1(o["w"][0], o["w"][1]) - _cj))) if _FOC else (lambda o: (_dest_plaus(o), bearing_to_j1(o["w"][0], o["w"][1]))))))
                if _FOC and _un and GROUND["place_queries"] and (a.dest_kind or "").strip().lower() in ("hole", "container", "flat") and any(o_.get("s_t") is not None and o_["s_t"] >= GROUND["floor"] for o_ in world.objs.values() if eligible(o_)):
                    _keepf = [o for o in _un if _dest_plaus(o) == 0]
                    if len(_keepf) < len(_un): log(f"정지 재채점: 대상은 이미 채점됨 -- 도착지 종류에 맞지 않는 물체 {len(_un) - len(_keepf)}개는 건너뜀(NERO_RESTILL_FOCUS)")
                    _un = _keepf
                if not _un: break
                _stop = float(np.clip(bearing_to_j1(_un[0]["w"][0], _un[0]["w"][1]), -J1_LIMIT + 0.05, J1_LIMIT - 0.05))
                log(f"정지 재채점 {_rs + 1}: 점수 없는 물체 {len(_un)}개 중 j1={_stop:+.2f} 방향을 정지 촬영")
                n.move_j1(_stop, j7_abs=a.j7); _rr = grab(n, a.floor_z)
                if _rr is None: break
                world.update(_rr[1], float(n.got["arm_q"][0])); pub()
                if _DESC and any(eligible(o_) for o_ in world.objs.values()): log("겉모습으로 대상을 확정 -- 남은 재채점 생략"); break
                if _STAT:
                    for _o in _un:
                        if abs(bearing_to_j1(_o["w"][0], _o["w"][1]) - _stop) < 0.3: _o["rs_n"] = _o.get("rs_n", 0) + 1
                for _o in _un:      # 이 방위에서 한 번 봤는데도 점수가 안 붙은 물체는 더 이상 되돌아보지 않는다
                    if abs(bearing_to_j1(_o["w"][0], _o["w"][1]) - _stop) < 0.3 and _need(_o): _o["s_t"] = -9.0 if GROUND["queries"] else _o.get("s_t"); _o["s_p"] = -9.0 if GROUND["place_queries"] else _o.get("s_p"); _o["s_x"] = -9.0 if GROUND["excludes"] else _o.get("s_x")
                if os.environ.get("NERO_RESTILL_EARLY", "1") == "1" and not (a.scan_only and os.environ.get("NERO_SCAN_FULL", "0") == "1") and not (_STAT and any(o_.get("tentative") and o_.get("rs_n", 0) < 2 for o_ in world.objs.values())) and (_refs_now() or True) and any(eligible(o_) for o_ in world.objs.values()) and _conf_ok() and (not GROUND["place_queries"] or any(o_.get("s_p") is not None and not eligible(o_) for o_ in world.objs.values())): log(f"정지 재채점 {_rs + 1}회에서 대상/도착지 확정 -- 나머지 점수 없는 물체는 건너뜀"); break      # 2026-10-13(기본 켜짐, =0 으로 끔): 관심 물체가 확정되면 나머지는 다시 보지 않는다
        for key, f in (("t", "s_t"), ("x", "s_x"), ("p", "s_p")):
            vals = [o[f] for o in world.objs.values() if o.get(f) is not None]
            GROUND["refs"][key] = max(vals) if vals else None
        if GROUND["place_queries"]:
            rp = GROUND["refs"]["p"]
            # 2026-10-07: 도착지 점수 = 접지 점수 s_p + 오목 가산(점군으로 잰 "위가 열린 용기" 여부, 문구와 무관한 형상 단서). 무늬 없는 바구니는 CLIP 점수만으로는 다른 물체에 졌다.
            _cb_ = float(os.environ.get("NERO_DEST_CONC_BONUS", "3.0"))
            # 2026-10-08: 재현성 시행(6회 중 1회)에서 바구니가 화면에 일부만 잡혀 소형 감점에 걸려 '도착지 없음'으로 끝났다. 용기로 측정됐는데 크기만 모자란(=일부만 본) 후보가 있고
            # 쓸 만한 용기 후보가 없으면, 그 방위로 팔을 돌려 다시 보고(최대 NERO_DEST_RELOOK 회) 상자를 합친 뒤 점수를 다시 낸다. 0 이면 끈다.
            _seen_rl = set()      # 같은 후보를 두 번 돌아보지 않는다(접지 점수는 흔들려서 점수 1위만 따라가면 엉뚱한 후보만 보게 된다)
            for _it in range(int(os.environ.get("NERO_DEST_RELOOK", "3"))):
                _ms0 = float(os.environ.get("NERO_DEST_MIN_SIDE", "0.16")); _part = None; _okc = False
                for _o in world.objs.values():
                    _o["concave"] = place_ops.concavity(_o); _cc0 = _o.get("concave"); _r0 = _o.get("raw")
                    if _cc0 and _cc0.get("concave") and _r0 is not None:
                        if min(_r0[0], _r0[1]) >= _ms0: _okc = True
                        elif _o.get("s_p") is not None and id(_o) not in _seen_rl and (_part is None or _o["s_p"] > _part["s_p"]): _part = _o
                if _okc or _part is None:
                    break
                _seen_rl.add(id(_part)); _j = float(bearing_to_j1(_part["xyz"][0], _part["xyz"][1])); log(f"도착지 후보가 일부만 보임(최소변 {min(_part['raw'][0], _part['raw'][1]):.2f}m) -- j1={_j:+.2f} 로 돌려 다시 본다({_it + 1})")
                n.move_j1(float(np.clip(_j, -J1_LIMIT + 0.05, J1_LIMIT - 0.05)), j7_abs=a.j7); _rr = grab(n, a.floor_z, a.target)
                if _rr is not None:
                    world.update(_rr[1], float(n.got["arm_q"][0])); pub()
            for _o in world.objs.values():
                _o["concave"] = place_ops.concavity(_o)
            if os.environ.get("NERO_OBJ_DUMP"):      # 분석용(2026-10-08): 등록부 전체(점군 포함)를 저장 -- 목적지 기술자 오프라인 검증용. 기본 꺼짐
                try:
                    import pickle as _pk2
                    _pk2.dump({k_: dict(pts=np.asarray(o_.get("pts")) if o_.get("pts") is not None else None, xyz=[float(v) for v in o_["xyz"]], raw=[float(v) for v in o_["raw"]], yaw=float(o_.get("yaw", 0.0)), label=o_.get("label", ""), s_p=o_.get("s_p"), concave=o_.get("concave"), why=o_.get("concave_why")) for k_, o_ in world.objs.items() if "raw" in o_}, open(os.environ["NERO_OBJ_DUMP"], "wb"))
                except Exception as e_: log(f"물체 덤프 실패 {e_!r}")
            _adj = {oid: o["s_p"] + (_cb_ if (o.get("concave") or {}).get("concave") else 0.0) for oid, o in world.objs.items() if o.get("s_p") is not None and not (o.get("role") == "placed" and os.environ.get("NERO_KEEP_PLACED", "0") == "1")}
            # 2026-10-08: 도착지 용량/미측정 보정(코드 수준). (a) 안쪽 한 변이 NERO_DEST_MIN_SIDE 미만이면 물체를 담을 수 없다고 보고 감점(작은 ramekin 이 바구니를 이긴 사고),
            # (b) 용기로 측정된 후보가 하나라도 있으면 오목 미측정 후보는 NERO_DEST_UNMEAS_PEN 만큼 감점(CLIP 점수만으로 1위가 되는 것 방지). 0 으로 두면 끈다.
            _ms_ = float(os.environ.get("NERO_DEST_MIN_SIDE", "0.16")); _sp_ = float(os.environ.get("NERO_DEST_SMALL_PEN", "3.0")); _up_ = float(os.environ.get("NERO_DEST_UNMEAS_PEN", "1.5"))
            _any_conc = any((world.objs[k].get("concave") or {}).get("concave") for k in _adj)
            for _k in list(_adj):
                _o = world.objs[_k]; _raw = _o.get("raw")
                if _sp_ > 0 and _raw is not None and min(_raw[0], _raw[1]) < _ms_:
                    _adj[_k] -= _sp_
                if _up_ > 0 and _any_conc and _o.get("concave") is None:
                    _adj[_k] -= _up_
            for oid in sorted(_adj, key=lambda k: -_adj[k])[:6]:
                _o = world.objs[oid]; _cc = _o.get("concave")
                log(f"  도착지 점수 {oid} '{_o.get('label')}' ({_o['xyz'][0]:+.2f},{_o['xyz'][1]:+.2f}) s_p={_o['s_p']:.2f} 오목={'예 깊이 %.1fcm' % (100 * _cc['depth']) if _cc and _cc['concave'] else ('아니오' if _cc else '미측정[' + str(_o.get('concave_why', '')) + ']')} -> {_adj[oid]:.2f}")
            _rpa = max(_adj.values()) if _adj else None
            # 2026-10-08: 용기로 측정되고 크기도 충분한 후보는 접지 점수 하한(floor)에 막히지 않는다(바구니 s_p 가 -2~-4 로 나와도 형상이 맞으면 도착지). 순위 규칙(1위 -0.5 이내)은 그대로.
            def _cont_ok(_k):
                _o = world.objs[_k]; _r = _o.get("raw")
                return bool((_o.get("concave") or {}).get("concave")) and _r is not None and min(_r[0], _r[1]) >= _ms_
            dests = [oid for oid in _adj if _rpa is not None and _adj[oid] >= _rpa - 0.5 and (_adj[oid] >= GROUND["floor"] or (_cont_ok(oid) and os.environ.get("NERO_DEST_CONT_BYPASS", "1") != "0"))]
            for oid in dests:
                world.objs[oid]["role"] = "dest"; world.objs[oid]["s_p_adj"] = _adj[oid]
            log(f"도착지 '{a.place_on}': {dests or '못 찾음'} (점수 기준 {rp}, 오목 가산 후 {_rpa})")
            def _estimate_hole_raw(_oid, _light=False):      # 2026-10-08: 후보 하나의 홀더 입구를 3시점으로 추정한다(입구 링이 잡히고 바깥 한 변이 5~14cm 이면 hole). 바구니 같은 큰 용기는 링 면적 상한에서 걸러진다
                import hole_estimator_v2 as _V2
                if _oid not in world.objs: return False
                _o = world.objs[_oid]; _Ps = []; _Ns = []; _cams = []
                _j0 = float(bearing_to_j1(_o["xyz"][0], _o["xyz"][1]))
                for _dj in ((0.0,) if _light else (0.0, 0.30, -0.30)):      # 2026-10-14(NERO_DEST_LIGHT, 기본 꺼짐): 도착지 종류가 용기/평면이면 홀 추정은 필요 없고 색/라벨 재채점 1시점만 한다
                    n.move_j1(float(np.clip(_j0 + _dj, -J1_LIMIT + 0.05, J1_LIMIT - 0.05)), j7_abs=a.j7); _rr = grab(n, a.floor_z)
                    if _rr is None or not _rr[1]: continue
                    if os.environ.get("NERO_DEST_RESCORE", "1") == "1":      # 2026-10-13(기본 꺼짐): 홀 추정용 정지 프레임에서 이 후보 위치의 검출을 찾아 점수/라벨/색을 후보에 직접 합친다(등록부 갱신/삭제 없음 -- world.update 는 후보 id 를 지워 KeyError 를 냈다)
                        _nd = min(_rr[1], key=lambda d_: np.hypot(d_["w"][0] - _o["xyz"][0], d_["w"][1] - _o["xyz"][1]))
                        if np.hypot(_nd["w"][0] - _o["xyz"][0], _nd["w"][1] - _o["xyz"][1]) < 0.12 and not _nd.get("gated"):
                            for _k in ("s_t", "s_x", "s_p", "p_t"):
                                _nv, _ov = _nd.get(_k), _o.get(_k)
                                if _nv is not None and (_ov is None or _ov <= -9.0 or _nv > _ov): _o[_k] = _nv
                            if _nd.get("label"):
                                _o["label"] = _nd["label"]; _o["top"] = _nd.get("top", [])
                                _lh = (_o.get("lab_hist") or []) + [_nd["label"]]; _o["lab_hist"] = _lh[-6:]; _o["label_disp"] = max(set(_o["lab_hist"]), key=lambda x_: (_o["lab_hist"].count(x_), len(_o["lab_hist"]) - 1 - _o["lab_hist"][::-1].index(x_)))
                            if _nd.get("hsv"): _hh = (_o.get("hsv_hist") or []) + [_nd["hsv"]]; _o["hsv_hist"] = _hh[-7:]; _o["hsv"] = [int(x) for x in np.median(np.array(_o["hsv_hist"]), axis=0)]
                            log(f"  재채점 {_oid}: 라벨 {_o.get('label')} s_p {_o.get('s_p')} HSV {_o.get('hsv')}"); pub()
                    _g, _dts = _rr; _cand = [d_ for d_ in _dts if len(d_["P"]) >= 3000 and np.hypot(d_["w"][0] - _o["xyz"][0], d_["w"][1] - _o["xyz"][1]) < 0.12]
                    if not _cand: continue
                    _d = max(_cand, key=lambda d_: len(d_["P"])); _P, _N = _V2.view_pn(_g["pts_base"][..., :3], _d["mask"]); _Ps.append(_P); _Ns.append(_N); _cams.append(_g["W_T_cam"][:3, 3])
                if _light: log(f"  도착지 후보 {_oid}: 종류가 홀이 아님 -- 홀 추정 생략, 재채점 1시점만(NERO_DEST_LIGHT)"); return False
                if not _Ps: log(f"  도착지 후보 {_oid} 홀 추정: 점군 없음"); return False
                _est = _V2.estimate_pn(np.concatenate(_Ps), np.concatenate(_Ns), np.mean(_cams, axis=0), inner=insert_ops.HOLE_INNER, wall=insert_ops.WALL, length=insert_ops.TUBE_LEN)
                if os.environ.get("NERO_HOLE_OPEN_RING", "0") == "1" and _est.get("ok") and _est.get("via") == "rim":      # 2026-11-03(기본 0=현행 불변): 원형 링 면적 보정 -- outer_est=√면적 은 지름의 √π/2=0.886배라 바깥 -14 mm·개구 -18 mm 과소. 원 등가 지름 2√(면적/π)=outer_est*1.1284 로 고치고 개구는 같은 벽 가정(insert_ops.WALL)으로 outer-2*wall
                    _o2 = _est.get("outer_est", 0) * (2.0 / math.sqrt(math.pi)); log(f"  도착지 후보 {_oid} 링 면적 보정(원형): 바깥 {100 * _est.get('outer_est', 0):.1f}->{100 * _o2:.1f}cm 개구 {100 * _est.get('open_est', 0):.1f}->{100 * (_o2 - 2 * insert_ops.WALL):.1f}cm (NERO_HOLE_OPEN_RING)"); _est["outer_est"] = _o2; _est["open_est"] = max(0.0, _o2 - 2 * insert_ops.WALL)
                if _est.get("ok") and _est.get("via") == "rim": log(f"  도착지 후보 {_oid} 홀 기하: 바깥 {100 * _est.get('outer_est', 0):.1f}cm 축방향 길이 {100 * _est.get('ext_axis', 0):.1f}cm 단면 최대 폭 {100 * _est.get('ext_cross', 0):.1f}cm")
                _san = os.environ.get("NERO_HOLE_SANITY", "0") == "1" and _est.get("ok") and _est.get("via") == "rim" and not (_est.get("ext_cross", 0) <= float(os.environ.get("NERO_HOLE_CROSS_RATIO", "1.35")) * _est.get("outer_est", 0) + 0.01 and _est.get("ext_axis", 0) <= float(os.environ.get("NERO_HOLE_AXIS_RATIO", "1.3")) * insert_ops.TUBE_LEN)      # 2026-10-12: 통이면 단면 폭 ~ 링 바깥 한 변, 축 방향 길이 <= 통 길이. 넓고 얕은 용기(바구니)의 벽 한 면이 링으로 오인되는 것을 거른다(기본 꺼짐)
                if _san: _o["hole_note"] = "홀 아님: 단면 %.1fcm / 축 %.1fcm 가 링 %.1fcm 과 불일치" % (100 * _est.get("ext_cross", 0), 100 * _est.get("ext_axis", 0), 100 * _est.get("outer_est", 0)); log(f"  도착지 후보 {_oid}: 홀 기하 불일치 -- 홀 아님으로 처리(NERO_HOLE_SANITY)"); return False
                if _est.get("ok") and _est.get("via") == "rim" and float(os.environ.get("NERO_HOLE_OUTER_MIN", "0.05")) <= _est.get("outer_est", 0) <= float(os.environ.get("NERO_HOLE_OUTER_MAX", "0.14")):
                    _o["hole"] = dict(ok=True, E=_est["E"], d=_est["d"], open=(_est["open_est"], _est["open_est"]), outer=_est["outer_est"], depth=insert_ops.TUBE_LEN, views=len(_Ps))
                    _o["hole_note"] = "홀 인정: 링 %.1fcm  단면 %.1fcm  축 %.1fcm" % (100 * _est.get("outer_est", 0), 100 * _est.get("ext_cross", 0), 100 * _est.get("ext_axis", 0))
                    log(f"  도착지 후보 {_oid} 홀 추정: E={np.round(_est['E'], 3).tolist()} d={np.round(_est['d'], 2).tolist()} 개구부 {100 * _est['open_est']:.1f}cm (시점 {len(_Ps)}개)"); return True
                log(f"  도착지 후보 {_oid}: 홀 아님/추정 실패 ({_est.get('via') or _est.get('why')}, 바깥 {None if _est.get('outer_est') is None else round(100 * _est['outer_est'], 1)}cm)"); return False
            _HOLE_DONE = {}
            def _estimate_hole(_oid, _light=False):      # 2026-10-13(NERO_DEST_REUSE=1, 기본 꺼짐, 등록부 기억과 함께): 이전 명령에서 확인한 도착지(홀 인정/홀 아님)는 같은 자리에서 다시 보이면 3시점 촬영을 생략하고 결과를 재사용한다
                if _oid not in world.objs: return False
                _o = world.objs[_oid]
                if _light: return _estimate_hole_raw(_oid, True)
                if os.environ.get("NERO_DEST_REUSE", "0") == "1" and _o.get("reg_xyz") and not _o.get("restored"):
                    _mv = math.hypot(_o["xyz"][0] - _o["reg_xyz"][0], _o["xyz"][1] - _o["reg_xyz"][1])
                    if (_o.get("hole") or {}).get("ok") and math.hypot(_o["hole"]["E"][0] - _o["xyz"][0], _o["hole"]["E"][1] - _o["xyz"][1]) < 0.08 and _mv < 0.12:
                        log(f"  도착지 후보 {_oid}: 이전 명령에서 확인한 홀 추정을 재사용(촬영 생략, 위치 변화 {100 * _mv:.1f}cm)"); return True
                    if _o.get("hole_neg") and _mv < 0.12:
                        log(f"  도착지 후보 {_oid}: 이전 명령에서 '홀 아님'으로 확인함 -- 촬영 생략(위치 변화 {100 * _mv:.1f}cm)"); return False
                if os.environ.get("NERO_HOLE_ONCE", "0") == "1" and _oid in _HOLE_DONE:      # 2026-10-14(기본 꺼짐): 같은 실행에서 같은 후보를 두 번 추정하지 않는다(실측: 바구니가 홀 추정 2회, 각 6~9초)
                    log(f"  도착지 후보 {_oid}: 이번 실행에서 이미 추정함 -- 재사용"); return _HOLE_DONE[_oid]
                _ok = _estimate_hole_raw(_oid); _HOLE_DONE[_oid] = _ok
                if _oid in world.objs: world.objs[_oid]["hole_neg"] = (not _ok)
                return _ok
            _dk0 = (a.dest_kind or "").strip().lower()
            def _kind_match_now(_k):      # 2026-10-14: 이미 측정된 형상으로 종류가 맞는 후보인가(홀 추정 없이 판단 가능한 경우만 light/skip 을 쓴다)
                _o = world.objs[_k]; _r = _o.get("raw")
                if _r is None: return False
                _ms = min(_r[0], _r[1]); _cc = bool((_o.get("concave") or {}).get("concave"))
                return (_cc and _ms >= _ms_) if _dk0 == "container" else ((_ms >= 0.12 and not _cc) if _dk0 == "flat" else False)
            _kind_ok0 = _dk0 in ("container", "flat") and any(_kind_match_now(k_) for k_, o_ in world.objs.items() if o_.get("role") != "placed")      # 2026-10-14: 문구 점수로 걸러진 dests 만 보면 접시처럼 CLIP 점수가 없는 평면 후보를 못 봐서 홀 폴백(40~50초)으로 갔다 -- 측정된 형상으로 종류가 맞는 물체가 장면에 있으면 가벼운 경로(종류 필터가 그 후보를 고른다)
            if os.environ.get("NERO_DEST_LIGHT", "0") == "1" and _dk0 in ("container", "flat") and not _kind_ok0: log(f"도착지 종류 '{_dk0}' 에 맞는 측정된 후보가 없음 -- 가벼운 경로 대신 전체 홀 추정 경로로 복귀")
            if AUTO:
                _hd = []      # 2026-10-13(기본 켜짐, NERO_HOLE_DEDUP=0 으로 끔): 같은 위치(10cm 이내)의 중복 검출은 홀 추정을 한 번만 하고 결과를 복사한다
                for _oid in list(dests):
                    if _oid not in world.objs: continue      # 재채점 프레임(NERO_DEST_RESCORE)이 등록부를 갱신하면서 지워졌을 수 있다
                    _o_ = world.objs[_oid]; _dup = next((k_ for k_ in _hd if np.hypot(world.objs[k_]["xyz"][0] - _o_["xyz"][0], world.objs[k_]["xyz"][1] - _o_["xyz"][1]) < 0.10), None) if os.environ.get("NERO_HOLE_DEDUP", "1") == "1" else None
                    if _dup is None:
                        _hd.append(_oid); _estimate_hole(_oid, _light=(os.environ.get("NERO_DEST_LIGHT", "0") == "1" and _kind_ok0))
                    else:
                        log(f"  도착지 후보 {_oid} 는 {_dup} 와 같은 위치 -- 홀 추정 결과를 복사(재촬영 생략)")
                        if world.objs[_dup].get("hole"): _o_["hole"] = dict(world.objs[_dup]["hole"])
                        if world.objs[_dup].get("hole_note"): _o_["hole_note"] = world.objs[_dup]["hole_note"]
                dests = [k_ for k_ in dests if k_ in world.objs]
                _kskip = os.environ.get("NERO_DEST_KIND_SKIP", "0") == "1" and (a.dest_kind or "").strip().lower() in ("container", "flat") and (os.environ.get("NERO_DEST_LIGHT", "0") != "1" or _kind_ok0)      # 2026-10-14(기본 꺼짐): 도착지 종류가 용기/평면이면 홀 후보 폴백(후보당 3시점 촬영 6~9초)을 생략한다
                if _kskip: log(f"도착지 종류 '{a.dest_kind}' -- 홀 후보 폴백 생략(NERO_DEST_KIND_SKIP)")
                if not _kskip and not any((world.objs[k_].get("hole") or {}).get("ok") for k_ in dests):      # 문구 접지가 홀더를 못 올려도 기하(홀 입구 링)가 확인되면 도착지로 인정한다
                    _tk = int(os.environ.get("NERO_AUTO_TOPK", "3"))
                    _byS = [k_ for k_, o_ in sorted(world.objs.items(), key=lambda kv: -(kv[1].get("s_p") if kv[1].get("s_p") is not None else -1e9)) if o_.get("s_p") is not None and "raw" in o_ and o_.get("role") != "placed"][:_tk]
                    _geo = [k_ for k_, o_ in world.objs.items() if "raw" in o_ and o_.get("role") not in ("placed",) and min(o_["raw"][0], o_["raw"][1]) >= (0.05 if os.environ.get("NERO_DEST_GEO_FIRST", "1") == "1" else 0.07) and o_["raw"][2] >= 0.10
                            and not ((o_.get("concave") or {}).get("concave") and min(o_["raw"][0], o_["raw"][1]) >= _ms_)]      # 홀더 모양(바닥 변 7cm 이상, 높이 10cm 이상, 큰 용기 아님). 막대(단면 3cm대)는 걸러진다
                    _gf = os.environ.get("NERO_DEST_GEO_FIRST", "1") == "1"      # 2026-10-13(기본 꺼짐): 모양 조건(바닥 변 5cm 이상, 높이 10cm 이상) 후보를 먼저, 문구 점수 상위는 뒤에, 상한 8개 -- 연필통이 크기 편차/6개 제한으로 후보에서 빠졌다
                    for _oid in list(dict.fromkeys((_geo + _byS) if _gf else (_byS + _geo)))[:(_tk + 5 if _gf else _tk + 3)]:
                        if _oid not in world.objs: continue
                        if (world.objs[_oid].get("hole") or {}).get("ok") or _estimate_hole(_oid):
                            world.objs[_oid]["role"] = "dest"; world.objs[_oid]["s_p_adj"] = world.objs[_oid].get("s_p") or 0.0
                            if _oid not in dests: dests.append(_oid)
                    log(f"도착지(기하 확인 후): {dests or '못 찾음'}")
                _dk = (a.dest_kind or "").strip().lower()
                if _dk in ("hole", "container", "flat"):      # 2026-10-12: 언어 쪽이 준 목적지 종류로 후보를 거른다(CLIP 문구 점수와 크기/오목 가감점은 같은 종류 후보끼리만 겨룬다). 맞는 후보가 없으면 거르지 않는다
                    def _kind_of(_o):
                        _r = _o.get("raw"); _ms2 = min(_r[0], _r[1]) if _r is not None else 0.0
                        if (_o.get("hole") or {}).get("ok"): return "hole"
                        if bool((_o.get("concave") or {}).get("concave")) and _ms2 >= _ms_: return "container"
                        if _r is not None and _ms2 >= 0.12 and not (_o.get("concave") or {}).get("concave"): return "flat"
                        return "other"
                    _pool = list(dict.fromkeys(dests + [k_ for k_, o_ in sorted(world.objs.items(), key=lambda kv: -(kv[1].get("s_p") if kv[1].get("s_p") is not None else -1e9)) if "raw" in o_ and o_.get("role") != "placed"][:12]))      # 2026-10-12: 점수가 없는 물체도 후보 풀에 넣는다(게이트가 스윕 프레임 채점을 막으면 접시가 풀에서 빠졌다)
                    _keep = [k_ for k_ in _pool if _kind_of(world.objs[k_]) == _dk]
                    log(f"목적지 종류 '{_dk}' 필터: 후보 { {k_: _kind_of(world.objs[k_]) for k_ in _pool} } -> {_keep or '맞는 후보 없음(필터 생략)'}")
                    if _keep:
                        for k_ in dests:
                            if k_ not in _keep: world.objs[k_]["role"] = "obstacle"
                        for k_ in _keep:
                            world.objs[k_]["role"] = "dest"; world.objs[k_]["s_p_adj"] = world.objs[k_].get("s_p") if world.objs[k_].get("s_p") is not None else -5.0
                        dests = list(_keep)
                        _noc = max(_keep, key=lambda k_: world.objs[k_].get("s_p_adj", -1e9))      # 색 없이 고른 후보(기록용)
                        _qc = query_colors(a.place_on) if os.environ.get("NERO_COLOR_MATCH", "1") == "1" else set()
                        _bon = {}
                        if _qc and any(world.objs[k_].get("hsv") and color_name(world.objs[k_]["hsv"]) in _qc for k_ in _keep):      # 종류 필터를 통과한 후보들 사이에서만, 그리고 질의 색과 맞는 후보가 하나라도 있을 때만 색을 반영한다(형태/종류가 먼저, 색은 보조)
                            for k_ in _keep:
                                _b = color_bonus(world.objs[k_], _qc); world.objs[k_]["s_p_adj"] += _b; _bon[k_] = _b
                                log(f"  색 비교 {k_}: HSV {world.objs[k_].get('hsv')} -> {color_name(world.objs[k_]['hsv']) if world.objs[k_].get('hsv') else '미측정'} / 질의 색 {sorted(_qc)} -> 가감 {_b:+.1f} (s_p_adj {world.objs[k_]['s_p_adj']:.2f})")
                        elif _qc: log(f"  색 비교 생략: 질의 색 {sorted(_qc)} 와 맞는 후보가 없음 -> 색 가감 없음")
                        _bst = max(_keep, key=lambda k_: world.objs[k_].get("s_p_adj", -1e9))
                        if os.environ.get("NERO_RVIZ_ENT") == "1":
                            STATE["dest_cands"] = list(_keep); STATE["dest_pick"] = _bst
                            STATE["dest_reason"] = dict(query=str(a.place_on).split("|")[0], kind=_dk, note=("점수 최고 선택" + (" · 색이 선택을 바꿈" if _noc != _bst else "")),
                                                        rows=sorted([dict(id=k_, label=world.objs[k_].get("label"), s_p=float(world.objs[k_].get("s_p") if world.objs[k_].get("s_p") is not None else -5.0),
                                                                          color=(color_name(world.objs[k_]["hsv"]) if (_qc and world.objs[k_].get("hsv")) else None), bonus=float(_bon.get(k_, 0.0)),
                                                                          adj=float(world.objs[k_].get("s_p_adj", -99)), chosen=(k_ == _bst)) for k_ in _keep], key=lambda r_: -r_["adj"]))
                            pub()
                        log(f"도착지 예상 선택 {_bst} xy=({world.objs[_bst]['xyz'][0]:+.3f},{world.objs[_bst]['xyz'][1]:+.3f}) s_p_adj={world.objs[_bst].get('s_p_adj'):.2f} 후보 { {k_: round(world.objs[k_].get('s_p_adj', -99), 2) for k_ in _keep} }" + (f" | 색이 선택을 바꿈(색 없이는 {_noc})" if _noc != _bst else ""))      # 2026-10-13: 최종 선택(점수 최고)을 미리 기록(로그만)
                        if llm_decide.enabled() and len(_keep) > 1:      # 2026-10-14: 같은 종류 목적지 후보가 여럿이면 Claude 가 고른다(선택 id 의 점수를 최고로 올려 place 단계가 그걸 쓰게 한다)
                            _co = sorted(_keep, key=lambda k_: -world.objs[k_].get("s_p_adj", -1e9))[:4]
                            _ch, _src, _why = llm_decide.ask("dest", f"목적지 '{str(a.place_on).split('|')[0]}' 후보가 {len(_co)}개다. 어느 것으로 할까?", dict(table=_llm_table(None, None, ()), 종류=_dk), [dict(id=k_, text=f"{world.objs[k_].get('label_disp') or world.objs[k_].get('label')} (CLIP+기하 점수 {world.objs[k_].get('s_p_adj', -99):.2f}, 위치 ({world.objs[k_]['xyz'][0]:+.2f},{world.objs[k_]['xyz'][1]:+.2f}))") for k_ in _co], _bst, log=log)
                            if _ch != _bst: world.objs[_ch]["s_p_adj"] = world.objs[_bst].get("s_p_adj", 0.0) + 1.0; log(f"[llm] 목적지를 {_bst} -> {_ch} 로 바꿈")
                        if os.environ.get("NERO_DEST_FREEZE", "0") == "1":      # 2026-10-14: 목적지 조사 1회 끝(홀 후보는 위에서 3시점 홀 추정을 이미 했다) -- 위치/형상을 고정하고 LLM 표에는 입구/깊이까지 넘긴다
                            for _fk, _fo in world.objs.items():
                                if _fo.get("role") == "dest" and "raw" in _fo:
                                    _fo["frozen"] = True; _h = _fo.get("hole") or {}; _c = _fo.get("concave") or {}
                                    log(f"  목적지 {_fk} 고정: xy=({_fo['xyz'][0]:+.3f},{_fo['xyz'][1]:+.3f}) 박스 {[round(100 * v_, 1) for v_ in _fo['raw']]}cm" + (f" 홀 입구 {100 * _h['open'][0]:.1f}cm 링 {100 * _h['outer']:.1f}cm" if _h.get("ok") else "") + (f" 오목 깊이 {100 * _c.get('depth', 0):.1f}cm" if _c.get("concave") else ""))
                for _k5, _f5 in (("t", "s_t"), ("x", "s_x")):      # 도착지로 확정된 물체는 대상 기준점 계산에서 뺀다(홀더가 대상 문구에 높은 점수를 받아 진짜 막대가 밀려나는 문제)
                    _v5 = [o_[_f5] for o_ in world.objs.values() if o_.get(_f5) is not None and o_.get("role") != "dest"]
                    GROUND["refs"][_k5] = max(_v5) if _v5 else None
            if os.environ.get("NERO_DEST_DUMP") and dests:      # 분석용: 도착지 점군과 박스를 저장(기본 꺼짐)
                import pickle as _pk
                _pk.dump({k: dict(pts=np.asarray(world.objs[k].get("pts")), xyz=world.objs[k].get("xyz"), raw=world.objs[k].get("raw"), yaw=world.objs[k].get("yaw"), concave=world.objs[k].get("concave")) for k in dests}, open("/tmp/dest_dump.pkl", "wb"))
            if not dests:
                log("도착지 없음 -- 종료"); set_state(stage="종료"); pub()
                json.dump(dict(events=[], picks_ok=0, found=0, why="no_dest", total_s=time.time() - t_run), open(a.out, "w"), indent=1)
                rclpy.shutdown(); return
        def _refs():
            for key, f in (("t", "s_t"), ("x", "s_x"), ("p", "s_p")):
                vals = [o[f] for o in world.objs.values() if o.get(f) is not None and o.get("role") not in ("dest", "placed")]
                GROUND["refs"][key] = max(vals) if vals else None
        tg = [oid for oid, o in world.objs.items() if eligible(o)]
        if FOCUS and early and tg:      # 발견한 대상/도착지 쪽만 정중앙으로 맞춰 다시 채점한다(전체를 다시 훑지 않는다)
            _rs = sorted(tg, key=lambda k: -(world.objs[k].get("s_t") or 0))[:2] + [oid for oid, o in world.objs.items() if o.get("role") == "dest"][:1]
            log(f"조기 종료 -- {_rs} 쪽만 재촬영")
            for oid in _rs:
                o = world.objs.get(oid)
                if o is None or "w" not in o:
                    continue
                n.move_j1(bearing_to_j1(o["w"][0], o["w"][1]), j7_abs=a.j7)
                r = grab(n, a.floor_z)
                if r is not None:
                    world.update(r[1], float(n.got["arm_q"][0])); pub()
            _refs(); tg = [oid for oid, o in world.objs.items() if eligible(o)]
            if os.environ.get("NERO_COVER_PATH", "1") == "1":      # 이송 경로가 훑는 방위 구간(홈~대상/도착지/삽입구)에서 아직 안 본 각도는 보고 간다: 못 본 장애물(홀더 등)을 월드에 등록
                _b = [0.0] + [bearing_to_j1(world.objs[k]["w"][0], world.objs[k]["w"][1]) for k in tg + [oid for oid, o in world.objs.items() if o.get("role") == "dest"] if "w" in world.objs[k]]
                if a.insert_at:
                    import insertion_geometry as _G0
                    _E0, _ = _G0.parse_insert_spec(a.insert_at); _b.append(bearing_to_j1(float(_E0[0]), float(_E0[1])))
                _lo, _hi = min(_b) - 0.25, max(_b) + 0.25
                _seen = [x for x in sorted(angles, key=lambda x: abs(x))]
                _seen = _seen[:_seen.index(ang) + 1] if ang in _seen else _seen
                if _seen_override is not None: _seen = _seen_override          # 연속 스윕 탐색이 훑은 범위
                _todo = sorted([x for x in angles if _lo <= x <= _hi and x not in _seen], key=lambda x: abs(x - _b[-1]))
                if _sk and os.environ.get("NERO_SKIP_COVER", "0") == "1" and _todo: log(f"[sweep-skip] 기억된 엔티티가 이송 경로의 장애물을 이미 알고 있어서 안 본 각도 {len(_todo)}개 재확인을 생략"); _todo = []      # 2026-10-14(기본 꺼짐)
                log(f"이송 경로 방위 [{_lo:+.2f},{_hi:+.2f}] 중 안 본 각도 {len(_todo)}개 확인: {[round(x, 2) for x in _todo]}")
                _fast_ok = False
                if _todo and os.environ.get("NERO_COVER_FAST", "0") == "1":      # 연속 스윕 1단계(2026-10-13 기본 켜짐, =0 으로 끔): 안 본 각도들을 한 번의 연속 스윕으로 훑고 도착 프레임을 이동 중에 처리한다
                    try:
                        import fast_scan
                        fast_scan.get(n).sweep(_todo, a.j7, lambda v_: grab(_scan_snap(v_), a.floor_z), lambda dets, j1: (world.update(dets, j1), pub()), log=log); _fast_ok = True
                    except Exception as e_:
                        log(f"연속 스윕 실패 -> 기존 정지-촬영으로 대체: {e_!r}")
                if not _fast_ok:
                    for x in _todo:
                        n.move_j1(x, j7_abs=a.j7)
                        r = grab(n, a.floor_z)
                        if r is not None:
                            world.update(r[1], float(n.got["arm_q"][0])); pub()
                _refs(); tg = [oid for oid, o in world.objs.items() if eligible(o)]
        if not tg and GROUND["queries"] and not a.scan_only and not GROUND.get("desc"):      # (겉모습 지정에서는 점수가 없어 이 폴백이 의미 없고 시간만 쓴다)
            # 대상이 하나도 안 잡히면 점수가 높은 후보 몇 개를 화면 중앙에 맞춰 다시 채점한다(스윕 중에는 잘리거나 비스듬히 보여 점수가 낮을 수 있다).
            cand = sorted((oid for oid, o in world.objs.items() if o.get("s_t") is not None and o.get("role") not in ("dest", "placed") and "w" in o), key=lambda k: -world.objs[k]["s_t"])[:3]
            log(f"대상 후보 없음 -- 점수 상위 {cand} 를 중앙에 맞춰 재채점")
            for oid in cand:
                o = world.objs.get(oid)
                if o is None:
                    continue
                n.move_j1(bearing_to_j1(o["w"][0], o["w"][1]), j7_abs=a.j7)
                r = grab(n, a.floor_z)
                if r is not None:
                    world.update(r[1], float(n.got["arm_q"][0])); pub()
            _refs(); tg = [oid for oid, o in world.objs.items() if eligible(o)]
        if os.environ.get("NERO_TARGET_COLOR", "0") == "1" and GROUND["queries"] and not GROUND.get("_tcolor"):      # 2026-10-14(기본 꺼짐, NERO_LEAN 이 켬): 모양이 비슷한 물체(안드로이드 모양 판다/공룡)는 CLIP 이 서로 바꿔 읽는다 -- 대상 문구의 색 단어와 측정한 엔티티 색(HSV)이 일치하면 가점, 불일치하면 감점해서 색이 뒤집는다(문구에 색 단어가 없으면 아무것도 안 한다)
            _qct = query_colors(a.target); _amt = float(os.environ.get("NERO_TARGET_COLOR_BONUS", "2.5"))
            if _qct:
                GROUND["_tcolor"] = True
                for _k, _o in world.objs.items():
                    if _o.get("s_t") is not None and _o.get("hsv") and _o.get("role") not in ("dest", "placed"):
                        _b = color_bonus(_o, _qct, _amt); _o["s_t"] += _b; log(f"  대상 색 보정 {_k}: HSV {_o['hsv']} -> {color_name(_o['hsv'])} / 문구 색 {sorted(_qct)} -> {_b:+.1f} (s_t {_o['s_t']:.1f})")
            _refs(); tg = [oid for oid, o in world.objs.items() if eligible(o)]
        log(f"사전 훑기 완료: 물체 {len(world.objs)}개, 점수 기준 t={GROUND['refs']['t']} x={GROUND['refs']['x']}, 대상 {len(tg)}개 "
            + str({oid: (world.objs[oid].get('label'), None if world.objs[oid].get('s_t') is None else round(world.objs[oid]['s_t'], 1)) for oid in world.objs}))
        if os.environ.get("NERO_SCORE_DUMP"):      # 분석용: 스윕 직후 등록부(위치/라벨/접지 점수)를 저장(기본 꺼짐)
            try: json.dump({k_: dict(xyz=[float(v) for v in o_["xyz"]], raw=[float(v) for v in o_["raw"]], label=o_.get("label", ""), s_t=o_.get("s_t"), s_p=o_.get("s_p"), p_t=o_.get("p_t")) for k_, o_ in world.objs.items() if "xyz" in o_ and "raw" in o_}, open(os.environ["NERO_SCORE_DUMP"], "w"), default=str)
            except Exception as _e: log(f"점수 덤프 실패 {_e!r}")
        scene = {oid: dict(label=o.get("label"), color=(color_name(o["hsv"]) if o.get("hsv") else None), top=o.get("top", []), s_t=o.get("s_t"), p_t=o.get("p_t"), s_x=o.get("s_x"), xy=[round(o["w"][0], 3), round(o["w"][1], 3)] if "w" in o else None,
                           size_cm=[round(100 * v, 1) for v in o["raw"]] if "raw" in o else None, feat=([round(float(x), 4) for x in o["feat"]] if o.get("feat") is not None and os.environ.get("NERO_SCAN_FEAT") == "1" else None), match=bool(eligible(o))) for oid, o in world.objs.items()}
        if os.environ.get("NERO_VLM_LABEL", "0") == "1":      # 2026-10-19: 승격 -- 플래너에는 VLM 자유 서술 라벨 + 일반 어휘(LVIS/ImageNet) CLIP 후보를 준다(52개 고정 사전 폐기). 대부분 스캔 중에 이미 끝나 있다.
            _tv = time.time()
            for oid, o in world.objs.items():
                if oid in scene:
                    scene[oid]["vlm"] = grounding.vlm_resolve(VLM_FUT.get(oid), wait=4.0); scene[oid]["general"] = grounding.general_labels(o["feat"]) if o.get("feat") is not None else []
            log(f"VLM 라벨 모으기 {time.time() - _tv:.1f}s 대기(스캔 중 제출한 결과): " + str({k_: v_.get("vlm") for k_, v_ in scene.items()}))
        reg_save("사전 훑기 완료")
        if a.scan_only:
            if os.environ.get("NERO_DEST_PERCEIVE", "0") == "1":      # 2026-10-28(기본 꺼짐): 목적지 인식 본체 -- 등록부(점군·박스)에서 기하로 유형 판정해 인식판 dests 를 저장(정답 입력 없음)
                try:
                    import dest_perceive as _DPC
                    _nrl = int(os.environ.get("NERO_PERCEIVE_RELOOK", "0")); _seen_rl = set(); _t_rl = time.time()      # 2026-10-28(기본 0=끔): 부분 시야 의심 후보(공동 깊이는 용기급인데 최소변이 작음)를 그 방위로 팔을 돌려 다시 본다(이름 무관)
                    for _it in range(_nrl):
                        _cs = _DPC.relook_candidates(world.objs, _seen_rl)
                        if not _cs: break
                        _seen_rl.add(_cs[0]); _ro = world.objs[_cs[0]]; log(f"[dest 인식] 부분 시야 의심 {_cs[0]} 크기 {[round(v, 3) for v in _ro['raw']]} -- 재관측 {_it + 1}/{_nrl}")
                        n.move_j1(float(np.clip(bearing_to_j1(_ro["xyz"][0], _ro["xyz"][1]), -J1_LIMIT + 0.05, J1_LIMIT - 0.05)), j7_abs=a.j7); _rr = grab(n, a.floor_z)
                        if _rr is not None: world.update(_rr[1], float(n.got["arm_q"][0]))
                    if _nrl: log(f"[dest 인식] 재관측 {len(_seen_rl)}회, {time.time() - _t_rl:.1f}s")
                    if os.environ.get("NERO_PERCEIVE_DUMP"):      # 2026-10-28(기본 꺼짐): 등록부 엔티티(점군·박스)를 저장 -- 인식 사양 오프라인 개선 시험용(개발 장면만)
                        import pickle as _pk3; _pk3.dump({k_: dict(pts=(None if o_.get("pts") is None else np.asarray(o_["pts"], np.float32)), raw=[float(v) for v in o_["raw"]], xyz=[float(v) for v in o_["xyz"]], yaw=float(o_.get("yaw", 0.0)), fobbs=o_.get("fobbs")) for k_, o_ in world.objs.items() if "raw" in o_ and "xyz" in o_}, open(os.environ["NERO_PERCEIVE_DUMP"], "wb"))
                    _pd = _DPC.perceive(world.objs); _pp = os.environ.get("NERO_PERCEIVED_DESTS") or (a.out + ".dests.json")
                    json.dump(_pd, open(_pp, "w"), ensure_ascii=False, indent=1); log(f"[dest 인식] {len(_pd)}개 -> {_pp}: " + str([(e_["id"], e_["kind"], e_["xy"], e_.get("d_c")) for e_ in _pd]))
                except Exception as _e: log(f"[dest 인식] 실패 {_e!r}")
            n.home(); wait_observation(n, 10)
            json.dump(dict(scene=scene, refs=GROUND["refs"], total_s=time.time() - t_run), open(a.out, "w"), indent=1, ensure_ascii=False)
            set_state(stage="종료"); rclpy.shutdown(); return
        if not tg and a.plan_file: log("[chain] 사전 훑기에서 첫 항목을 못 찾음 -- 종료하지 않고 본 탐색으로 이어 감(못 찾으면 다음 항목으로)")      # 2026-10-15: 첫 항목이 잘린 검출/각도 때문에 임베딩이 안 맞으면 전체 정리가 11초 만에 끝났다
        elif not tg:
            log("대상 없음 -- 종료"); set_state(stage="종료"); pub()
            json.dump(dict(events=[], picks_ok=0, found=0, scene=scene, refs=GROUND["refs"], total_s=time.time() - t_run), open(a.out, "w"), indent=1, ensure_ascii=False)
            rclpy.shutdown(); return

    def _holder_dims():      # 2026-10-14(NERO_HOLE_MEASURED, NERO_LEAN 이 켬): 삽입 계획의 홀더 치수를 고정 상수(안지름 6cm, 벽 1cm)가 아니라 초반 조사에서 잰 값(개구부, 링, 깊이)으로 넣는다 -- 입구 8.5cm 컵을 6cm 구멍으로 계산해 여유가 음수가 되던 문제
        _hd = next((o_ for o_ in world.objs.values() if o_.get("role") == "dest" and (o_.get("hole") or {}).get("ok")), None); _hh = (_hd or {}).get("hole") or {}
        if os.environ.get("NERO_HOLE_MEASURED", "0") == "1" and _hh.get("ok") and _hh.get("open"):
            _in = float(_hh["open"][0]); return dict(inner=_in, wall=max(0.005, 0.5 * (float(_hh.get("outer", _in + 0.02)) - _in)), len=float(_hh.get("depth", insert_ops.TUBE_LEN)))
        return dict(inner=insert_ops.HOLE_INNER, wall=insert_ops.WALL, len=insert_ops.TUBE_LEN)
    def to_insert_req(req):
        _r = _to_insert_req(req)
        if _r is not None:      # 2026-10-14: 이송 중 이웃과의 여유/집은 물체 외접구를 삽입 요청에도 전달한다(기본 서버값)
            if os.environ.get("NERO_PLACE_MARGIN"): _r["margin"] = float(os.environ["NERO_PLACE_MARGIN"])
            if os.environ.get("NERO_HELD_COVER", "0") == "1": _r["held_cover"] = "bound"
        return _r
    def _to_insert_req(req):
        """/place 요청 -> /insert 요청(같은 파지/막대 정보 + 홀 입구/축). 홀더 근처 장애물(홀더/기둥이 등록부에 박스로 잡힘)은 뺀다: 홀더 벽은 서버가 속이 빈 충돌체로 따로 만든다."""
        import insertion_geometry as _G
        E_, dd_ = _G.parse_insert_spec(a.insert_at); ex_, ey_ = float(E_[0]), float(E_[1])
        obs_ = [o for o in req.get("obstacles", []) if math.hypot(o["xyz"][0] - ex_, o["xyz"][1] - ey_) > 0.22]
        pc_ = req["obj"].get("pca")
        if not pc_ or not pc_.get("ok"):
            return None                                              # 삽입 불가: 길쭉하지 않음(PCA 신뢰 불가)
        return dict(start_q=req["start_q"], grasp=req["grasp"], peg=dict(center=pc_["center"], dims=req["obj"]["dims"], yaw=req["obj"]["yaw"], axis=pc_["axis"], length=pc_["length"], cross=pc_["cross"]), E=[float(v) for v in E_], d=[float(v) for v in dd_],
                    holder=_holder_dims(), obstacles=obs_, budget_s=float(os.environ.get("NERO_RT_PLACE_BUDGET", "10.0")),
                    rolls=[float(r_) for r_ in range(0, 360, int(os.environ.get("NERO_INSERT_ROLL_STEP", "45")))],
                    depth=float(os.environ.get("NERO_INSERT_PLAN_DEPTH", "0.02")), retreat_mode=os.environ.get("NERO_INSERT_RETREAT", "rev"), rim_screen=os.environ.get("NERO_RIM_SCREEN", "1"))         # 2026-10-16: 계획 깊이 기본 2cm(판정 깊이 --insert-depth 와 분리). 입구에서 얕게 넣고 놓아 중력으로 넣는 방식이 검증값(bar 실험 2~4cm). 판정+1cm(6cm)로 맞췄더니 E134 컵 후보 0/16 계획 실패, 2cm 로 되돌리니 15/16 계획·3/3 삽입

    def publish_insert(req, pr):
        """UI 에 삽입 정보를 낸다: 홀 입구/축/홀더 치수, 막대 PCA 축, 팁 경로, 선택된 롤/후퇴/궤적 안정 지표(또는 실패 사유)."""
        if not a.insert_at:
            return
        try:
            import insertion_geometry as _G
            E_, dd_ = _G.parse_insert_spec(a.insert_at); al_, _ = _G.angles_from_axis(dd_)
            ins = dict(((k_, v_) for k_, v_ in (STATE.get("insert") or {}).items() if k_ in ("hole_pts", "hole_pca")), E=[float(v) for v in E_], d=[float(v) for v in dd_], alpha=round(float(al_), 1), inner=insert_ops.HOLE_INNER, wall=insert_ops.WALL, len=insert_ops.TUBE_LEN, peg=(req.get("obj") or {}).get("pca") or {})
            if pr and pr.get("ok"):
                ins["plan"] = dict(roll=pr.get("yaw_deg"), tip_sign=pr.get("tip_sign"), retreat=pr.get("retreat_used"), depth=float(os.environ.get("NERO_INSERT_PLAN_DEPTH", "0.02")), metrics=pr.get("traj_metrics"))
                ins["tip_path"] = pr.get("tip_path")
            else:
                tr_ = (pr or {}).get("tried") or []
                ins["fail"] = (((pr or {}).get("why") or [None])[0]) or (tr_[0].get("fail") if tr_ else None) or "후보 없음(IK 후보 %s개)" % (pr or {}).get("n_ik_ok")
            STATE["insert"] = ins; flush_state()
        except Exception as e_:
            log(f"UI 삽입 정보 발행 실패 {e_!r}")

    def finish_insert(verdict, contained):
        try:
            if STATE.get("insert"):
                STATE["insert"]["verdict"] = verdict; STATE["insert"]["contained"] = contained; flush_state()
        except Exception:
            pass

    def call_plan(req):
        """place/insert 계획 호출. 삽입인데 PCA 가 막대를 길쭉하다고 보지 못하면 서버를 부르지 않고 "삽입 불가"를 돌려준다(Claude 가 텍스트로 대응)."""
        if req.get("_nofit"):      # 2026-10-08: decide() 가 안 들어간다고 판정 -> 서버를 부르지 않고 사유를 돌려준다
            return dict(ok=False, mode="none", n_cands=0, n_ik_ok=0, tried=[], why=["안 들어감: " + str(req["_nofit"])])
        if a.insert_at:
            rq = to_insert_req(req)
            if rq is None:
                pc_ = req["obj"].get("pca") or {}
                return dict(ok=False, mode="insert", n_cands=0, n_ik_ok=0, tried=[], why=["삽입 불가: 물체가 길쭉하지 않음 (PCA elong=%s, %s)" % (pc_.get("elong"), pc_.get("why", ""))])
            resp_ = post_json(a.curobo + "/insert", rq, timeout=60)
            for k_try in range(1, int(os.environ.get("NERO_INSERT_RETRY", "3"))):       # IK 후보는 있는데 계획이 실패한 경우만 재시도(trajopt 확률적 실패: 같은 요청 6회 중 18/21 이 한 번은 성공)
                if resp_.get("ok") or (resp_.get("n_ik_ok") or 0) == 0:
                    break
                resp_ = post_json(a.curobo + "/insert", rq, timeout=60); resp_["attempts"] = k_try + 1
            try:        # 분석용: 삽입 요청/응답을 저장(NO_PLAN 재현). 실험 판정에는 영향 없음
                os.makedirs("" + NERO_RUNS_DIR + "/insert_reqs", exist_ok=True)
                json.dump(dict(req=rq, resp={k: v for k, v in resp_.items() if k not in ("transit", "descend", "retreat")}), open("" + NERO_RUNS_DIR + "/insert_reqs/%d.json" % int(time.time() * 1000), "w"))
            except Exception:
                pass
            return resp_
        return post_json(a.curobo + "/place", req, timeout=60)

    def _nearest_prim(xy_):
        try:
            import place_rank as _PRK2
            return _PRK2.nearest_prim(xy_)
        except Exception as e_: return dict(name=None, err=repr(e_))
    _POOL = {}; _RANK_INFO = {}      # 2026-11-04(NERO_LOG_TRIAL=1): oid -> 최신 후보 풀 기록 / PLACE_RANK 정보
    def _theta_of(c_):
        """접근각(연직 대비, 도). approach 벡터에서 계산; 값이 없거나 잘못되면 None (0.0 과 구분)."""
        try:
            v_ = np.array(c_.get("approach"), float); n_ = float(np.linalg.norm(v_))
            return None if n_ <= 0 else float(math.degrees(math.acos(max(-1.0, min(1.0, -v_[2] / n_)))))
        except Exception: return None
    def _pool_record(oid, cl, max_c):
        """C 요청 직전의 후보 풀(= B 최종 keep + PLACE_RANK 정렬 결과) 전체를 기록하고, Thor 의 원 CGN 출력(물체별 덮어쓰기됨) 사본을 비동기로 저장한다."""
        ri = _RANK_INFO.pop(oid, None) or {}; ps = ri.get("p"); od = ri.get("order")
        pn = ([float(ps[i_]) for i_ in od] if ps and od and len(ps) == len(cl) else None)
        pool = []
        for i_, c_ in enumerate(cl):
            pool.append(dict(idx=i_, theta_deg=_theta_of(c_), theta_raw=c_.get("approach_theta"), approach=c_.get("approach"), flange=c_.get("flange"), quat=c_.get("quat"), contact=c_.get("contact"),
                             opening=c_.get("opening"), cgn_score=c_.get("score"), cost=c_.get("cost"), kind=c_.get("kind"), twin=c_.get("twin"), bin_fill=bool(c_.get("bin_fill")),
                             b_pass=True, sent_to_c=bool(i_ < max_c), place_rank_p=(pn[i_] if pn else None), place_rank_fallback=ri.get("fallback")))
        rec_ = dict(oid=oid, t=time.time(), max_cands=max_c, n=len(cl), pool=pool, rank_used=bool(pn))
        _td = os.environ.get("NERO_TRIAL_DIR")
        if _td:
            try:
                os.makedirs(_td + "/cgn", exist_ok=True); ms_ = int(time.time() * 1000)
                for suf_ in ("grasps", "pick"):      # B 가 쓰는 두 파일: 원 CGN 출력과 B 결과(keep 후보)
                    fp_ = "%s/cgn/%d_%s_%s.json" % (_td, ms_, oid, suf_)
                    subprocess.Popen(["ssh", "-o", "ConnectTimeout=5", "thor", "cat ~/grasp/%s_%s.json" % (oid, suf_)], stdout=open(fp_, "wb"), stderr=subprocess.DEVNULL)
                    rec_["cgn_copy_" + suf_] = fp_
            except Exception as e_: rec_["cgn_copy_err"] = repr(e_)
        _POOL[oid] = rec_
    def _trial_ext(oid, plan, req, dest_o, obj, H, top_z):      # 2026-11-04(NERO_LOG_TRIAL=1 일 때만 호출, 기본 꺼짐): 학습용 시행 기록 추가 항목 -> place_log 행의 trial 필드
        ext = {}
        try: ext.update(json.loads(os.environ.get("NERO_TRIAL_META", "{}")))      # 러너가 run/scene/rep/seed 를 넘긴다
        except Exception: pass
        ext["registry_id"] = oid; ext["label"] = world.objs.get(oid, {}).get("label", "")      # 계획 id 와는 다른 번호 공간: gt.prim(시뮬 prim 이름)이 공통 키
        cg = dict(idx=plan.get("idx"), score=plan.get("score"), rank=plan.get("rank"), opening=plan.get("opening"), n_cands=len(STATE.get("cands") or []))
        try:
            if cg["score"] is None and plan.get("idx") is not None: cg["score"] = next((c_.get("score") for c_ in (STATE.get("cands") or []) if c_.get("idx") == plan.get("idx")), None)
        except Exception: pass
        ext["cgn"] = cg
        try:
            from scipy.spatial.transform import Rotation as _R
            ext["wrist_rpy_deg"] = [round(float(v), 1) for v in _R.from_quat(plan["quat"]).as_euler("xyz", degrees=True)]
        except Exception: pass
        ext["obj_wh_ratio"] = round(float(min(obj["dims"][0], obj["dims"][1])) / max(float(H), 1e-6), 3)
        if dest_o is not None:
            cc = dest_o.get("concave") or {}; raw = dest_o.get("raw") or [0, 0, 0]; wall = float(dest_o.get("wall_thick", 0.02) or 0.02)
            ext["dest"] = dict(kind=dest_o.get("kind") or (a.dest_kind or None), concave=bool(cc.get("concave")), depth=cc.get("depth"), rim_z=cc.get("rim_z"), floor_z=cc.get("floor_z"),
                               size=[round(float(v), 4) for v in raw], inner_open=[round(max(0.0, float(raw[0]) - 2 * wall), 4), round(max(0.0, float(raw[1]) - 2 * wall), 4)],
                               xyz=[round(float(v), 4) for v in dest_o["xyz"][:3]], yaw=dest_o.get("yaw"), meas=dest_o.get("dest_meas"), given_id=dest_o.get("given_id"), top_z=top_z,
                               dist_xy=round(float(math.hypot(obj["centre"][0] - dest_o["xyz"][0], obj["centre"][1] - dest_o["xyz"][1])), 4))
        _td = os.environ.get("NERO_TRIAL_DIR")      # place 요청 덤프(NO_PLAN 재질의용) -- HDD 디렉터리 지정
        if _td:
            try:
                os.makedirs(_td + "/reqs", exist_ok=True); _fn = "%s/reqs/%d_%s.json" % (_td, int(time.time() * 1000), oid)
                json.dump(req, open(_fn, "w"), default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_)); ext["req_dump"] = _fn
            except Exception as _e: ext["req_dump_err"] = repr(_e)
            pr_ = _POOL.get(oid)      # 후보 풀: place 요청 덤프와 같은 키(<ms>_<oid>)로 .pool.json 저장 + 선택된 후보 인덱스
            if pr_ is not None and ext.get("req_dump"):
                try:
                    fl_ = plan.get("flange"); sel_ = None
                    for c_ in pr_["pool"]:
                        if fl_ is not None and c_.get("flange") is not None and max(abs(float(x) - float(y)) for x, y in zip(c_["flange"], fl_)) < 1e-4: sel_ = c_["idx"]; break
                    pr_ = dict(pr_, selected_idx=sel_, episode_id="%s/%s/%s" % (ext.get("run"), ext.get("rep"), os.path.basename(ext["req_dump"])[:-5]), prim_scene_ref=_nearest_prim(obj["centre"][:2]))
                    _pf = ext["req_dump"][:-5] + ".pool.json"; json.dump(pr_, open(_pf, "w"), default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_)); ext["pool_dump"] = _pf; ext["pool_selected_idx"] = sel_
                except Exception as _e: ext["pool_dump_err"] = repr(_e)
        return ext

    def place_request(oid, d, plan, q_start):
        """place 계획 요청 구성(팔 자세 q_start 에서 출발). pick 실행 중에 미리 호출할 수 있도록 팔/ROS 상태를 읽지 않는다. -> (rec, req, dest_o, top_z, obj, H)"""
        rec = dict(ts=time.time(), oid=oid, label=world.objs.get(oid, {}).get("label", ""), target=a.target, place_on=a.place_on, place_free=a.place_free, place_xy=a.place_xy)
        if AUTO: a.insert_at = ""      # 물체마다 새로 결정한다(이전 물체의 삽입 설정을 끌고 오지 않는다)
        try:
            rec["trial"] = json.loads(os.environ.get("NERO_TRIAL_INFO", "{}"))      # insert_queue/place_queue 가 시행 태그, 번호, 홀 위치/방향, 파라미터를 넘긴다
        except Exception:
            pass
        P_ = d["P"]; box = d["box"]
        base_z = float(max(0.0, min(0.03, P_[:, 2].min()))); top_o = float(np.percentile(P_[:, 2], 98)); H = max(0.02, top_o - base_z)
        obj = dict(centre=[box["xyz"][0], box["xyz"][1], base_z + H / 2], base_z=base_z, height=H, dims=[box["raw"][0], box["raw"][1], H], yaw=box["yaw"])
        if a.insert_at or AUTO:        # 삽입: 막대 장축을 점군 PCA 로 측정(OBB 눕/섬 가정 대체). 길쭉하지 않으면 ok=False -> to_insert_req 가 None 을 돌려 "삽입 불가"
            import axis_pca as _AP
            _pc = _AP.pca_axis(P_)
            if _pc.get("ok"):      # 중심: 축 방향 위치는 PCA(점군 양끝의 중간), 단면 중심은 박스 기반 추정을 유지한다(점군은 보이는 윗면뿐이라 PCA 평균은 막대 반 굵기만큼 높게 잡힌다)
                _c0 = np.array(obj["centre"], float); _ax = np.array(_pc["axis"], float); _pc["mid"] = _c0 + _ax * float((np.array(_pc["mid"], float) - _c0) @ _ax)
            obj["pca"] = dict(ok=True, axis=[float(v) for v in _pc["axis"]], length=float(_pc["length"]), center=[float(v) for v in _pc["mid"]], cross=[float(v) for v in _pc["cross"]],
                              elong=round(_pc["elong"], 1), tilt_deg=round(_pc["tilt_deg"], 1)) if _pc.get("ok") else dict(ok=False, why=_pc.get("why", "PCA 실패"), elong=_pc.get("elong"))
        obs = world.obstacles(oid)
        dest_o = None; dest_ids = []
        _dcs = [(oid2, o2) for oid2, o2 in world.objs.items() if o2.get("role") == "dest" and "raw" in o2]
        if _dcs:      # 2026-10-07: 도착지 후보가 여럿이면 접지 점수(s_p)가 가장 높은 것을 쓴다(이전: 등록부 순서상 마지막 후보가 덮어써서 점수와 무관하게 골라졌다). 나머지 후보는 장애물로 남긴다
            _sc = lambda kv: kv[1].get("s_p_adj", kv[1].get("s_p") if kv[1].get("s_p") is not None else -1e9)
            if os.environ.get("NERO_DEST_FIT", "1") == "1":      # 2026-10-08: 들고 있는 물체가 개구부에 들어가는지(place_decide)를 보고, 들어가는 후보를 점수 순으로 먼저 쓴다. 전부 안 들어가면 점수 최고를 쓰되 사유를 남긴다(중단은 아직 안 연결)
                import place_decide as _PD
                _od = _PD.describe_obj(pca=obj.get("pca"), dims=obj["dims"]); _fit = {}
                for _k3, _o3 in _dcs:
                    _dd = place_ops.describe_dest(_o3); _dec = _PD.decide(_od, _dd, "in" if _dd["kind"] == "receptacle" else "on"); _fit[_k3] = _dec
                    log(f"  도착지 {_k3} 기술자: {_dd['kind']} 개구부 {None if _dd['open'] is None else tuple(round(x, 3) for x in _dd['open'])} 깊이 {_dd['depth']} ({_dd['confidence']}) -> 방식 {_dec['mode']}/{_dec['how']}: {_dec['reason']}")
                _ok = [kv for kv in _dcs if _fit[kv[0]]["mode"] is not None]
                _bi, dest_o = max(_ok or _dcs, key=_sc)
                try:
                    _ddh = place_ops.describe_dest(dest_o); _fh = _fit[_bi]; _cl = lambda a_, b_: _PD._fits(_ddh["open"][0], _ddh["open"][1], a_, b_, _PD.MIN_CLEAR)[1] if _ddh.get("open") else None
                    set_state(decision=dict(mode=_fh["mode"], how=_fh["how"], obj=dict(kind=_od["kind"], L=_od["L"], c1=_od["c1"], c2=_od["c2"]), dest=dict(kind=_ddh["kind"], open=_ddh["open"], tilt=_ddh.get("tilt_deg", 0.0)),
                                             clear_lie=_cl(_od["L"], _od["c2"]), clear_up=_cl(_od["c1"], _od["c2"]), min_clear=_PD.MIN_CLEAR, hole_note=None, hole_check=dest_o.get("hole_note"), applied=False,
                                             note=(_fh["reason"] + " -- 홀이 아니라서 모드 전환 없이 기존 놓기 경로")))
                except Exception as _e: log(f"HUD 판단 기록 실패 {_e!r}")
                if not _ok: log(f"  모든 도착지 후보에 안 들어감 -- 점수 최고 {_bi} 로 진행(사유 위)")
            else:
                _bi, dest_o = max(_dcs, key=_sc)
            dest_ids.append("other_" + _bi)
            if os.environ.get("NERO_RVIZ_ENT") == "1": STATE["dest_pick"] = _bi; flush_state()
            log(f"도착지 선택 {_bi} xy=({dest_o['xyz'][0]:+.3f},{dest_o['xyz'][1]:+.3f}) 박스 {[round(100 * v, 1) for v in dest_o['raw']]}cm 라벨={dest_o.get('label')} s_p={dest_o.get('s_p')} 후보 {[k for k, _ in _dcs]}")      # 2026-10-13: 어느 도착지를 골랐는지 기록(로그만)
            if len(_dcs) > 1: log(f"도착지 후보 {[k for k, _ in _dcs]} 중 점수 최고 {_bi}(s_p={dest_o.get('s_p')}) 선택")
        _clr = (world.objs.get(oid) or {}).get("clearing")
        if _clr: dest_o = None; dest_ids = []; log(f"{oid}: 막은 물체 치우기 -- 도착지 대신 빈 자리에 놓는다")
        if _clr and float(os.environ.get("NERO_CLEAR_DEST_PAD", "0") or 0) > 0:      # 2026-10-14(기본 꺼짐): 치우는 동작이 목적지 위/옆을 스치지 않게 목적지(입력 명세)를 사방으로 부풀려 충돌체로 넣는다
            _p = float(os.environ["NERO_CLEAR_DEST_PAD"]); _n = 0
            for _ob in obs:
                _src = world.objs.get(str(_ob["id"])[6:]) or {}
                if _src.get("given") or _src.get("role") == "dest":
                    _ob["raw"] = [_ob["raw"][0] + 2 * _p, _ob["raw"][1] + 2 * _p, _ob["raw"][2] + _p]; _ob["xyz"] = [_ob["xyz"][0], _ob["xyz"][1], _ob["xyz"][2] + _p / 2]; _n += 1
            log(f"치우기: 목적지 {_n}개 충돌체를 {100 * _p:.0f}cm 부풀림(목적지를 스치지 않는 경로)")
        if dest_o is None and a.place_xy:      # 2026-10-08: 좌표로 지정한 놓을 곳이 등록 물체(바구니 등) 안이면 그 물체를 도착지로 본다 -- 장애물에서 빼고, 오목하면 아래에서 벽 4면을 넣는다(접지 없이도 용기 안에 놓을 수 있게)
            _xy0 = np.array([float(v) for v in a.place_xy.split(",")]); _cc = []
            for _k2, _o2 in world.objs.items():
                if _k2 == oid or "raw" not in _o2 or _o2.get("role") == "placed": continue
                if World._in_box(_xy0, dict(xyz=_o2["xyz"], raw=_o2["raw"], yaw=float(_o2.get("yaw", 0.0))), 0.02)[0]: _cc.append((_o2["raw"][0] * _o2["raw"][1], _k2, _o2))
            if _cc:
                _, _bi, dest_o = max(_cc, key=lambda t: t[0]); dest_ids.append("other_" + _bi); dest_o["concave"] = place_ops.concavity(dest_o)
                log(f"놓을 곳 ({_xy0[0]:.2f},{_xy0[1]:.2f}) 는 등록 물체 {_bi} 안 -> 도착지로 취급(장애물에서 제외)")
        if dest_o is not None and (dest_o.get("concave") or {}).get("concave"):      # 위가 열린 용기: 벽 4면을 충돌체로 넣는다(바닥 높이는 슬롯별 support_z 가 점군에서 잰다)
            if os.environ.get("NERO_PLACED_SOFT", "0") == "1":      # 2026-10-13(기본 꺼짐): 용기 풋프린트 안에 이미 놓은 물체는 놓기 계획의 장애물에서 뺀다(겹쳐 놓여도 괜찮다 -- 담는 것이 목적)
                _boxd = dict(xyz=dest_o["xyz"], raw=dest_o["raw"], yaw=float(dest_o.get("yaw", 0.0)))
                _n0 = len(obs); obs = [o3 for o3 in obs if not (str(o3.get("id", "")).replace("other_", "") in world.objs and world.objs[str(o3["id"]).replace("other_", "")].get("role") == "placed" and World._in_box(np.array(o3["xyz"][:2]), _boxd, 0.02)[0])]
                if len(obs) < _n0: log(f"{oid}: 용기 안에 이미 놓은 물체 {_n0 - len(obs)}개는 놓기 계획의 장애물에서 제외")
            if os.environ.get("NERO_CONTAINER_NOSOLID", "0") == "1":      # 2026-10-14(기본 꺼짐): 용기는 벽 4면이 이미 충돌체다. 용기 전체를 속이 찬 직육면체로도 넣으면 이송 목표(손/든 물체)가 그 상자 안에 들어가 이송 계획이 막힌다(실패 덤프 4/4: 이 상자를 빼면 1초 만에 성공)
                _n1 = len(obs); obs = [o3 for o3 in obs if str(o3.get("id", "")) != "other_" + str(_bi)]
                if len(obs) < _n1: log(f"{oid}: 용기 {_bi} 의 속 찬 상자 충돌체를 뺌(벽 4면만 사용)")
            obs = list(obs) + place_ops.container_walls(dest_o); log(f"도착지 {_bi} 는 위가 열린 용기(깊이 {100 * dest_o['concave']['depth']:.1f}cm, 림 {100 * dest_o['concave']['rim_z']:.1f}cm) -> 벽 4면 충돌체 추가" + (lambda io_: f" | 안쪽 실측 폭 {100 * io_[0]:.1f}x{100 * io_[1]:.1f}cm (바깥 {100 * dest_o['raw'][0]:.0f}x{100 * dest_o['raw'][1]:.0f}cm)" if io_ else " | 안쪽 폭 미측정")(place_ops.inner_open(dest_o)))
        if a.place_xy:
            xy = [float(v) for v in a.place_xy.split(",")]; slots = [xy]; top_z = float(os.environ.get("NERO_PLACE_TOPZ", "0.0") or 0.0)      # 2026-11-06(기본 0 = 이전과 동일): 놓을 받침면 높이(m). 아래 막대 위에 올리기 등 2층 이상 쌓기용
            if os.environ.get("NERO_PLACE_DEST_BELOW", "0") == "1" and top_z > 0:      # 2026-11-12(기본 꺼짐, 바닥 기둥4+뚜껑 실행용): 좌표로 준 놓을 곳 아래에서 물체를 받치는 등록 물체(윗면이 받침면 높이 ±3cm, 중심이 들고 있는 물체 놓을 자리 안)를 도착지로 본다 -- 장애물에서 빠지지 않고 dest_ids 에만 들어감(하강·안착 접촉 허용, 서버 계획과 같은 처리)
                _rr = 0.5 * math.hypot(*sorted([float(v) for v in obj["dims"]])[-2:]); _sup = []
                for _k4, _o4 in world.objs.items():
                    if _k4 == oid or "raw" not in _o4: continue
                    _tp = float(_o4["xyz"][2]) + 0.5 * float(_o4["raw"][2])
                    if abs(_tp - top_z) <= 0.03 and math.hypot(float(_o4["xyz"][0]) - xy[0], float(_o4["xyz"][1]) - xy[1]) <= _rr: _sup.append(_k4); dest_ids.append("other_" + _k4)
                log(f"{oid}: 놓을 곳 아래 받침 물체 {len(_sup)}개를 도착지로 {[(k_, [round(v, 3) for v in world.objs[k_]['xyz']], [round(v, 3) for v in world.objs[k_]['raw']]) for k_ in _sup]} (반경 {_rr:.3f}, 받침면 {top_z:.3f})")
        elif dest_o is not None:
            _occ = None
            if os.environ.get("NERO_CONTAINER_SPREAD", "0") == "1":      # 2026-10-13(기본 꺼짐): 도착지 풋프린트 안에 이미 있는 다른 물체/놓은 물체의 자리를 피해서 칸을 고른다
                _occ = [(float(o3["xyz"][0]), float(o3["xyz"][1]), 0.5 * max(float(o3["raw"][0]), float(o3["raw"][1]))) for k3, o3 in world.objs.items()
                        if "raw" in o3 and o3 is not dest_o and k3 != oid and o3.get("role") != "dest" and float(o3["raw"][2]) < float(dest_o["raw"][2]) + 0.12 and World._in_box(np.array(o3["xyz"][:2]), dict(xyz=dest_o["xyz"], raw=dest_o["raw"], yaw=float(dest_o.get("yaw", 0.0))), 0.0)[0]]
                if _occ: log(f"{oid}: 도착지 안에 이미 있는 물체 {len(_occ)}개의 자리를 피해서 놓을 칸을 고른다: {[(round(x, 2), round(y, 2)) for x, y, _ in _occ]}")
            slots, top_z = place_ops.dest_slots(dest_o, obj["dims"][:2], _occ)
            if os.environ.get("NERO_DROP_IN", "0") == "1" and (dest_o.get("concave") or {}).get("concave"):      # 2026-10-14(기본 꺼짐): 용기에는 안으로 내려가지 않고 림 위 NERO_DROP_H(기본 3cm)에서 열어 떨어뜨린다(벽에 걸림/그리퍼 안 열림 방지)
                _rz = float(dest_o["concave"]["rim_z"]); _dh = float(os.environ.get("NERO_DROP_H", "0.03"))
                slots = [[float(sl[0]), float(sl[1]), _rz + _dh] for sl in slots]; log(f"{oid}: 용기에는 드롭 -- 림({100 * _rz:.1f}cm) 위 {100 * _dh:.0f}cm 에서 놓는다")
        else:
            others = [o3 for o3 in world.objs.values() if "raw" in o3 and o3 is not world.objs.get(oid)]
            if _clr and (world.objs.get(_clr_for := (world.objs.get(oid) or {}).get("clear_for")) or None): others = others + [dict(raw=[0.22, 0.22, 0.10], xyz=world.objs[_clr_for]["xyz"])]      # 치운 물체가 원래 대상 바로 옆에 다시 놓이지 않게 대상 주변을 비운다
            slots = place_ops.free_slots(others, 0.5 * math.hypot(*obj["dims"][:2]), box["xyz"][:2]); top_z = 0.0
        _nofit = None
        if AUTO and dest_o is not None and (dest_o.get("hole") or {}).get("ok"):      # 2026-10-08: 목적지 기술자 + 물체 기술자 -> decide() -> place / insert / 안 들어감
            import place_decide as _PD
            _dd = place_ops.describe_dest(dest_o); _od = _PD.describe_obj(pca=obj.get("pca"), dims=obj["dims"]); _dec = _PD.decide(_od, _dd, "in")
            log(f"{oid}: 모드 결정 -- 물체 {_od['kind']} L={_od['L']:.3f} 단면 {_od['c1']:.3f}x{_od['c2']:.3f} / 목적지 개구부 {tuple(round(x, 3) for x in _dd['open'])} 기울기 {_dd['tilt_deg']}도 -> {_dec['mode']}/{_dec['how']}: {_dec['reason']}")
            if _dec["mode"] == "insert":
                a.insert_at = "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % tuple(list(_dd["E"]) + list(_dd["d"]))
            elif _dec["mode"] is None:
                _nofit = _dec["reason"]
            rec["auto_mode"] = dict(mode=_dec["mode"], how=_dec["how"], reason=_dec["reason"])
            try:
                _cl2 = lambda a_, b_: _PD._fits(_dd["open"][0], _dd["open"][1], a_, b_, _PD.MIN_CLEAR)[1]
                set_state(decision=dict(mode=_dec["mode"], how=_dec["how"], obj=dict(kind=_od["kind"], L=_od["L"], c1=_od["c1"], c2=_od["c2"]), dest=dict(kind=_dd["kind"], open=_dd["open"], tilt=_dd.get("tilt_deg", 0.0)),
                                         clear_lie=_cl2(_od["L"], _od["c2"]), clear_up=_cl2(_od["c1"], _od["c2"]), min_clear=_PD.MIN_CLEAR, hole_check=dest_o.get("hole_note"), applied=True, note=_dec["reason"]))
            except Exception as _e: log(f"HUD 판단 기록 실패 {_e!r}")
        rec.update(obj=obj, n_slots=len(slots), top_z=top_z, dest=(dict(xyz=dest_o["xyz"], raw=dest_o["raw"], yaw=dest_o.get("yaw", 0.0)) if dest_o else None),
                   grasp=dict(quat=plan.get("quat"), approach=plan.get("approach"), flange=plan.get("flange"), contact=plan.get("contact"), opening=plan.get("opening"), theta=plan.get("approach_theta"),
                              held_ratio=(plan["contact"][2] - base_z) / H if plan.get("contact") else None), q_start=q_start)
        req = dict(start_q=q_start, grasp=dict(quat=plan["quat"], approach=plan["approach"], flange=plan["flange"]), obj=obj, slots=slots, top_z=top_z, clear=a.place_clear,
                   obstacles=obs, dest_ids=dest_ids, budget_s=float(os.environ.get("NERO_RT_PLACE_BUDGET", "8.0")))
        if os.environ.get("NERO_SHELF_CELL", "0") == "1":      # 2026-11-04(기본 꺼짐): 선반 판이 장애물에 있으면 서버의 슬롯 겹침 재배치(slot_filter)를 끈다(판을 슬롯 막힘으로 보지 않게)
            import shelf_cell as _SC
            if _SC.has_plates(obs): req["slot_filter"] = False
        if os.environ.get("NERO_PLACE_MARGIN"): req["margin"] = float(os.environ["NERO_PLACE_MARGIN"])      # 2026-10-14(기본: 서버 5mm): 이송 중 집은 물체와 이웃 사이 최소 여유(m). 5mm 로는 림을 스치고 지나가 통을 밀었다(S5 손전등)
        if os.environ.get("NERO_HELD_COVER", "0") == "1": req["held_cover"] = "bound"      # 2026-10-14(기본 꺼짐): 집은 물체 구를 상자 외접으로(모서리 삐져나옴 방지)
        if os.environ.get("NERO_PLACE_REQ_DUMP"):      # 분석용: place 요청 저장(실패 재현). 기본 꺼짐
            try:
                _rd = os.environ["NERO_PLACE_REQ_DUMP"]; _rd = (_rd.rstrip("/") + "/%d_%s.json" % (int(time.time() * 1000), oid)) if os.path.isdir(_rd) else _rd      # 2026-11-04: 값이 디렉터리면 요청별 파일 <ms>_<oid>.json(덮어쓰기 방지), 파일 경로면 기존 동작
                json.dump(req, open(_rd, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            except Exception as _e: log(f"place 요청 덤프 실패 {_e!r}")
        if os.environ.get("NERO_PLACE_RETREAT"):
            req["retreat_mode"] = os.environ["NERO_PLACE_RETREAT"]        # up: 놓은 뒤 월드 수직으로 먼저 후퇴 / (기본) 접근축 후퇴
        if os.environ.get("NERO_PLACE_MODE"):
            req["mode"] = os.environ["NERO_PLACE_MODE"]        # same: 집을 때 손 자세 유지(쥔 물체는 처음 서 있던 자세 그대로) / plumb: 손을 수직으로 세움(물체가 같이 기운다)
        if os.environ.get("NERO_PLACE_YAWS"):      # 2026-11-06(기본 꺼짐, 설정 안 하면 이 줄은 실행되지 않아 요청 동일): 놓을 yaw 후보(도, 쉼표 구분, 서버 요청의 yaws). 서버는 집은 자세 대비 손을 z 로 yw 만큼 돌려 놓으므로 물체 최종 yaw = 집기 직전 물체 yaw + yw
            req["yaws"] = [float(v) for v in os.environ["NERO_PLACE_YAWS"].split(",") if v.strip() != ""]
        if os.environ.get("NERO_PLACE_WORLD_YAW"):      # 2026-11-06(기본 꺼짐): 물체의 긴 축을 world 좌표 yaw(도, +x 기준 반시계)로 맞춘다. 서버 yaws 는 집기 직전 대비 상대각이라 (목표 - 현재 긴 축 각)을 대칭 주기(NERO_PLACE_YAW_SYM, 기본 180; 정사각에 가까우면 90)로 접어 후보로 보낸다.
            _tw = float(os.environ["NERO_PLACE_WORLD_YAW"]); _d0, _d1 = float(obj["dims"][0]), float(obj["dims"][1])
            _sym = float(os.environ.get("NERO_PLACE_YAW_SYM", "0") or 0) or (90.0 if abs(_d0 - _d1) < 0.12 * max(_d0, _d1) else 180.0)
            _l0 = float(os.environ["NERO_PLACE_YAW0"]) if os.environ.get("NERO_PLACE_YAW0") else (math.degrees(float(obj.get("yaw", 0.0))) + (90.0 if _d1 > _d0 else 0.0))      # 현재 긴 축 각: NERO_PLACE_YAW0(시험용 주입) 또는 인식 OBB(obj.yaw, dims)
            _rel = ((_tw - _l0 + _sym / 2.0) % _sym) - _sym / 2.0; _tol = float(os.environ.get("NERO_PLACE_YAW_TOL", "0") or 0)
            _c = [_rel + k * _sym for k in range(-int(360 // _sym), int(360 // _sym) + 1)]; _c = [((v + 180.0) % 360.0) - 180.0 for v in _c]
            if _tol > 0: _c = _c + [v + o for o in (_tol / 2.0, -_tol / 2.0, _tol, -_tol) for v in _c]
            _seen = []; [_seen.append(round(v, 3)) for v in _c if round(v, 3) not in _seen]; req["yaws"] = sorted(_seen, key=lambda v: abs(((v - _rel + 180.0) % 360.0) - 180.0))
            rec["yaw_plan"] = dict(target_world=_tw, long_axis_now=round(_l0, 2), sym=_sym, rel=round(_rel, 2), yaws=req["yaws"][:12]); log(f"{oid}: 월드 yaw 정렬 목표 {_tw:.1f}° | 현재 긴 축 {_l0:.1f}° (대칭 {_sym:.0f}°) -> 상대 yaw 후보 {req['yaws'][:6]}")
        if os.environ.get("NERO_PLACE_OBJ_ROT"):      # 2026-11-06(기본 꺼짐, 설정 안 하면 이 블록은 실행되지 않아 요청 동일): 놓을 때 물체 방향 변경(goalpose 서버 8021 전용 req["obj_rot"], xyzw). 값: "x90","x-90","y90","x180"(축+도, '_' 로 이으면 왼쪽부터 적용) 또는 쿼터니언 JSON [x,y,z,w]. 월드 기준, 물체 중심 둘레, yaw 보다 먼저 적용
            _ov = os.environ["NERO_PLACE_OBJ_ROT"].strip()
            if _ov.startswith("["):
                _oq = [float(v) for v in json.loads(_ov)]
            else:
                _oq = [0.0, 0.0, 0.0, 1.0]
                for _tk in [t for t in _ov.lower().replace(",", "_").split("_") if t.strip()]:
                    _ax = "xyz".index(_tk[0]); _h = math.radians(float(_tk[1:])) / 2.0; _b = [0.0, 0.0, 0.0, math.cos(_h)]; _b[_ax] = math.sin(_h)
                    _x1, _y1, _z1, _w1 = _b; _x2, _y2, _z2, _w2 = _oq      # _oq <- _b * _oq (새 회전을 왼쪽에 곱함 = 나중에 적용)
                    _oq = [_w1 * _x2 + _x1 * _w2 + _y1 * _z2 - _z1 * _y2, _w1 * _y2 - _x1 * _z2 + _y1 * _w2 + _z1 * _x2, _w1 * _z2 + _x1 * _y2 - _y1 * _x2 + _z1 * _w2, _w1 * _w2 - _x1 * _x2 - _y1 * _y2 - _z1 * _z2]
            _n = math.sqrt(sum(v * v for v in _oq)); req["obj_rot"] = [v / _n for v in _oq]; rec["obj_rot"] = dict(spec=_ov, quat=req["obj_rot"]); log(f"{oid}: 놓기 물체 회전 {_ov} -> obj_rot {[round(v, 4) for v in req['obj_rot']]}")
        if os.environ.get("NERO_PLACE_POSE_LIST_FILE") and not os.environ.get("NERO_PLACE_POSE_LIST"):      # 2026-11-09: LLM 판단과 인식을 겹치기 -- 목록 파일이 생길 때까지(최대 180초) 기다렸다가 읽는다(인식·CGN 은 그동안 이미 끝남)
            _pf = os.environ["NERO_PLACE_POSE_LIST_FILE"]; _tw = time.time()
            try: (os.environ.get("NERO_PERC_EARLY", "0") == "1" and os.path.exists(_pf + ".perc.json")) or json.dump(dict(obj=req.get("obj"), oid=oid, t=time.time()), open(_pf + ".perc.json", "w"), default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_))      # 2026-11-10: 인식한 물체 상자(중심·치수·yaw)를 판단 쪽에 알림 -> 면 이름->회전을 정답 yaw 대신 인식 yaw 로
            except Exception as _e: log(f"인식 상자 파일 실패 {_e!r}")
            while not os.path.exists(_pf) and time.time() - _tw < 180: time.sleep(0.2)
            if os.path.exists(_pf):
                time.sleep(0.2); os.environ["NERO_PLACE_POSE_LIST"] = open(_pf).read(); log(f"{oid}: 자세 목록 파일 받음(대기 {time.time() - _tw:.1f}s)")
        if os.environ.get("NERO_PLACE_POSE_LIST"):      # 2026-11-09(기본 꺼짐, 8021 사본 전용): 순위 있는 놓기 자세 목록 [{name, rot(xyzw 또는 null=그대로), yaws(또는 null)}]. 서버가 한 번의 계획에서 순위대로 모두 확인해 되는 것 중 가장 앞 순위를 고른다
            _plj = json.loads(os.environ["NERO_PLACE_POSE_LIST"])
            if isinstance(_plj, dict) and _plj.get("place_spec"):      # 2026-11-10: 조건+순서만 보내고 후보 목록은 플래너 서버(Thor)가 만든다
                req["place_spec"] = _plj["place_spec"]; req.pop("obj_rot_list", None); req.pop("obj_rot", None); rec["pose_list"] = list(_plj["place_spec"].get("order") or []); rec["place_spec"] = _plj["place_spec"]
            else:
                req["obj_rot_list"] = _plj; req.pop("obj_rot", None); rec["pose_list"] = [e_.get("name") for e_ in req["obj_rot_list"]]
        if os.environ.get("NERO_PLACE_AUTO_CLEAR", "0") == "1":      # 2026-11-06(기본 꺼짐, 복사본 서버 8021 전용): 놓는 높이 자동 -- 서버가 후보마다 손가락 충돌 구 최저점이 받침면보다 margin 위가 되도록 clear 를 올린다(요청 clear 가 하한)
            req["auto_clear"] = True
            if os.environ.get("NERO_PLACE_AUTO_CLEAR_MARGIN"): req["auto_clear_margin"] = float(os.environ["NERO_PLACE_AUTO_CLEAR_MARGIN"])      # 기본 0.005 m
            if os.environ.get("NERO_PLACE_AUTO_CLEAR_MAX"): req["auto_clear_max"] = float(os.environ["NERO_PLACE_AUTO_CLEAR_MAX"])      # 기본 0.03 m (clear 상한)
        if os.environ.get("NERO_PLACE_FINGER_REJECT", "0") in ("1", "report"):      # 2026-11-08(기본 꺼짐, 복사본 서버 8021 전용): 놓는 자세에서 손가락이 받침면 아래로 파고드는 후보를 서버가 버린다(하강은 도착지를 충돌에서 빼고 계획하므로 따로 검사)
            req["finger_reject"] = (True if os.environ["NERO_PLACE_FINGER_REJECT"] == "1" else "report")
            if os.environ.get("NERO_PLACE_FINGER_REJECT_MARGIN"): req["finger_reject_margin"] = float(os.environ["NERO_PLACE_FINGER_REJECT_MARGIN"])
        if os.environ.get("NERO_PLACE_DESC_DEST"):      # 2026-11-09 v4(기본 꺼짐): 하강을 도착지 포함 월드로 계획(옮기다 물체가 받침대에 걸리는 문제). 값 strict = 안 되면 후보 버림, 1 = 안 되면 예전 방식
            req["descend_with_dest"] = ("strict" if os.environ["NERO_PLACE_DESC_DEST"] == "strict" else True)
        if os.environ.get("NERO_PLACE_NEIGHBOR_REJECT", "0") == "1": req["neighbor_reject"] = True      # 2026-11-10(기본 꺼짐, 8021 사본): 놓은 물체 상자가 이웃 장애물과 겹치는 후보 버림
        if os.environ.get("NERO_PLACE_HAND_REJECT", "0") == "1":      # 2026-11-09(기본 꺼짐): 놓는 자세에서 카메라/손바닥/link7 구가 도착지 속으로 들어가는 후보를 버림(서버 8021 사본, finger_reject 블록 안에서 동작하므로 진단도 켬)
            req["hand_dest_reject"] = True; req.setdefault("finger_reject", "report")
        if _nofit: req["_nofit"] = _nofit
        if os.environ.get("NERO_LOG_TRIAL", "0") == "1":      # 2026-11-04(기본 꺼짐): 학습용 시행 기록 -- 꺼짐이면 아래는 실행되지 않아 출력 동일
            try: rec["trial"] = dict(rec.get("trial") or {}, **_trial_ext(oid, plan, req, dest_o, obj, H, top_z))
            except Exception as _e: log(f"{oid}: [trial 기록 실패] {_e!r}")
        return rec, req, dest_o, top_z, obj, H

    def plan_place_bg(oid, d, plan, pre):
        """pick 이 실행되는 동안 백그라운드로 place 계획을 미리 세운다. 시작 자세 = pick 계획의 lift 끝(HELD 로 끝나면 팔이 있게 될 자세)."""
        t0 = time.time()
        try:
            q0 = [float(x) for x in plan["lift"][-1]]
            pre["built"] = place_request(oid, d, plan, q0)
            req = pre["built"][1]
            if not req["slots"]:
                pre["pr"] = None; return
            pre["pr"] = (plan.get("place_plan") or call_plan(req)); publish_insert(req, pre["pr"])      # 동시 선택에서 이미 계획된 place 가 있으면 그대로 쓴다
            _prx = pre["pr"]
            if _prx and _prx.get("ok"):
                try:       # UI 에 place 궤적/놓을 곳 표시(pick 이 실행되는 동안 이미 보인다)
                    qs_ = []
                    for k_ in ("transit", "descend", "retreat"):
                        qs_ += _prx.get(k_, [])
                    path_ = post_json(a.curobo + "/fk", dict(q=qs_), timeout=10)["pos"]
                    STATE["place_plan"] = dict(oid=oid, ee_path=path_, slot=_prx.get("slot"), tcp=_prx.get("tcp"), slot_z=_prx.get("slot_z"), mode=_prx.get("mode"), relax=_prx.get("relax_level")); flush_state()
                except Exception:
                    pass
        except Exception as e:
            pre["err"] = repr(e)
        pre["t_plan"] = time.time() - t0

    def _tilt_free_retry(oid, req, dest_o, pr, rec):
        """2026-11-04 NERO_PLACE_TILT_FREE=1(기본 꺼짐): 기본 /place 가 NO_PLAN 일 때만, 쥔 물체를 접촉점 둘레로 pitch/roll 기울여(요청의 grasp.quat/approach/flange, obj.centre 회전; mode=same)
        다시 /place 에 보낸다. 작은 기울기부터(NERO_TILT_STEP 간격, NERO_TILT_MAX 까지), 경로 계획까지 ok(plan ok) 인 첫 해만 채택(IK 만 통과는 불채택).
        기울여 놓아도 되는 목적지만: 오목 용기(dest.concave) — 평면 목적지는 NERO_TILT_FLAT=1 일 때만. 시간 NERO_TILT_TIME_CAP(30 s)·횟수 NERO_TILT_MAX_TRIES(16) 상한. 서버 수정 없음.
        -> (새 pr 또는 None, 기록 dict)"""
        info = dict(tried=0, adopted=False)
        if "grasp" not in req or pr.get("mode") == "insert" or rec.get("auto_mode", {}).get("mode") == "insert" or a.insert_at:
            info["skip"] = "삽입/비일반 놓기"; return None, info
        conc = bool(((dest_o or {}).get("concave") or {}).get("concave"))
        if not conc and os.environ.get("NERO_TILT_FLAT", "0") != "1":
            info["skip"] = "목적지가 오목 용기 아님(기록 없음/평면)"; return None, info
        from scipy.spatial.transform import Rotation as _R
        import copy
        step = float(os.environ.get("NERO_TILT_STEP", "15")); tmax = float(os.environ.get("NERO_TILT_MAX", "30")); cap = float(os.environ.get("NERO_TILT_TIME_CAP", "30")); maxn = int(os.environ.get("NERO_TILT_MAX_TRIES", "16"))
        D = 0.1358; g0 = req["grasp"]; A0 = np.array(g0["approach"], float); A0 /= np.linalg.norm(A0); C0 = np.array(g0["flange"], float) + D * A0
        levels = [step * k for k in range(1, int(tmax // step) + 1)]; dirs = []
        for L_ in levels:
            dirs += [(L_, 0.0), (-L_, 0.0), (0.0, L_), (0.0, -L_)] + [(L_, L_), (L_, -L_), (-L_, L_), (-L_, -L_)]      # 같은 수준에서 축 방향(노름 L) 먼저, 대각(노름 √2·L) 뒤
        t0 = time.time(); info["tilts"] = []
        for px, ry in dirs:
            if info["tried"] >= maxn or time.time() - t0 > cap: info["stopped"] = "상한(횟수/시간)"; break
            r2 = copy.deepcopy(req); g = r2["grasp"]; Rt = _R.from_euler("xy", [px, ry], degrees=True)
            g["quat"] = (Rt * _R.from_quat(g0["quat"])).as_quat().tolist(); A2 = Rt.apply(A0); g["approach"] = A2.tolist(); g["flange"] = (C0 - D * A2).tolist()
            o = r2["obj"]; o["centre"] = (C0 + Rt.apply(np.array(o["centre"], float) - C0)).tolist()      # 서버는 centre 의 x,y 만 쓴다(z 는 base_z+H/2 고정): z 오프셋 회전은 반영 안 됨(근사)
            r2["mode"] = "same"; r2["budget_s"] = min(float(r2.get("budget_s", 8.0)), 4.0)
            r2["sphere_pad"] = float(r2.get("sphere_pad", 0.004)) + float(os.environ.get("NERO_TILT_PAD", "0.012"))      # 서버의 쥔 물체 구는 세워진 상자 기준 -> 기울기 근사 오차를 여유로 보강(기본 +1.2 cm)
            info["tried"] += 1
            try: p2 = post_json(a.curobo + "/place", r2, timeout=60)
            except Exception as e_: info["tilts"].append(dict(pitch=px, roll=ry, err=repr(e_)[:60])); continue
            info["tilts"].append(dict(pitch=px, roll=ry, n_ik_ok=p2.get("n_ik_ok"), ok=bool(p2.get("ok"))))
            if p2.get("ok"):
                info.update(adopted=True, pitch=px, roll=ry, norm=round(float(math.hypot(px, ry)), 1)); return p2, info
        return None, info

    def do_place(oid, d, plan, snap_a, pre=None, t_pick_end=None, warm_place=None):
        """pick 직후(HELD) 쥔 물체를 도착지에 놓는다. 계획은 pick 중에 미리 계산해 둔 것을 쓰고(팔이 계획 시작 자세에 있을 때), 없거나 어긋나면 지금 계산한다.
        모든 시행을 place_log.jsonl 에 기록한다. -> 기록 dict"""
        t_p0 = time.time()
        for _ in range(8):
            rclpy.spin_once(n, timeout_sec=0.03)
        q_now = [float(x) for x in n.got["arm_q"]]
        used_pre = False
        if pre is not None and pre.get("thread") is not None:
            pre["thread"].join(timeout=12.0)            # 보통 pick 이 끝나기 전에 이미 끝나 있다
        if pre and pre.get("pr") is not None and pre.get("built") and max(abs(x - y) for x, y in zip(q_now, pre["built"][1]["start_q"])) < 0.15:
            rec, req, dest_o, top_z, obj, H = pre["built"]; pr = pre["pr"]; used_pre = True
        else:
            set_state(stage="place 계획")
            rec, req, dest_o, top_z, obj, H = place_request(oid, d, plan, q_now)
            if not req["slots"]:
                rec.update(outcome="NO_SLOT", verdict="NO_SLOT"); place_ops.log_place(rec, PLACE_LOG); return rec
            try:
                pr = call_plan(req); publish_insert(req, pr)
            except Exception as e:
                log(f"{oid}: cuRobo place 호출 실패 {e!r}"); rec.update(outcome="SERVER_ERR", verdict="NO_PLAN"); place_ops.log_place(rec, PLACE_LOG); return rec
        if not req["slots"]:
            rec.update(outcome="NO_SLOT", verdict="NO_SLOT"); place_ops.log_place(rec, PLACE_LOG); return rec
        t_plan = time.time()
        rec.update(server={k: pr.get(k) for k in ("ok", "mode", "n_cands", "n_ik_ok", "ik_rejects", "tried", "slot", "yaw_deg", "cost", "plan_sec", "slot_index", "why")},
                   t_plan=(pre or {}).get("t_plan") if used_pre else t_plan - t_p0, plan_precomputed=used_pre, wait_for_plan=t_plan - t_p0)
        # 정밀도 연구(2026-10-02): (a) 인식 중심 추정 오차를 기록한다(pick 전 스냅샷의 물체 상자 중심 대 인식이 낸 중심).
        if snap_a:
            try:
                prim_g = min(snap_a, key=lambda k: math.dist(snap_a[k]["p"], plan["contact"]))
                pg = snap_a[prim_g]["p"]; ec = req["obj"]["centre"]
                rec["pick_gt"] = dict(prim=prim_g, p_gt=pg[:2], est=ec[:2], e_perc_mm=round(1000 * math.hypot(pg[0] - ec[0], pg[1] - ec[1]), 1))
            except Exception as e:
                log(f"인식 오차 기록 실패 {e!r}")
        # (b) 오라클 보정(NERO_PLACE_GTCORR=1, 시뮬 전용 상한 실험): 들어 올린 뒤 물체의 실제 중심을 스냅샷으로 읽어, 계획이 가정한 중심(인식 추정 + 파지 오프셋)과의 차이를 구하고
        #     그만큼 보정한 중심으로 place 를 다시 계획한다. "손 안 위치를 완벽히 알면 최종 오차가 어디까지 줄어드나"의 상한. 실물에서는 손 안 물체 재측정(깊이)이 이 역할을 해야 한다.
        if snap_a and os.environ.get("NERO_PLACE_GTCORR", "0") == "1" and pr.get("ok") is not None:
            try:
                set_state(stage="손 안 위치 측정(오라클)")
                subprocess.run([sys.executable, os.path.join(HERE, "snap_pose.py"), f"/tmp/snap_pose_{oid}_c.json"], cwd=HERE, capture_output=True, timeout=40)
                sc_ = json.load(open(f"/tmp/snap_pose_{oid}_c.json")); p_true = sc_[prim_g]["p"]
                A_ = np.array(plan["approach"], float); A_ /= np.linalg.norm(A_)
                fk_ = np.array(post_json(a.curobo + "/fk", dict(q=[q_now]), timeout=10)["pos"][0], float)
                c_est = np.array(req["obj"]["centre"], float)
                O_est = (np.array(plan["flange"], float) + 0.1358 * A_) - np.array([c_est[0], c_est[1], req["obj"]["base_z"] + req["obj"]["height"] / 2.0])
                exp_c = (fk_ + 0.1358 * A_) - O_est
                dlt = np.array(p_true[:2]) - exp_c[:2]
                obj2 = dict(req["obj"]); obj2["centre"] = [c_est[0] + float(dlt[0]), c_est[1] + float(dlt[1]), c_est[2]]
                req2 = dict(req); req2["obj"] = obj2
                pr = post_json(a.curobo + "/place", req2, timeout=60)
                rec["gtcorr"] = dict(delta_mm=[round(1000 * float(dlt[0]), 1), round(1000 * float(dlt[1]), 1)], norm_mm=round(1000 * float(np.hypot(*dlt)), 1), replan_ok=bool(pr.get("ok")))
                log(f"{oid}: 오라클 손 안 보정 {rec['gtcorr']['norm_mm']}mm -> place 재계획 ok={pr.get('ok')}")
                rec["server"] = {k: pr.get(k) for k in ("ok", "mode", "n_cands", "n_ik_ok", "tried", "slot", "yaw_deg", "cost", "plan_sec", "slot_index", "relax_level")}
            except Exception as e:
                log(f"오라클 보정 실패 {e!r}")
        # 손 안 물체 자세(시뮬 정답, 선택): 놓기 방식을 정하는 변수. 지연을 만들므로 NERO_INHAND_GT=1 일 때만 잰다.
        if snap_a and os.environ.get("NERO_INHAND_GT", "0") == "1":
            try:
                subprocess.run([sys.executable, os.path.join(HERE, "snap_pose.py"), f"/tmp/snap_pose_{oid}_h.json"], cwd=HERE, capture_output=True, timeout=40)
                sh = json.load(open(f"/tmp/snap_pose_{oid}_h.json")); prim_h = min(snap_a, key=lambda k: math.dist(snap_a[k]["p"], plan["contact"]))
                rec["in_hand"] = dict(prim=prim_h, tilt=round(place_ops.tilt_deg(snap_a[prim_h]["q"], sh[prim_h]["q"]), 1), p=sh[prim_h]["p"], q0=snap_a[prim_h]["q"], q=sh[prim_h]["q"], p0=snap_a[prim_h]["p"])
            except Exception as e:
                log(f"{oid}: 손 안 자세 측정 실패 {e!r}")
        if not pr.get("ok") and os.environ.get("NERO_PLACE_TILT_FREE", "0") == "1":      # 2026-11-04(기본 꺼짐): 기본 NO_PLAN 일 때만 기울기 재시도
            try:
                _p2, _ti = _tilt_free_retry(oid, req, dest_o, pr, rec)
                rec["tilt_free"] = _ti
                if _p2 is not None:
                    log(f"{oid}: [TILT_FREE] 기본 NO_PLAN → pitch {_ti['pitch']:+.0f}° roll {_ti['roll']:+.0f}° 로 plan ok (시도 {_ti['tried']}회)"); pr = _p2
                    rec["server"] = {k: pr.get(k) for k in ("ok", "mode", "n_cands", "n_ik_ok", "tried", "slot", "yaw_deg", "cost", "plan_sec", "slot_index", "why")}
                else:
                    log(f"{oid}: [TILT_FREE] 기본 NO_PLAN → 기울기 해 없음 ({_ti.get('skip') or '시도 %d회' % _ti['tried']}{', ' + _ti['stopped'] if _ti.get('stopped') else ''})")
                if (rec.get("trial") or {}).get("pool_dump"):      # NERO_LOG_TRIAL 후보 풀 파일에 선택 기울기 기록
                    try:
                        _pj = json.load(open(rec["trial"]["pool_dump"])); _pj["tilt_free"] = _ti; json.dump(_pj, open(rec["trial"]["pool_dump"], "w"), default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_))
                    except Exception as _e: log(f"{oid}: [TILT_FREE] 풀 파일 기록 실패 {_e!r}")
            except Exception as _e: log(f"{oid}: [TILT_FREE] 재시도 실패 -- 기존 NO_PLAN 유지 {_e!r}")
        if not pr.get("ok"):
            log(f"{oid}: place 계획 실패 -- 후보 {pr.get('n_cands')}개 중 IK {pr.get('n_ik_ok')}개, {pr.get('tried')}")
            try:      # 2026-10-13: 계획 실패 때 등록부 전체(위치/크기/역할)를 남긴다(로그만, 동작 영향 없음) -- 어떤 장애물이 삽입 구간과 겹쳤는지 사후 분석용
                log(f"{oid}: [NO_PLAN 등록부] " + " | ".join(f"{k_}({o_.get('role') or '-'},{o_.get('label')}) xyz=({o_['xyz'][0]:+.3f},{o_['xyz'][1]:+.3f}) 박스 {[round(100 * v, 1) for v in o_['raw']]}cm yaw={math.degrees(float(o_.get('yaw', 0.0))):.0f}" for k_, o_ in world.objs.items() if "raw" in o_ and "xyz" in o_))
            except Exception as _e: log(f"NO_PLAN 등록부 기록 실패 {_e!r}")
            rec.update(outcome="NO_PLAN", verdict="NO_PLAN"); place_ops.log_place(rec, PLACE_LOG); return rec
        if pr.get("pose_name") is not None: log(f"{oid}: 놓기 자세 선택 {json.dumps(dict(name=pr.get('pose_name'), rank=pr.get('pose_rank')), ensure_ascii=False)}")      # 2026-11-09: NERO_PLACE_POSE_LIST 일 때 서버가 고른 자세
        if pr.get("fdiag"): log(f"{oid}: 놓기 손 진단 {json.dumps(pr['fdiag'])}")      # 2026-11-08: NERO_PLACE_FINGER_REJECT 켰을 때만 서버가 줌(접근 z, 닫힘축 z, 손가락 최저-받침면 mm, 손이 물체 밑)
        log(f"{oid}: place 계획 {'(pick 중 미리 계산)' if used_pre else '(지금 계산)'} {pr.get('plan_sec')}s 슬롯#{pr.get('slot_index')} yaw {pr.get('yaw_deg')} 모드 {pr.get('mode')} (후보 {pr.get('n_cands')} 중 IK {pr.get('n_ik_ok')}) | pick 끝->place 시작 대기 {time.time() - (t_pick_end or t_p0):.2f}s")
        pr["ok"] = True
        top_used = float(pr.get("slot_z", top_z)); rec["top_used"] = top_used       # 슬롯 위치에서 추정한 면 높이(가장자리 top_z 와 다를 수 있다)
        try:
            json.dump(pr, open(os.path.expanduser(f"~/grasp/{oid}_place_plan.json"), "w"))
        except Exception as e:
            log(f"{oid}: place 계획 저장 실패 {e!r}"); rec.update(outcome="SAVE_ERR", verdict="NO_PLAN"); place_ops.log_place(rec, PLACE_LOG); return rec
        set_state(stage="place")
        rec["gap_pick_to_place_s"] = time.time() - (t_pick_end or t_p0)
        if snap_a and os.environ.get("NERO_PLACE_SNAPS", "0") == "1":
            try:
                prim_s = min(snap_a, key=lambda k: math.dist(snap_a[k]["p"], plan["contact"]))
                json.dump(dict(prim=prim_s, q0=snap_a[prim_s]["q"]), open(f"/tmp/{oid}.place_prim.json", "w"))
            except Exception as e:
                log(f"place_prim 기록 실패 {e!r}")
        t_x0 = time.time()
        try:
            if warm_place is not None and warm_place["p"].poll() is None:
                warm_go(warm_place)                                    # 예열된 place 실행기에 신호 -> 즉시 출발
                warm_place["p"].wait(timeout=180)
                keep_exec_log(warm_place, oid, "place")
                for ln in open(warm_place["log"], errors="replace").read().splitlines()[-6:]:
                    print(ln, file=sys.stderr, flush=True)
            else:
                rp = subprocess.run([sys.executable, os.path.join(HERE, "exec_place.py"), oid], cwd=HERE, env=dict(os.environ), capture_output=True, text=True, timeout=180)
                for ln in (rp.stdout or "").splitlines()[-6:]:
                    print(ln, file=sys.stderr, flush=True)
        except Exception as e:
            log(f"{oid}: exec_place 실패 {e!r}")
        ex = {}
        try:
            ex = json.load(open(f"/tmp/{oid}.place_same.json"))
        except Exception:
            pass
        rec.update(exec=ex, t_exec=time.time() - t_x0, snaps=ex.get("snaps"))
        verdict = "EXEC_FAIL" if ex.get("status") != "success" else "EXEC_OK"
        if not a.no_place_gt and ex.get("status") == "success":
            time.sleep(1.5)
            try:
                subprocess.run([sys.executable, os.path.join(HERE, "snap_pose.py"), f"/tmp/snap_pose_{oid}_b.json"], cwd=HERE, capture_output=True, timeout=40)
                snap_b = json.load(open(f"/tmp/snap_pose_{oid}_b.json"))
                if a.insert_at:
                    import insertion_geometry as _G
                    E_, dd_ = _G.parse_insert_spec(a.insert_at)
                    prim_i = min(snap_a, key=lambda k: math.dist(snap_a[k]["p"], plan["contact"]))
                    time.sleep(0.6)       # 정지 확인용 두 번째 스냅샷(0.6초 뒤)
                    subprocess.run([sys.executable, os.path.join(HERE, "snap_pose.py"), f"/tmp/snap_pose_{oid}_c.json"], cwd=HERE, capture_output=True, timeout=40)
                    snap_c = json.load(open(f"/tmp/snap_pose_{oid}_c.json"))
                    j = insert_ops.judge_insert(snap_c[prim_i], E_, dd_, depth_min=a.insert_depth, snap_prev=snap_b[prim_i])
                    j["prim"] = prim_i; rec["gt"] = j; verdict = "INSERTED" if j["success"] else "NOT_INSERTED"; finish_insert(verdict, j.get("contained_len_m"))
                else:
                    j = place_ops.judge(snap_a, snap_b, plan["contact"], pr["slot"], top_used, dest_o, H)
                    rec["gt"] = j; verdict = j["verdict"]
            except Exception as e:
                log(f"{oid}: place 정답 판정 실패 {e!r}")
        if a.no_place_gt and verdict == "EXEC_OK":      # 2026-10-16: GT 없는 모드 -- 실행이 끝났다는 것뿐(검증 전). 놓였는지는 끝난 뒤 재조사(비전)로 확인하고 재시도한다
            verdict = "INSERTED" if a.insert_at else "PLACED"; rec["unverified"] = True
        if a.no_place_gt and verdict in ("PLACED", "INSERTED") and VBASE.get(a.dest_id) is not None and a.dest_id in DSPEC:      # 2026-10-16: 놓은 직후 같은 자세에서 목적지 영역 높이 지도 차이로 안/바깥 판정(GT 없음)
            try:
                _t0v = time.time(); _now = _vlook(DSPEC[a.dest_id]); _jv = place_verify.judge(VBASE[a.dest_id], _now, DSPEC[a.dest_id], floor_z=a.floor_z); rec["vision"] = _jv
                log(f"{oid}: 놓은 뒤 비전 검증 {_jv['verdict']} {_jv} ({time.time() - _t0v:.1f}s)")
                if _jv["verdict"] == "outside": verdict = "MISSED"
                VBASE[a.dest_id] = _now
            except Exception as e_: log(f"{oid}: 놓은 뒤 비전 검증 실패 {e_!r}")
        rec.update(verdict=verdict, outcome=verdict)
        place_ops.log_place(rec, PLACE_LOG)
        log(f"{oid}: place verdict={verdict} {rec.get('gt', {})}")
        _bp = (rec.get("gt") or {}).get("bumped") or []
        if _bp:      # 2026-10-16: 놓은 직후 다른 물체/용기가 밀림 -- 로그와 결과(info)로 알린다(동작은 바꾸지 않음)
            log(f"{oid}: [경고] 놓은 뒤 밀림 감지: " + ", ".join(f"{b_['prim']}({b_['moved_cm']}cm, 기울기 {b_['tilt']:.0f}도)" for b_ in _bp))
        # 놓은 물체는 장애물로 등록(다시 집지 않는다)
        if ex.get("status") == "success" and not a.insert_at and not os.environ.get("NERO_PLACED_VANISH"):      # NERO_PLACED_VANISH: 놓은 물체를 시뮬에서 치우는 실험 -- 등록부에 장애물로 남기지 않는다(2026-10-08)
            sx, sy = pr["slot"]; zt = top_used + H
            _pid = "placed_%d" % int(time.time() * 1000 % 100000); rec["placed_id"] = _pid
            world.objs[_pid] = dict(fails=99, defer_gen=-1, seen=1, role=(None if (world.objs.get(oid) or {}).get("clearing") else "placed"), xyz=[sx, sy, zt / 2], raw=[obj["dims"][0], obj["dims"][1], zt], yaw=obj["yaw"], w=[sx, sy, 0.05], label=rec["label"])
            if os.environ.get("NERO_PLACED_MEM"):      # 2026-11-12(기본 꺼짐): 놓은 물체 상자 = 계획의 물체 목표 회전(obj_goal.quat)으로 돌린 인식 치수, 받침면 위에 선 것으로(시뮬 정답 안 씀)
                try:
                    from scipy.spatial.transform import Rotation as _Rt
                    _og = pr.get("obj_goal") or {}; _dm = np.array([float(obj["dims"][0]), float(obj["dims"][1]), float(H)])
                    if _og.get("centre") and pr.get("half_h_rot"):      # r6 고침: obj_goal.quat 의 로컬 축 규약이 인식 dims 순서와 달라(판에서 폭을 연직으로 잡음) 서버가 쓴 회전 후 반높이(half_h_rot)로 연직 길이를 정하고, 그 길이에 가장 가까운 치수를 뺀 나머지 둘을 수평으로
                        _vz = 2.0 * float(pr["half_h_rot"]); _kz = int(np.argmin(np.abs(_dm - _vz))); _ii, _jj = [i_ for i_ in range(3) if i_ != _kz]
                        _cx, _cy = float(_og["centre"][0]), float(_og["centre"][1]); _raw = [float(_dm[_ii]), float(_dm[_jj]), _vz]; _yw = float(obj["yaw"]) + math.radians(float(pr.get("yaw_deg") or 0.0))      # 수평 방향은 근사(인식 yaw + 놓기 yaw)
                    else:
                        _cx, _cy = float(sx), float(sy); _raw = [float(_dm[0]), float(_dm[1]), float(H)]; _yw = float(obj["yaw"])
                    _xyz = [_cx, _cy, top_used + _raw[2] / 2]; world.objs[_pid].update(xyz=_xyz, raw=_raw, yaw=_yw, w=[_cx, _cy, 0.05])
                    _L = json.load(open(os.environ["NERO_PLACED_MEM"])) if os.path.exists(os.environ["NERO_PLACED_MEM"]) else []
                    _L.append(dict(id=_pid, xyz=_xyz, raw=_raw, yaw=_yw, label=rec.get("label"), src=("obj_goal" if _og.get("quat") else "slot")))
                    json.dump(_L, open(os.environ["NERO_PLACED_MEM"], "w"), indent=1); log(f"{oid}: [placed-mem] 놓은 물체 기억 {_pid} 중심 ({_cx:+.3f},{_cy:+.3f},{_xyz[2]:.3f}) 상자 {[round(v, 3) for v in _raw]} ({'obj_goal' if _og.get('quat') else 'slot'})")
                except Exception as _e: log(f"{oid}: [placed-mem] 기억 저장 실패 {_e!r}")
        return rec

    # ── 2026-10-14: 다물체 체인(--plan-file). 항목을 한 프로세스 안에서 차례로 처리한다: 월드 등록부/임베딩 유지, 실패해도 홈 복귀·재시작·재탐색 없음 ──
    PL = dict(items=[], i=0, out=[], retry=[], paused=None, path="/tmp/multi_exec_chain.json")
    _CAPT = float(os.environ.get("NERO_CHAIN_ITEM_CAP_TOTAL", "0")); _CAPT = (100.0 if (_HOOK and _CAPT <= 0) else _CAPT)      # 2026-10-22(기본 0=꺼짐): 항목 누적 시간 상한(초) -- 본 라운드와 미룬 라운드, 홈 경유, 목적지 후보 전환의 시간을 합산한다
    def _over_total(it):
        return _CAPT > 0 and (PL.get("acc", {}).get(it["id"], 0.0) + time.time() - PL.get("t0", t_run)) > _CAPT
    def _trk_new(i_):
        return dict(id=i_, c_reasons=[], nocand=[], cgn=[], coll=[], c_secs=[], views=0, blocked="", xy=None)
    if _HOOK: PL["trk"] = _trk_new(None); PL["hk_cnt"] = {}; PL["hk_ev"] = []; PL["view_n"] = {}
    def _hook_decide(cur):
        """2026-10-22: 계획 실패 확정 직후 LLM 에게 다음 행동을 묻는다(사양 LLM_HOOK_SPEC.md). 반환: 판단 dict(act=giveup|defer|alt|view) 또는 None(=규칙 현행 동작). 입력은 지금까지 알려진 값만."""
        trk = PL["trk"]; i_ = cur["id"]; spent = PL.get("acc", {}).get(i_, 0.0) + time.time() - PL.get("t0", t_run); cap = float(_CAPT); left = max(0.0, cap - spent)
        PL["hk_cnt"][i_] = PL["hk_cnt"].get(i_, 0) + 1; xy = list(trk["xy"]) if trk["xy"] else None
        try:
            ev = llm_recover.make_event(reasons=list(trk["c_reasons"]) + list(trk["nocand"]), no_cgn=" ".join(trk["nocand"]).strip(), cgn_cands_per_attempt=",".join(str(c_) for c_ in trk["cgn"]), collision_objects_per_attempt=",".join(trk["coll"]),
                                        view_retries_done=int(trk["views"]), planning_fail_count_this_item=PL["hk_cnt"][i_], planning_seconds_spent=round(sum(trk["c_secs"]), 2), blocked_note=trk["blocked"].strip(), target_xy=xy,
                                        target_radius_from_base=(round(math.hypot(*xy), 3) if xy else None), seconds_already_spent_on_this_item=round(spent, 1), item_time_cap_seconds=cap, item_seconds_left_before_cap=round(left, 1),
                                        run_elapsed_seconds=round(time.time() - _T_LOG0, 1), items_in_this_run=int(PL.get("n_items0") or len(PL["items"])), planning_failure_events_earlier_in_this_run=len(PL["hk_ev"]),
                                        planning_seconds_spent_on_those_events=round(sum(PL["hk_ev"]), 1))
            PL["hk_ev"].append(sum(trk["c_secs"]))
            r = llm_recover.decide(ev, left, bool(cur.get("alt")) and not cur.get("_nocand") and not cur.get("_pickside"))
        except Exception as e_:
            log(f"[noplan-log] llm_recover item={i_} error={e_!r} fallback=true"); return None
        log("[noplan-log] " + llm_recover.log_line(i_, r))
        llm_recover.log_record(llm_recover.jsonl_path(a.out), ts=round(time.time(), 1), run_elapsed=round(time.time() - _T_LOG0, 1), item=i_, dest=cur.get("dest"), prompt_sha256=llm_recover.PROMPT_SHA256, event=ev, result=r)
        return r if r["act"] else None
    if a.plan_file:
        PL["items"] = json.load(open(a.plan_file)); a.max_picks = 99; PL["t0"] = time.time(); PL["n_items0"] = len(PL["items"])
    ADV = llm_advisor.Advisor() if (a.plan_file and os.environ.get("NERO_LLM_RECOVER", "0") == "1") else None      # 2026-10-18: 실패 항목 조언을 LLM 에게 비동기로 받는다(응답 시간은 로봇 동작 뒤에 숨는다)
    DESTK = {}
    if ADV and a.dests:
        try: DESTK = {d_["id"]: d_.get("kind") for d_ in json.load(open(a.dests))}
        except Exception: DESTK = {}
    def _chain_save():
        try: json.dump(dict(out=PL["out"], current=(PL["items"][PL["i"]]["id"] if PL["i"] < len(PL["items"]) else None), paused=PL["paused"], advice=(ADV.log if ADV else None), remaining=[x["id"] for x in PL["items"][PL["i"] + 1:]] + [x["id"] for x in PL["retry"]], t=round(time.time() - t_run, 1)), open(PL["path"], "w"), ensure_ascii=False)
        except Exception: pass
    def _b1_stash(ang_, r_):
        if _B1: PL["lf"] = dict(ang=float(ang_), r=r_, arm0=float(n.got["arm_q"][0]) if n.got.get("arm_q") is not None else 1e9)
    def _c1_on():
        if not _C1 or os.environ.get("NERO_C1_OFF_FINAL", "0") == "1": return False
        return not (PL["items"] and PL["i"] < len(PL["items"]) and PL["items"][PL["i"]].get("_tried"))
    def _chain_local_find(it):
        """새 항목의 대상을 힌트 방위 주변에서 찾는다(정지 촬영 최대 3곳 + 임베딩 없는 후보 재촬영). 360도 스윕 없음."""
        _B1 and PL.pop("lf", None)
        if not it.get("hint"): return False
        hb = float(np.clip(bearing_to_j1(float(it["hint"][0]), float(it["hint"][1])) + PL.pop("view_off", 0.0), -J1_LIMIT + 0.05, J1_LIMIT - 0.05)); t0_ = time.time()
        for ang in [hb, float(np.clip(hb - 0.4, -J1_LIMIT + 0.05, J1_LIMIT - 0.05)), float(np.clip(hb + 0.4, -J1_LIMIT + 0.05, J1_LIMIT - 0.05))]:
            set_state(stage="탐색", angle=float(ang), target=None, plan=None); n.move_j1(ang, j7_abs=a.j7); r_ = grab(n, a.floor_z)
            if r_ is not None: world.update(r_[1], float(n.got["arm_q"][0])); pub()
            if any(eligible(o_) for o_ in world.objs.values()): log(f"[chain] {it['id']} 국소 탐색으로 확정 (j1={ang:+.2f}, {time.time() - t0_:.1f}s)"); _b1_stash(ang, r_); return True
            _dsz = sorted(GROUND["desc"].get("size_cm") or [])
            for _k in range(2):
                _un = sorted((o_ for o_ in world.objs.values() if "raw" in o_ and o_.get("role") not in ("dest", "placed") and not o_.get("given") and o_.get("feat") is None and (not _dsz or sum(abs(x - 100 * y) for x, y in zip(_dsz, sorted(float(v) for v in o_["raw"]))) < 14.0)), key=lambda o_: abs(bearing_to_j1(o_["w"][0], o_["w"][1]) - ang))
                if not _un: break
                _st = float(np.clip(bearing_to_j1(_un[0]["w"][0], _un[0]["w"][1]), -J1_LIMIT + 0.05, J1_LIMIT - 0.05)); n.move_j1(_st, j7_abs=a.j7); r_ = grab(n, a.floor_z)
                if r_ is not None: world.update(r_[1], float(n.got["arm_q"][0])); pub()
                if any(eligible(o_) for o_ in world.objs.values()): log(f"[chain] {it['id']} 재촬영으로 확정 ({time.time() - t0_:.1f}s)"); _b1_stash(_st, r_); return True
        return False
    def _chain_switch(it):
        PL["pickside"] = False; PL["nocand"] = False; PL.pop("c1", None)
        if _HOOK:
            if PL["trk"].get("id") != it["id"]: PL["trk"] = _trk_new(it["id"])
            PL["trk"]["blocked"] = ""
        """현재 항목을 it 로: 대상 겉모습, 목적지 역할(입력 명세 엔티티의 dest/obstacle), 홀 목적지면 옆 파지 설정."""
        if _CAPT > 0 and PL.get("cur_id") is not None and PL.get("t0") is not None: PL.setdefault("acc", {}); PL["acc"][PL["cur_id"]] = PL["acc"].get(PL["cur_id"], 0.0) + time.time() - PL["t0"]
        if _CAPT > 0: PL["cur_id"] = it["id"]
        PL["t0"] = time.time()      # 항목(후보 목적지)별 시간 상한 기준
        GROUND["desc"] = it["desc"]; GROUND["hint"] = (list(it["hint"]) if it.get("hint") else None); a.dest_id = it["dest"]; a.dest_kind = it.get("kind", a.dest_kind)
        for o_ in world.objs.values():
            if o_.get("given_id") is not None: o_["role"] = "dest" if o_["given_id"] == it["dest"] else "obstacle"
            elif o_.get("role") not in ("dest", "placed") and o_.get("fails", 0) >= a.max_fail: o_["fails"] = 0; o_["defer_gen"] = -1; o_.pop("stale", None)      # 앞 항목에서 실패 표시된 물체도 다시 후보가 된다
        if it.get("kind") == "hole":      # 2026-10-16: 파지 높이 대역은 10-15 에 임시로 넣은 미검증 규칙이라 기본을 끈다(NERO_HOLE_BAND="0.30,0.70" 으로 켤 수 있음). 후보는 놓기 사전 선별(손-홀더 여유)이 고른다
            _hb = os.environ.get("NERO_HOLE_BAND", "")
            if _hb: os.environ["NERO_GRASP_BAND"] = _hb
            else: os.environ.pop("NERO_GRASP_BAND", None)
            os.environ["NERO_PLACE_AWARE"] = "1"
        else: os.environ.pop("NERO_GRASP_BAND", None); os.environ["NERO_PLACE_AWARE"] = "0"
        log(f"[chain] 다음 항목 {it['id']} -> {it['dest']}({it.get('kind')})")
    def _chain_nudge(it):
        """2026-10-15: 항목에 nudge 가 있으면(이웃과 틈이 좁아 집기 계획이 막히는 물체) 집기 전에 가장 가까운 이웃 반대쪽으로 밀어 틈을 벌린다(닫은 그리퍼, cuRobo /push)."""
        nd = it.get("nudge")
        if not nd or it.get("_nudged"): return
        _t = next(((k_, o_) for k_, o_ in world.objs.items() if eligible(o_) and "xyz" in o_ and "raw" in o_), None)
        if _t is None: log(f"[chain] {it['id']} 밀기 생략: 대상 위치를 모름"); return
        k_, o_ = _t; _ot = [(kk, oo) for kk, oo in world.objs.items() if kk != k_ and oo.get("role") not in ("dest", "placed") and "xyz" in oo and "raw" in oo and not oo.get("stale")]
        if not _ot: return
        c_ = np.array(o_["xyz"][:2], float); nb_ = min(_ot, key=lambda t_: math.hypot(t_[1]["xyz"][0] - c_[0], t_[1]["xyz"][1] - c_[1]))
        a_ = c_ - np.array(nb_[1]["xyz"][:2], float); a_ = a_ / np.linalg.norm(a_) if np.linalg.norm(a_) > 1e-3 else np.array([1.0, 0.0])
        d_ = float(nd) if (isinstance(nd, (int, float)) and not isinstance(nd, bool)) else float(os.environ.get("NERO_NUDGE_DIST", "0.07"))
        r_ = 0.5 * max(float(o_["raw"][0]), float(o_["raw"][1])); lead_ = 0.05; _obs = world.obstacles(k_)
        def _rad(ob): return 0.5 * max(float(ob["raw"][0]), float(ob["raw"][1]))
        def _free(p, rr):      # 점 p 에서 반경 rr 가 다른 물체(밀 물체 제외)와 겹치지 않는가 -> 가장 가까운 표면 간격
            return min((math.hypot(p[0] - ob["xyz"][0], p[1] - ob["xyz"][1]) - _rad(ob) - rr for ob in _obs), default=9.0)
        cands = []      # 좁은 틈 안에는 못 서므로 이웃 반대쪽(+접선) 방향 후보를 30도 간격으로 평가: 서는 자리(밀 물체 뒤)와 끝 자리가 비어야 한다
        for ang_ in range(0, 360, 30):
            u_ = np.array([math.cos(math.radians(ang_)), math.sin(math.radians(ang_))])
            if float(u_ @ a_) < -0.01: continue
            s_ = c_ - u_ * (r_ + lead_); e_o = c_ + u_ * d_; pend = s_ + u_ * (lead_ + d_)
            if _free(s_, 0.03) < 0.005 or _free(pend, 0.03) < 0.005: continue
            gap_end = min((math.hypot(e_o[0] - ob["xyz"][0], e_o[1] - ob["xyz"][1]) - _rad(ob) - r_ for ob in _obs), default=9.0)
            if gap_end < 0.02: continue
            cands.append((gap_end + 0.01 * float(u_ @ a_), u_))
        cands.sort(key=lambda t_: -t_[0]); ok_ = False
        for sc_, u_ in cands[:8]:
            ok_ = n.push_object(c_, u_, r_, d_, _obs, a.curobo)
            log(f"[chain] 밀기 후보 방향 ({u_[0]:+.2f},{u_[1]:+.2f}) 끝 틈 {100 * sc_:.0f}cm -> {'성공' if ok_ else '실패'}")
            if ok_: break
        if not cands: log(f"[chain] {it['id']} 밀기 불가: 서 있을 빈 자리가 있는 방향이 없음")
        it["_nudged"] = True; log(f"[chain] {it['id']} 이웃 {nb_[0]} 반대쪽으로 {100 * d_:.0f}cm 밀기 ok={ok_}")
        if ok_: _chain_local_find(it)
    def _adv_submit(cur_, outcome, info):
        """실패 항목에 대한 조언 요청(비동기). 로봇은 기다리지 않는다."""
        if ADV is None: return
        try:
            x_, y_ = cur_["hint"]; sz_ = (cur_.get("desc") or {}).get("size_cm"); fill_ = {}
            for o_ in PL["out"]:
                if o_.get("outcome") == "ok": fill_[o_["dest"]] = fill_.get(o_["dest"], 0) + 1
            tries_ = sum(1 for o_ in PL["out"] if o_["id"] == cur_["id"]); remain_ = len(PL["items"]) - PL["i"] - 1 + len(PL["retry"])
            L_ = ["너는 로봇 정리 플래너다. 아래 물체를 집거나 놓는 데 실패했다. 텍스트만 보고(도구/파일 사용 금지) 이 물체를 나중에 다시 시도할 때의 처리를 고른다. 로봇은 기다리지 않고 다음 물체로 가 있으며, 이 물체 차례가 오면 네 답을 적용한다.", "",
                  f"물체 {cur_['id']}: 크기 {sz_}cm 위치 ({x_:+.2f},{y_:+.2f}) 반경 {100 * math.hypot(x_, y_):.0f}cm 방위 {math.degrees(math.atan2(y_, x_)):+.0f}°",
                  f"이번 결과: {outcome} {json.dumps(info or {}, ensure_ascii=False)[:160]} | 지금 목적지 {cur_['dest']} | 이 물체 누적 시도 {tries_}회 | 실패 사유: {PL.get('lastwhy')}",
                  "목적지: " + ", ".join(f"{k}({v}, 현재 {fill_.get(k, 0)}개 넣음)" for k, v in DESTK.items()) + f" | 남은 물체 {remain_}개",
                  "", "선택지: keep(같은 목적지로 한 번 더), change_dest(다른 목적지 dest 로), abandon(포기: 반경 55cm 이상이라 팔이 못 닿거나 같은 사유로 이미 두 번 실패했을 때만).",
                  "규칙: 큰/긴 물체는 높이 13cm 의 bin, 작은 것은 bin2(낮음). 용기가 많이 찼으면 덜 찬 곳. 집기 계획 실패(접근/손목 한계)는 목적지를 바꿔도 안 풀린다.",
                  '출력(JSON 하나만): {"action":"keep|change_dest|abandon","dest":"<change_dest 일 때만>","why":"<한 줄>"}']
            ADV.submit(cur_["id"], "\n".join(L_))
        except Exception as e_: log(f"[advice] 요청 실패 {e_!r}")
    def _adv_apply(it):
        """차례가 온 미룬 항목에 조언을 적용한다. 포기면 True."""
        if ADV is None or not it.get("_tried"): return False
        rec_ = next((r_ for r_ in reversed(ADV.log) if r_["id"] == it["id"] and "applied" not in r_), None); adv_ = ADV.take(it["id"])
        if adv_ is None:
            if rec_ is not None: rec_["applied"] = "late"
            if rec_ is not None: log(f"[advice] {it['id']} 응답이 아직 안 옴 -- 기본 동작(변경 없음)")
            return False
        act_ = adv_.get("action"); d_ = adv_.get("dest")
        if act_ == "abandon": rec_["applied"] = "abandon"; log(f"[advice] {it['id']} 포기: {adv_.get('why')}"); return True
        if act_ == "change_dest" and d_ in DESTK and d_ != it["dest"]:
            log(f"[advice] {it['id']} 목적지 {it['dest']} -> {d_}: {adv_.get('why')}"); it["dest"] = d_; it["kind"] = DESTK[d_]; it["alt"] = [x_ for x_ in (it.get("alt") or []) if x_["dest"] != d_]; rec_["applied"] = "change:" + d_
        else: rec_["applied"] = "keep"; log(f"[advice] {it['id']} 유지: {adv_.get('why')}")
        return False
    def _chain_advance(outcome, info=None):
        """현재 항목 결과를 기록하고 다음 항목(보류했던 것은 끝에서 한 번 더)으로 넘어간다. 다음이 있으면 True."""
        cur_ = PL["items"][PL["i"]]; PL["out"].append(dict(id=cur_["id"], dest=cur_["dest"], outcome=outcome, t=round(time.time() - t_run, 1), **(info or {})))
        _NPL and nlog(f"chain_advance item={cur_['id']} outcome={outcome} defer_skipped_nocand={int(outcome == 'planfail' and bool(cur_.get('_nocand')) and os.environ.get('NERO_CHAIN_DEFER_NOCAND', '1') == '0')} over_total={int(_over_total(cur_))} acc_s={PL.get('acc', {}).get(cur_['id'], 0.0) + time.time() - PL.get('t0', t_run):.1f}")
        _pf = outcome == "planfail" or (outcome == "stopped" and str((info or {}).get("place")) in ("NO_PLAN", "planfail"))
        if _pf and cur_.get("alt") and not cur_.get("_nocand") and not cur_.get("_pickside") and not _over_total(cur_) and not cur_.get("_llm_nodefer") and not _att_capped(cur_):      # 2026-10-15: 후보 1 목적지가 안 풀리면(계획 실패/놓기 불가) 후보 2 로 같은 자리에서 바로 다시 시도한다(홈 복귀/다른 항목 거치지 않음)
            _nx = cur_["alt"].pop(0); log(f"[chain] {cur_['id']} 목적지 후보 {cur_['dest']} 실패({outcome}) -> 후보 {_nx['dest']} 로 바로 재시도")
            PL["out"][-1]["outcome"] = outcome + "->alt:" + _nx["dest"]; cur_["dest"] = _nx["dest"]; cur_["kind"] = _nx.get("kind") or cur_.get("kind"); PL["paused"] = None
            _chain_switch(cur_); _chain_local_find(cur_); _chain_save(); return True
        if outcome in ("not_found", "planfail") and not cur_.get("_tried") and os.environ.get("NERO_CHAIN_DEFER", "1") == "1" and not (outcome == "planfail" and cur_.get("_nocand") and os.environ.get("NERO_CHAIN_DEFER_NOCAND", "1") == "0") and not _over_total(cur_) and not cur_.get("_llm_nodefer"): cur_["_tried"] = True; cur_["_fails"] = 0; PL["retry"].append(cur_); _adv_submit(cur_, outcome, info)      # 2026-10-17: NERO_CHAIN_DEFER=0 이면 뒤로 미루지 않는다(NERO_REPLAN 의 마지막 폴백 라운드: 홈 재시도만, 미루기 없음 -- multi_exec.run_chain)
              # 계획 실패/못 찾음은 뒤로 미루었다가 한 번 더
        if _FIN and PL.get("c1"):      # 2026-10-30(NERO_FINAL_RETRY_ONCE, 기본 꺼짐): C1 로 포기한 항목은 끝 재시도에 한 번 올라가 있어야 한다(이미 올라갔으면 그대로; _tried 로 항목당 1회 상한)
            _q = any(x is cur_ for x in PL["retry"])
            _add = (not _q) and not cur_.get("_tried")
            if _add: cur_["_tried"] = True; PL["retry"].append(cur_); _adv_submit(cur_, outcome, info)
            log(f"[final] item={cur_['id']} dest={cur_['dest']} prior={outcome} queued={'already' if _q else ('new' if _add else 'no')} c1_off_in_retry=true")
        PL.pop("c1", None)
        PL["i"] += 1
        if PL["i"] >= len(PL["items"]) and PL["retry"]: PL["items"] = PL["items"] + PL["retry"]; PL["retry"] = []
        _chain_save()
        while PL["i"] < len(PL["items"]) and _adv_apply(PL["items"][PL["i"]]):      # 2026-10-18: LLM 이 포기하라고 한 미룬 항목은 건너뛴다
            _ab = PL["items"][PL["i"]]; PL["out"].append(dict(id=_ab["id"], dest=_ab["dest"], outcome="abandoned", t=round(time.time() - t_run, 1))); PL["i"] += 1
        if PL["paused"] or PL["i"] >= len(PL["items"]): _chain_save(); return False
        it = PL["items"][PL["i"]]; _chain_switch(it); _chain_local_find(it); _chain_nudge(it); _chain_save(); return True
    def _chain_done(oid, verdict, rec_):
        """집고 놓은 뒤 호출. 정상이면 다음 항목, 상태가 불확실하면(놓기 실패/미끄러짐 등) 멈추고 LLM 에게 돌려준다."""
        pl_ = rec_.get("place"); ok_ = verdict == "HELD" and pl_ in ("PLACED", "INSERTED")
        _hard = set(x for x in os.environ.get("NERO_CHAIN_STOP", "TOPPLED").split(",") if x)      # 2026-10-15: 놓기 실패(MISSED/NOT_INSERTED 등)는 멈추지 않고 기록만 남기고 다음 항목으로 -- 멈춤은 쓰러짐 같은 하드 실패만(NERO_CHAIN_STOP 로 조정)
        if not ok_ and (pl_ in _hard or verdict == "SLIPPED" and "SLIPPED" in _hard): PL["paused"] = f"집기 {verdict}, 놓기 {pl_}"
        _bump = [str(x) for x in (rec_.get("bumped") or [])] if isinstance(rec_, dict) else []
        more = _chain_advance("ok" if ok_ else ("stopped" if PL["paused"] else "failed"), dict(pick=verdict, place=pl_, label=rec_.get("label"), **({"bumped": _bump} if _bump else {})))
        if not more: a.max_picks = picks_ok + 1 if False else 0      # 루프 종료 조건
        return more
    forced_retry = None; recover_n = 0
    VBASE = {}; DSPEC = {}
    def _vlook(d):
        """목적지 d 쪽으로 관측 자세를 돌려 목적지 영역 높이 지도를 찍는다(기준과 검증이 같은 자세라 비교가 된다)."""
        n.move_j1(bearing_to_j1(d["xy"][0], d["xy"][1]), j7_abs=a.j7); time.sleep(0.8)
        def _cap():
            n.fresh_frame(); g_ = n.capture_geom(); return place_verify.heightmap(g_["pts_base"], g_["zf"], d)
        h_ = place_verify.stable_heightmap(_cap, d)
        if not h_.get("stable"): log(f"[verify] {d.get('id')} 영상이 안정되지 않음(흔들림 {h_.get('jitter')})")
        return h_
    if a.no_place_gt and a.dests and os.environ.get("NERO_VERIFY", "1") == "1" and PL["items"]:      # 2026-10-16: GT 없는 모드 -- 놓은 직후 비전 검증용 기준 영상(목적지마다 1회)
        try:
            DSPEC = {d_["id"]: d_ for d_ in json.load(open(a.dests)) if d_.get("size") and d_.get("xy")}; _tv0 = time.time()
            for _id, _d in DSPEC.items(): VBASE[_id] = _vlook(_d)
            log(f"[verify] 목적지 {len(VBASE)}곳 기준 영상 {time.time() - _tv0:.0f}s (덮임 {[round(v['cover'], 2) for v in VBASE.values()]})")
        except Exception as e_: log(f"[verify] 기준 영상 실패 {e_!r}"); VBASE = {}
    while picks_ok < a.max_picks and attempts < a.max_picks * 3 + 4 * CLEAR["used"] or forced_retry is not None:
        engaged = False
        _cap = float(os.environ.get("NERO_CHAIN_ITEM_CAP", "45"))      # 2026-10-15: 체인 항목당 시간 상한(초) -- 집기 계획이 반복해서 막히는 물체 하나가 1~2분씩 쓰던 문제(향신료 통 97초+67초)
        if PL["items"] and PL["i"] < len(PL["items"]) and not PL["paused"] and _cap > 0 and time.time() - PL.get("t0", t_run) > _cap:
            _it = PL["items"][PL["i"]]; log(f"[chain] {_it['id']} -> {_it['dest']} 시간 상한 {_cap:.0f}s 초과 -- 포기하고 다음(후보 2/다음 항목)으로")
            if _chain_advance("planfail", dict(why="시간 상한")): attempts += 1; forced_retry = None; continue
            break
        if _CAPT > 0 and PL["items"] and PL["i"] < len(PL["items"]) and not PL["paused"] and _over_total(PL["items"][PL["i"]]):
            _it = PL["items"][PL["i"]]; log(f"[chain] {_it['id']} -> {_it['dest']} 누적 시간 상한 {_CAPT:.0f}s 초과 -- 포기하고 다음 항목으로")
            if _chain_advance("planfail", dict(why="누적 시간 상한")): attempts += 1; forced_retry = None; continue
            break
        try:
            _order = order
            if os.environ.get("NERO_FOCUS_SCAN", "1") == "1":      # 필요한 부분만: 이미 발견한 대상의 방위각을 먼저 방문하고 나머지 각도는 폴백
                _cur = float(n.got["arm_q"][0]) if n.got.get("arm_q") is not None else 0.0
                _ks = sorted({round(bearing_to_j1(o["w"][0], o["w"][1]), 2) for o in world.objs.values() if "w" in o and o.get("role") not in ("dest", "placed") and o["fails"] < a.max_fail and eligible(o)}, key=lambda k: abs(k - _cur))
                _order = _ks if (_ks or world.objs) else order      # 2026-10-14: 360도 폴백 삭제 -- 알려진 물체(마스크가 잡혀 있던 방위)만 본다. 아무것도 모르는 첫 탐색(등록부 비어 있음)일 때만 전체 각도. 한 바퀴 재탐색이 꼭 필요하면 LLM 이 rt_scan 도구로 부른다
                if _ks: log(f"필요한 부분만 탐색: 발견한 방위각 {_ks} 먼저")
                if os.environ.get("NERO_LEAN", "0") == "1":      # 2026-10-14: 치우는 중에는 등록부에 마스크가 잡혀 있던 그 물체의 방위만 보고 다른 각도는 훑지 않는다
                    _cb = sorted({round(bearing_to_j1(o["w"][0], o["w"][1]), 2) for o in world.objs.values() if o.get("clearing") and "w" in o}, key=lambda k: abs(k - _cur))
                    if _cb: _order = _cb; log(f"치우는 중 -- 마스크가 잡혀 있던 방위 {_cb} 만 본다(다른 각도는 훑지 않음)")
            for ang in ((['__forced__'] if forced_retry else []) + _order):
                _forced = None
                if ang != '__forced__' and clean.get(round(ang, 3)):
                    continue            # 마지막 pick 이후 바뀐 게 없는 구역은 다시 훑지 않는다(마무리 확인 스윕 단축)
                if ang == '__forced__':
                    _forced = forced_retry; forced_retry = None
                    geom, dets = _forced["geom"], [_forced["det"]]; cur = float(n.got["arm_q"][0])
                    _db = _forced["det"]["box"]
                    world.objs[_forced["oid"]] = dict(fails=0, defer_gen=-1, seen=1, forced=True, xyz=_db["xyz"], raw=_db["raw"], yaw=_db["yaw"], w=[float(x) for x in _forced["det"]["w"]], label=_forced.get("label", ""))
                    pub()
                else:
                    _lf = PL.get("lf") if _B1 else None; _b1_skip = False
                    if _lf is not None:
                        _a0 = float(n.got["arm_q"][0]) if n.got.get("arm_q") is not None else 1e9
                        _b1_skip = _lf["r"] is not None and abs(float(ang) - _lf["ang"]) <= 0.06 and abs(_a0 - _lf["arm0"]) <= 0.02
                        if not _b1_skip: log(f"[b1] keep ang={float(ang):+.2f} lf_ang={_lf['ang']:+.2f} arm0={_a0:+.3f} lf_arm0={_lf['arm0']:+.3f} reason={'ang_mismatch' if abs(float(ang) - _lf['ang']) > 0.06 else 'moved'}")
                    elif _B1: log(f"[b1] keep ang={float(ang):+.2f} reason=no_lf")
                    if _b1_skip:
                        PL.pop("lf", None); geom, dets = _lf["r"]; cur = float(n.got["arm_q"][0])      # world.update 는 국소 탐색에서 이미 수행했다
                        log(f"[b1] skip ang={float(ang):+.2f} lf_ang={_lf['ang']:+.2f} arm0={_a0:+.3f} saved_stops=1")
                    else:
                        set_state(stage="탐색", angle=float(ang), target=None, plan=None)
                        n.move_j1(ang, j7_abs=a.j7)
                        r = grab(n, a.floor_z, a.target)
                        if r is None:
                            continue
                        geom, dets = r
                        cur = float(n.got["arm_q"][0])
                        world.update(dets, cur); pub()
                # 목표: 아직 시도 안 한/실패 한도 안 넘은/막힘 보류가 풀린 물체 중 화면 중심에 가장 가까운 것
                cands = []
                if _forced is not None:
                    cands = [(0.0, _forced["oid"], _forced["det"])]
                for d in ([] if _forced is not None else dets):
                    oid = world.match(d["w"][:2]); o = world.objs.get(oid)
                    if o is None or (not (o.get("clearing") and os.environ.get("NERO_LEAN", "0") == "1") and (o["fails"] >= a.max_fail or o["defer_gen"] == gen)):      # 2026-10-14(NERO_LEAN): 치우라고 정해진 물체는 자신의 실패/보류 기록과 상관없이 집는다
                        continue
                    if not eligible(o):
                        continue
                    cands.append((abs(bearing_to_j1(d["w"][0], d["w"][1]) - cur), oid, d))
                if not cands:
                    clean[round(ang, 3)] = True
                    continue
                _, oid, d = min(cands, key=lambda t: t[0])
                warm_cleanup(); STATE["place_plan"] = None
                warm_pick = warm_start("exec_pick.py", oid, dict(NERO_NO_PHYSICS=os.environ.get("NERO_NO_PHYSICS", "0"))) if a.warm else None
                t_found = time.time(); attempts += 1; engaged = True
                set_state(stage="발견", target=oid, timing={}); pub(oid)
                log(f"{oid} 발견: xy=({d['w'][0]:.3f},{d['w'][1]:.3f}) j1={cur:.2f} 화면 여유 {d['margin']}px | 같은 화면의 다른 물체 {len(dets) - 1}개는 장애물로 등록"); _HOOK and PL["trk"].__setitem__("xy", (round(float(d["w"][0]), 3), round(float(d["w"][1]), 3)))
                # 1) 정렬: 물체 방위각으로 j1 을 맞추고 1회 촬영(중간 j7 이라 세로로도 중앙)
                j1t = bearing_to_j1(d["w"][0], d["w"][1])
                if abs(j1t - cur) > 0.03 and _forced is None:
                    set_state(stage="정렬")
                    n.move_j1(j1t, j7_abs=a.j7)
                    r2 = grab(n, a.floor_z, a.target)
                    if r2 is None:
                        bump(world, oid); continue
                    geom2, dets2 = r2
                    m2 = [dd for dd in dets2 if np.hypot(dd["w"][0] - d["w"][0], dd["w"][1] - d["w"][1]) < 0.08]
                    # 2026-10-01 CGN 에 쓰는 정렬 후 프레임으로 충돌 객체(같은 화면의 다른 물체 박스)도 같이 갱신한다. 대상이 다시 안 잡혀도(폴백) 이웃은 이 프레임이 최신이다.
                    world.update(dets2, float(n.got["arm_q"][0])); pub(oid)
                    if m2:
                        geom = geom2
                        d = min(m2, key=lambda dd: np.hypot(dd["w"][0] - d["w"][0], dd["w"][1] - d["w"][1]))
                    else:
                        # 2026-10-01 정렬 뒤 같은 물체가 안 잡히면(화면 아래쪽 그리퍼 제외선 등) 포기하지 않고 정렬 전 프레임(검출 성공)으로 진행한다.
                        # 점군은 base 좌표라 카메라 자세와 무관하다. 등록부에도 이 물체를 되살려 둔다.
                        log(f"{oid}: 정렬 후 재검출 실패 -- 정렬 전 프레임으로 진행")
                        world.objs.setdefault(oid, dict(fails=0, defer_gen=-1, seen=1, xyz=d["box"]["xyz"], raw=d["box"]["raw"], yaw=d["box"]["yaw"], w=[float(x) for x in d["w"]]))
                t_aligned = time.time()
                log(f"{oid} 정렬 완료({t_aligned - t_found:.1f}s): 화면 여유 {d['margin']}px" + ("  [잘림 주의]" if d["margin"] < a.margin_px else ""))
                # 2,3) CGN+B -> C. 후보가 없거나 C 가 막힘 없이 실패하면 j1 을 틀어 다른 시점에서 바로 한 번 더 시도한다(스윕이 한 바퀴 돌아 재방문하길 기다리면 수십 초가 버려진다:
                #      야간 413시행 중 116시행이 첫 시도 실패, 그중 36시행은 재방문 뒤 성공 -- 시점/점군 변동으로 후보가 달라진다).
                def attempt(geom, d):
                    q0 = arm_q_str(n); t_a = time.time()
                    _pf0 = os.environ.get("NERO_PLACE_POSE_LIST_FILE")
                    if os.environ.get("NERO_PERC_EARLY", "0") == "1" and _pf0 and not os.environ.get("NERO_PLACE_POSE_LIST") and not os.path.exists(_pf0 + ".perc.json"):      # 2026-11-11(기본 꺼짐): 인식 상자를 CGN 전에 알림(판단이 CGN 을 기다리지 않게). 값은 place_request 와 같은 식(이 프레임 d)
                        try:
                            _P0 = d["P"]; _b0 = d["box"]; _bz0 = float(max(0.0, min(0.03, _P0[:, 2].min()))); _H0 = max(0.02, float(np.percentile(_P0[:, 2], 98)) - _bz0)
                            json.dump(dict(obj=dict(centre=[_b0["xyz"][0], _b0["xyz"][1], _bz0 + _H0 / 2], base_z=_bz0, height=_H0, dims=[_b0["raw"][0], _b0["raw"][1], _H0], yaw=_b0["yaw"]), oid=oid, t=time.time(), early=True), open(_pf0 + ".perc.json.tmp", "w"), default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_)); os.replace(_pf0 + ".perc.json.tmp", _pf0 + ".perc.json")
                            log(f"{oid}: 인식 상자 조기 알림(CGN 전)")
                        except Exception as e_: log(f"{oid}: 인식 상자 조기 알림 실패 {e_!r}")
                    set_state(stage="CGN+B")
                    body = npz_bytes(geom, d["mask"], oid, [float(x) for x in d["w"]])
                    _npzn = ""
                    if _NPZK:
                        try: os.makedirs(_NPZK, exist_ok=True); _npzn = f"{oid}_{int(time.time() * 1000)}.npz"; open(f"{_NPZK}/{_npzn}", "wb").write(body)
                        except Exception as e_: log(f"npz 보존 실패 {e_!r}")
                    url = f"{a.thor}/plan_upload?name={oid}&arm_q={urllib.parse.quote(q0)}&ik_url={urllib.parse.quote(a.ik_url)}&chunk={os.environ.get('NERO_B_IK_CHUNK', '32')}"
                    if os.environ.get("NERO_REPLAN_REUSE_CB", "0") == "1" and PL.get("cb_geom") is geom and PL.get("cb_d") is d and PL.get("cb_q0") is not None and max(abs(float(x_) - float(y_)) for x_, y_ in zip(PL["cb_q0"].split(), q0.split())) < 2e-3 and PL.get("cb_resp") is not None:      # 2026-11-11(기본 꺼짐, 사용자 승인): 풂 뒤 다시 계획(같은 프레임·같은 팔 자세)은 CGN·B 후보를 재사용하고 C 만 새 조건으로. 시점(각도)이 바뀌면 새 프레임이라 여기 안 걸림 -> CGN·B·C 모두 새로
                        import copy as _cp; resp = _cp.deepcopy(PL["cb_resp"]); log(f"{oid}: [replan] 같은 프레임·같은 팔 자세 -- CGN·B 후보 재사용(C 만 새 조건으로)")
                    else:
                        try:
                            resp = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/octet-stream"}), timeout=60).read())
                        except Exception as e:
                            log(f"{oid}: CGN+B 호출 실패 {e!r}"); return "final", None
                        if os.environ.get("NERO_REPLAN_REUSE_CB", "0") == "1":
                            import copy as _cp; PL["cb_geom"], PL["cb_d"], PL["cb_q0"], PL["cb_resp"] = geom, d, q0, _cp.deepcopy(resp)
                    t_cb = time.time()
                    STATE["timing"] = dict(align=t_aligned - t_found, cgn_ms=resp.get("cgn_ms"), b_ms=resp.get("b_ms"), cgn_b_rtt=t_cb - t_a); flush_state()
                    pick = resp.get("pick"); cl = (pick or {}).get("candidates") or []
                    if os.environ.get("NERO_C_ALL_CGN", "0") == "1" and cl:      # 2026-11-12(기본 꺼짐, 사용자 지시): B 의 8개 상한을 풀어 집기 가능한 CGN 후보 전부를 C 로(원래 8개는 순서 그대로 앞에). cgn_all.py 참고
                        import cgn_all as _CA; _n8 = len(cl); cl, _cai = _CA.expand(oid, q0, cl, os.path.dirname(os.path.abspath(a.out)), a.ik_url)
                        log(f"{oid}: [cgn_all] 후보 {_n8} -> {len(cl)}개 {_cai}")
                    if os.environ.get("NERO_C_TWIST_MAX") and cl:      # 2026-11-12(기본 꺼짐, 사용자 승인): 손가락 닫힘축이 인식 물체의 얇은 수평축과 크게 어긋난 후보 제외 -- 쥐는 순간 물체가 손가락 방향으로 돌아가는데(판 38도 실측) 계획기는 쥔 뒤 자세를 그대로 가정해 기울게 놓는다
                        try:
                            from scipy.spatial.transform import Rotation as _Rt
                            _tm = float(os.environ["NERO_C_TWIST_MAX"]); _bx = d["box"]; _r0, _r1 = float(_bx["raw"][0]), float(_bx["raw"][1]); _yw = float(_bx["yaw"]); _kept = []; _drop = []
                            _thin = np.array([math.cos(_yw), math.sin(_yw)]) if _r0 < _r1 else np.array([-math.sin(_yw), math.cos(_yw)])      # raw[0] = yaw 방향 길이
                            for _c in cl:
                                _cl = _Rt.from_quat(_c["quat"]).as_matrix()[:, 1]; _h = float(np.hypot(_cl[0], _cl[1]))
                                if _h < 0.8 or min(_r0, _r1) / max(_r0, _r1) > 0.75: _kept.append(_c); continue      # 닫힘축이 수평이 아니거나 수평 두 변이 비슷하면 판정 안 함(그대로 둠)
                                _ang = math.degrees(math.acos(min(1.0, abs(float((_cl[:2] / _h) @ _thin)))))
                                (_kept if _ang <= _tm else _drop).append(_c); _c["twist_deg"] = round(_ang, 1)
                            log(f"{oid}: [twist] 닫힘축-얇은축 어긋남 {_tm:.0f}도 초과 후보 {len(_drop)}개 제외 {[(c_.get('kind'), c_['twist_deg']) for c_ in _drop][:8]} -> {len(_kept)}개 (물체 {100 * _r0:.1f}x{100 * _r1:.1f}cm yaw {math.degrees(_yw):.1f})")
                            cl = _kept
                        except Exception as _e: log(f"{oid}: [twist] 판정 실패 {_e!r} -- 후보 그대로")
                    log(f"{oid} CGN {resp.get('cgn_ms')}ms + B {resp.get('b_ms')}ms (왕복 {t_cb - t_a:.2f}s): 후보 {len(cl)}개"); PL["nocand"] = (len(cl) == 0); PL["c_ik_only"] = False; _HOOK and PL["trk"]["cgn"].append(len(cl)); _NPL and nlog(f"cgn oid={oid} cands={len(cl)} cgn_ms={resp.get('cgn_ms')} b_ms={resp.get('b_ms')}")      # 2026-10-15: 파지 후보 자체가 없으면 목적지/홈 경유와 무관한 실패
                    if _TRC:      # 2026-10-23(기본 꺼짐, NERO_C_TRACE 확장): 깔때기 S0~S5 계측 -- CGN 문구(S0 N), B 의 pick 요약(S1 n_cand, S4 n_reach, 제외 수, S5 후보 수), B 출력 마지막 10줄(S2/S3 필터 줄이 들어오는지 확인용), 보존 점군 파일명
                        try: log(f"[funnel] oid={oid} item={(PL['items'][PL['i']]['id'] if PL['items'] and PL['i'] < len(PL['items']) else None)} t={time.time():.3f} npz={_npzn or '-'} cgn_ms={resp.get('cgn_ms')} b_ms={resp.get('b_ms')} cgn={json.dumps(resp.get('cgn'), ensure_ascii=False)} pick={json.dumps({k_: (pick or {}).get(k_) for k_ in ('n_cgn_raw', 'n_cand', 'n_reach', 'n_reach_cgn', 'n_excluded', 'unified_rank')}, ensure_ascii=False)} n_keep={len(cl)} resp_keys={sorted(resp)} b_tail={json.dumps(resp.get('b_tail'), ensure_ascii=False)}")
                        except Exception as e_: log(f"[funnel] 기록 실패 {e_!r}")
                    try:        # RViz 그리퍼 모양 표시용(2026-10-07): 후보 상위 12개를 상태 파일에 내보낸다(접촉점/접근/쿼터니언/열림/점수). 실패해도 루프에는 영향 없음
                        STATE["cands"] = [dict(idx=i_, contact=c_.get("contact"), approach=c_.get("approach"), quat=c_.get("quat"), opening=c_.get("opening"), score=c_.get("score"), cost=c_.get("cost"), kind=c_.get("kind"), twin=c_.get("twin")) for i_, c_ in enumerate(cl[:12])]; flush_state()
                    except Exception:
                        pass
                    _mx = float(os.environ.get("NERO_MAX_APPROACH_DEG", "0"))      # 2026-10-14(기본 꺼짐, NERO_LEAN 이 60 으로 켬): 수평에 가까운 옆 접근 후보(접근각 > 한도, 수직=0°)는 손가락이 물체를 먼저 밀어 넘어뜨린다(Nesquik 88°, 손전등 69° 가 MISSED) -- 한도 안의 후보가 하나라도 있으면 나머지는 버린다
                    if _mx > 0 and cl:
                        def _ang(c_):
                            try: v_ = np.array(c_.get("approach"), float); v_ /= np.linalg.norm(v_); return math.degrees(math.acos(max(-1.0, min(1.0, float(-v_[2])))))
                            except Exception: return 0.0
                        _ok_ = [c_ for c_ in cl if _ang(c_) <= _mx]
                        if _ok_ and len(_ok_) < len(cl): log(f"{oid}: 수평에 가까운 접근 후보 {len(cl) - len(_ok_)}개 제외(접근각 > {_mx:.0f}°, 남은 {len(_ok_)}개, 최소각 {min(_ang(c_) for c_ in cl):.0f}°)"); cl = _ok_
                    if not cl:
                        log(f"{oid}: 도달 가능한 파지 후보 없음 ({str(resp.get('cgn'))[:80]})"); PL["nocand"] = True; _HOOK and PL["trk"]["nocand"].append("(" + str(resp.get("cgn"))[:80] + ")")
                        if os.environ.get("NERO_LEAN", "0") == "1" and _clear_decision(oid, str(resp.get("cgn"))[:80]): return "deferred", None
                        return "retry", None
                    obs_now = world.obstacles(oid)
                    log(f"{oid}: 충돌 객체 {len(obs_now)}개 = 정렬 후 프레임 기준으로 갱신된 등록부({', '.join(o['id'].replace('other_', '') for o in obs_now)})"); _HOOK and PL["trk"]["coll"].append(str(len(obs_now)))
                    _bd = os.environ.get("NERO_GRASP_BAND", "")      # 2026-10-14(기본 꺼짐): 긴 물체를 꽂을 때는 끝단을 손끝으로 집는(핀치) 파지가 아니라 몸통 가운데를 옆에서 감싸 쥐는 후보만 쓴다. 접촉 높이 비율 lo,hi
                    if _bd and cl:
                        try:
                            _lo, _hi = (float(v) for v in _bd.split(",")); _P = d["P"]; _bz = float(max(0.0, min(0.03, _P[:, 2].min()))); _H = max(0.02, float(np.percentile(_P[:, 2], 98)) - _bz)
                            _rr = lambda c_: (float(c_["contact"][2]) - _bz) / _H
                            _ok2 = [c_ for c_ in cl if c_.get("contact") is not None and _lo <= _rr(c_) <= _hi]
                            log(f"{oid}: 파지 높이 대역 {_lo:.2f}~{_hi:.2f} -- 후보 {len(cl)}개 중 {len(_ok2)}개 통과(비율 분포 {sorted(round(_rr(c_), 2) for c_ in cl if c_.get('contact') is not None)})")
                            if _ok2: cl = _ok2
                        except Exception as _e: log(f"파지 높이 대역 적용 실패 {_e!r}")
                    if os.environ.get("NERO_PAT", "0") == "1" and cl:      # 2026-11-03(개발 사본, 기본 꺼짐): place-aware 파지 -- 높이/길이가 큰 물체는 파지 접촉 높이를 띠로 제한하고 놓기 도달 점수순으로 재정렬(삭제 없음, 실패·비대상이면 현행 순위). place_aware.py
                        try:
                            import place_aware as _PAT
                            _dst = [o_ for o_ in world.objs.values() if o_.get("role") == "dest" and "xyz" in o_][:1]
                            _dd_ = (dict(xy=(float(_dst[0]["xyz"][0]), float(_dst[0]["xyz"][1])), top_z=float((_dst[0].get("concave") or {}).get("rim_z", _dst[0]["raw"][2]))) if _dst else None)
                            cl, _pi = _PAT.rerank(cl, _PAT.geom_from_points(d["P"]), _dd_, place_clear=a.place_clear, log=log, oid=oid)
                        except Exception as _e: log(f"{oid}: [PAT] 재정렬 실패 -- 현행 순위 유지 {_e!r}")
                    if os.environ.get("NERO_PLACE_RANK", "0") == "1" and len(cl) > 1:      # 2026-11-04(기본 꺼짐): 확정 모델 p(NO_PLAN) 낮은 순으로 파지 후보 정렬(필터 아님, 순서만; 실패하면 기존 순서). place_rank.py
                        try:
                            import place_rank as _PRK, axis_pca as _AP2
                            _bz = float(max(0.0, min(0.03, d["P"][:, 2].min()))); _H = max(0.02, float(np.percentile(d["P"][:, 2], 98)) - _bz); _pc2 = _AP2.pca_axis(d["P"])
                            _g = dict(centre=[d["box"]["xyz"][0], d["box"]["xyz"][1], _bz + _H / 2], base_z=_bz, height=_H, dims=[d["box"]["raw"][0], d["box"]["raw"][1], _H], elong=_pc2.get("elong"))
                            _dst2 = [o_ for o_ in world.objs.values() if o_.get("role") == "dest" and "xyz" in o_ and "raw" in o_][:1]
                            cl, _pri = _PRK.rerank(cl, _g, (dict(raw=[float(v) for v in _dst2[0]["raw"]], xyz=[float(v) for v in _dst2[0]["xyz"]]) if _dst2 else None), log=log, oid=oid)
                            _RANK_INFO[oid] = _pri
                        except Exception as _e: log(f"{oid}: [PLACE_RANK] 실패 -- 현행 순위 유지 {_e!r}")
                    if os.environ.get("NERO_PLACE_TOPFIRST", "0") == "1" and len(cl) > 1:      # 2026-11-06(기본 꺼짐): 접근이 수직에 가까운(탑다운) 후보를 앞으로(필터 아님, 같은 묶음 안에서는 CGN 순위 유지). 방향 요구가 있는 정밀 놓기는 사이드 파지에서 손목 한계로 놓기 IK 가 막힌다.
                        try:
                            _lim = float(os.environ.get("NERO_PLACE_TOPFIRST_DEG", "30")); _th = [math.degrees(math.acos(max(-1.0, min(1.0, -float(c_["approach"][2]))))) for c_ in cl]
                            _ix = sorted(range(len(cl)), key=lambda i_: (0 if _th[i_] <= _lim else 1, i_)); log(f"{oid}: 탑다운 우선 정렬(접근각 {_lim:.0f}° 이내 먼저): 접근각(도) 원래순 {[round(t_) for t_ in _th[:12]]} -> 순서 {_ix[:12]}"); cl = [cl[i_] for i_ in _ix]
                        except Exception as _e: log(f"{oid}: [TOPFIRST] 실패 -- 현행 순위 유지 {_e!r}")
                    pcheck = None
                    if placing and os.environ.get("NERO_PLACE_AWARE", "0") == "1":      # 파지-place 동시 선택(서버가 후보마다 place 가능 여부를 확인). 기본 OFF: 직렬 지연(후보당 최대 6초)이 크고, 확인된 place 실패는 파지와 무관한 원인(장애물 위 슬롯)이었다. NERO_PLACE_AWARE=1 로 켠다
                        try:
                            _dummy = dict(quat=[0.0, 0.0, 0.0, 1.0], approach=[0.0, 0.0, -1.0], flange=[0.0, 0.0, 0.2], contact=None)
                            _pr_t = place_request(oid, d, _dummy, [float(x) for x in n.got["arm_q"]])[1]
                            if a.insert_at:
                                _rq = to_insert_req(_pr_t)
                                if _rq is None:
                                    log(f"{oid}: 삽입 불가(물체가 길쭉하지 않음) -- 집지 않고 건너뜀"); (world.objs.get(oid) or {}).__setitem__("fails", a.max_fail); return "retry", None
                                pcheck = dict(kind="insert", req=_rq)
                            else:
                                pcheck = dict(kind="place", req=_pr_t)
                            for k_ in ("grasp", "start_q"):
                                pcheck["req"].pop(k_, None)
                        except Exception as e_:
                            log(f"{oid}: place 확인 템플릿 실패 {e_!r} -- 기존 방식(pick 후 계산)"); pcheck = None
                    if os.environ.get("NERO_LOG_TRIAL", "0") == "1":      # 2026-11-04(기본 꺼짐): 후보 풀 전체 기록 + 원 CGN 출력 사본
                        try: _pool_record(oid, cl, int(os.environ.get("NERO_C_MAX_CANDS", "4")))
                        except Exception as _e: log(f"{oid}: [trial 후보 풀 기록 실패] {_e!r}")
                    req = dict(start_q=[float(x) for x in n.got["arm_q"]], candidates=cl, obstacles=obs_now, target=aabb_box(d["P"]),
                               tol=0.0, margin=0.01, max_cands=int(os.environ.get("NERO_C_MAX_CANDS", "4")), budget_s=float(os.environ.get("NERO_RT_C_BUDGET", "2.0")) + (14.0 if pcheck else 0.0))
                    if os.environ.get("NERO_C_ALL_CGN", "0") == "1": req["max_cands"] = max(req["max_cands"], min(len(cl), int(os.environ.get("NERO_C_ALL_CGN_MAX", "64"))))      # 늘린 후보를 C 가 실제로 보게
                    if pcheck:
                        req["place_check"] = pcheck
                    if os.environ.get("NERO_C_PRESCREEN", "0") == "1": req["prescreen"] = True      # 2026-10-22(기본 꺼짐, 서버 재기동 필요): 후보를 체인 IK 로 먼저 선별
                    if os.environ.get("NERO_C_HARD_DEADLINE", "0") == "1": req["hard_deadline"] = True      # 2026-10-22(기본 꺼짐, 서버 재기동 필요): 후보 내부 루프도 예산에서 끊음
                    if os.environ.get("NERO_C_PLACE_PRESCREEN", "0") == "1" and req.get("place_check"): req["place_prescreen"] = True      # 2026-11-06(기본 꺼짐, 복사본 서버 8021 전용): 집기+놓기 목표 IK 배치 사전 선별로 후보 순서만 바꿈(빼지 않음)
                    if os.environ.get("NERO_C_PLACE_PRESCREEN_DROP", "0") == "1" and req.get("place_prescreen"): req["place_prescreen_drop"] = True
                    if os.environ.get("NERO_C_PLACE_PRESCREEN_POSE", "0") == "1" and req.get("place_check"): req["place_prescreen_pose"] = True      # 2026-11-09(기본 꺼짐, 8021 전용): 파지별 놓기 IK 사전 선별(판단 시험과 같은 설정)
                    if os.environ.get("NERO_C_TOTAL_BUDGET") and req.get("place_check"): req["total_budget_s"] = float(os.environ["NERO_C_TOTAL_BUDGET"])      # 2026-11-09: 요청 전체 시간 상한(자세 목록 전체)      # 측정 전용(권장하지 않음): 놓기 IK 0개 후보를 뺀다. IK 는 확률적이라 해를 놓칠 수 있다
                    if _RJL: req["reject_log"] = True
                    if _TRC:
                        _ti = (PL["items"][PL["i"]]["id"] if PL["items"] and PL["i"] < len(PL["items"]) else None); _tc = PL.setdefault("c_n", {}); _tc[(_ti, oid)] = _tc.get((_ti, oid), 0) + 1
                        req["trace"] = dict(oid=oid, item=_ti, call=_tc[(_ti, oid)], run=os.environ.get("NERO_C_TRACE_RUN", ""), t=time.time(), stage=("deferred_round" if (PL["items"] and PL["i"] < len(PL["items"]) and PL["items"][PL["i"]].get("_tried")) else "main"))
                    if _NPL: req["log_extra"] = True; nlog(f"c_req oid={oid} budget_s={req['budget_s']} cands_sent={len(cl)} max_cands={req['max_cands']} prescreen={int(bool(req.get('prescreen')))} hard_deadline={int(bool(req.get('hard_deadline')))} start_j1={req['start_q'][0]:.3f} n_obstacles={len(obs_now)} stage={'deferred_round' if (PL['items'] and PL['i'] < len(PL['items']) and PL['items'][PL['i']].get('_tried')) else 'main'}")
                    set_state(stage="C")
                    if os.environ.get("NERO_C_REQ_DUMP"):      # 분석용(2026-10-08): C 요청 저장 -- 실패를 서버에 그대로 다시 던져 변인(대상 상자/이웃/시작 자세)을 하나씩 바꿔 본다. 기본 꺼짐
                        try: json.dump(req, open(f"{os.environ['NERO_C_REQ_DUMP']}/{oid}_{int(time.time())}.json", "w"), default=lambda o_: o_.tolist() if hasattr(o_, "tolist") else str(o_))
                        except Exception as e_: log(f"C 요청 저장 실패 {e_!r}")
                    try:
                        if PL["items"] and os.environ.get("NERO_CHAIN_DIAG", "0") != "1": req["diagnose"] = False      # 2026-10-16: 체인 모드는 막힘 진단(후보 루프를 이웃 물체 하나씩 빼고 최대 5번 재실행, 실패 한 번에 10초 이상)을 건너뛴다. 막힌 물체를 치우지 않으므로 결과를 안 쓴다
                        cr = post_json(a.curobo + "/plan", req, timeout=max(90.0, float(req["budget_s"]) + 60.0, float(req.get("total_budget_s") or 0.0) + 60.0))      # 2026-11-06: 예산이 길 때(새 설정 134초) 클라이언트가 90초에 먼저 끊던 문제. 기존 예산(16초)은 90초 그대로
                    except Exception as e:
                        log(f"{oid}: cuRobo C 호출 실패 {e!r}"); return "final", None
                    t_c = time.time()
                    _TRC and log(f"[c-trace] oid={oid} item={req['trace']['item']} call={req['trace']['call']} t={req['trace']['t']:.3f} ok={int(bool(cr.get('ok')))} sec={cr.get('sec')} n_cands={len(cl)} stage={req['trace']['stage']}")
                    _RJL and log(f"[c-reject] oid={oid} cand_reject={json.dumps(cr.get('cand_reject'), ensure_ascii=False)}")
                    _NPL and nlog(f"c_res oid={oid} ok={int(bool(cr.get('ok')))} sec={cr.get('sec')} class={_c_why_class(cr.get('why')) if not cr.get('ok') else 'ok'} n_why={len(cr.get('why') or [])} cand_secs={cr.get('cand_secs')} n_used={cr.get('n_cands_used')}")
                    if not cr.get("ok"):
                        blk = [b.replace("other_", "") for b in cr.get("blockers", [])]
                        PL["lastwhy"] = str(cr.get("why"))[:200]; log(f"{oid}: C 실패({cr.get('sec')}s) {cr.get('why')} 막힘 {blk or '없음'}"); _HOOK and (PL["trk"]["c_reasons"].append(str(cr.get("why"))), PL["trk"]["c_secs"].append(float(cr.get("sec") or 0.0)), PL["trk"].__setitem__("blocked", f"막힘 {blk or '없음'}"))
                        PL["pickside"] = not any("place 불가" in str(w_) for w_ in (cr.get("why") or [])); PL["c_ik_only"] = (_c_why_class(cr.get("why")) == "ik_only"); _cw_ = [str(x_) for x_ in (cr.get("why") or []) if str(x_).startswith("cand")]; PL["c_nb_only"] = bool(_cw_) and all("다른 물체가 있어" in x_ for x_ in _cw_)      # 2026-11-11 c_nb_only: 모든 후보가 놓은 물체-이웃 겹침(파지·시점과 무관)
                        # 2026-10-16: pick 쪽 계획 실패면 목적지를 바꿔도 안 풀린다(E129: 목적지 전환 + 재시도로 항목당 20~25초 낭비)
                        # 2026-10-08: 도착지(홀더/바구니)나 이미 놓은 물체는 치워질 일이 없으니 '막던 물체가 치워질 때까지 보류' 대상이 아니다. 자동 모드 180시행 중 5건이 홀더를 막힘으로 보고 막대를 영구 보류해 "더 집을 물체 없음"으로 끝났다.
                        _blk_src = "server" if blk else ""
                        if not blk and os.environ.get("NERO_LEAN", "0") == "1" and oid in world.objs:
                            blk = _infer_blockers(oid); _blk_src = "infer"
                            if blk: log(f"{oid}: 서버는 막힘을 못 짚음 -- 등록부(깊이 마스크 위치/크기)로 막은 물체 추정: {[(b, str(world.objs[b].get('label'))) for b in blk]}")
                        if not blk and llm_decide.enabled() and oid in world.objs and "xyz" in world.objs[oid]:      # 2026-10-14: LLM 모드는 서버/추정 모두 비어도 대상 15cm 이내의 치울 수 있는 물체를 선택지로 올린다(막은 물체 판단은 Claude 가 한다)
                            blk = [k_ for k_ in world.objs if k_ != oid and "xyz" in world.objs[k_] and world.objs[k_].get("role") not in ("dest", "placed") and _clearable(k_) and math.hypot(world.objs[k_]["xyz"][0] - world.objs[oid]["xyz"][0], world.objs[k_]["xyz"][1] - world.objs[oid]["xyz"][1]) < 0.15]; _blk_src = "near"
                        _blk_all = list(blk); blk = [b for b in blk if (world.objs.get(b) or {}).get("role") not in ("dest", "placed")] if os.environ.get("NERO_BLK_SKIP_DEST", "1") == "1" else blk
                        if _blk_all and not blk: log(f"{oid}: 막힘이 모두 도착지/놓은 물체({_blk_all}) -- 보류하지 않고 재시도로 처리")
                        if blk and (os.environ.get("NERO_CLEAR_BLOCKER", "0") == "1" or llm_decide.enabled()) and CLEAR["used"] < int(os.environ.get("NERO_CLEAR_MAX", "2")):      # 2026-10-14(기본 꺼짐): 막은 물체가 치울 수 있는 크기면 먼저 집어서 빈 자리에 놓고 원래 대상을 다시 시도한다
                            _CMD_SET.add(_cmd_target(oid)); _cl = [b for b in blk if _clearable(b)]
                            if _cl:
                                _bt = min(_cl, key=lambda b: math.hypot(world.objs[b]["xyz"][0] - world.objs[oid]["xyz"][0], world.objs[b]["xyz"][1] - world.objs[oid]["xyz"][1])) if oid in world.objs else _cl[0]
                                if llm_decide.enabled():      # 2026-10-14: 무엇을 치울지 Claude 가 고른다(블로커 외에 대상 근처의 치울 수 있는 물체도 선택지로 준다). 'defer' = 치우지 않고 대상을 보류
                                    _near = [k_ for k_ in world.objs if k_ not in _cl and k_ != oid and "xyz" in world.objs[k_] and _clearable(k_) and math.hypot(world.objs[k_]["xyz"][0] - world.objs[oid]["xyz"][0], world.objs[k_]["xyz"][1] - world.objs[oid]["xyz"][1]) < 0.15]
                                    _opts = [dict(id=k_, text=f"{world.objs[k_].get('label_disp') or world.objs[k_].get('label')} 를 먼저 집어 빈 자리에 치운 뒤 대상을 다시 시도" + (({"server": " (계획 서버가 막힘으로 보고)", "infer": " (서버는 막힘을 못 짚었으나 깊이 마스크 기준 대상 가장자리 6cm 이내)", "near": " (대상 15cm 이내)"}.get(_blk_src, "")) if k_ in _cl else " (막힘 보고는 없지만 대상 근처)")) for k_ in _cl + _near] + [dict(id="defer", text="아무것도 치우지 않고 이 대상은 보류(다른 물체 먼저, 없으면 종료)")]
                                    _ch, _src, _why = llm_decide.ask("clear_blocker", f"{oid} 를 집는 계획이 막혔다(C 실패: {cr.get('why')}). 무엇을 치울까?", dict(table=_llm_table(oid, oid, blk), 치운횟수=f"{CLEAR['used']}/{os.environ.get('NERO_CLEAR_MAX', '2')}", 명령=f"대상 {a.target if hasattr(a, 'target') else ''} -> 목적지 {a.place_on}"), _opts, _bt, log=log)
                                    if _ch == "defer": _cl = []
                                    else: _bt = _ch
                            if _cl or (llm_decide.enabled() and False):
                                _bt = _bt
                                world.objs[_bt]["clearing"] = True; world.objs[_bt]["clear_for"] = oid; CLEAR["used"] += 1
                                log(f"{oid}: 막은 물체 {_bt}({world.objs[_bt].get('label')}) 를 먼저 치운다({CLEAR['used']}/{os.environ.get('NERO_CLEAR_MAX', '2')}) -- 치울 수 있는 후보 {_cl}")
                            elif os.environ.get("NERO_LEAN", "0") == "1" and _clear_decision(oid, str(cr.get("why"))[:80]):      # 2026-10-14: 서버가 지목한 물체가 치울 수 없는 것(막대 등)뿐이어도 가까운 치울 수 있는 이웃을 같은 방식으로 정한다 -- 두 번째 치우기가 느리던 원인(그냥 보류 -> 360도 재탐색)
                                blk = [oid]
                            else: log(f"{oid}: 막은 물체 {blk} 는 치울 수 없는 것(고정물/큰 물체/도착지) -- 치우지 않고 보류")
                        if blk:
                            (world.objs.get(oid) or {}).__setitem__("defer_gen", gen); return "deferred", None        # 막던 물체가 먼저 치워질 때까지 보류
                        if os.environ.get("NERO_LEAN", "0") == "1": return "planfail", None      # 막힘이 없는 C 실패 = 계획이 안 되는 자세/이웃 배치. 시점을 바꿔도 달라지지 않는다
                        return "retry", None
                    if cr.get("plan") is not None and cr["plan"].get("place_plan"):
                        log(f"{oid}: 파지-place 동시 선택 -- 후보 {cr['plan'].get('cand_index')} 로 pick 과 place 모두 계획됨 ({cr['plan'].get('place_kind')})")
                    return "ok", dict(plan=cr["plan"], cr=cr, resp=resp, t_cb=t_cb, t_c=t_c, t_a=t_a)

                res = None; last = "retry"; _c1_nc = []
                for _rp in range(1 + (int(os.environ.get("NERO_REPLAN_MAX", "4")) if os.environ.get("NERO_REPLAN_LOOP", "0") == "1" else 0)):      # 2026-11-11(NERO_REPLAN_LOOP, 기본 꺼짐 -> 한 번만 돔 = 기존과 같음): C 실패 시 판단 쪽에 새 자세 목록을 받아 같은 인식(같은 프레임)으로 다시 계획
                    for vt in range(1 + a.retry_views):
                        if vt > 0:
                            off = (0.30, -0.30, 0.55, -0.55)[(vt - 1) % 4]
                            log(f"{oid}: 시점을 바꿔 즉시 재시도 {vt}/{a.retry_views} (j1 {off:+.2f}rad)"); _HOOK and PL["trk"].__setitem__("views", PL["trk"]["views"] + 1)
                            set_state(stage="재조준")
                            n.move_j1(float(np.clip(j1t + off, -J1_LIMIT + 0.05, J1_LIMIT - 0.05)), j7_abs=a.j7)
                            rr = grab(n, a.floor_z, a.target)
                            if rr is None:
                                break
                            geom_r, dets_r = rr
                            mm = [dd for dd in dets_r if np.hypot(dd["w"][0] - d["w"][0], dd["w"][1] - d["w"][1]) < 0.10]
                            world.update(dets_r, float(n.got["arm_q"][0])); pub(oid)
                            if not mm:
                                log(f"{oid}: 재조준 시점에서 물체가 안 보임"); continue
                            geom = geom_r; d = min(mm, key=lambda dd: np.hypot(dd["w"][0] - d["w"][0], dd["w"][1] - d["w"][1]))
                        PL["c_nb_only"] = False
                        last, res = attempt(geom, d); _C1 and _c1_nc.append(bool(PL.get("nocand")) and last == "retry")
                        if last == "ok":
                            break
                        if last == "retry" and PL.get("c_nb_only") and os.environ.get("NERO_NB_NORETRY", "0") == "1":      # 2026-11-11(기본 꺼짐): 실패 이유가 전부 '놓은 물체가 이웃과 겹침'이면 시점을 바꿔도 같은 결과(87경우 기록 9/9 재시도로 안 풀림) -> 시점 재시도 생략
                            log(f"{oid}: C 실패 이유가 모두 이웃 겹침 -- 시점 재시도 생략(NERO_NB_NORETRY)"); break
                        if last in ("final", "deferred", "planfail"):
                            break
                    if not (res is None and last == "retry" and os.environ.get("NERO_REPLAN_LOOP", "0") == "1" and _rp < int(os.environ.get("NERO_REPLAN_MAX", "4"))): break
                    if not _replan_wait(oid, _rp + 1): break
                if _c1_on() and res is None and last == "retry" and len(_c1_nc) == 1 + a.retry_views and all(_c1_nc) and oid in world.objs:      # 2026-10-30(NERO_C1_FASTFAIL, 기본 꺼짐): 한 패스(최초 + 시점 재시도) 모두 파지 후보 없음 -> 두 번째 패스 없이 포기(아래 bump 로 fails 가 max_fail 에 닿는다)
                    world.objs[oid]["fails"] = max(int(world.objs[oid].get("fails", 0)), a.max_fail - 1); PL["c1"] = True
                    log(f"[c1] item={PL['items'][PL['i']]['id'] if PL['items'] and PL['i'] < len(PL['items']) else '-'} oid={oid} reason=nocand_pass1 views={len(_c1_nc)} second_pass=skip tried={int(bool(PL['items'] and PL['i'] < len(PL['items']) and PL['items'][PL['i']].get('_tried')))}")
                if res is None and last == "planfail":      # 2026-10-14(NERO_LEAN): 도달 가능한 후보가 없으면 그 물체를 포기하고, 다른 물체가 있으면 거기로, 없으면 다시 훑지 않고 끝낸다
                    if oid in world.objs: world.objs[oid]["fails"] = a.max_fail
                    _oth = [k_ for k_, o_ in world.objs.items() if k_ != oid and o_.get("role") not in ("dest", "placed") and o_["fails"] < a.max_fail and eligible(o_)]
                    if _oth or any(o_.get("clearing") for o_ in world.objs.values()):
                        log(f"{oid}: 도달 가능한 후보 없음(막힘 없음) -- 시점 재시도/재훑기 없이 포기하고 다음 물체로"); engaged = True; attempts += 1; break
                    log(f"{oid}: 도달 가능한 후보 없음(막힘 없음), 다른 물체 없음 -- 다시 훑지 않고 종료"); break
                if res is None and last == "deferred" and os.environ.get("NERO_BLOCKED_FREEZE", "0") == "1":      # 2026-10-14(기본 꺼짐): 막힘으로 보류됐으면 남은 각도를 훑지 않는다(실측: C 실패 뒤 한 바퀴 스윕 20~40초 낭비). 치울 물체가 정해졌으면 그 방위각으로 바로, 아니면 다른 물체가 없을 때 즉시 종료
                    _clr_now = any(o_.get("clearing") for o_ in world.objs.values())
                    _oth = [k_ for k_, o_ in world.objs.items() if k_ != oid and o_.get("role") not in ("dest", "placed") and o_["fails"] < a.max_fail and o_["defer_gen"] != gen and eligible(o_)]
                    if _clr_now or _oth:
                        log(f"{oid}: 막힘으로 보류 -- 남은 각도를 훑지 않고 {'막은 물체 치우기' if _clr_now else '다른 물체'}로 바로 이동"); engaged = True; attempts += 1; break
                    log(f"{oid}: 막힘으로 보류, 치울/다른 물체 없음 -- 스윕 없이 종료"); break
                if res is None:
                    if last != "deferred":
                        bump(world, oid)
                        # 2026-10-08: 한 바퀴(시점 바꿔 재시도 포함) 모두 실패한 물체는 다른 물체를 하나 집을 때까지 보류한다(이웃이 치워지면 풀리는 경우가 과거 로그에서 57%).
                        # 불가능 물체에 같은 자리에서 연달아 시간(C 실패 한 번 3.9~5s)을 쓰는 것을 막는다. 남은 대상이 이 물체뿐이면 보류하지 않는다. 끄기: NERO_RT_DEFER_EXHAUSTED=0
                        if last == "retry" and os.environ.get("NERO_RT_DEFER_EXHAUSTED", "1") != "0" and oid in world.objs:
                            _others = [k_ for k_, o_ in world.objs.items() if k_ != oid and o_.get("role") not in ("dest", "placed") and o_["fails"] < a.max_fail and o_["defer_gen"] != gen and eligible(o_)]
                            if _others:
                                world.objs[oid]["defer_gen"] = gen; log(f"{oid}: 한 바퀴 모두 실패 -- 다른 물체({len(_others)}개) 먼저 하고 보류")
                    continue
                plan, cr, resp, t_cb, t_c = res["plan"], res["cr"], res["resp"], res["t_cb"], res["t_c"]
                STATE["timing"].update(c_s=cr.get("sec"), ready=t_c - t_found); STATE["plan"] = dict(oid=oid, ee_path=ee_path(plan), grasp=plan.get("flange"), approach=plan.get("approach"), cand=plan.get("cand_index"), pre_used=plan.get("pre_used"), lift_used=plan.get("lift_used"), n_wp={k: len(plan.get(k, [])) for k in ("transit", "advance", "lift")}); flush_state()
                log(f"{oid} C 완료 {cr.get('sec')}s (후보 {plan.get('cand_index')}) | 발견→pick 준비 {t_c - t_found:.2f}s = 정렬 {t_aligned - t_found:.1f} + CGN·B {t_cb - res['t_a']:.2f} + C {t_c - t_cb:.2f}")
                top = dict(plan); top["plans"] = [plan]
                try:
                    os.makedirs(os.path.expanduser("~/grasp"), exist_ok=True)
                    json.dump(top, open(os.path.expanduser(f"~/grasp/{oid}_step6_pick.json"), "w"))     # 2026-10-01 PC 에 직접 저장(Thor 왕복 생략)
                except Exception as e:
                    log(f"{oid}: 계획 저장 실패 {e!r}"); continue
                ev = dict(id=oid, t_found=t_found - t_run, aligned=t_aligned - t_found, cgn_b=t_cb - res["t_a"], c=t_c - t_cb, ready=t_c - t_found, views=vt + 1)
                if a.no_exec:
                    events.append(ev); bump(world, oid); continue
                # 4) 실행 -- 현재 자세(정렬 자세)에서 곧바로(chain 모드: 홈 복귀 없음)
                disc = f"/tmp/rt_disc_{oid}.json"
                json.dump({"boxes": {oid: {"world": [float(x) for x in d["w"]]}}, "precomputed": {oid: True}}, open(disc, "w"))
                env = dict(os.environ); env["NERO_CHAIN_PLAN_TAG"] = "_rt"; env["NERO_CHAIN_START"] = arm_q_str(n)
                env["NERO_RT_LOCAL"] = "1"; env["NERO_RT_NO_HOME"] = "1"; env["EXEC_LOCAL_PLAN"] = "1"
                snap_a = None; pre = {}
                if placing:
                    env["SIM_REMOVE"] = "0"          # place 할 물체는 시뮬이 치우지 않는다(손에 쥔 채 이어서 놓는다)
                    env["NERO_RT_PLACE"] = "1"; env["SNAP"] = "0"; env["NERO_GT_HOLD"] = "0"; env["NERO_GT_VERDICT"] = "0"     # pick 직후 검사(유지 대기/스냅샷)로 place 가 늦어지지 않게: 규칙 판정으로 바로 진행
                    pre["thread"] = threading.Thread(target=plan_place_bg, args=(oid, d, plan, pre), daemon=True); pre["thread"].start()      # pick 이 실행되는 동안 place 계획을 미리 계산
                    if not a.no_place_gt:
                        snap_a_box = {}
                        def _snap_a(box=snap_a_box):
                            try:
                                subprocess.run([sys.executable, os.path.join(HERE, "snap_pose.py"), f"/tmp/snap_pose_{oid}_a.json"], cwd=HERE, capture_output=True, timeout=40)
                                box["a"] = json.load(open(f"/tmp/snap_pose_{oid}_a.json"))
                            except Exception as e:
                                log(f"pick 전 스냅샷 실패 {e!r}")
                        th_a = threading.Thread(target=_snap_a, daemon=True); th_a.start()     # 물체를 건드리기 전(pick 접근 중)에 병렬로 찍는다
                set_state(stage="pick")
                if not placing:
                    STATE["place_plan"] = None
                warm_place = None
                if warm_pick is not None and warm_pick["p"].poll() is None:
                    import shutil
                    shutil.copy(os.path.expanduser(f"~/grasp/{oid}_step6_pick.json"), os.path.expanduser("~/grasp/step6_pick.json"))
                    env["NERO_WARM_EXEC"] = "1"; warm_go(warm_pick); warm_finish_pick(warm_pick, oid)      # 신호 -> 즉시 출발
                    if placing:
                        warm_place = warm_start("exec_place.py", oid)                                       # place 실행기도 pick 중에 미리 띄운다
                log(f"{oid}: pick 시작" + (" (예열 실행기: 신호 즉시 출발)" if env.get("NERO_WARM_EXEC") == "1" else ""))
                p = subprocess.Popen([a.pick_script, disc, oid, "0"], cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                removed_ok = False; t_pick0 = time.time(); _tl = {}
                for line in p.stdout:
                    print(line.rstrip(), file=sys.stderr, flush=True)
                    for _k in ("exec pick", "node-init", "grip-open", "p0-transit", "p0-advance", "grip-close", "lift", "grip-hold"):
                        if _k in line and _k not in _tl:
                            _tl[_k] = round(time.time() - t_pick0, 2)
                    if "치움: removed" in line:
                        removed_ok = True
                p.wait(); t_pick_end = time.time()
                log(f"{oid}: pick 시간표(pick_one 시작 기준 초): {_tl} | C 완료->pick_one 시작 {t_pick0 - t_c:.2f}s")
                verdict = "?"
                try:
                    verdict = json.load(open(f"/tmp/{oid}.exec_res.json")).get("verdict")
                except Exception:
                    pass
                ev["verdict"] = verdict; events.append(ev)
                STATE["picked"] = STATE.get("picked", []) + [dict(id=oid, verdict=verdict, t=round(time.time() - t_run, 1), label=world.objs.get(oid, {}).get("label", ""))]; flush_state()
                _pk_idx = len(STATE['picked']) - 1
                prec = None
                _slip_stop = os.environ.get("NERO_SLIP_STOP", "0") == "1" and verdict == "SLIPPED" and not (world.objs.get(oid) or {}).get("clearing")      # 2026-10-14(기본 꺼짐): 미끄러짐 판정이면 빈손으로 놓기를 하지 않고 그 자리에 놓은 뒤 LLM 에게 결과를 돌려준다
                if _slip_stop: log(f"{oid}: 파지 판정 SLIPPED -- 놓기를 하지 않고 결과를 돌려준다(재시도 여부는 LLM 이 결정)")
                if placing and verdict in ("HELD", "SLIPPED") and not _slip_stop:          # 규칙 판정(그리퍼 폭)으로 바로 진행: SLIPPED 는 경계값 거짓 판정이 잦다. 실제 안착은 place 정답이 가린다.
                    if not a.no_place_gt:
                        try:
                            th_a.join(timeout=15); snap_a = snap_a_box.get("a")
                        except NameError:
                            snap_a = None
                    prec = do_place(oid, d, plan, snap_a, pre, t_pick_end, warm_place)
                    if prec.get("verdict") in ("NO_PLAN", "NO_SLOT", "EXEC_FAIL", "HELD_STILL"):
                        log(f"{oid}: place 불가/실패({prec.get('verdict')}) -- 그 자리에서 놓고 계속")
                        try:
                            subprocess.run([sys.executable, os.path.join(HERE, "release_here.py")], cwd=HERE, timeout=30, capture_output=True)
                        except Exception as e:
                            log(f"release 실패 {e!r}")
                    ev["place"] = prec.get("verdict")
                    if a.recover and prec.get("placed_id") and recover_n < a.max_recover and prec.get("verdict") not in ("NO_PLAN", "NO_SLOT", "EXEC_FAIL", "HELD_STILL"):
                        try:
                            t_r0 = time.time(); n.fresh_frame(); gr_ = n.capture_geom()
                            ob_ = prec.get("obj") or {}; dd_ = prec.get("dest"); sl_ = (prec.get("server") or {}).get("slot")
                            as_ = recover_ops.assess(gr_["pts_base"], gr_["zf"], dd_, sl_, ob_.get("height", 0.1), floor_z=a.floor_z) if sl_ else dict(status="unknown")
                            prec["vision_status"] = as_["status"]; prec["vision_info"] = as_.get("info"); prec["vision_sec"] = round(time.time() - t_r0, 2)
                            log(f"{oid}: 놓은 뒤 비전 판정 {as_['status']} {as_.get('info')} (GT {prec.get('verdict')}, {prec['vision_sec']}s)")
                            if as_["status"] in ("toppled", "off_dest") and as_.get("P") is not None:
                                world.objs.pop(prec["placed_id"], None); recover_n += 1
                                forced_retry = dict(oid=f"{oid}_rt{recover_n}", geom=gr_, det=recover_ops.make_det(as_, obb_box), label=(world.objs.get(oid) or {}).get("label", ""))
                                log(f"{oid}: 국소 재시도 예약 -- 스윕 없이 현재 자세에서 바로 다시 집는다")
                        except Exception as e:
                            log(f"{oid}: 놓은 뒤 비전 판정 실패 {e!r}")
                    try:
                        STATE["picked"][_pk_idx]["place"] = prec.get("verdict"); STATE["picked"][_pk_idx]["bumped"] = [f"{b_['prim']} {b_['moved_cm']}cm" for b_ in ((prec.get("gt") or {}).get("bumped") or [])]; flush_state()
                    except Exception:
                        pass
                elif placing and (verdict not in ("HELD", "SLIPPED") or _slip_stop):
                    try:
                        subprocess.run([sys.executable, os.path.join(HERE, "release_here.py")], cwd=HERE, timeout=30, capture_output=True)
                    except Exception as e:
                        log(f"release 실패 {e!r}")
                if verdict in ("HELD", "SLIPPED") and not removed_ok and not placing:
                    # 2026-10-01 시뮬이 집은 물체를 치우지 못했다(이름 필터 등) -> 물체를 쥔 채 스캔하면 시야가 가려지므로 그 자리에서 놓는다.
                    log(f"{oid}: 시뮬 치움 실패 -- 그 자리에서 물체를 놓고 계속(쥔 채 스캔 방지)")
                    try:
                        subprocess.run([sys.executable, os.path.join(HERE, "release_here.py")], cwd=HERE, timeout=30, capture_output=True)
                    except Exception as e:
                        log(f"release 실패 {e!r}")
                if verdict in ("HELD", "SLIPPED"):
                    _was_clr = bool((world.objs.get(oid) or {}).get("clearing"))
                    if _was_clr: CLEAR["done"] += 1; gen += 1; world.objs.pop(oid, None); log(f"{oid}: 막은 물체 치우기 완료 verdict={verdict} -- 원래 대상을 다시 시도(집은 개수에는 안 센다)")
                    else:
                        picks_ok += 1; gen += 1
                        world.objs.pop(oid, None)
                        log(f"{oid}: pick 완료 verdict={verdict} (누적 {picks_ok})")
                        if PL["items"]: _chain_done(oid, verdict, STATE["picked"][_pk_idx])
                else:
                    bump(world, oid)
                    if oid in world.objs and os.environ.get("NERO_STALE_ON_FAIL", "1") == "1": world.objs[oid]["stale"] = True      # 다시 보이기 전까지 장애물 제외
                    log(f"{oid}: pick 실패 verdict={verdict}")
                # 2026-10-01 복귀 대기 제거: pick_one.sh 가 홈 복귀를 하지 않으므로(NERO_RT_NO_HOME) 다음 move_j1 이 곧바로 탐색 자세로 간다.
                # 집은 물체 주변(이웃이 밀렸을 수 있음)과 보류된 물체가 있던 각도만 다시 훑게 표시를 지운다.
                for aa in list(clean):
                    if abs(aa - cur) < 1.0:
                        clean[aa] = False
                for o2 in world.objs.values():
                    if o2.get("defer_gen", -1) >= 0 and "w" in o2:
                        jb = bearing_to_j1(o2["w"][0], o2["w"][1])
                        for aa in list(clean):
                            if abs(aa - jb) < 0.9:
                                clean[aa] = False
                order = order_from(cur)
                break
        except Exception:
            traceback.print_exc()
            log("예외 발생 -- 이 물체는 건너뛰고 다음 바퀴로 계속"); engaged = True; attempts += 1; time.sleep(1.0)
        if not engaged and os.environ.get("NERO_LEAN", "0") == "1" and any(o_.get("clearing") for o_ in world.objs.values()):
            for k_, o_ in list(world.objs.items()):
                if o_.get("clearing"): o_.pop("clearing", None); log(f"{k_}: 치울 물체를 그 방위에서 못 찾음 -- 치우기를 취소하고 보류했던 대상을 다시 시도")
            gen += 1; attempts += 1
            continue
        if not engaged:
            _dfr = [k_ for k_, o_ in world.objs.items() if o_.get("defer_gen") == gen and o_["fails"] < a.max_fail and o_.get("role") not in ("dest", "placed")]
            if os.environ.get("NERO_CLEAR_RETRY", "0") == "1" and _dfr and CLEAR["used"] > 0 and not any(o_.get("clearing") for o_ in world.objs.values()) and CLEAR.get("retry", 0) < 1:      # 2026-10-14(기본 꺼짐): 막은 물체를 치우려다 그 물체가 안 보여 사라졌으면(치우기 불발) 보류했던 원래 대상을 한 번 다시 시도한다 -- 지금까지는 그대로 종료했다
                CLEAR["retry"] = CLEAR.get("retry", 0) + 1; gen += 1
                for aa in list(clean): clean[aa] = False
                log(f"막은 물체를 치우지 못함(안 보임/사라짐) -- 보류했던 {_dfr} 를 한 번 다시 시도")
                attempts += 1; continue
            if PL["items"] and PL["i"] < len(PL["items"]) and not PL["paused"]:      # 체인: 현재 항목의 대상을 못 집었다 -> 홈 복귀/재시작 없이 그 자리에서 다음 항목으로
                _seen = any(eligible(o_) for o_ in world.objs.values())
                _cur = PL["items"][PL["i"]]
                if _seen and PL.get("pickside") and os.environ.get("NERO_CHAIN_PICKSIDE_NOALT", "1") == "1": _cur["_pickside"] = True
                if _seen and PL.get("nocand"): _cur["_nocand"] = True; log(f"[chain] {_cur['id']} 파지 후보 자체가 없음 -- 목적지 후보 전환/홈 경유로 달라지지 않으므로 바로 다음 항목으로(끝에서 한 번만 재시도)")
                if _ATT_CAP > 0:      # 2026-10-30(NERO_ITEM_ATTEMPT_CAP): 이 항목의 실패 패스 수 +1, 상한이면 홈 재시도·목적지 후보 교체를 건너뛴다(아래 조건·_chain_advance)
                    _cur["_fails"] = _cur.get("_fails", 0) + 1
                    if _att_capped(_cur): log(f"[cap] item={_cur['id']} dest={_cur['dest']} fails={_cur['_fails']} cap={'final' if _cur.get('_tried') else _ATT_CAP} stage={'deferred_round' if _cur.get('_tried') else 'main'} -> 홈 재시도·후보 교체 생략")
                _hk = None
                if _HOOK and _seen: _hk = _hook_decide(_cur)      # 2026-10-22(기본 꺼짐): LLM 이 포기/미룸/목적지 전환/시점 변경 중 고른다. None 이면 규칙 현행 동작
                if _hk is not None and _hk["act"] == "giveup": _cur["_llm_nodefer"] = True
                if _hk is not None and _hk["act"] == "view":
                    _off = (0.30, -0.30, 0.55, -0.55)[PL["view_n"].get(_cur["id"], 0) % 4]; PL["view_n"][_cur["id"]] = PL["view_n"].get(_cur["id"], 0) + 1; PL["view_off"] = _off
                    _cur["_homed"] = _cur["dest"]; log(f"[chain] {_cur['id']} LLM 판단: 시점 변경{'(다음 후보 대체)' if _hk['degraded'] else ''} -- 홈 자세를 거쳐 j1 {_off:+.2f}rad 에서 다시 관측")
                    try: n.home()
                    except Exception as e_: log(f"[chain] 홈 복귀 실패: {e_}")
                    _chain_switch(_cur); _chain_local_find(_cur); _chain_save(); attempts += 1; continue
                _NPL and nlog(f"chain_fail item={_cur['id']} dest={_cur['dest']} seen={int(_seen)} nocand={int(bool(_cur.get('_nocand')))} c_ik_only={int(bool(PL.get('c_ik_only')))} homed={int(_cur.get('_homed') == _cur['dest'])} stage={'deferred_round' if _cur.get('_tried') else 'main'} acc_s={PL.get('acc', {}).get(_cur['id'], 0.0) + time.time() - PL.get('t0', t_run):.1f}")
                if _seen and not _cur.get("_nocand") and os.environ.get("NERO_CHAIN_HOME_RETRY", "1") == "1" and _cur.get("_homed") != _cur["dest"] and os.environ.get("NERO_CHAIN_HOME_RETRY_REASON", "0") == "1" and PL.get("c_ik_only"): nlog(f"skip_home_retry item={_cur['id']} reason=ik_only")      # 2026-10-22(기본 꺼짐): 연속 IK 전용 거부는 시작 자세와 무관하므로 홈 경유 재시도를 생략
                if _seen and not _cur.get("_nocand") and os.environ.get("NERO_CHAIN_HOME_RETRY", "1") == "1" and _cur.get("_homed") != _cur["dest"] and not _att_capped(_cur) and not (os.environ.get("NERO_CHAIN_HOME_RETRY_REASON", "0") == "1" and PL.get("c_ik_only")) and not (_hk is not None and _hk["act"] in ("giveup", "defer", "alt")):      # 2026-10-15: 용기 쪽에서 막 놓고 반대편으로 크게 도는 시작 자세 때문에 접근 계획이 전멸하는 경우(obj_8: 홈에서는 34초에 성공) -- 홈(관측) 자세를 거쳐 같은 항목을 한 번 더
                    _cur["_homed"] = _cur["dest"]; log(f"[chain] {_cur['id']} 계획 실패 -- 홈 자세를 거쳐 같은 항목을 한 번 더 시도")
                    try: n.home()
                    except Exception as e_: log(f"[chain] 홈 복귀 실패: {e_}")
                    _chain_switch(_cur); _chain_local_find(_cur); _chain_save(); attempts += 1; continue
                log(f"[chain] {PL['items'][PL['i']]['id']} {'계획 실패' if _seen else '대상을 못 찾음'} -- 바로 다음 항목으로")
                if _chain_advance("planfail" if _seen else "not_found"): attempts += 1; continue
            log("한 바퀴 훑는 동안 더 집을 물체 없음 -- 종료")
            break
    warm_cleanup(only_unused=False)
    set_state(stage="종료", target=None, plan=None); pub()
    json.dump(dict(events=events, picks_ok=picks_ok, total_s=time.time() - t_run), open(a.out, "w"), indent=1)
    reg_save("종료")
    log(f"종료: pick 성공 {picks_ok}, 시도 {attempts}, 총 {time.time() - t_run:.0f}s")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
