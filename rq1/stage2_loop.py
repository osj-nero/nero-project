"""RQ1 2단계 연결(2026-11-09): Gemma 가 쓴 조건(면/yaw/위치) -> 플래너가 그 조건 안에서만 풀기 -> 실행 -> 조건 지켰나 자동 확인.
조건 안에서 계획이 하나도 안 서면 Gemma 에게 '무엇을 시도했고 왜 안 됐는지' 알려 주고 조건을 다시 쓰게 함(풀기 또는 포기), 최대 MAX_ROUND 번.
1단계(조건이 명령 뜻에 맞나)는 사람이 채점(GRADING_*.md). 이 스크립트는 2단계 + 다시 묻기만.
조건 -> 플래너 번역:
 - 면 조건(up/vertical/not_down): 안정 자세 6개 중 조건 만족하는 것만 (NERO_PLACE_OBJ_ROT)
 - yaw 조건: 서버 yaw 후보(30도 간격) 중 최종 긴 축이 조건 만족하는 것만 (NERO_PLACE_YAWS). 없으면 서버 기본(전부)
 - region: 작업대 위 구역 점(--place-xy). 없으면 기존 목적지 슬롯
사용: python3 stage2_loop.py test [items=a,b] [scenes=bar_y0,...]
결과: stage2/episodes.jsonl, stage2/runs/<key>/..."""
import os, sys, json, re, time, subprocess
import numpy as np
from scipy.spatial.transform import Rotation as Rot
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "contplace")); sys.path.insert(0, H)
from constraints import check, GRID_DIMS, CENTER
import exec_main as EM
PR = json.load(open(os.path.join(H, "prereg", "RQ1_PREREG_v3.json"))); ITEMS = {i["id"]: i for i in PR["items"]}
from place_spec_core import ORDER, YAWS, SPOTS      # 2026-11-10 공통 모듈로
MAX_ROUND = 3
EM.ENV["NERO_PLACE_HAND_REJECT"] = "1"      # 2026-11-09 v3: 카메라/손바닥/link7 이 작업대 속으로 들어가는 놓기 후보 제외(실측: 카메라 -26~-28mm 파고듦 -> 덜 내려가 3cm 위에서 놓침)
EM.ENV["NERO_PLACE_DESC_DEST"] = os.environ.get("NERO_PLACE_DESC_DEST", "strict")
EM.ENV["NERO_C_PLACE_PRESCREEN_POSE"] = "1"      # 2026-11-09: 판단 시험과 같게 파지별 놓기 IK 사전 선별(검증: 18조건 잘못 거름 0)
EM.ENV["NERO_C_TOTAL_BUDGET"] = os.environ.get("NERO_C_TOTAL_BUDGET", "180")      # 요청 전체 시간 상한(판 해가 114~126s 에 나와 90s 부족)      # 2026-11-09 v4: 하강을 받침대 포함 월드로 계획(옮기다 물체가 받침대에 걸림 -- 사용자 보고). strict = 안 되면 그 후보 버림
OUT = os.path.join(H, "stage2"); os.makedirs(OUT + "/runs", exist_ok=True)
MEMO_F = OUT + "/plan_memo.json"; MEMO = json.load(open(MEMO_F)) if os.path.exists(MEMO_F) else {}


FACE_NAMES = ("윗면", "바닥면", "앞면", "뒷면", "왼면", "오른면")      # = constraints.FACES 이름(Thor/PC 같음)


class BadSpec(ValueError): pass


def bad_faces(sp):
    """2026-11-11: Gemma 가 조건 언어에 없는 면 이름(예 'PARTS 표시 면')을 내면 서버 /candidates 가 KeyError(500)로 죽었음.
    고쳐 주지 않고(모델 출력 수정 = 꼼수) 그 에피소드를 'invalid_spec'(판단 실패, 실행 실패와 별도)으로 기록하기 위한 검사. 없으면 []."""
    out = []
    for c in (sp.get("hard") or []):
        fs = ([c["face"]] if "face" in c else []) + (list(c["faces"]) if isinstance(c.get("faces"), list) else ([c["faces"]] if "faces" in c else []))
        out += [x for x in fs if x not in FACE_NAMES]
    return out


def norm_spec(sp):
    sp = dict(hard=[dict(c) for c in (sp.get("hard") or [])], manner=sp.get("manner") or ["place"])
    for c in sp["hard"]:
        if c.get("type") in ("up", "vertical", "yaw") and c.get("tol") not in ("엄격", "보통", "느슨"): c["tol"] = "보통"
    return sp


import place_spec_core as _C      # 2026-11-10: 후보 계산은 PC/Thor 공통 모듈 하나(place_spec_core.py)로 -- 같은 코드를 옮김(1386 경우 결과 동일 확인)
final_R = _C.final_R
def candidates(spec, asset, yaw0, prefer): return _C.candidates(spec, asset, yaw0, prefer)


CLEAN = ("import omni.usd\nst=omni.usd.get_context().get_stage()\n"
         "for n in ('WhitePackerBottle_A04_PR_NVD_01','SM_Mug_B1','OV_CreateBook_02'):\n    p=st.GetPrimAtPath('/World/'+n)\n    if p.IsValid() and p.IsActive(): p.SetActive(False)\n"
         "for p in list(st.GetPrimAtPath('/World').GetChildren()):\n    if p.GetName().startswith('gso_') or p.GetName().startswith('Platform_'): st.RemovePrim(p.GetPath())\n")


GHOST = ("import omni.usd\nfrom pxr import UsdGeom\nst=omni.usd.get_context().get_stage()\n"
         "for n in ('box','bar','plate','block','eggbox'):\n    q='/World/gso_0_'+n\n    if not st.GetPrimAtPath(q).IsValid(): UsdGeom.Cube.Define(st,q).CreateSizeAttr(0.01)\n")      # 2026-11-11: 지운 동적 물체가 물리·화면에 남는 문제(USD 에선 사라짐). 같은 경로를 다시 만들고(1초 뒤) 지우면 사라짐(실측). 상자는 매번 같은 경로라 덮였지만 막대·판은 다음 장면까지 남았음


def clean_scene():
    """2026-11-09(사용자 지시): 장면 배치 전 정리 -- 동결 조건과 같게 작업 영역 소품 3개 비활성화 + 이전 물체/작업대 제거. Isaac 재시작 후에도 같은 상태."""
    from isaac_exec import run_code
    run_code(CLEAN, timeout=40); time.sleep(1)      # 1) 지움(무늬 껍데기 물체는 여기서 물리·화면에 남을 수 있음)
    run_code(GHOST, timeout=40); time.sleep(1)      # 2) 같은 경로를 다시 만듦
    run_code(CLEAN, timeout=40); time.sleep(1)      # 3) 다시 지움 -> 남은 것까지 사라짐(2026-11-11 실측)


def safe_home(s, asset):
    """2026-11-09(사용자 보고): 충돌 무시 홈 복귀가 손/카메라를 작업대에 걸어 로봇이 멈춤 -> 작업대 + 놓인 물체를 피하는 복귀(home_safe.py).
    계획 실패 시에만 기존 복귀로 넘어가고 기록(판정은 복귀 전 상태라 결과에는 영향 없음)."""
    # 2026-11-09 v4: 실패 10/27 원인 = 놓은 직후 손이 물체/받침대에 붙어 있는데 (a) 놓인 물체를 최대치수 정육면체로, (b) 여유 2cm 로 잡아 시작 자세가 충돌 판정(진단: 손가락 -2~-3cm).
    # -> 놓인 물체는 실제 치수·방향 상자(연직축 찾고 yaw), 여유 5mm. 안 되면 놓인 물체를 빼고 한 번 더(받침대 유지). 그래도 안 되면 기존 복귀(기록).
    obs = [dict(id="platform", xyz=[EM.PX, EM.PY, EM.TOP / 2], raw=[0.24, 0.24, EM.TOP], yaw=0.0)]
    if s:
        Rm = np.array(s["R"], float); dm = list(GRID_DIMS[asset]); k = int(np.argmax(np.abs(Rm[2, :]))); i, j = [a for a in range(3) if a != k]
        obs.append(dict(id="placed", xyz=s["xyz"], raw=[dm[i], dm[j], dm[k]], yaw=float(np.arctan2(Rm[1, i], Rm[0, i]))))
    tries = [("", obs)] + ([("(놓인 물체 제외) ", obs[:1])] if len(obs) > 1 else []); parts = []; ok = False
    for tag, ob in tries:
        out = subprocess.run(["python3", os.path.join(H, "home_safe.py"), json.dumps(ob), "0.005"], cwd=EM.PC, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
        m_ = out[-1] if out else "FAIL 출력 없음"; parts.append(tag + m_)
        if not m_.startswith("FAIL"): ok = True; break
    msg = " | ".join(parts)
    if not ok:
        subprocess.run(["python3", "home_and_get_armq.py"], cwd=EM.PC, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, timeout=90); msg += " -> 기존 복귀"
    return msg


def plan_exec(scene, asset, yaw0, cand):
    key = f"{scene}|{cand['pose']}|{cand['yaws']}|{cand['xy']}"
    if key in MEMO: return MEMO[key], True
    D = f"{OUT}/runs/{len(MEMO):03d}"; os.makedirs(D, exist_ok=True)
    clean_scene(); EM.spawn(asset, yaw0); time.sleep(5)
    e = dict(EM.ENV); Rw = EM.world_rot(cand["pose"], yaw0)
    if Rw is not None: e["NERO_PLACE_OBJ_ROT"] = json.dumps([float(v) for v in Rw.as_quat()])
    if cand["xy"] is not None: e["NERO_PLACE_TOPZ"] = str(EM.TOP)      # --place-xy 는 받침면 높이 기본 0(바닥) -> 작업대 윗면 30cm 를 준다(2026-11-09 원인 확인)
    if cand["yaws"] is not None: e["NERO_PLACE_YAWS"] = ",".join(str(v) for v in cand["yaws"])
    dest = ["--dest-id", "platform", "--dest-kind", "flat"] if cand["xy"] is None else [f"--place-xy={cand['xy'][0]},{cand['xy'][1]}"]
    t0 = time.time()
    try:
        subprocess.run(["./rt_run.sh", f"--target-at={EM.OX},{EM.OY}", "--dests", EM.DESTS] + dest + ["--vocab", EM.VOC, "--max-picks", "1", "--place-clear", "0.015", "--out", D + "/run.json"],
                       cwd=EM.PC, env=e, stdout=open(D + "/run.log", "w"), stderr=subprocess.STDOUT, timeout=600)
    except subprocess.TimeoutExpired: pass
    sec = round(time.time() - t0); time.sleep(5)
    s = EM.final_state(); EM.shot(D + "/before_home.png")
    home = safe_home(s, asset); time.sleep(2)
    log = open(D + "/run.log").read(); mfd = re.search(r"놓기 손 진단 (\{.*\})", log)
    r = dict(dir=D, planned=("pick 과 place 모두 계획됨" in log), state=s, sec=sec, home=home, fdiag=(json.loads(mfd.group(1)) if mfd else None),
             why=summ_fail(log))
    MEMO[key] = r; json.dump(MEMO, open(MEMO_F, "w"), ensure_ascii=False, indent=1)
    return r, False


def summ_fail(log):
    """계획 실패 이유를 짧게(Gemma 에게 줄 말). C 실패 후보 사유 빈도."""
    reasons = re.findall(r"cand\d+: ([^']+)'", log)
    if not reasons: return None
    from collections import Counter
    c = Counter(("놓을 자세에 팔이 닿지 않음" if "place 불가" in x else "집은 뒤 들어 올리는 동작에서 손목 관절 한계" if "손목" in x else "집기 자세에 팔이 닿지 않음" if "pick 불가" in x else x[:30]) for x in reasons)
    return ", ".join(f"{k} {v}회" for k, v in c.most_common(3))


def ask_relax(it, spec, why, tried):
    import rq1_decide3_prompt as P3, relax_common as RC      # 2026-11-09: 판단 시험(blocked_plan_only)과 같은 질문
    p = RC.relax_prompt(it, spec, why, tried)
    txt, rs, ct, sec, fin = P3.call(p, 0)
    j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    return j, txt, sec


POSE_DESC = {"윗면이 위": "지금 모양 그대로(윗면이 위)", "바닥면이 위": "위아래를 뒤집음(바닥면이 위)", "앞면이 위": "앞면이 하늘을 보게 굴림(앞면이 위)",
             "뒷면이 위": "뒷면이 하늘을 보게 굴림(뒷면이 위)", "왼면이 위": "왼면이 하늘을 보게 굴림(왼면이 위)", "오른면이 위": "오른면이 하늘을 보게 굴림(오른면이 위)"}
RANK_F = OUT + "/rank_cache.json"; RANK = json.load(open(RANK_F)) if os.path.exists(RANK_F) else {}


def rank(it, spec, why, cands):
    """2026-11-09(사용자 지시): 조건을 만족하는 자세들의 시도 순서를 고정 규칙이 아니라 Gemma 가 정한다. 실행 가능성은 묻지 않음(의미·안전·자연스러움만).
    장면과 무관(문항+조건만 봄) -> 캐시. 답에 빠진 자세는 뒤에 기본 순서로 붙이고 기록."""
    names = [c["pose"] for c in cands]
    if len(names) <= 1: return cands, None
    key = it["id"] + "|" + json.dumps(spec, ensure_ascii=False, sort_keys=True) + "|" + ",".join(names)
    if key not in RANK:
        import rq1_decide3_prompt as P3
        lst = "\n".join(f"- {n}: {POSE_DESC[n]}" for n in names)
        p = f"""너는 로봇이 물체를 놓을 자세의 우선순위를 정하는 판단기다.
물체: {it['obj']}
사람의 명령: "{it['cmd']}"
{PR['face_vocab']}
너는 앞에서 이 조건을 정했다: {json.dumps(spec['hard'], ensure_ascii=False)} (근거: {why})
아래 자세는 모두 그 조건을 만족한다. 명령의 뜻, 물체와 내용물의 안전, 사람이 보기에 자연스러운 정도를 기준으로 가장 바람직한 것부터 순서를 정해라.
로봇이 실제로 할 수 있는지는 고려하지 마라(로봇은 이 순서대로 시도하고 안 되면 다음으로 넘어간다).
{lst}
출력은 JSON 하나만: {{"order": ["자세 이름", ...], "why": "<근거 한두 문장>"}}  (자세 이름은 괄호 안 이름 그대로, 예: "윗면이 위")"""
        txt, rs, ct, sec, fin = P3.call(p, 0)
        try: j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except Exception: j = {}
        RANK[key] = dict(order=j.get("order") or [], why=j.get("why", ""), raw=txt, sec=sec); json.dump(RANK, open(RANK_F, "w"), ensure_ascii=False, indent=1)
    od = [n for n in RANK[key]["order"] if n in names]; od = list(dict.fromkeys(od)); missing = [n for n in names if n not in od]
    by = {c["pose"]: c for c in cands}
    return [by[n] for n in od + missing], dict(order=od, missing=missing, why=RANK[key]["why"])


def episode(item_id, scene, spec0, why0, prefer):
    it = ITEMS[item_id]; asset = scene.split("_y")[0]; yaw0 = int(scene.split("_y")[1])
    spec = norm_spec(spec0); why = why0; hist = []
    for rnd in range(MAX_ROUND):
        cands, rk = rank(it, spec, why, candidates(spec, asset, yaw0, prefer)); tried = []
        for c in cands:
            r, cached = plan_exec(scene, asset, yaw0, c)
            if r["planned"]:
                s = r["state"]; ok = None
                if s and s["on_platform"]:
                    R = Rot.from_matrix(np.array(s["R"])); ok = check(R, spec, "place", pos=s["xyz"][:2], dims=GRID_DIMS[asset], center=CENTER, half=0.12)
                hist.append(dict(round=rnd, spec=spec, why=why, rank=rk, tried=tried, chosen=c, run=r["dir"], on_platform=bool(s and s["on_platform"]), meets_own_spec=(ok[0] if ok else False),
                                 broken=(ok[1] if ok else None), close_z=(r["fdiag"] or {}).get("close_z")))
                return dict(item=item_id, scene=scene, outcome=("executed_ok" if ok and ok[0] else "executed_bad" if s and s["on_platform"] else "dropped"), hist=hist)
            tried.append(dict(c, where=next((x["where"] for x in spec["hard"] if x["type"] == "region"), None), why=r["why"]))
        hist.append(dict(round=rnd, spec=spec, why=why, rank=rk, tried=tried, chosen=None))
        j, raw, sec = ask_relax(it, spec, why, tried)
        if j.get("abstain"):
            hist.append(dict(round=rnd + 1, abstain=True, why=j.get("why"), raw=raw)); return dict(item=item_id, scene=scene, outcome="abstain", hist=hist)
        new = norm_spec(j)
        if new == spec:      # 같은 조건을 다시 씀 = 풀 수 있는 조건이 없다고 판단(더 시도해도 같은 결과) -> 놓지 않음
            hist.append(dict(round=rnd + 1, abstain=True, kept_same=True, why=j.get("why"), raw=raw)); return dict(item=item_id, scene=scene, outcome="abstain", hist=hist)
        spec = new; why = j.get("why", ""); prefer = j.get("prefer")
    return dict(item=item_id, scene=scene, outcome="gave_up_rounds", hist=hist)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    scenes = sys.argv[3].split(",") if len(sys.argv) > 3 else [f"{o}_y{y}" for o in ("plate", "box", "bar") for y in (0, 60, 120)]
    cf = {}
    for l in open(os.path.join(H, "out3", f"rq1v3_g31_{split}_direct.jsonl")):
        r = json.loads(l)
        if r.get("sample", 0) == 0 and r["id"] not in cf:
            j = json.loads(r["raw"][r["raw"].index("{"):r["raw"].rindex("}") + 1]); cf[r["id"]] = (j, j.get("why", ""), j.get("prefer"))
    fo = open(OUT + "/episodes.jsonl", "a")
    for scene in scenes:
        asset = scene.split("_y")[0]
        for iid, (j, why, pref) in cf.items():
            if ITEMS[iid]["asset"] != asset or (only and iid not in only): continue
            ep = episode(iid, scene, j, why, pref); fo.write(json.dumps(ep, ensure_ascii=False) + "\n"); fo.flush()
            print(f"{scene} {iid}: {ep['outcome']} 라운드 {len([h for h in ep['hist'] if 'spec' in h])}", flush=True)
    print("STAGE2_DONE", flush=True)
