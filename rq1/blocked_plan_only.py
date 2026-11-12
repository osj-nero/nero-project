"""막히는 장면 판단 시험 -- 로봇을 움직이지 않고 플래너(8021 /plan, 집기+놓기 동시)만 부른다(2026-11-09, 설계: DESIGN_BLOCKED_SCENES_20261109.md).
한 경우의 흐름: Gemma 조건 -> 조건 만족 자세 목록 -> Gemma 순서 -> 저장된 실제 C 요청(덤프, CGN 파지 후보 포함)에 목록·장면 장애물을 넣어 /plan
  -> 되면 '놓을 수 있음'(실행은 별도 확인), 덤프 MAX_DUMPS 개 모두 안 되면 실제 실패 이유를 Gemma 에게 -> 풀기/포기 (최대 MAX_ROUND).
덤프 = 지금 파이프라인(NERO_C_REQ_DUMP)으로 받은 장면별 C 요청. 재시도 3회 제한과 같게 덤프 3개(서로 다른 CGN 샘플)까지 본다.
모든 Gemma 입력(조건 질문, 순서 질문, 실패 알림)을 기록 -> 다른 모델에 같은 입력을 다시 넣어 비교(모델 비교).
사용: python3 blocked_plan_only.py ITEMS.json SCENES.json OUT.jsonl [item_id,...]"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import os, sys, json, re, time, copy, glob
from collections import Counter
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
sys.path.insert(0, "" + NERO_ROOT + "/goalpose_impl")
import stage2_loop as S
from validate_goalpose import post
EM = S.EM
os.makedirs(H + "/blocked_v4", exist_ok=True); S.RANK_F = H + "/blocked_v4/rank_cache.json"; S.RANK.clear()      # 다른 실험 캐시와 섞지 않음
MAX_ROUND = 3; MAX_DUMPS = int(os.environ.get("BP_MAX_DUMPS", "3"))
FLAGS = dict(hand_dest_reject=True, finger_reject="report", descend_with_dest="strict")      # 지금 파이프라인과 같은 놓기 설정(stage2_loop ENV 와 일치)

from relax_common import REASON_KO, reason_summary, per_pose, pose_msg, relax_prompt      # 2026-11-09: 로봇 실행 경로(stage2_fast)와 같은 코드


def where_name(xy):
    if xy is None: return None
    return next((k for k, v in S.SPOTS.items() if abs(v[0] - xy[0]) < 1e-6 and abs(v[1] - xy[1]) < 1e-6), str(xy))


def spot_occupied(pose, yaw0, xy, asset, obstacles):
    """2026-11-09: 놓을 점에 물체를 어떤 yaw 로 놓아도 받침대 위 장애물과 겹치는가(시스템이 아는 고정 장애물 위치로 판단 -- 사실만 알려 주고 판단은 Gemma).
    보수적으로 물체 바닥 윤곽의 가장 짧은 반폭(r)만 써서, 그것도 겹치면 '자리 막힘'. 위치 조건이 없으면(xy None) 판단하지 않음."""
    if xy is None or not obstacles: return False
    import numpy as np
    R = S.final_R(pose, yaw0, 0).as_matrix(); ext = np.abs(R) @ np.array(S.GRID_DIMS[asset]) / 2.0      # 월드 축 반폭
    r = float(min(ext[0], ext[1]))
    for o in obstacles:
        if o["xyz"][2] - o["raw"][2] / 2 > EM.TOP + 0.05: continue      # 받침면 위로 떠 있는 것은 제외
        if abs(xy[0] - o["xyz"][0]) < o["raw"][0] / 2 + r and abs(xy[1] - o["xyz"][1]) < o["raw"][1] / 2 + r: return True
    return False


def dumps_for(scene):
    okf = H + "/blocked_v4/dump_ok.json"      # 사전 확인(screen_dumps_v4.py)을 통과한 덤프만: 막힘이 장면 구조 때문임이 확인된 것
    if os.path.exists(okf) and scene["id"] in json.load(open(okf)): return json.load(open(okf))[scene["id"]][:MAX_DUMPS]
    d = sorted(glob.glob(f"{H}/dumps_v4/{scene['asset']}_y{scene.get('dump_yaw', scene['yaw'])}/obj_*.json"))      # dump_yaw = 폴더 이름(장면), yaw = 자세 계산에 쓰는 값(인식 모드면 인식 yaw)
    return d[:MAX_DUMPS]


SERVER_CAND = os.environ.get("BP_SERVER_CAND", "1") == "1"      # 2026-11-10: 조건 -> 후보 계산을 플래너 서버(Thor)에서(place_spec_core.py, PC 와 같은 파일). 0 = 예전처럼 PC 에서


def server_candidates(spec, asset, yaw0):
    o = post(8021, "/candidates", dict(hard=spec.get("hard") or [], asset=asset, yaw0=float(yaw0)))
    if not o.get("ok"): raise RuntimeError(f"/candidates 실패: {o}")
    return o["cands"]


def make_req(base, pl, scene, spec=None):
    r = copy.deepcopy(base); pr = r["place_check"]["req"]
    if SERVER_CAND and spec is not None:      # 서버가 조건+순서로 목록을 만든다(PC 는 이름 순서만 보냄)
        pr["place_spec"] = dict(hard=spec.get("hard") or [], asset=scene["asset"], yaw0=float(scene["yaw"]), order=[e["name"] for e in pl]); pr.pop("obj_rot_list", None)
    else: pr["obj_rot_list"] = pl
    pr.pop("obj_rot", None); pr.update(FLAGS); r["place_prescreen_pose"] = True; r["total_budget_s"] = float(os.environ.get("BP_TOTAL_BUDGET", "90"))      # 요청 전체 시간 상한(초)      # 2026-11-09: 파지별 놓기 IK 사전 선별(성공 손실 0/8, 실패 4~6배 빠름)
    extra = [dict(o, id=o["id"]) for o in scene.get("obstacles", [])]
    r["obstacles"] = list(r.get("obstacles") or []) + copy.deepcopy(extra)       # 집기·이송 장면에도
    pr["obstacles"] = list(pr.get("obstacles") or []) + copy.deepcopy(extra)     # 놓기 장면(받침대 위 장애물은 도착지가 아니라 이웃)
    return r


def pose_list(spec, asset, yaw0, it, why):
    cands = server_candidates(spec, asset, yaw0) if SERVER_CAND else S.candidates(spec, asset, yaw0, None)
    t0 = time.time(); ranked, rk = S.rank(it, spec, why, cands); sec = round(time.time() - t0, 1)
    out = []
    for c in ranked:
        if SERVER_CAND: out.append(dict(name=c["pose"], rot=None, yaws=c["yaws"], slots=([c["xy"]] if c["xy"] else None)))      # 회전은 서버가 계산(기록·메시지용 이름·점만)
        else:
            Rw = EM.world_rot(c["pose"], yaw0)
            out.append(dict(name=c["pose"], rot=(None if Rw is None else [float(v) for v in Rw.as_quat()]), yaws=c["yaws"], slots=([c["xy"]] if c["xy"] else None)))
    return out, rk, sec


def try_plan(pl, scene, spec=None):
    res = []
    for f in dumps_for(scene):
        t0 = time.time()
        try: o = post(8021, "/plan", make_req(json.load(open(f)), pl, scene, spec))
        except Exception as e: o = dict(ok=False, why=repr(e)[:200])
        why = o.get("why") or o.get("reason") or ""
        whys = re.findall(r"cand\d+: ([^']+)'", str(why)) if not o.get("ok") else []
        if not o.get("ok") and "시간 예산" in str(why): whys.append("시간 예산 초과")      # 요청 전체 상한에 걸림 -> Gemma 에게 그대로 알림
        plan = o.get("plan") or {}
        res.append(dict(dump=os.path.basename(f), ok=bool(o.get("ok")), pose=plan.get("pose_name"), sec=round(time.time() - t0, 1), whys=whys, raw_why=str(why)[:600],
                        per_pose={k: dict(reasons=v[0], state=v[1]) for k, v in per_pose(o.get("why")).items()} if not o.get("ok") else None, timing=o.get("timing")))
        if o.get("ok"): break
    return res


def episode(it, scene):
    import rq1_decide3_prompt as P3
    asset, yaw0 = scene["asset"], scene["yaw"]; hist = []; t_ep = time.time()
    p0 = P3.prompt(it); txt, rs, ct, sec, fin = P3.call(p0, 0)
    j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]); spec = S.norm_spec(j); why = j.get("why", "")
    hist.append(dict(step="condition", prompt=p0, raw=txt, sec=sec, spec=spec))
    if S.bad_faces(spec): return dict(item=it["id"], scene=scene["id"], outcome="invalid_spec", bad=S.bad_faces(spec), final_spec=spec, relaxed=False, hist=hist, sec=round(time.time() - t_ep, 1))      # 2026-11-11: 조건 언어에 없는 면 이름 -> 판단 실패로 기록(고치지 않음, 서버 500 방지)
    for rnd in range(MAX_ROUND):
        pl, rk, rsec = pose_list(spec, asset, yaw0, it, why)
        h = dict(step="plan", round=rnd, spec=spec, pl=[e["name"] for e in pl], rank_sec=rsec)
        if not pl:
            h["note"] = "조건을 만족하는 자세 없음"; tried = []
        else:
            res = try_plan(pl, scene, spec); h["plan"] = res
            if any(r["ok"] for r in res):
                h["feasible"] = True; hist.append(h)
                return dict(item=it["id"], scene=scene["id"], outcome="placed_plan", final_spec=spec, chosen=next(r["pose"] for r in res if r["ok"]), relaxed=(rnd > 0), hist=hist, sec=round(time.time() - t_ep, 1))
            tried = []
            for e in pl:
                xy = e["slots"][0] if e["slots"] else None
                occ = spot_occupied(e["name"], yaw0, xy, asset, scene.get("obstacles", []))
                msg = pose_msg(e["name"], [r.get("per_pose") or {} for r in res], occ)
                tried.append(dict(pose=e["name"], where=where_name(xy), why=msg))
            h["tried_msgs"] = tried
        hist.append(h)
        rp = relax_prompt(it, spec, why, tried); txt, rs, ct, sec, fin = P3.call(rp, 0)
        jj = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        hist.append(dict(step="relax", round=rnd, prompt=rp, raw=txt, sec=sec))
        if jj.get("abstain"): return dict(item=it["id"], scene=scene["id"], outcome="abstain", final_spec=spec, why=jj.get("why"), relaxed=(rnd > 0), hist=hist, sec=round(time.time() - t_ep, 1))
        new = S.norm_spec(jj)
        if S.bad_faces(new): return dict(item=it["id"], scene=scene["id"], outcome="invalid_spec", bad=S.bad_faces(new), final_spec=spec, spec_raw=jj, relaxed=(rnd > 0), hist=hist, sec=round(time.time() - t_ep, 1))
        if new == spec: return dict(item=it["id"], scene=scene["id"], outcome="abstain", kept_same=True, final_spec=spec, why=jj.get("why"), relaxed=(rnd > 0), hist=hist, sec=round(time.time() - t_ep, 1))
        spec, why = new, jj.get("why", "")
    return dict(item=it["id"], scene=scene["id"], outcome="gave_up_rounds", final_spec=spec, relaxed=True, hist=hist, sec=round(time.time() - t_ep, 1))


if __name__ == "__main__":
    items = {i["id"]: i for i in json.load(open(sys.argv[1]))}; scenes = json.load(open(sys.argv[2])); out = sys.argv[3]
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    fo = open(out, "a")
    for sc in scenes:
        for iid in sc["items"]:
            if only and iid not in only: continue
            ep = episode(items[iid], sc); fo.write(json.dumps(ep, ensure_ascii=False) + "\n"); fo.flush()
            print(f"{sc['id']} {iid}: {ep['outcome']} {ep.get('chosen') or ep.get('why', '')} {ep['sec']}s", flush=True)
    print("BLOCKED_PLAN_ONLY_DONE", flush=True)
