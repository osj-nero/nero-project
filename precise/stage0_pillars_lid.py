"""0단계(2026-11-12, 사용자 'ㄱㄱ'): 세운 막대 기둥 4개 + 그 위 판 뚜껑 -- 계획만(cuRobo 8021 /plan, Isaac 동작 없음).
기반 요청 = 11/11 실행(exec87_v5)에서 실제로 계획·성공한 C 요청(같은 인식·파지 후보): 막대 = runs/078(R_bar_y117.3, 세워 놓기 성공), 판 = runs/067(R_plate_y38.5, 눕혀 놓기 성공).
바꾸는 것: 놓을 자세 목록(obj_rot_list: 세움 / 눕힘), 놓을 점(꼭짓점 1개 / 정사각형 가운데 1개), 받침 높이(top_z), 장애물(앞서 세운 기둥 추가), 도착지(뚜껑은 기둥 4개).
변인: 정사각형 한 변(기둥 중심 간격) s, 받침 높이 base(0.30 = 30cm 작업대). 기둥 놓는 순서 = 먼 쪽(-x) 먼저.
결과: precise/stage0_pillars_lid/results_v2.json
2026-11-12 v2: 뚜껑 도착지 = 기둥 + 작업대(other_obj_1). v1 은 작업대가 '자리 점유 물체'로 잡혀 사전 선별에서 자리 후보가 전부 빠졌음(IK 0 의 원인)."""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import os, sys, json, glob, copy, time, itertools
H = "" + NERO_ROOT + ""; sys.path.insert(0, H + "/rq1"); sys.path.insert(0, H + "/goalpose_impl")
import place_spec_core as C
from validate_goalpose import post
R1 = H + "/rq1/exec87_v5_20261111"
OUT = H + "/precise/stage0_pillars_lid"; os.makedirs(OUT, exist_ok=True)
EP = {json.loads(l)["hist"][-1].get("run", "").split("/")[-1]: json.loads(l) for l in open(R1 + "/episodes.jsonl") if json.loads(l)["hist"]}
BAR_L, BAR_W = 0.20, 0.03


def base(run):
    f = sorted(glob.glob(f"{R1}/runs/{run}/creq/*.json"))[-1]; r = json.load(open(f))
    h = [x for x in EP[run]["hist"] if "run" in x][-1]; return r, float(h["perc"]["yaw"]), f


def pillar_obs(k, xy, top):
    return dict(id=f"pillar_{k}", xyz=[xy[0], xy[1], top + BAR_L / 2], raw=[BAR_W, BAR_W, BAR_L], yaw=0.0)


def req_for(r0, yaw0, poses, slot, top_z, extra_obs, dest_ids):
    r = copy.deepcopy(r0); pr = r["place_check"]["req"]
    pr.pop("place_spec", None); pr.pop("obj_rot", None)
    pr["obj_rot_list"] = [dict(name=n, rot=(None if C.world_rot(n, yaw0) is None else [float(v) for v in C.world_rot(n, yaw0).as_quat()]), yaws=None, slots=[list(slot)]) for n in poses]
    pr["slots"] = [[slot[0], slot[1], top_z]]; pr["top_z"] = top_z; pr["dest_ids"] = list(dest_ids)
    for L in (r["obstacles"], pr["obstacles"]): L.extend(copy.deepcopy(extra_obs))
    return r


def plan(r):
    t = time.time()
    try: o = post(8021, "/plan", r)
    except Exception as ex: o = dict(ok=False, why=repr(ex)[:200])
    pc = (o.get("plan") or {}).get("place_plan") or {}      # v2: 응답의 plan.place_plan 에서 꺼냄(v1 은 키를 잘못 봐서 빈 칸)
    return dict(ok=bool(o.get("ok")), sec=round(time.time() - t, 1), why=str(o.get("why"))[:300],
                place={k: pc.get(k) for k in ("pose_name", "slot", "slot_index", "yaw_deg", "n_cands", "n_ik_ok", "mode", "fdiag", "tcp", "obj_goal") if isinstance(pc, dict) and k in pc}, keys=sorted(o.keys())[:30])


if __name__ == "__main__":
    bar, bar_yaw, bf = base("078"); plate, plate_yaw, pf = base("067")
    print("기반 요청:", bf, bar_yaw, "|", pf, plate_yaw, flush=True)
    res = []
    for base_z, s in itertools.product([0.30], [0.07, 0.08, 0.09]):
        cx, cy = -0.46, 0.0      # 작업대 가운데
        corners = sorted([(cx + dx * s / 2, cy + dy * s / 2) for dx in (-1, 1) for dy in (-1, 1)], key=lambda p: p[0])      # 먼 쪽(-x) 먼저
        placed = []
        for k, xy in enumerate(corners):
            r = req_for(bar, bar_yaw, ["왼면이 위", "오른면이 위"], xy, base_z, placed, ["other_obj_1"])
            o = plan(r); o.update(kind="pillar", k=k, xy=xy, s=s, base=base_z, n_prev=len(placed)); res.append(o)
            print(f"s={s} 기둥{k} {xy}: ok={o['ok']} {o['sec']}s {o['place']} {'' if o['ok'] else o['why']}", flush=True)
            placed.append(pillar_obs(k, xy, base_z))      # 계획 실패해도 다음 단계를 보려고 세워진 것으로 가정(결과에 표시)
        top = base_z + BAR_L
        for poses in (["앞면이 위", "뒷면이 위"],):
            r = req_for(plate, plate_yaw, poses, (cx, cy), top, placed, [p["id"] for p in placed] + ["other_obj_1"])
            o = plan(r); o.update(kind="lid", s=s, base=base_z, top=top); res.append(o)
            print(f"s={s} 뚜껑 top={top}: ok={o['ok']} {o['sec']}s {o['place']} {'' if o['ok'] else o['why']}", flush=True)
        json.dump(res, open(OUT + "/results_v2.json", "w"), ensure_ascii=False, indent=1)
    print("STAGE0_DONE", flush=True)
