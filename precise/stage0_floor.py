"""0단계 바닥판(2026-11-12, 사용자: '작업대 위가 아니라 땅바닥에서'): 세운 막대 기둥 4개(바닥) + 그 위 판 뚜껑(높이 20cm) -- 계획만(8021 /plan, Isaac 동작 없음).
기반 요청 = 11/11 실행의 실제 C 요청(같은 인식·파지 후보, 물체는 바닥 (-0.30,0.30) 에서 집음): 막대 = runs/078, 판 = runs/061·063·067(판 시작 자세 3개).
바닥판으로 바꾸는 것: 작업대 장애물(other_obj_1) 제거, 놓는 높이 top_z = 0(테이블 면, 서버 월드에 항상 있음), 도착지 없음(11/06 우물 정과 같음). 뚜껑: top_z = 0.20, 도착지 = 기둥 4개.
구조 중심 x {-0.38,-0.42,-0.46}(첫 시도 -0.34 는 로봇 쪽 기둥 IK 0), y 0. 변인: 기둥 중심 간격 s, 정사각형 방향 θ. 뚜껑 yaw = 정사각형 축과 맞는 4개만(판이 기둥 4개를 덮게).
결과 precise/stage0_pillars_lid/floor.json"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import json, sys, math, copy
sys.path.insert(0, "" + NERO_ROOT + "/precise")
import stage0_pillars_lid as S, os
CY = 0.0
SYNTH = os.environ.get("SYNTH", "0") == "1"      # 2026-11-12: 합성 파지 후보(phase_c_L1/synth_grasps.py)로 후보를 늘린 비교
THETAS = tuple(int(x) for x in os.environ.get("THETAS", "0,30,60,90,120,150").split(",")); CXS = tuple(float(x) for x in os.environ.get("CXS", "-0.38,-0.42,-0.46").split(","))


def floorize(r0):
    r = copy.deepcopy(r0)
    if SYNTH:
        sys.path.insert(0, "" + NERO_ROOT + "/phase_c_L1"); import synth_grasps as SG
        r["candidates"] = SG.augment(r["candidates"], r["place_check"]["req"]["obj"]); r["max_cands"] = max(int(r.get("max_cands", 40)), len(r["candidates"]))
    on_plat = lambda o: -0.60 < o["xyz"][0] < -0.32 and abs(o["xyz"][1]) < 0.14      # 작업대와 그 위 장애물 블록(11/11 장면 것) 모두 제거, 로봇 뒤 고정물만 남김
    for L in (r["obstacles"], r["place_check"]["req"]["obstacles"]): L[:] = [o for o in L if not on_plat(o)]
    return r


if __name__ == "__main__":
    bar, bar_yaw, _ = S.base("078"); bar = floorize(bar)
    plates = [(floorize(S.base(run)[0]), S.base(run)[1], run) for run in ("061", "063", "067")]
    res = []; out = S.OUT + ("/floor_synth.json" if SYNTH else "/floor.json")
    for CX, s, th in __import__("itertools").product(CXS, (0.07, 0.09), THETAS):
        if True:
            t = math.radians(th); R = lambda x, y: (CX + x * math.cos(t) - y * math.sin(t), CY + x * math.sin(t) + y * math.cos(t))
            corners = sorted([R(dx * s / 2, dy * s / 2) for dx in (-1, 1) for dy in (-1, 1)], key=lambda p: p[0])      # 먼 쪽(-x) 먼저
            placed = []; n_ok = 0
            for k, xy in enumerate(corners):
                o = S.plan(S.req_for(bar, bar_yaw, ["왼면이 위", "오른면이 위"], xy, 0.0, placed, []))
                o.update(kind="pillar", k=k, xy=xy, s=s, theta=th, cx=CX); res.append(o); n_ok += o["ok"]
                print(f"cx={CX} s={s} θ={th} 기둥{k} ({xy[0]:.3f},{xy[1]:.3f}): ok={o['ok']} {o['sec']}s {o['place'].get('pose_name')} yaw {o['place'].get('yaw_deg')} {'' if o['ok'] else o['why'][:120]}", flush=True)
                placed.append(dict(S.pillar_obs(k, xy, 0.0), yaw=t))
            for pr, py, run in plates:
                r = S.req_for(pr, py, ["앞면이 위", "뒷면이 위"], (CX, CY), S.BAR_L, placed, [p["id"] for p in placed])
                yws = sorted({int(round(((th - py + 90 * k + 180) % 360) - 180)) for k in range(4)})
                for e in r["place_check"]["req"]["obj_rot_list"]: e["yaws"] = yws
                o = S.plan(r); o.update(kind="lid", cx=CX, s=s, theta=th, run=run, yaws=yws, pillars_ok=n_ok); res.append(o)
                print(f"cx={CX} s={s} θ={th} 뚜껑(판 {run}): ok={o['ok']} {o['sec']}s {o['place'].get('pose_name')} yaw {o['place'].get('yaw_deg')} {'' if o['ok'] else o['why'][:120]}", flush=True)
            json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    P = [o for o in res if o["kind"] == "pillar"]; L = [o for o in res if o["kind"] == "lid"]
    print(f"FLOOR_DONE 기둥 {sum(o['ok'] for o in P)}/{len(P)} 뚜껑 {sum(o['ok'] for o in L)}/{len(L)}", flush=True)
