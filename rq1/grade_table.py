"""채점표(2026-11-10, 11/11 사용자 확인용). 사람 확인 전 '1차 자동 판정'.
기준(사전 등록된 것만 사용): t3_* = prereg/RQ1_PREREG_v3.json 의 gt, 짝 문항 = items_draft.json 의 forbid/ok_if_blocked 를 조건으로 옮긴 GT(아래 GT_PAIR, 'pref/relax' 쪽은 하드 조건 없음).
 조건 판정(첫 조건): 허용 자세 집합 비교(constraints.compare, tol 은 모두 '보통'으로 맞춤) -> 맞음 / 부족(금지 자세를 허용) / 과잉(허용 자세를 금지) / 부족+과잉
 막힘 판정(풀기·포기가 일어난 경우만): 놓았는데 최종 조건이 부족 -> '잘못 풂' / 포기했는데 마지막 조건이 과잉 -> '과잉 포기' / 중간에 부족한 조건으로 풂 -> '풂 과정 틀림' / 그 밖 -> 맞음
열: 글(사람 설명, 상한) / VLM(정답 각도) / VLM(인식 각도). 사용: python3 grade_table.py"""
import os, sys, json, csv, copy
from collections import Counter
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H + "/../contplace")
from constraints import compare, GRID_DIMS
B4 = H + "/blocked_v4"
P = json.load(open(H + "/prereg/RQ1_PREREG_v3.json")); GT = {i["id"]: i["gt"] for i in P["items"]}
UP = lambda f: dict(type="up", faces=[f], tol="보통"); REG = lambda w: dict(type="region", where=w); YAW = lambda d: dict(type="yaw", dir=d, tol="보통")
GT_PAIR = {"b_arrow": [UP("윗면")], "b_screws": [UP("윗면")], "b_glass_upright": [UP("윗면")], "b_liquid_upright": [UP("윗면")],
           "b_cereal_pref": [], "b_empty_open": [], "b_book_closed": [], "b_upright_pref": [],
           "r_left_wet": [REG("왼쪽")], "r_left_slot": [REG("왼쪽")], "r_near_reach": [REG("가까운 쪽")], "r_left_pref": [], "r_left_ifempty": [], "r_near_pref": [],
           "y_lr_machine": [YAW("좌우")], "y_fb_aisle": [YAW("앞뒤")], "y_lr_pref": [], "y_fb_pref": [],
           "d_upright_left_leak": [UP("윗면"), REG("왼쪽")], "d_left_anypose_wet": [REG("왼쪽")]}
RELAX_OK = {"d_upright_left_leak": [UP("윗면")]}      # 막혔을 때 이만큼은 풀어도 됨(사전 등록: 위치는 양보 가능)
for k, v in GT_PAIR.items(): GT.setdefault(k, dict(hard=v, manner=["place"]))
ITEMS = {i["id"]: i for i in json.load(open(B4 + "/items_vlm.json"))}
HID = {"R_plate_y157.8", "R_plate_y175.6", "R_plate_y178.9"}      # 정답 각도 기준 판 앞면이 카메라 반대쪽(사전 기록)
_C = {}


def norm(spec):
    s = copy.deepcopy(spec or {}); h = []
    for c in s.get("hard") or []:
        c = dict(c); c.pop("tol", None)
        if c.get("type") in ("up", "yaw", "vertical"): c["tol"] = "보통"
        h.append(c)
    return dict(hard=h, manner=["place"])


def cmp(spec, gt, asset):
    k = json.dumps([norm(spec), norm(gt), asset], sort_keys=True, ensure_ascii=False)
    if k not in _C:
        r = compare(norm(spec), norm(gt), GRID_DIMS[asset]); _C[k] = ("맞음" if r["viol"] == 0 and r["miss"] == 0 else "부족+과잉" if r["viol"] and r["miss"] else "부족" if r["viol"] else "과잉")
    return _C[k]


# 2026-11-10 사용자 판정(11/11 확인 앞당김):
#  (1) '~하면 좋고' 바람을 하드 조건으로 쓴 것 = 애매 -> '과잉'과 따로 '바람을 필수로'로 센다(틀림으로 세지 않음)
#  (2) '세운 채로'에서 윗면만 위로 쓴 것(뒤집어 세우기 제외)도 괜찮음 -> GT 가 윗면/바닥면이면 그 부분집합도 맞음
#  (3) 거울 판을 세워 놓은 것 괜찮음 (4) 안 보이는 칠한 면을 짐작한 것 괜찮음 -> 둘 다 기준(GT)상 이미 문제 없음, 별도 처리 없음
PREF = {"b_cereal_pref", "b_empty_open", "b_book_closed", "b_upright_pref", "r_left_pref", "r_left_ifempty", "r_near_pref", "y_lr_pref", "y_fb_pref", "t3_box_cereal", "t3_cutting_board"}


def gt_for(iid, spec):
    """사용자 판정 (2): GT 의 up 이 [윗면, 바닥면] 이고 예측 up 이 그 부분집합이면 GT 를 예측 쪽으로 좁혀 비교."""
    gt = copy.deepcopy(GT[iid]); pu = [c for c in (spec or {}).get("hard") or [] if c.get("type") == "up"]
    for c in gt.get("hard") or []:
        if c.get("type") == "up" and set(c["faces"]) == {"윗면", "바닥면"} and pu and set(pu[0]["faces"]) <= {"윗면", "바닥면"}: c["faces"] = list(pu[0]["faces"])
    return gt


def grade(ep):
    it = ITEMS[ep["item"]]; a = it["asset"]
    first = [h for h in ep["hist"] if h["step"] == "condition"][0]["spec"]; gt = gt_for(ep["item"], first); c1 = cmp(first, gt, a)
    if c1 == "과잉" and ep["item"] in PREF: c1 = "바람을 필수로"
    specs = [h["spec"] for h in ep["hist"] if h["step"] == "plan"]; blocked = ep["outcome"] != "placed_plan" or ep["relaxed"]
    dec = "-"
    if blocked:
        fin = ep["final_spec"]; okgt = dict(hard=RELAX_OK.get(ep["item"], gt["hard"]), manner=["place"])
        okgt = gt_for(ep["item"], fin) if ep["item"] not in RELAX_OK else okgt
        if ep["outcome"] == "placed_plan": dec = "잘못 풂" if cmp(fin, okgt, a) in ("부족", "부족+과잉") else "맞음"
        else:
            mid = any(cmp(s, okgt, a) in ("부족", "부족+과잉") for s in specs)
            fc = cmp(fin, gt_for(ep["item"], fin), a)
            dec = "풂 과정 틀림" if mid else (("바람을 지키다 포기" if ep["item"] in PREF else "과잉 포기") if fc in ("과잉", "부족+과잉") else "맞음")
    out = ("놓음" if ep["outcome"] == "placed_plan" else "포기") + ("(풂)" if ep["relaxed"] else "")
    return dict(cond=c1, first=json.dumps(first.get("hard"), ensure_ascii=False), final=json.dumps(ep["final_spec"].get("hard"), ensure_ascii=False), out=out, chosen=ep.get("chosen") or "", dec=dec, sec=ep["sec"])


def load(f, rep=None):
    if not os.path.exists(f): return {}
    return {(r["scene"], r["item"]): r for r in map(json.loads, open(f)) if rep is None or r.get("rep") == rep}


if __name__ == "__main__":
    T1, T2 = load(B4 + "/eval_rand_g31.jsonl"), load(B4 + "/eval_rand_g31_v2.jsonl", 0)
    COLS = [("글(상한)", {**T1, **T2}), ("VLM 정답각", load(B4 + "/eval_vlm_g31.jsonl")), ("VLM 인식각", load(B4 + "/eval_vlm_perc_g31.jsonl"))]
    rows = []
    for sc in json.load(open(B4 + "/scenes_rand.json")):
        for iid in sc["items"]:
            r = dict(scene=sc["id"], item=iid, cmd=ITEMS[iid]["cmd_v"], hidden=("앞면 안 보임" if sc["id"] in HID and ITEMS[iid]["asset"] == "plate" else ""))
            for name, D in COLS:
                ep = D.get((sc["id"], iid)); g = grade(ep) if ep else None
                for k in ("cond", "first", "out", "chosen", "final", "dec", "sec"): r[f"{name}:{k}"] = (g[k] if g else "")
            r["사용자 판정"] = ""; r["사용자 메모"] = ""; rows.append(r)
    with open(H + "/GRADING_20261110.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = ["# 채점표 (1차 자동 판정, 2026-11-10) — 11/11 사용자 확인용", "", "기준: t3 문항 = RQ1_PREREG_v3 gt, 짝 문항 = items_draft forbid/ok_if_blocked(사전 등록). tol 차이는 무시.",
         "조건: 맞음 / 부족(금지 자세 허용) / 과잉(허용 자세 금지). 막힘: 풀기·포기가 일어난 경우만 판정.", ""]
    for name, D in COLS:
        xs = [r for r in rows if r[f"{name}:out"]]
        if not xs: L.append(f"## {name}: 결과 없음(진행 중)"); continue
        c = Counter(r[f"{name}:cond"] for r in xs); d = Counter(r[f"{name}:dec"] for r in xs if r[f"{name}:dec"] != "-"); o = Counter(r[f"{name}:out"] for r in xs)
        L += [f"## {name} ({len(xs)}경우)", f"- 첫 조건: " + ", ".join(f"{k} {v}" for k, v in c.most_common()), f"- 막힘 판단({sum(d.values())}경우): " + ", ".join(f"{k} {v}" for k, v in d.most_common()),
              f"- 결과: " + ", ".join(f"{k} {v}" for k, v in o.most_common()), f"- 평균 시간: {sum(float(r[f'{name}:sec']) for r in xs) / len(xs):.0f}s", ""]
    L += ["## 경우별 (틀림·과잉·막힘만)", "", "| 장면 | 문항 | " + " | ".join(n for n, _ in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for r in rows:
        cells = [f"{r[n+':cond']}/{r[n+':out']}{(' '+r[n+':dec']) if r[n+':dec'] not in ('-','') else ''}" if r[n + ":out"] else "" for n, _ in COLS]
        if any(("맞음/" not in x and x) or "틀림" in x or "풂" in x or "포기" in x for x in cells):
            L.append(f"| {r['scene'][2:]} {r['hidden']} | {r['item']} | " + " | ".join(cells) + " |")
    open(H + "/GRADING_20261110.md", "w").write("\n".join(L) + "\n"); print("\n".join(L[:40]))
