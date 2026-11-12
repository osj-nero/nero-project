"""exec87_v5_20261111 채점 (규칙 = exec87_v5_20261111/DENOMINATORS.md, 2026-11-11). 실행은 다시 하지 않고 기록만 읽는다.
판단 채점 = grade_table.grade() 그대로(계획만 시험과 같은 규칙). 실행 = episodes 의 outcome + run.log.
사용: python3 score_exec87.py [episodes.jsonl]"""
import json, sys, os, re, collections
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
F = sys.argv[1] if len(sys.argv) > 1 else H + "/exec87_v5_20261111/episodes.jsonl"
sys.argv = sys.argv[:1]
import grade_table as G

EXEC = ("executed_ok", "executed_bad", "dropped_or_offplatform")


def to_plan_ep(e):
    """stage2 에피소드 -> grade() 가 읽는 계획만 시험 형식."""
    specs = [h for h in e["hist"] if "spec" in h and h.get("spec")]
    if not specs: return None
    hist = [dict(step="condition", spec=specs[0]["spec"])] + [dict(step="plan", spec=h["spec"]) for h in specs]
    out = "placed_plan" if e["outcome"] in EXEC or (e["outcome"] == "error" and any(h.get("planned") for h in specs)) else e["outcome"]
    fin = e.get("final_spec") or specs[-1]["spec"]
    return dict(item=e["item"], hist=hist, outcome=out, relaxed=bool(e.get("relaxed")), final_spec=fin, sec=e["sec"])


def pick_failed(e):
    runs = [h.get("run") for h in e["hist"] if h.get("run")]
    if not runs: return None
    try: t = open(runs[-1] + "/run.log", errors="ignore").read()
    except OSError: return None
    m = re.findall(r"pick 성공 (\d+)", t)
    return (m[-1] == "0") if m else None


E = [json.loads(l) for l in open(F)]; N = len(E)
rows = []
for e in E:
    r = dict(scene=e["scene"], item=e["item"], outcome=e["outcome"], sec=e["sec"], st=e.get("stage_sec", {}))
    pe = to_plan_ep(e)
    if e["outcome"] == "invalid_spec" and pe is None: r.update(cond="invalid_spec", dec="-", blocked=False)
    elif pe is None: r.update(cond="판단 못 함", dec="-", blocked=False)
    else:
        g = G.grade(pe); r.update(cond=g["cond"], dec=g["dec"], blocked=g["dec"] != "-")
        if e["outcome"] == "invalid_spec": r["dec"] = "invalid_spec"; r["blocked"] = True
        if e["outcome"] in EXEC:
            fin = pe["final_spec"]; r["gt_final"] = G.cmp(fin, G.gt_for(e["item"], fin), G.ITEMS[e["item"]]["asset"])
            if e["item"] in G.RELAX_OK and r["blocked"]: r["gt_final"] = G.cmp(fin, dict(hard=G.RELAX_OK[e["item"]], manner=["place"]), G.ITEMS[e["item"]]["asset"])
    if e["outcome"] == "dropped_or_offplatform": r["pick_fail"] = pick_failed(e)
    rows.append(r)

def fail_type(e, r):
    """2026-11-12: 실행 실패 원인 라벨(로그만 읽음, 성공 판정은 그대로). 이전 라벨은 '놓기 전 이미 손이 빔'을 EXEC_FAIL/executed_bad 로, 파지 자세 미도달을 '떨어짐·받침 밖' 과 같은 칸으로 셌음."""
    runs = [h.get("run") for h in e["hist"] if h.get("run")]
    try: t = open(runs[-1] + "/run.log", errors="ignore").read()
    except (OSError, IndexError): return "기록 없음"
    if re.search(r"pick 실패 verdict=NO_REACH", t): return "파지 자세 미도달(NO_REACH)"
    if r.get("pick_fail"): return "집기 실패(기타)"
    g = re.findall(r"pre_open 그리퍼 관절 ([0-9.]+)", t)
    if g and float(g[-1]) < 0.01: return "놓기 전 손에서 빠짐(운반·하강 중)"
    pv = re.findall(r"place verdict=(\w+)", t); pv = pv[-1] if pv else None
    if pv == "MISSED":
        sn = re.findall(r"\[snap\] (\w+): 기울기 ([0-9.]+)도 z=([0-9.]+)", t); po = [x for x in sn if x[0] == "post_open"]
        if po and float(po[-1][2]) < 0.2: return "놓기 전 손에서 빠짐(운반·하강 중)"
        return "놓은 뒤 떨어짐(해제·후퇴 중)" if po else "떨어짐(시점 미확인)"
    if pv == "PLACED": return "놓았으나 기울어 안착"
    if pv == "TOPPLED": return "놓은 뒤 넘어짐"
    return f"기타({pv})"


pct = lambda a, b: f"{a}/{b} = {100 * a / b:.1f}%" if b else f"{a}/0"
C = collections.Counter(r["cond"] for r in rows)
blk = [r for r in rows if r["blocked"]]; D = collections.Counter(r["dec"] for r in blk)
att = [r for r in rows if r["outcome"] in EXEC or (r["outcome"] == "error" and r.get("gt_final"))]
ok = [r for r in att if r["outcome"] == "executed_ok"]
fail = collections.Counter(("pick 실패(집기)" if r.get("pick_fail") else ("떨어짐·받침 밖" if r.get("pick_fail") is False else "떨어짐(집기 여부 미확인)")) if r["outcome"] == "dropped_or_offplatform" else r["outcome"] for r in att if r["outcome"] != "executed_ok")


def end_ok(r):
    if r["outcome"] == "executed_ok": return r.get("gt_final") not in ("부족", "부족+과잉")
    if r["outcome"] in ("abstain", "gave_up_rounds", "abstain_nopose"): return r["dec"] == "맞음"
    return False


B2 = [r for r in rows if end_ok(r)]
why = collections.Counter()
for r in rows:
    if end_ok(r): continue
    judg_bad = r["cond"] in ("부족", "invalid_spec") or r["dec"] not in ("-", "맞음") or (r["outcome"] == "executed_ok" and r.get("gt_final") in ("부족", "부족+과잉"))
    why["판단 탓" if judg_bad else ("시스템(오류)" if r["outcome"] == "error" else "실행 탓")] += 1
mean = lambda k, xs: sum(x["st"].get(k, 0) for x in xs) / len(xs) if xs else 0
print(f"# exec87_v5 채점 ({N}/87경우 진행)")
print(f"A1 첫 조건: 엄격 {pct(C['맞음'], N)}, 사용자 판정 포함 {pct(C['맞음'] + C['바람을 필수로'], N)}  | {dict(C)}")
print(f"A2 막힘 판단: {pct(D['맞음'], len(blk))}  | {dict(D)}")
print(f"B1 pick&place: {pct(len(ok), len(att))}  | 실패 {dict(fail)}")
print(f"B1 실패 원인(2026-11-12 라벨): {dict(collections.Counter(fail_type(E[rows.index(r)], r) for r in att if r['outcome'] != 'executed_ok'))}")
print(f"B2 끝까지 맞음: {pct(len(B2), N)}  | 틀림 원인 {dict(why)}")
print(f"결과 분포: {dict(collections.Counter(r['outcome'] for r in rows))}")
print("평균 시간(s): 전체 %.1f, cond %.1f, rank %.1f, relax %.1f, robot %.1f, harness %.1f" % (sum(r["sec"] for r in rows) / N, *(mean(k, rows) for k in ("cond", "rank", "relax", "robot", "harness"))))
print("\n## 틀린·막힌·실패 경우")
for r in rows:
    if r["blocked"] or not end_ok(r) or r["cond"] not in ("맞음", "바람을 필수로"):
        print(f"- {r['scene']} {r['item']}: {r['outcome']} | 조건 {r['cond']} | 막힘 {r['dec']} | GT기준 최종 {r.get('gt_final', '')} | pick실패 {r.get('pick_fail', '')}")
