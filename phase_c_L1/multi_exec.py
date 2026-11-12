#!/usr/bin/env python3
"""multi_exec.py <plan.json> <dests.json> [--status 상태.json] -- LLM 계획(물체 id/위치 -> 목적지, 순서)을 한 물체씩 실행한다(2026-10-14).
plan: {"order": [{"id": "obj_3", "xy": [x, y], "dest": "bin"|"tray"|"cup"|"leave", "label": "..."}, ...]}  -- 순서대로 실행, dest=leave 는 건너뜀.
한 물체 = rt_run.sh --target-at x,y --dests dests.json --dest-id <dest> (기존 pick and place 루프 1회). 대상이 없으면(위치가 바뀜) 장면을 다시 조사해 같은 라벨의 가장 가까운 물체로 한 번 재시도한다."""
import collections, os, sys, json, re, time, subprocess, math, signal
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sort_tools as ST
plan = json.load(open(sys.argv[1])); dests = sys.argv[2]; status = sys.argv[sys.argv.index("--status") + 1] if "--status" in sys.argv else "/tmp/multi_exec_status.json"
LOGDIR = os.environ.get("NERO_MULTI_LOG", "/tmp/multi_exec_logs"); os.makedirs(LOGDIR, exist_ok=True)

REF = {}
DK = {}
try: REF = json.load(open("/tmp/sort_objects.json"))      # rt_survey 가 저장한 처음 조사 결과(물체 id -> 크기/색/라벨/위치)
except Exception: pass
def match_identity(st, sc):
    ref = REF.get(st["id"]) or {}; best = None
    for k, v in sc.items():
        if done_xy and any(math.hypot(v["xy"][0] - x, v["xy"][1] - y) < 0.04 for x, y in done_xy): continue      # 이미 옮긴 물체(목적지 안)는 제외
        d = math.hypot(v["xy"][0] - st["xy"][0], v["xy"][1] - st["xy"][1]); sc_ = 0.3 * d
        if ref.get("size_cm") and v.get("size_cm"): sc_ += 0.03 * sum(abs(a - b) for a, b in zip(sorted(ref["size_cm"]), sorted(v["size_cm"])))
        if ref.get("color") and v.get("color") and ref["color"] != v["color"]: sc_ += 0.6
        if ref.get("top") and v.get("top") and ref["top"][:1] == v["top"][:1]: sc_ -= 0.2
        if best is None or sc_ < best[0]: best = (sc_, k)
    return best[1] if best and best[0] < 1.0 else None
done_xy = []
try: DK = {d["id"]: d["kind"] for d in json.load(open(sys.argv[2]))}
except Exception: pass
steps = [s for s in plan["order"]]; res = []; t0 = time.time()
PENDING = {}
def write(stage):      # 2026-10-15: 임시 파일에 쓰고 교체(RViz 브리지가 쓰는 도중의 빈 파일을 읽어 라벨이 깜박이던 문제)
    _tmp = status + ".tmp"; json.dump(dict(stage=stage, pending=PENDING or None, t=round(time.time() - t0, 1), n=len(steps), results=res, plan=steps, dests=dests, updated=time.time()), open(_tmp, "w"), ensure_ascii=False, indent=1); os.replace(_tmp, status)

import threading
import signal as _sg
_KIDS = []
def _kill_kids(*_a):      # 2026-10-15: rt_stop 이 multi_exec 만 죽이면 별도 세션의 rt_run/discover_object 가 계속 돌았다 -> 자식 프로세스 그룹까지 종료
    for _p in _KIDS:
        try: os.killpg(_p.pid, _sg.SIGTERM)
        except Exception: pass
    time.sleep(1.5)
    for _p in _KIDS:
        try: os.killpg(_p.pid, _sg.SIGKILL)
        except Exception: pass
    os._exit(143)
_sg.signal(_sg.SIGTERM, _kill_kids)

class Tracer:
    """한 물체를 옮기는 동안 Isaac 의 gso_* 위치를 0.1초마다 기록한다(진단). 처음으로 2cm 넘게 밀린 '다른' 물체와, 그때 들려 있던 물체(가장 높이 올라간 것)를 찾아 충돌 시점을 알아낸다."""
    def __init__(self): self.s = []; self.run = False
    def start(self):
        self.s = []; self.run = True; self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
    def _loop(self):
        t0 = time.time()
        n_ = 0; g0_ = None; seen_ = set()
        while self.run:
            g = ST.gt_snapshot()
            if g:
                self.s.append((round(time.time() - t0, 2), g)); n_ += 1
                if g0_ is None: g0_ = g
                for k_ in g0_:      # 2026-10-15: 용기(bin/바구니/접시/연필꽂이)가 3cm 넘게 밀린 첫 순간을 실행 로그 옆에 바로 남긴다(중지/강제 종료돼도 남음)
                    if k_ not in seen_ and k_ in g and any(t_ in k_ for t_ in ("Curver", "Basket", "Saucer", "Pencil")) and math.hypot(g[k_][0] - g0_[k_][0], g[k_][1] - g0_[k_][1]) > 0.03:
                        seen_.add(k_)
                        try:
                            last_ = [l_ for l_ in open(f"{LOGDIR}/chain.log", errors="ignore").read().splitlines() if l_.startswith("[rt]") or l_.startswith("[jmove]") or l_.startswith("[place") or l_.startswith("[snap")][-6:]
                            open(f"{LOGDIR}/displace.log", "a").write(f"t={self.s[-1][0]:.1f}s {k_} -> {[round(c, 3) for c in g[k_]]}\n" + "\n".join("   " + x_[:200] for x_ in last_) + "\n")
                        except Exception: pass
                if n_ % 100 == 0:      # 5초마다 시계열도 저장(중지돼도 남게)
                    try: json.dump(dict(series=self.s[::4]), open(f"{LOGDIR}/chain_trace.live.json", "w"))
                    except Exception: pass
            time.sleep(0.05)
    def stop(self, path):
        self.run = False; self.th.join(timeout=5)
        if len(self.s) < 2: return {}
        g0 = self.s[0][1]; zmax = {k: max(g[k][2] for _, g in self.s if k in g) - g0[k][2] for k in g0}; carried = max(zmax, key=zmax.get) if zmax and max(zmax.values()) > 0.05 else None
        hit = None
        for t, g in self.s:
            for k in g0:
                if k == carried or k not in g: continue
                d = math.hypot(g[k][0] - g0[k][0], g[k][1] - g0[k][1])
                if d > 0.02 and hit is None: hit = dict(victim=k, t=t, d=round(d, 3), carried=carried, carried_xyz=[round(v, 3) for v in g[carried]] if carried else None, victim_xyz=[round(v, 3) for v in g[k]], carried_dist=round(math.hypot(g[carried][0] - g0[k][0], g[carried][1] - g0[k][1]), 3) if carried else None)
        try: json.dump(dict(carried=carried, hit=hit, series=self.s[::2]), open(path, "w"))
        except Exception: pass
        return hit or {}
TR = Tracer()
def tgt_args(st, xy):
    ref = REF.get(st["id"])
    if ref and ref.get("size_cm"): return ["--target-desc", json.dumps(dict(size_cm=ref["size_cm"], color=ref.get("color"), top=ref.get("top"), feat=ref.get("feat")), ensure_ascii=False)] + ([f"--hint-xy={xy[0]},{xy[1]}"] if os.environ.get("NERO_SORT_FULLSCAN", "0") != "1" else [])      # 위치는 대상 지정이 아니라 탐색 구역 힌트(국소 스캔)로만 쓴다      # 집을 물체는 겉모습으로(위치 안 씀)
    return [f"--target-at={xy[0]},{xy[1]}"]
def _nap(p, dt):
    """2026-10-29(NERO_EXEC_EVENT=1, 기본 0): 자식 프로세스 종료를 기다리며 dt 초 쉰다. 기본(0)은 기존 time.sleep(dt) 그대로. 켜면 종료 즉시 깨어나(Popen.wait(timeout)) 폴링 지연을 없앤다 -- 종료 전에는 같은 간격으로 루프가 돈다."""
    if os.environ.get("NERO_EXEC_EVENT", "0") == "1":
        try: p.wait(timeout=dt)
        except subprocess.TimeoutExpired: pass
    else: time.sleep(dt)
def run_one(i, st, xy, tag="", _retry=False):
    logp = f"{LOGDIR}/{i:02d}_{st['id']}{tag}.log"; env = dict(os.environ, **ST.base_env(), NERO_FOCUS_SCAN="1")
    if DK.get(st["dest"]) == "hole" and os.environ.get("NERO_HOLE_BAND"): env["NERO_GRASP_BAND"] = os.environ["NERO_HOLE_BAND"]
    if DK.get(st["dest"]) == "hole": env["NERO_PLACE_AWARE"] = "1"      # 꽂기는 옆 파지여야 놓기 IK 가 있다: 파지 후보를 고를 때 놓기 가능 여부도 함께 본다(느리지만 홀 목적지에만)
    cap = float(os.environ.get("NERO_SORT_SEARCH_CAP", "30")); aborted = False      # 대상을 이 시간(초) 안에 못 찾으면 헤매지 않고 중단해 다음 판단으로 넘긴다(2026-10-14: 꼬인 뒤 스캔만 30초 넘게 반복하던 문제)
    with open(logp, "w") as lf:
        p = subprocess.Popen([os.path.join(HERE, "rt_run.sh"), *tgt_args(st, xy), "--dests", dests, "--dest-id", st["dest"], "--vocab", ",".join(ST.VOCAB), "--max-picks", "1", "--out", logp + ".json"], cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); _KIDS.append(p)
        t0_ = time.time()
        while p.poll() is None:
            _nap(p, 0.5)
            if time.time() - t0_ > cap and p.poll() is None:
                _t = open(logp, errors="replace").read()
                if "발견" not in _t and "pick 시작" not in _t:
                    try: os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=10)
                    except Exception:
                        try: os.killpg(p.pid, signal.SIGKILL)
                        except Exception: pass
                    aborted = True; subprocess.run([sys.executable, os.path.join(HERE, "force_home.py")], cwd=HERE, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, timeout=60); break
            if time.time() - t0_ > 420:
                try: os.killpg(p.pid, signal.SIGKILL)
                except Exception: pass
                break
    tx = open(logp, errors="replace").read()
    if ("시작:" not in tx) and not _retry:      # 시작 단계(초기 홈)에서 멈추거나 죽은 경우: 앞 단계의 홈 복귀와 겹친 것으로 보고 홈을 확실히 한 뒤 한 번 다시 시도한다
        subprocess.run([sys.executable, os.path.join(HERE, "force_home.py")], cwd=HERE, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, timeout=90)      # 시작 직후 ROS/home 오류로 죽은 경우(탐색 실패가 아니다)는 한 번 다시 시도한다
        time.sleep(3); return run_one(i, st, xy, tag=tag + "r", _retry=True)
    pk = re.findall(r"pick 완료 verdict=(\w+)", tx); vd = [(m[0], (m[1] or m[2] or "False")) for m in re.findall(r"place verdict=(\w+) \{(?:[^}]*'on_dest': (\w+))?(?:[^}]*'success': (\w+))?", tx)]; nt = "대상 없음" in tx; ss = re.findall(r"\+\s*([\d.]+)s 사전 훑기 완료", tx)
    return dict(pick=(pk[-1] if pk else "NONE"), place=(vd[-1][0] + ":" + vd[-1][1] if vd else "NONE"), no_target=(nt or aborted), aborted=aborted, log=logp, scan_s=(float(ss[0]) if ss else None))

def _inside(p, d, pad=0.015):
    dx, dy = p[0] - d["xy"][0], p[1] - d["xy"][1]; yaw = float(d.get("yaw", 0.0)); c, sn = math.cos(-yaw), math.sin(-yaw)
    return abs(dx * c - dy * sn) <= d["size"][0] / 2 + pad and abs(dx * sn + dy * c) <= d["size"][1] / 2 + pad

def _missing_steps(items, g0, g1):
    """계획 항목 중 최종 위치가 어느 목적지 안도 아닌 것 -> 재시도용 steps(새 위치 힌트). 물체 id -> 시뮬 프림은 시작 스냅샷에서 계획 좌표와 가장 가까운 것으로 매긴다(시뮬 전용 진단 정보, 실물에서는 재조사로 대체)."""
    D = json.load(open(dests)); dp = {d["prim"] for d in D if d.get("prim")}; cand = {k: v for k, v in g0.items() if k not in dp}
    out = []; used = set()
    for it in items:
        if not cand: break
        k = min((k for k in cand if k not in used), key=lambda k: math.hypot(cand[k][0] - it["hint"][0], cand[k][1] - it["hint"][1]), default=None)
        if k is None or math.hypot(cand[k][0] - it["hint"][0], cand[k][1] - it["hint"][1]) > 0.10 or k not in g1: continue
        used.add(k); p = g1[k]
        if any(_inside(p, d) for d in D if d.get("size") and d.get("xy")): continue          # 어딘가 목적지 안: 완료
        if p[2] < -0.05 or math.hypot(p[0], p[1]) > 0.75 or p[2] > 0.25: continue          # 책상 밖으로 떨어졌거나 손에 든 채: 시도하지 않는다
        st_ = next((x for x in steps if x["id"] == it["id"]), {}); out.append(dict(id=it["id"], dest=it["dest"], dest2=st_.get("dest2"), xy=[round(p[0], 3), round(p[1], 3)], label_top1=st_.get("label_top1")))
    return out

def _missing_steps_vision(items):
    """GT 없이: 끝난 뒤 장면을 다시 조사(rt_survey 와 같은 방법)해서 아직 테이블 위에 있는 계획 물체를 찾는다. 계획 물체와 조사 물체는 처음 조사 때 저장한 기준 임베딩(REF)과의 코사인 거리 + 크기로 맞춘다. 용기 안에 든 물체는 용기에 흡수돼 조사에 안 나온다."""
    import numpy as np
    D = json.load(open(dests)); sc2 = ST.objects_only(ST.survey(dests), dests); cand = []
    for it in items:
        ref = REF.get(it["id"]) or {}; f0 = ref.get("feat")
        if f0 is None or not ref.get("size_cm"): continue
        f0 = np.asarray(f0, float); f0 /= (np.linalg.norm(f0) + 1e-9)
        for k, v in sc2.items():
            f = v.get("feat")
            if f is None or not v.get("size_cm"): continue
            f = np.asarray(f, float); dc = 1.0 - float(f0 @ (f / (np.linalg.norm(f) + 1e-9)))
            se = max(abs(a_ - b_) for a_, b_ in zip(sorted(ref["size_cm"]), sorted(v["size_cm"])))
            if dc <= 0.35 and se <= 3.5: cand.append((dc + 0.05 * se, it, k))
    cand.sort(key=lambda x: x[0]); used_i = set(); used_k = set(); out = []
    for _, it, k in cand:
        if it["id"] in used_i or k in used_k: continue
        used_i.add(it["id"]); used_k.add(k); xy = sc2[k]["xy"]
        if any(_inside([xy[0], xy[1]], d) for d in D if d.get("size") and d.get("xy")): continue      # 목적지 위(접시 등): 완료
        if math.hypot(xy[0], xy[1]) > 0.58: continue      # 팔이 닿지 않는 곳으로 튕겨 나간 물체는 시도하지 않는다
        st_ = next((x for x in steps if x["id"] == it["id"]), {}); out.append(dict(id=it["id"], dest=it["dest"], dest2=st_.get("dest2"), xy=[round(float(xy[0]), 3), round(float(xy[1]), 3)], label_top1=st_.get("label_top1")))
    print("[retry] 재조사: 조사 물체 %d개, 계획 항목과 맞은 것 %d개 중 재시도 %d개" % (len(sc2), len(used_i), len(out)), file=sys.stderr)
    return out

def _replan(miss):
    """2026-10-17(NERO_REPLAN=1): 첫 사이클(항목당 최대 2회 시도)이 끝나고 남은 물체(miss: 새 위치 포함)의 계획을 LLM 에게 다시 받는다. 실패 사유/반경/크기/목적지별 현재 수용 수를 텍스트로 주고 목적지·순서(dest/dest2)를 새로 받는다. 실패하면 원래 계획을 그대로 쓴다."""
    try:
        last = {}
        for r in res:
            if r.get("id") and r.get("dest") != "leave": last[r["id"]] = r
        D = json.load(open(dests)); fill = collections.Counter(r["dest"] for r in res if r.get("ok") and r.get("dest"))
        L = ["너는 로봇 정리 플래너다. 첫 번째 정리 사이클이 끝났고 아래 물체들이 아직 목적지 밖에 남았다(항목당 최대 2회 시도하고 다음으로 넘어간 결과). 텍스트만 보고(다른 파일/도구 사용 금지) 남은 물체의 목적지와 순서를 다시 정한다.", "", "## 규칙",
             "- 이전 실패 사유를 보고 다른 목적지를 고르거나 순서를 바꿔라. 가능하면 다른 목적지/순서로 한 번 더 시도하게 하고, dest=\"leave\" 는 반경 55cm 이상으로 팔이 못 닿는 것처럼 정말 가망이 없을 때만 쓴다.",
             "- 반경 47cm 이상/방위 |b| 큰 곳은 손목 한계로 집기 계획이 자주 실패. 이웃 틈 <= 5cm 면 접근 후보가 0개가 되는 일이 많다. 가늘고 긴 물체는 이송 중 빠질 수 있다.",
             "- 바구니(bin2, 높이 10cm)는 낮아 큰/긴 물체가 밖으로 빠질 수 있다. 용기가 이미 많이 찼으면 덜 찬 곳을 고른다. 같은 용기로 가는 것은 가까운 것끼리 이어서.", "", "## 목적지(종류, 위치, 현재 넣은 수)"]
        for d in D: L.append(f"- {d['id']}: {d.get('label', '')}({d.get('kind')}) 위치 ({d['xy'][0]:+.2f},{d['xy'][1]:+.2f}) 이미 넣은 물체 {fill.get(d['id'], 0)}개")
        L += ["", "## 남은 물체(위치는 지금 값)"]
        for m in miss:
            ref = REF.get(m["id"]) or {}; r = last.get(m["id"]) or {}; x, y = m["xy"]
            L.append(f"- {m['id']}: 라벨 {m.get('label_top1')} 크기 {ref.get('size_cm')}cm 위치 ({x:+.2f},{y:+.2f}) 반경 {100 * math.hypot(x, y):.0f}cm 방위 {math.degrees(math.atan2(y, x)):+.0f}° | 이전 목적지 {m['dest']} | 이전 결과 pick={r.get('pick')} place={r.get('place')} outcome={r.get('outcome')}")
        L += ["", "## 출력(JSON 하나만)", '{"order":[{"id":"obj_N","dest":"<id|leave>","dest2":"<대안 id 또는 null>","why":"<한 줄>"}...]}  (남은 물체를 모두 포함)']
        t0 = time.time(); out = subprocess.run(["claude", "-p", "--tools", "", "--output-format", "text"], input="\n".join(L), capture_output=True, text=True, timeout=180, cwd="/tmp").stdout
        js = json.loads(out[out.index("{"): out.rindex("}") + 1])["order"]; byid = {m["id"]: m for m in miss}; new = []; seen = set(); dropped = []
        for o in js:
            m = byid.get(o.get("id"))
            if not m or o["id"] in seen: continue
            seen.add(o["id"]); d1 = o.get("dest")
            if d1 == "leave": dropped.append(o["id"]); continue
            if d1 not in DK: d1 = m["dest"]
            d2 = o.get("dest2"); d2 = d2 if d2 in DK and d2 != d1 else None
            new.append(dict(m, dest=d1, dest2=d2))
        new += [m for m in miss if m["id"] not in seen]
        msg = "재계획(%.0fs): " % (time.time() - t0) + ", ".join(f"{x['id']}→{x['dest']}" for x in new) + (f" | 포기 {dropped}" if dropped else "")
        print("[replan]", msg, file=sys.stderr)
        try: ST.llm_log("llm>robot", "plan", msg)
        except Exception: pass
        return new
    except Exception as e_:
        print("[replan] 실패 -- 원래 계획으로 재시도", repr(e_), file=sys.stderr); return miss

def run_chain(steps_in=None, _retry_round=False):
    """한 프로세스에서 물체 목록 전체를 이어서 처리한다(실패해도 홈 복귀/재시작/재탐색 없이 그 자리에서 다음으로). 상태 불확실(놓기 실패, 미끄러짐)이면 멈추고 판단을 돌려준다."""
    items = []
    for s_ in (steps if steps_in is None else steps_in):
        if s_.get("dest", "leave") == "leave": res.append(dict(id=s_["id"], dest="leave", result="건너뜀")); continue
        ref = REF.get(s_["id"]) or {}
        _alts = s_.get("dest_alt") or s_.get("dest2") or []; _alts = [_alts] if isinstance(_alts, str) else list(_alts); _alts = [x for x in _alts if x and x != "leave" and x in DK and x != s_["dest"]]      # 2026-10-15: 후보 2 -- 후보 1 이 안 풀리면(계획 실패/놓기 불가) 같은 자리에서 후보 2 로
        items.append(dict(id=s_["id"], dest=s_["dest"], nudge=s_.get("nudge"), kind=DK.get(s_["dest"]), alt=[dict(dest=x, kind=DK.get(x)) for x in _alts], hint=s_["xy"], desc=dict(size_cm=ref.get("size_cm"), color=ref.get("color"), top=ref.get("top"), feat=ref.get("feat"))))
    if not items: write("종료"); return
    pf = "/tmp/sort_chain_plan.json"; json.dump(items, open(pf, "w"), ensure_ascii=False); cj = "/tmp/multi_exec_chain.json"
    try: os.remove(cj)
    except Exception: pass
    f0 = items[0]; env = dict(os.environ, **ST.base_env(), NERO_FOCUS_SCAN="1")
    if _retry_round and os.environ.get("NERO_FINAL_RETRY_ONCE", "0") == "1": env["NERO_C1_OFF_FINAL"] = "1"      # 2026-10-30(기본 꺼짐): 끝 재시도 프로세스에서는 C1 을 끈다(전체 노력으로 한 번)
    if os.environ.get("NERO_REPLAN", "0") == "1":      # 2026-10-17: 1~2 라운드(처음 + 미룬 항목 한 바퀴)는 홈 재시도 없이 항목당 총 2회, 재계획 후 마지막 라운드에서만 홈 재시도(폴백)를 켜고 미루기는 끈다
        env["NERO_CHAIN_HOME_RETRY"] = "1" if _retry_round else "0"; env["NERO_CHAIN_DEFER"] = "0" if _retry_round else "1"
    if f0["kind"] == "hole":
        if os.environ.get("NERO_HOLE_BAND"): env["NERO_GRASP_BAND"] = os.environ["NERO_HOLE_BAND"]
        env["NERO_PLACE_AWARE"] = "1"
    logp = f"{LOGDIR}/chain{'_retry' if _retry_round else ''}.log"; g0 = ST.gt_snapshot(); TR.start(); t1 = time.time()
    with open(logp, "w") as lf:
        p = subprocess.Popen([os.path.join(HERE, "rt_run.sh"), "--target-desc", json.dumps(f0["desc"], ensure_ascii=False), f"--hint-xy={f0['hint'][0]},{f0['hint'][1]}", "--dests", dests, "--dest-id", f0["dest"], "--plan-file", pf, "--vocab", ",".join(ST.VOCAB), "--max-picks", "99", "--out", logp + ".json"], cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True); _KIDS.append(p)
        seen = 0; last = None
        while p.poll() is None:
            _nap(p, 1.0)
            try: last = json.load(open(cj))
            except Exception: continue
            while seen < len(last["out"]):
                o = last["out"][seen]; ok = o["outcome"] == "ok"
                res.append(dict(**({"bumped": o["bumped"]} if o.get("bumped") else {}), id=o["id"], dest=o["dest"], pick=o.get("pick", o["outcome"]), place=o.get("place") or o["outcome"], ok=ok, secs=round(o["t"] - (last["out"][seen - 1]["t"] if seen else 0), 1), log=logp, outcome=o["outcome"])); seen += 1
            write(f"{seen}/{len(items)} 처리 중 -- 현재 {last.get('current')}")
            if time.time() - t1 > 900: os.killpg(p.pid, signal.SIGKILL); break
    hit = TR.stop(f"{LOGDIR}/chain_trace{'_retry' if _retry_round else ''}.json"); g1 = ST.gt_snapshot()
    try: last = json.load(open(cj))
    except Exception: last = None
    while last and seen < len(last["out"]):
        o = last["out"][seen]; res.append(dict(**({"bumped": o["bumped"]} if o.get("bumped") else {}), id=o["id"], dest=o["dest"], pick=o.get("pick", o["outcome"]), place=o.get("place") or o["outcome"], ok=(o["outcome"] == "ok"), secs=round(o["t"] - (last["out"][seen - 1]["t"] if seen else 0), 1), log=logp, outcome=o["outcome"])); seen += 1
    if not _retry_round and os.environ.get("NERO_FINAL_RETRY", "1") == "1":      # 2026-10-16: 끝난 뒤 목적지 밖에 있는 물체(놓치거나 튕겨 나간 것)를 찾아 새 위치 힌트로 국소 탐색(마스크를 새로 받음)부터 다시 시도한다. 1회
        try: miss = _missing_steps_vision(items) if os.environ.get("NERO_NO_GT", "0") == "1" else _missing_steps(items, g0, g1)
        except Exception as e_: miss = []; print("[retry] 놓친 물체 판정 실패", repr(e_), file=sys.stderr)
        if miss:
            try: ST.llm_log("robot>llm", "warn", "끝난 뒤 목적지 밖 물체 재시도: " + ", ".join(f"{m['id']}→{m['dest']}(새 위치 {m['xy']})" for m in miss))
            except Exception: pass
            if os.environ.get("NERO_REPLAN", "0") == "1": miss = _replan(miss)
            write(f"놓친 물체 {len(miss)}개 재시도"); run_chain(miss, _retry_round=True); return
    moved_to = {k: [round(g1[k][0], 2), round(g1[k][1], 2)] for k in g0 if k in g1 and math.hypot(g1[k][0] - g0[k][0], g1[k][1] - g0[k][1]) > 0.02}
    done_ok = {r["id"] for r in res if r.get("ok")}; bad = [r for r in res if not r.get("ok") and r["id"] not in done_ok and r.get("dest") != "leave"]
    why = []
    if last and last.get("paused"): why.append("멈춤: " + last["paused"])
    for r in bad: why.append(f"{r['id']}->{r['dest']} {r['outcome']}")
    left = [x["id"] for x in items if x["id"] not in {r["id"] for r in res}]
    if why or left:
        PENDING.update(failed=(bad[-1]["id"] if bad else (res[-1]["id"] if res else items[0]["id"])), reason=" / ".join(why) + (f" / 처리 못 한 항목 {left}" if left else "") + (" / 이 동안 움직인 물체: " + ", ".join(f"{k.split('_',2)[-1][:14]}→({v[0]:+.2f},{v[1]:+.2f})" for k, v in moved_to.items()) if moved_to else "") + ". 재시도 여부는 판단 필요: 쓰러졌거나 옮겨졌으면 rt_survey 로 다시 보고 계획을 고쳐 rt_sort", remaining=left)
    _bumps = {r["id"]: r["bumped"] for r in res if r.get("bumped")}      # 2026-10-16: 놓은 직후 다른 물체/용기가 밀린 항목
    if _bumps:
        try: ST.llm_log("robot>llm", "warn", f"놓은 뒤 밀림 감지(용기/다른 물체가 2cm 이상 움직임): {_bumps}")
        except Exception: pass
    res.append(dict(id="_summary", dest="-", pick="-", place="-", ok=None, bumped=_bumps or None, secs=round(time.time() - t1, 1), moved_to=moved_to, first_hit=hit))
    try:
        _okn = len({r["id"] for r in res if r.get("ok")}); ST.llm_log("robot>llm", "result", (f"판단 필요: {PENDING.get('failed')} 실패 -- " + str(PENDING.get("reason"))[:200]) if PENDING else f"실행 완료: {_okn}/{len(items)}개 성공", ok=_okn, total=len(items), pending=bool(PENDING))
    except Exception: pass
    write("종료"); print(json.dumps(dict(ok=sum(1 for x in res if x.get("ok")), total=len(items)), ensure_ascii=False))

try: os.remove(f"{LOGDIR}/displace.log")      # 2026-10-15: 이전 실행의 용기 이동 기록이 남아 헷갈리지 않게 시작할 때 지운다
except Exception: pass
write("시작")
if plan.get("chain", True):
    run_chain(); sys.exit(0)
QUEUE = list(enumerate(steps)); DEFERRED = []
while QUEUE or DEFERRED:
    if not QUEUE: QUEUE, DEFERRED = DEFERRED, []      # 한 바퀴 돈 뒤 보류한 물체를 한 번 더 시도한다(그래도 실패하면 멈추고 판단을 돌려준다)
    i, st = QUEUE.pop(0)
    if st.get("dest", "leave") == "leave": res.append(dict(id=st["id"], dest="leave", result="건너뜀")); write(f"{i + 1}/{len(steps)} 건너뜀"); continue
    write(f"{i + 1}/{len(steps)} {st['id']} -> {st['dest']} 실행 중"); t1 = time.time(); g0 = ST.gt_snapshot(); TR.start(); r = run_one(i, st, st["xy"]); hit = TR.stop(f"{LOGDIR}/{i:02d}_{st['id']}_trace.json"); g1 = ST.gt_snapshot()
    moved = {k: round(math.hypot(g1[k][0] - g0[k][0], g1[k][1] - g0[k][1]), 3) for k in g0 if k in g1 and math.hypot(g1[k][0] - g0[k][0], g1[k][1] - g0[k][1]) > 0.02}      # 이번 물체를 옮기는 동안 2cm 넘게 움직인 물체(집은 물체 포함)
    ok = (r["place"].endswith("True") and r["place"].startswith("PLACED")) or r["place"].startswith("INSERTED")  # on_dest 가 참인 PLACED/INSERTED 만 성공
    res.append(dict(id=st["id"], dest=st["dest"], pick=r["pick"], place=r["place"], ok=bool(ok), secs=round(time.time() - t1, 1), log=r["log"], retried=r.get("retried"), moved=moved, moved_to={k: [round(g1[k][0], 2), round(g1[k][1], 2)] for k in moved if k in g1}, first_hit=hit, scan_s=r.get("scan_s")))
    write(f"{i + 1}/{len(steps)} {st['id']} 완료")
    if not ok and r["pick"] not in ("HELD", "SLIPPED") and not r.get("no_target") and not moved and not st.get("_deferred") and plan.get("stop_on_fail", True):      # 계획 실패(도달 불가 등)이고 아무것도 안 움직였으면 멈추지 않고 이 물체를 뒤로 미루어 옆 물체부터 한다(재탐색 없이 다음으로)
        st["_deferred"] = True; DEFERRED.append((i, st)); res[-1]["deferred"] = True; write(f"{i + 1}/{len(steps)} {st['id']} 계획 실패 -- 보류하고 다음 물체로"); continue
    if not ok and plan.get("stop_on_fail", True):      # 실패하면 다음 물체로 맹목적으로 넘어가지 않고 멈춰서 판단(LLM)에게 돌려준다
        _tx = open(r["log"], errors="replace").read(); _gr = re.findall(r"파지 기록: 접촉 ([\d.]+)m / 물체높이 ([\d.]+)m = (\d+)%, 접근각 (-?\d+)도, verdict=(\w+)", _tx); _pv = re.findall(r"pick 완료 verdict=(\w+)", _tx); _cf = re.findall(r"C 실패\([\d.]+s\) (\[.*?\])", _tx)
        if r.get("aborted"): why = f"대상을 못 찾음({res[-1]['secs']:.0f}초, 탐색 상한에서 중단) -- 위치가 바뀌었거나 가려졌을 수 있음"
        elif r.get("no_target"): why = "대상을 못 찾음(국소 탐색에서 임베딩 일치 없음) -- 위치가 바뀌었을 수 있음"
        elif r["pick"] != "HELD":
            why = f"집기 실패(판정 {r['pick']})"
            if _gr: g_ = _gr[-1]; why += f": 마지막 파지 접촉 높이 {int(g_[2])}%, 접근각 {g_[3]}도, 파지 판정 {g_[4]}"
            if _cf: why += f"; 계획 실패 사유 {_cf[-1][:140]}"
        else:
            why = f"놓기 실패({r['place']}); 집기는 {r['pick']}"
            if _gr: g_ = _gr[-1]; why += f"(접촉 높이 {int(g_[2])}%, 접근각 {g_[3]}도)"
        if moved: why += " / 이 동안 움직인 물체: " + ", ".join(f"{k.split('_',2)[-1][:14]} {v * 100:.0f}cm→({g1[k][0]:+.2f},{g1[k][1]:+.2f})" for k, v in moved.items() if k in g1)
        why += ". 재시도 여부는 판단 필요: 같은 계획 항목을 다시 rt_sort 로 보내면 재시도, 물체가 쓰러졌거나 옮겨졌으면 rt_survey 로 다시 보고 위치/목적지를 고쳐서 보내거나 dest=leave"
        PENDING.update(failed=st["id"], reason=why, remaining=[s_["id"] for s_ in steps[i + 1:] if s_.get("dest") != "leave"]); write(f"판단 대기: {st['id']} 실패"); break
write("종료"); print(json.dumps(dict(ok=sum(1 for x in res if x.get("ok")), total=len([x for x in res if x["dest"] != "leave"])), ensure_ascii=False))
