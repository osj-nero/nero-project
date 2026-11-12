#!/usr/bin/env python3
"""rt_mcp_server.py -- 실시간 pick 루프(discover_object.py)를 Claude 에게 노출하는 MCP 서버 (HTTP :9100).
Claude 는 두뇌(명령 해석, 영어 문구/어휘 확장, 도구 호출 순서)이고 눈은 검출기다: 이 서버의 모든 응답은 텍스트/수치뿐이며 이미지는 반환하지 않는다.
  rt_scan   : 팔이 한 바퀴 훑어 보이는 물체와 라벨 후보/점수를 텍스트로 돌려줌(집지 않음)
  rt_pick   : 대상/제외 문구로 pick 루프를 백그라운드로 시작
  rt_place  : "A 를 B 에 넣어줘/꽂아줘/올려줘" -- 집어서 목적지에 놓거나 꽂는다(방식은 크기 규칙이 정하고 근거를 돌려줌)
  rt_status : 진행 상황(단계, 물체별 라벨/대상 여부, 집은 결과)
  rt_stop   : 중지 + 팔 홈 복귀
실행: python3 rt_mcp_server.py   (포트 RT_MCP_PORT, 기본 9100)"""
import os, json, time, signal, subprocess
from mcp.server.fastmcp import FastMCP

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.expanduser(os.environ.get("NERO_RT_STATE", "~/grasp/rt_state.json"))
mcp = FastMCP("nero-rt")
_run = dict(proc=None, out=None, log=None, args=None, t0=0.0)


def _alive():
    p = _run["proc"]
    return p is not None and p.poll() is None


def _state():
    try:
        return json.load(open(STATE_PATH))
    except Exception:
        return {}


# 놓기/꽂기 자동 판단 평가(auto_rounds.sh)와 같은 설정 + 홀 기하 점검(바구니를 홀로 오인하지 않게)
AUTO_ENV = dict(NERO_AUTO_MODE="1", NERO_HOLE_SANITY="1", NERO_PLACE_SNAPS="1", NERO_PLACE_AWARE="0", NERO_INSERT_PLAN_DEPTH="0.04", NERO_INSERT_ROLL_STEP="15", NERO_FOCUS_SCAN="1", NERO_COND="E")


LLM_ENV = dict(NERO_LLM_DECIDE="1", NERO_LLM_TIMEOUT="20", NERO_DEST_LIGHT="1", NERO_RESTILL_FOCUS="1", NERO_ENT_MERGE_FOOT="1", NERO_CLEAR_RETRY="1", NERO_LEAN="1", NERO_CONTAINER_NOSOLID="1", NERO_DEST_FREEZE="1", NERO_CLEAR_MAX="3", NERO_BLOCKED_FREEZE="1", NERO_DEST_KIND_SKIP="1", NERO_HOLE_ONCE="1", NERO_RESTILL_KIND_ORDER="1")      # 2026-10-14: 판단을 Claude 에게 묻는 모드(기본 꺼짐)
import llm_decide


def _start(args, tag, env_extra=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"/tmp/rt_{tag}_{ts}.json"; logp = f"/tmp/rt_{tag}_{ts}.log"
    f = open(logp, "w")
    p = subprocess.Popen([os.path.join(HERE, "rt_run.sh")] + args + ["--out", out], stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, **(env_extra or {})))
    _run.update(proc=p, out=out, log=logp, args=args, t0=time.time())
    return p


def _args(target, exclude, vocab, extra=()):
    a = []
    if target:
        a += ["--target", target]
    if exclude:
        a += ["--exclude", exclude]
    if vocab:
        a += ["--vocab", vocab]
    return a + list(extra)


def _decision_text(st=None):
    """상태 파일의 판단 기록(어느 분기, 근거)을 한 줄 문장으로. 없으면 빈 문자열."""
    st = st or _state(); dc = st.get("decision")
    if not dc:
        return ""
    md = dc.get("mode"); how = dc.get("how"); ob = dc.get("obj") or {}; ds = dc.get("dest") or {}
    lab = {"insert": "꽂기(세워서 삽입)", "place": "놓기" + (f"({how})" if how else ""), None: "안 들어감 -> 중단"}.get(md, str(md))
    parts = [f"판단: {lab}"]
    if ob:
        parts.append(f"물체 {ob.get('kind')} 길이 {100 * float(ob.get('L', 0)):.1f}cm 단면 {100 * float(ob.get('c1', 0)):.1f}x{100 * float(ob.get('c2', 0)):.1f}cm")
    if ds:
        parts.append(f"목적지 {ds.get('kind')}" + (f" 개구부 {100 * ds['open'][0]:.1f}x{100 * ds['open'][1]:.1f}cm" if ds.get("open") else " (평면)"))
    if dc.get("clear_lie") is not None:
        parts.append(f"한 변 여유 눕힘 {1000 * float(dc['clear_lie']):.0f}mm / 단면 {1000 * float(dc['clear_up']):.0f}mm (기준 {1000 * float(dc.get('min_clear', 0.006)):.0f}mm 이상)")
    if dc.get("hole_check"):
        parts.append(str(dc["hole_check"]))
    parts.append("모드 전환 적용" if dc.get("applied") else "홀이 아니라 모드 전환 없이 기존 놓기 경로")
    return " | ".join(parts)


@mcp.tool()
def rt_scan(vocab: str = "", target: str = "", exclude: str = "") -> str:
    """팔이 j1 로 한 바퀴 훑어 테이블 위 물체를 찾고, 물체별 위치/크기/라벨 후보(top-3, 점수)를 텍스트로 돌려준다. 아무것도 집지 않는다(약 20~30초, 블로킹).
    이미지는 오지 않는다 -- 라벨은 CLIP 검출기가 붙인 것이다. 모호하면 vocab 에 후보 문구를 넣어 다시 스캔한다.
    Args:
        vocab: 라벨 후보 어휘(영어 쉼표 구분, 예 "cup,bottle,scissors"). 비우면 기본 어휘.
        target: (선택) 이 문구에 해당하는 물체를 match=True 로 표시(예 "red cup|mug", 동의어는 | 로)
        exclude: (선택) 이 문구에 해당하는 물체를 제외 표시
    Returns: 물체 목록(id, xy[m], 크기[cm], 라벨 후보 (점수), match)."""
    if _alive():
        return "실행 중인 pick 루프가 있음 -- rt_status 로 확인하거나 rt_stop 후 다시 호출"
    p = _start(_args(target, exclude, vocab, ["--scan-only"]), "scan")
    try:
        p.wait(timeout=150)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM); return "스캔 시간 초과(150s) -- 로그 " + _run["log"]
    try:
        d = json.load(open(_run["out"]))
    except Exception:
        return "스캔 실패 -- 로그 " + _run["log"]
    lines = [f"물체 {len(d['scene'])}개 ({d['total_s']:.0f}s)"]
    for oid, o in d["scene"].items():
        tops = ", ".join(f"{k}({v:+.1f})" for k, v in o.get("top", []))
        lines.append(f"- {oid}: xy={o['xy']} 크기={o['size_cm']}cm 라벨후보: {tops}" + ("  [match]" if o.get("match") and (target or exclude) else ""))
    if target or exclude:
        lines.append(f"점수 기준 최고: 대상 {d['refs']['t']} 제외 {d['refs']['x']}")
    return "\n".join(lines)


@mcp.tool()
def rt_pick(target: str = "", exclude: str = "", vocab: str = "", max_picks: int = 30, no_exec: bool = False,
            place_on: str = "", place_free: bool = False, place_xy: str = "") -> str:
    """대상 물체를 찾아 집는 실시간 루프를 백그라운드로 시작한다(즉시 반환). 진행은 rt_status 로 본다.
    먼저 사전 훑기로 장면 안 점수 기준을 잡고, 대상만 하나씩 정렬->CGN->B->C->pick 한다. 나머지는 장애물로만 등록한다.
    명령 해석 규칙(Claude 가 번역):
      - "컵 집어" -> target="cup|mug"   (영어 명사구, 동의어는 | 로)
      - "X 빼고 다 치워" -> target="" exclude="X의 영어"   (X 에 해당하는 물체만 건드리지 않음)
      - "빨간 물체만" -> target="a red object"
      - "책상 치워" -> target="" exclude=""  (보이는 물체 전부)
    vocab 에는 대상과 혼동될 만한 다른 물체의 영어 명사를 넣는다(대상의 동의어/상위어는 넣지 말 것: toy 를 찾으면서 vocab 에 'a toy' 금지).
    Args:
        target: 집을 대상 문구('|' 로 동의어). 비우면 제외 문구를 뺀 전부.
        exclude: 건드리지 않을 문구('|' 또는 ',' 로 여러 개)
        vocab: 라벨링용 경쟁 어휘(영어 쉼표 구분)
        max_picks: 최대 pick 수
        no_exec: True 면 계획까지만(팔 실행 없음)
        place_on: 집은 물체를 이 문구에 해당하는 물체 위/안에 놓는다(예 "box|bin", "plate"). 도착지는 같은 검출기로 찾는다
        place_free: 집은 물체를 책상의 빈 자리에 놓는다("치워줘/옮겨줘"에 도착지 지정이 없을 때)
        place_xy: "x,y"(m) 좌표에 놓는다(시험용). place_* 를 하나도 안 주면 집기만 한다(시뮬은 집은 물체를 치움)"""
    if _alive():
        return "이미 실행 중 -- rt_status 확인 또는 rt_stop"
    extra = ["--max-picks", str(max_picks)] + (["--no-exec"] if no_exec else [])
    if place_on:
        extra += ["--place-on", place_on]
    if place_free:
        extra += ["--place-free"]
    if place_xy:
        extra += ["--place-xy", place_xy]
    _start(_args(target, exclude, vocab, extra), "pick")
    return f"시작함: target={target!r} exclude={exclude!r} (로그 {_run['log']}). rt_status 로 진행 확인"


def _wait_next_decision(wait_s):
    """다음 판단 요청이 오거나 실행이 끝날 때까지(최대 wait_s) 기다린다. 요청이 오면 그 텍스트를, 끝나면 상태 요약을 돌려준다."""
    t_end = time.time() + wait_s
    while time.time() < t_end:
        p = llm_decide.pending()
        if p: return llm_decide.render(p)
        if not _alive(): return "실행 끝남\n" + rt_status()
        time.sleep(0.2)
    return "대기 중인 판단 없음(실행은 계속 중, rt_decision(wait_s=60) 또는 rt_status 로 확인)"


@mcp.tool()
def rt_place(target: str, destination: str, dest_kind: str = "", vocab: str = "", no_exec: bool = False, wait_s: int = 0, llm_decide: bool = False) -> str:
    """"A 를 B 에 넣어줘 / 꽂아줘 / 올려줘 / 옮겨줘" 명령을 실행한다. 대상 A 를 집어서 목적지 B 에 놓거나 꽂는다.
    어떻게 놓을지(그냥 놓기 / 눕혀 넣기 / 세워 꽂기 / 안 들어가면 중단)는 Claude 가 정하지 않는다: 물체와 목적지 개구부의 크기 관계 규칙이 정하고, 그 근거를 응답에 돌려준다.
    "넣어줘/꽂아줘/올려줘" 의 표현 차이로 방식을 바꾸지 말고 목적지 문구만 정확히 번역해서 넘겨라.
    **물체 이름이 아니라 눈에 보이는 외형을 한 문장으로 묘사해서 넘겨라**(색, 크기, 모양): CLIP 은 에셋 이름과 실제 모양이 다른 물체("공룡"이 초록 안드로이드 모양)를 못 맞추고,
    외형 문장은 맞춘다. 예: target="a small green android toy", destination="a flat green square plate". 사용자가 이름만 말하면 먼저 rt_scan 으로 보이는 물체와 모양을 확인한다.
    명령 해석 규칙(Claude 가 번역):
      - "파란 막대를 주황 홀더에 꽂아줘" -> target="blue bar|blue rectangular block" destination="orange holder|orange tube" dest_kind="hole"
      - "파란 막대를 바구니에 넣어줘" -> target="blue bar|blue rectangular block" destination="basket|tray|bin" dest_kind="container"
      - "파란 막대를 접시 위에 올려줘" -> target="blue bar|blue rectangular block" destination="plate|white plate" dest_kind="flat"
      - 장면에 무엇이 있는지 모르면 먼저 rt_scan 으로 보이는 물체와 라벨 후보를 확인한다.
    Args:
        target: 집을 물체의 영어 명사구('|' 로 동의어)
        destination: 놓을 곳의 영어 명사구('|' 로 동의어). 같은 검출기(CLIP)로 찾는다
        dest_kind: 목적지의 기하 종류를 명령의 뜻에서 골라 넘긴다. "hole" = 좁은 입구에 꽂아 넣는 통/홀더/관/구멍, "container" = 위가 열린 큰 용기(바구니, 상자, 그릇, 트레이), "flat" = 위에 올리는 평면(접시, 판, 책상). 장면에 목적지 후보가 여러 개일 때 오접지를 막는다. 모르면 비운다(그러면 문구 점수로만 고른다)
        vocab: 라벨 후보 어휘(영어 쉼표 구분, 선택)
        no_exec: True 면 계획까지만(팔 실행 없음)
        llm_decide: True 면 루프가 판단 지점(막은 물체 선택, 같은 종류 목적지 후보 여럿)에서 멈추고 Claude 에게 묻는다. 이때는 wait_s 를 쓰지 않는다: 이 응답에 첫 판단 요청이 바로 포함되고, rt_decide 의 응답에 다음 판단 요청(두 번째 치우기 등)이 포함된다. 요청을 읽는 즉시 rt_decide 로 답한다. '실행 끝남' 이 나오면 멈춘다(안 답하면 20초 뒤 규칙 기본값으로 진행). 한 바퀴 재탐색은 코드에서 없앴으므로 정말 필요하면 rt_scan 으로 직접 부른다
        wait_s: 0 이면 시작만 하고 즉시 반환(rt_status 로 확인). 양수면 최대 이 초(상한 150) 동안 끝나기를 기다렸다가 판단 근거와 결과를 함께 돌려준다
    Returns: 시작/완료 메시지. 판단(분기)과 근거(크기, 여유, 홀 점검)는 완료 후 rt_status 또는 이 응답에 텍스트로 나온다."""
    if _alive():
        return "이미 실행 중 -- rt_status 확인 또는 rt_stop"
    extra = ["--max-picks", "1", "--place-on", destination] + (["--dest-kind", dest_kind] if dest_kind in ("hole", "container", "flat") else []) + (["--no-exec"] if no_exec else [])
    _start(_args(target, "", vocab, extra), "place", dict(AUTO_ENV, **(LLM_ENV if llm_decide else {})))
    if llm_decide and not (wait_s and wait_s > 0):      # 2026-10-14: 첫 판단 요청이 올 때까지 기다렸다가 그 요청을 응답에 바로 포함한다(rt_place 와 rt_decision 호출 사이의 지연 제거)
        return f"시작함(판단 모드): target={target!r} destination={destination!r}\n" + _wait_next_decision(60)
    if wait_s and wait_s > 0:
        t_end = time.time() + min(int(wait_s), 150)
        while _alive() and time.time() < t_end:
            time.sleep(2)
        done = not _alive()
        return ("완료" if done else "아직 실행 중(시간 초과, rt_status 로 확인)") + "\n" + rt_status()
    return f"시작함: target={target!r} destination={destination!r} (자동 판단 모드, 로그 {_run['log']}). rt_status 로 진행과 판단 근거 확인"


@mcp.tool()
def rt_status() -> str:
    """현재/직전 pick 루프 상태: 실행 여부, 단계, 집은 결과(verdict), 물체별 라벨/대상 여부/상태, 최근 로그. 텍스트만."""
    st = _state(); alive = _alive()
    if not st:
        return "상태 없음(아직 실행한 적 없음)"
    lines = [f"실행 중={alive} 단계={st.get('stage')} 경과={time.time() - _run['t0']:.0f}s" if _run["t0"] else f"실행 중={alive} 단계={st.get('stage')}"]
    pk = st.get("picked", [])
    lines.append(f"집은 물체 {len(pk)}개: " + ", ".join(f"{p.get('label') or p['id']}={p['verdict']}" + (f"->{p['place']}" if p.get('place') else "") for p in pk))
    for oid, o in (st.get("objs") or {}).items():
        lines.append(f"- {oid}: {o.get('label') or '?'} 상태={o.get('status')} 대상={o.get('match')} 실패={o.get('fails')}")
    _pd = llm_decide.pending()
    if _pd: lines.append(f"판단 대기: 요청 {_pd['id']} ({_pd['kind']}) -- rt_decision 으로 상황 확인 후 rt_decide 로 답")
    dt = _decision_text(st)
    if dt:
        lines.append(dt)
    lines += ["로그: " + l for l in st.get("log", [])[-5:]]
    return "\n".join(lines)


@mcp.tool()
def rt_decision(wait_s: int = 0) -> str:
    """루프가 Claude 의 판단을 기다리고 있으면 그 상황을 텍스트로 돌려준다(엔티티 표: 라벨 후보/색/크기/위치/대상과의 관계/치울 수 있는지, 질문, 선택지). 없으면 '대기 중인 판단 없음'.
    rt_place(llm_decide=True) 실행 중에 rt_status 로 '판단 대기' 가 보이면 이걸 부른다. wait_s>0 이면 요청이 올 때까지 최대 그 초(상한 120) 기다린다(rt_place(llm_decide=True) 직후 wait_s=60 으로 부르면 지연 없이 받는다). 이미지는 없다: 라벨은 CLIP 추정이라 틀릴 수 있으니 크기/위치/관계 수치를 우선 근거로 삼는다."""
    p = llm_decide.pending()
    if not p and wait_s and wait_s > 0:      # 2026-10-14: 롱폴 -- rt_place(llm_decide=True) 직후 이걸 wait_s=60 으로 부르면 판단 요청이 생기는 즉시(0.3초 이내) 돌려준다(폴링 간격만큼 늦던 지연 제거). 실행이 끝나면 바로 돌아온다
        t_end = time.time() + min(int(wait_s), 120)
        while time.time() < t_end:
            p = llm_decide.pending()
            if p or not _alive(): break
            time.sleep(0.3)
    return llm_decide.render(p) if p else ("대기 중인 판단 없음" + ("" if _alive() else " (실행 끝남)"))


@mcp.tool()
def rt_decide(decision_id: str, choice: str, reason: str = "") -> str:
    """rt_decision 이 보여준 판단 요청에 답한다.
    Args:
        decision_id: 요청 id(대괄호 안 8자리)
        choice: 선택지 id 중 하나(그대로)
        reason: 고른 이유 한 문장(기록용; 평가/논문에서 판단 근거로 쓰임)"""
    ok, msg = llm_decide.respond(decision_id, choice, reason)
    if not ok: return "실패: " + msg
    time.sleep(0.5)      # 2026-10-14: 답한 뒤 다음 판단 요청(두 번째 치우기 등)이 오거나 실행이 끝날 때까지 기다렸다가 같은 응답에 돌려준다
    return "전달함: " + msg + "\n\n" + _wait_next_decision(90)


@mcp.tool()
def rt_stop() -> str:
    """실행 중인 pick 루프를 중지하고 팔을 홈(관측) 자세로 되돌린다."""
    try: sort_tools.llm_log("llm>robot", "stop", "실행 중지 지시(rt_stop)")
    except Exception: pass
    if _alive():
        try:
            os.killpg(_run["proc"].pid, signal.SIGTERM); _run["proc"].wait(timeout=10)
        except Exception:
            try:
                os.killpg(_run["proc"].pid, signal.SIGKILL)
            except Exception:
                pass
    sp = _sort.get("proc")      # 2026-10-15: 다물체 정리(multi_exec)도 같이 중지 -- 이전엔 pick 루프만 죽이고 정리 프로세스는 계속 움직였다
    if sp is not None and sp.poll() is None:
        try: os.killpg(sp.pid, signal.SIGTERM); sp.wait(timeout=10)
        except Exception:
            try: os.killpg(sp.pid, signal.SIGKILL)
            except Exception: pass
    subprocess.run(["bash", "-lc", "pkill -f '^bash ./pick_one.sh' ; pkill -f '^python3 .*exec_pick.py'; true"], capture_output=True)
    r = subprocess.run(["python3", os.path.join(HERE, "force_home.py")], cwd=HERE, capture_output=True, text=True, timeout=60, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"))
    return "중지하고 홈 복귀 " + ("완료" if r.returncode == 0 else "실패: " + r.stderr[-200:])


# ---- 다물체 정리(2026-10-14): 목적지는 입력 명세, 물체는 조사로 얻고, 어느 물체를 어디로 보낼지는 Claude 가 정한다 ----
import sort_tools
_sort = dict(proc=None, status="/tmp/multi_exec_status.json", t0=0.0)


@mcp.tool()
def rt_survey(dests_path: str) -> str:
    """장면 전체를 한 번 훑어 물체 표를 돌려준다(물체별 id, 대략적 라벨 top3(틀릴 수 있음), 색, 크기 cm, 위치 xy m) + 입력으로 준 목적지 목록(종류/크기). dests_path = 목적지 명세 JSON.
    이 표를 보고 각 물체를 어느 목적지로 보낼지(또는 보내지 않을지) 정해 rt_sort 에 넘긴다. 약 30초."""
    if _alive() or (_sort["proc"] is not None and _sort["proc"].poll() is None):
        return "실행 중인 작업이 있음 -- rt_stop 또는 끝날 때까지 대기"
    sort_tools.llm_log("llm>robot", "survey", "장면 조사 요청(rt_survey)"); _t0 = time.time()
    sc = sort_tools.objects_only(sort_tools.survey(dests_path), dests_path)
    json.dump(sc, open("/tmp/sort_objects.json", "w"), ensure_ascii=False)
    _tt = sort_tools.table_text(sc, dests_path)
    sort_tools.llm_log("robot>llm", "survey", f"조사 결과 전달: 물체 {len(sc)}개 표(id·라벨 top3·색·크기·위치·이웃 틈) + 목적지 목록, 조사 {time.time() - _t0:.0f}s", n=len(sc))
    return _tt


@mcp.tool()
def rt_sort(plan: str, dests_path: str) -> str:
    """정한 계획을 순서대로 실행한다(물체 하나씩 집어서 목적지에 놓기, 백그라운드). plan = JSON 문자열
    {"order":[{"id":"obj_3","xy":[x,y],"dest":"<목적지 id 또는 leave>","dest2":"<후보 2 목적지 id(선택): dest 가 계획 실패/놓기 불가일 때 같은 자리에서 바로 시도>","nudge":"<true 또는 밀 거리 m(선택): 이웃과 틈이 좁아 집기가 막히는 물체를 집기 전에 가장 가까운 이웃 반대쪽으로 밀어 틈을 벌림(첫 항목 제외)>","label_top1":"표의 첫 라벨"}, ...]}. 한 물체가 실패하면 멈추고 사유를 돌려준다(stop_on_fail, 기본): 그때 rt_survey 로 장면을 다시 보고 남은 계획을 고쳐 다시 부른다. 놓기 쉬운 순서(앞쪽/큰 것 먼저)로 배열하고, 어디에도 맞지 않는 물체는 dest="leave".
    진행은 rt_sort_status 로 본다."""
    if _alive() or (_sort["proc"] is not None and _sort["proc"].poll() is None):
        return "이미 실행 중"
    pj = "/tmp/sort_plan.json"; json.dump(json.loads(plan), open(pj, "w"), ensure_ascii=False); _sort["dests"] = dests_path
    try:
        _pl = json.loads(plan)["order"]; sort_tools.llm_log("llm>robot", "plan", f"계획 {len(_pl)}개: " + ", ".join(f"{i + 1}.{x.get('label_top1') or x['id']}→{x.get('dest')}" + (f"(대안 {x['dest2']})" if x.get("dest2") else "") for i, x in enumerate(_pl)), n=len(_pl))
    except Exception: pass
    f = open("/tmp/sort_run.log", "w")
    _sort.update(proc=subprocess.Popen(["python3", os.path.join(HERE, "multi_exec.py"), pj, dests_path, "--status", _sort["status"]], cwd=HERE, stdout=f, stderr=subprocess.STDOUT, start_new_session=True), t0=time.time())
    return "시작함. rt_sort_status 로 진행 확인"


@mcp.tool()
def rt_resume(retry_failed: bool = False, drop: str = "", dest: str = "") -> str:
    """멈춘 정리를 재조사 없이 바로 이어 간다(직전 rt_sort 계획의 남은 항목, 물체 id 그대로). retry_failed=True 면 실패한 항목도 다시 맨 앞에 넣는다.
    drop = 뺄 id 를 쉼표로(예 "obj_5,obj_7"), dest = 목적지를 바꿀 항목 JSON(예 {"obj_9":"bin2"}). 물체가 쓰러지거나 크게 옮겨졌으면 rt_survey 로 다시 보고 rt_sort 를 쓴다."""
    if _alive() or (_sort["proc"] is not None and _sort["proc"].poll() is None):
        return "이미 실행 중"
    try:
        st = json.load(open(_sort["status"])); plan = json.load(open("/tmp/sort_plan.json"))
    except Exception as e:
        return f"이어 갈 계획 없음: {e}"
    pd_ = st.get("pending") or {}
    done = {r["id"] for r in st.get("results", []) if r.get("ok")}
    left = set(pd_.get("remaining") or [])
    if retry_failed and pd_.get("failed"): left.add(pd_["failed"])
    if not pd_:      # 판단 대기 없이 끝났어도 못 한 항목(성공 아님)은 이어 간다
        left = {it["id"] for it in plan["order"] if it["id"] not in done and it.get("dest") != "leave"}
    dr = {x.strip() for x in drop.split(",") if x.strip()}; ov = json.loads(dest) if dest.strip() else {}
    order = []
    for it in plan["order"]:
        if it["id"] in left and it["id"] not in dr:
            it = dict(it); it["dest"] = ov.get(it["id"], it["dest"]); order.append(it)
    if not order: return "남은 항목 없음"
    dests_path = "/tmp/dests_cur.json"
    for cand in (st.get("dests_path"), plan.get("dests_path")):
        if cand: dests_path = cand
    pj = "/tmp/sort_plan.json"; json.dump(dict(plan, order=order), open(pj, "w"), ensure_ascii=False)
    f = open("/tmp/sort_run.log", "w")
    _sort.update(proc=subprocess.Popen(["python3", os.path.join(HERE, "multi_exec.py"), pj, _sort.get("dests", dests_path), "--status", _sort["status"]], cwd=HERE, stdout=f, stderr=subprocess.STDOUT, start_new_session=True), t0=time.time())
    sort_tools.llm_log("llm>robot", "resume", f"재조사 없이 이어 가기: {len(order)}개 " + ", ".join(f"{i.get('label_top1') or i['id']}→{i['dest']}" for i in order), n=len(order))
    return f"이어 감: {[i['id'] + '->' + i['dest'] for i in order]}. rt_sort_status 로 확인"


@mcp.tool()
def rt_sort_status() -> str:
    """rt_sort 진행: 단계, 물체별 결과(집기 verdict, 놓기 verdict, 성공 여부, 소요 초)."""
    try:
        st = json.load(open(_sort["status"]))
    except Exception:
        return "상태 없음"
    alive = _sort["proc"] is not None and _sort["proc"].poll() is None
    L = [f"실행 중={alive} 단계={st['stage']} 경과={st['t']}s"]
    if st.get("pending"):
        pd_ = st["pending"]; L.append(f"판단 필요: {pd_['failed']} 실패 -- {pd_['reason']}. 남은 계획 {pd_['remaining']}. rt_resume 으로 재조사 없이 바로 이어 가거나(물체가 크게 안 움직였을 때), rt_survey 로 다시 보고 계획을 고쳐 rt_sort 로 이어 간다")
    for r in st["results"]:
        L.append(f"- {r['id']}->{r['dest']}: " + (r["result"] if "result" in r else f"집기={r['pick']} 놓기={r['place']} 성공={r['ok']} {r['secs']}s"))
    return "\n".join(L)



if __name__ == "__main__":
    mcp.settings.host = os.environ.get("RT_MCP_HOST", "127.0.0.1"); mcp.settings.port = int(os.environ.get("RT_MCP_PORT", "9100"))
    mcp.run(transport="streamable-http")
