#!/usr/bin/env python3
"""rt_plus.py -- nero-rt MCP 서버(rt_mcp_server.py)를 **수정하지 않고** 감싸서 개편 기능을 붙인다(2026-11-03, code_pro).
phase_c 는 후보 코드·해시 대상이라 건드리지 않는다: 이 파일은 phase_c 밖(~/nero/mcp/)에 있고 rt_mcp_server 를 import 해서 도구 설명 갱신·호출 로깅·응답 보완·신규 도구를 붙인 뒤 같은 포트(9100)로 띄운다.
선택: 환경변수 NERO_RT_MCP_PLUS=1 일 때만 nero_pc_control.sh 가 이 파일을 실행(기본 0 = 기존 rt_mcp_server.py 그대로).
1단계(11/3): ① 모든 호출을 HDD jsonl 로 기록(명령·인자·모델 ID 추정·응답 요약) ② rt_place 응답 첫 줄을 '결과: 성공/실패(사유)'로 보완 ③ 도구 설명·서버 instructions 갱신.
기록 위치: ~/nero/analysis/nl_runs/<날짜>.jsonl (NERO_NL_LOG_DIR 로 변경, /tmp 금지).
시험: python3 rt_plus.py --selftest   (서버를 띄우지 않고 등록·설명·instructions·분류기·로깅을 확인)"""
import os, sys, json, time, glob, asyncio, hashlib, threading
PHASE_C = os.environ.get("NERO_PHASE_C") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase_c_L1"); sys.path.insert(0, PHASE_C)
LOG_DIR = os.path.expanduser(os.environ.get("NERO_NL_LOG_DIR", "~/nero/analysis/nl_runs"))
import rt_mcp_server as R
_lock = threading.Lock(); _seq = [0]; NOTE = dict(id=None, text=None, model=None, t=0.0)      # rt_note(3단계)가 채움

# ---------------------------------------------------------------- 기록
def _model_guess():
    """MCP 서버는 호출한 모델을 직접 알 수 없다 -> (1) 환경변수 NERO_NL_MODEL_ID, (2) rt_note 로 받은 값, (3) 가장 최근 갱신된 Claude Code 대화 기록(jsonl)의 마지막 assistant 모델 ID 를 '추정'으로 적는다."""
    if os.environ.get("NERO_NL_MODEL_ID"): return os.environ["NERO_NL_MODEL_ID"], "env"
    if NOTE.get("model"): return NOTE["model"], "rt_note"
    try:
        cands = glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")) + glob.glob(os.path.expanduser(os.environ.get("NERO_NL_TRANSCRIPT_GLOB", "~/.config/claude/projects/*/*.jsonl")))
        cands = [(os.path.getmtime(f), f) for f in cands]; cands = [c for c in cands if time.time() - c[0] < 600]
        if not cands: return "", "none"
        f = max(cands)[1]; sz = os.path.getsize(f)
        with open(f, "rb") as fh: fh.seek(max(0, sz - 300000)); tail = fh.read().decode("utf-8", "ignore")
        i = tail.rfind('"model":"')
        if i < 0: return "", "transcript(no model)"
        m = tail[i + 9:i + 80].split('"')[0]
        return m, "transcript_tail_guess:" + os.path.basename(os.path.dirname(f))[-24:] + "/" + os.path.basename(f)[:8]
    except Exception as e:
        return "", f"err:{e!r}"[:60]

def _run_id():
    lp = R._run.get("log") or ""
    return os.path.splitext(os.path.basename(lp))[0] if lp else ""

def _log(rec):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with _lock, open(os.path.join(LOG_DIR, time.strftime("%Y%m%d") + ".jsonl"), "a", buffering=1) as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
    except Exception: pass

# ---------------------------------------------------------------- 결과 분류(N8)
PLACE_OK = ("PLACED", "INSERTED")
def _tail_text():
    lp = R._run.get("log")
    try: return open(lp, errors="ignore").read()[-8000:] if lp else ""
    except Exception: return ""

def classify(st=None, logtxt=None, rc=None, no_exec=None):
    """실행이 끝난 뒤 상태 파일·실행 로그로 결과를 분류한다. 반환 dict(ok: True/False/None, label, reason). 판정은 도구 쪽 기록(verdict, 로그 문구)만 쓰며 새 측정은 없다."""
    st = st if st is not None else R._state(); txt = (logtxt if logtxt is not None else _tail_text()) + "\n" + "\n".join(st.get("log", []))
    if no_exec is None: no_exec = "--no-exec" in (R._run.get("args") or [])
    if rc is None and R._run.get("proc") is not None: rc = R._run["proc"].poll()
    if rc not in (None, 0): return dict(ok=False, label="비정상 종료", reason=f"프로세스 종료코드 {rc}" + (" (중지됨)" if rc in (-15, -9) else ""))
    pk = st.get("picked") or []
    if pk:
        p = pk[-1]; pv = str(p.get("verdict") or ""); pl = str(p.get("place") or ""); who = p.get("label") or p.get("id")
        if pl in PLACE_OK: return dict(ok=True, label="성공", reason=f"{who}: 집기 {pv} -> 놓기 {pl}")
        if pl == "NO_PLAN": return dict(ok=False, label="실패", reason=f"{who}: 놓기 계획 실패(NO_PLAN: 후보 중 IK/도달 가능한 놓기 자세 없음)")
        if pl in ("MISSED", "TOPPLED", "NOT_INSERTED", "EXEC_FAIL", "HELD_STILL", "NO_SLOT", "SERVER_ERR", "SAVE_ERR"): return dict(ok=False, label="실패", reason=f"{who}: 놓기 {pl} (집기 {pv})")
        if pv in ("SLIPPED", "MISSED"): return dict(ok=False, label="실패", reason=f"{who}: 파지 실패({pv})")
        if pv == "NO_REACH": return dict(ok=False, label="실패", reason=f"{who}: 도달 불가(NO_REACH)")
        if pv == "HELD" and not pl: return dict(ok=True, label="성공", reason=f"{who}: 집기 HELD(놓기 요청 없음)")
        return dict(ok=None, label="불명", reason=f"{who}: 집기 {pv} 놓기 {pl or '-'} — 로그 확인")
    for key, label, why in (("도착지 없음 -- 종료", "실패", "도착지 없음(목적지를 못 찾음 — 장면에 해당 용기가 없거나 문구/종류가 안 맞음)"),
                            ("대상 없음 -- 종료", "실패", "대상 없음(대상 물체를 못 찾음 — 문구를 바꾸거나 rt_scan 으로 확인)"),
                            ("한 바퀴 훑는 동안 더 집을 물체 없음", "실패", "집을 물체 없음(대상 후보가 계획 단계에서 모두 제외됨)")):
        if key in txt: return dict(ok=False, label=label, reason=why)
    if "계획 실패" in txt and not no_exec: return dict(ok=False, label="실패", reason="계획 실패(집기 후보/IK 없음 — 로그 확인)")
    if no_exec and str(st.get("stage")) == "종료": return dict(ok=True, label="계획 완료(실행 없음)", reason="no_exec=True: 접지·계획 단계까지 끝남(팔 실행·판정 없음)")
    return dict(ok=None, label="불명", reason="상태 파일/로그에서 결과를 판별하지 못함 — rt_status 와 로그 확인")

def _first_line(c): return f"결과: {c['label']} — {c['reason']}"

# ---------------------------------------------------------------- 래퍼
def _wrap_logging(name, tool):
    orig = tool.fn
    def wrapper(*a, **kw):
        t0 = time.time(); _seq[0] += 1; seq = _seq[0]; mid, msrc = _model_guess(); rid0 = _run_id()
        rec = dict(t=t0, iso=time.strftime("%F %T", time.localtime(t0)), seq=seq, tool=name, args=kw, model=mid, model_src=msrc, note_id=NOTE.get("id"), run_id_before=rid0)
        try:
            out = orig(*a, **kw); rec.update(ok=True, sec=round(time.time() - t0, 2), run_id=_run_id(), result_len=len(str(out)), result_head=str(out)[:400]); return out
        except Exception as e:
            rec.update(ok=False, sec=round(time.time() - t0, 2), error=repr(e)[:300]); raise
        finally: _log(rec)
    wrapper.__name__ = orig.__name__; wrapper.__doc__ = orig.__doc__; tool.fn = wrapper

def _wrap_rt_place(tool):
    orig = tool.fn
    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        try:
            s = str(out)
            if s.startswith("완료\n") or s.startswith("완료"):           # wait_s>0 로 끝났을 때만 원래 응답 첫 줄이 '완료'(프로세스 종료)
                c = classify(); return _first_line(c) + "\n(프로세스 종료)\n" + s.split("\n", 1)[1] if "\n" in s else _first_line(c)
        except Exception: pass
        return out
    wrapper.__name__ = orig.__name__; wrapper.__doc__ = orig.__doc__; tool.fn = wrapper


# ---------------------------------------------------------------- 완화안 5: dest 상세 (rt_status 에 덧붙임)
import re as _re
def dest_detail(st=None, logtxt=None):
    """상태=dest 인 물체의 id·월드 xy·CLIP 라벨/점수·종류·개구와 선정 근거(실행 로그의 도착지 단계 줄)를 한 블록으로. 상태 파일·실행 로그만 읽고 새 계산은 없음."""
    st = st if st is not None else R._state(); L = logtxt if logtxt is not None else _tail_text_full()
    objs = st.get("objs") or {}; dests = [(k, v) for k, v in objs.items() if v.get("status") == "dest"]
    if not dests and "도착지" not in L: return ""
    out = ["목적지(dest) 상세:"]
    kinds = {}; m = _re.search(r"목적지 종류 '(\w+)' 필터: 후보 (\{.*?\})", L)
    if m: kinds = dict(_re.findall(r"'(obj_\d+)': '(\w+)'", m.group(2))); out.append(f"  종류 필터 '{m.group(1)}' 적용: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    first = _re.search(r"도착지 '([^']*)': (\[[^\]]*\])", L); geo = _re.search(r"도착지\(기하 확인 후\): (\[[^\]]*\])", L); sel = _re.findall(r"도착지 예상 선택 (obj_\d+) xy=\(([+-][\d.]+),([+-][\d.]+)\) s_p_adj=([-\d.]+)", L)
    if first: out.append(f"  1차(CLIP 문구 '{first.group(1)}' + 오목 가산) 후보: {first.group(2)}")
    if geo: out.append(f"  기하 확인 후 후보: {geo.group(1)}")
    for k, v in dests:
        xy = v.get("xyz") or [None, None]; hole = _re.findall(rf"도착지 후보 {k} 홀 추정: E=\[([^\]]*)\].*?개구부 ([\d.]+)cm", L); ring = _re.findall(rf"도착지 후보 {k} 링 반지름 방식: 바깥 ([\d.]+)->([\d.]+)cm 개구 ([\d.]+)->([\d.]+)cm", L)
        sc = _re.findall(rf"도착지 점수 {k} '([^']*)' \(([^)]*)\) s_p=([-\d.]+)", L); raw = v.get("raw") or []
        txt = f"  - {k}: 월드 xy=({xy[0]:+.3f},{xy[1]:+.3f}) 라벨(CLIP)={v.get('label') or '?'} CLIP점수={v.get('s')} VLM라벨={(vlm_map(L).get(k) or 'VLM 없음')} 종류={kinds.get(k, '?')} 크기(cm)={[round(100 * x, 1) for x in raw]}"
        if hole: txt += f" 홀 추정 E=[{hole[-1][0]}] 개구 {hole[-1][1]}cm"
        if ring: txt += f" (링 반지름 보정 전 개구 {ring[-1][2]}cm)"
        if not sc and not first: txt += ""
        out.append(txt)
        why = []
        if not sc: why.append("CLIP 점수 없음 -> 1차 문구 후보가 아니라 홀 기하 확인으로 추가됨")
        if kinds.get(k) == "hole": why.append("dest_kind 필터에서 hole 로 판정")
        if why: out.append("    선정 근거: " + "; ".join(why))
    if sel: out.append("  예상 선택: " + ", ".join(f"{a} xy=({b},{c}) 조정점수 {d}" for a, b, c, d in sel[-1:]))
    return "\n".join(out)

def _tail_text_full():
    lp = R._run.get("log")
    try: return open(lp, errors="ignore").read() if lp else ""
    except Exception: return ""

def _wrap_rt_status(tool):
    orig = tool.fn
    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        try:
            d = dest_detail()
            if d: return str(out) + "\n" + d
        except Exception as e: return str(out) + f"\n(dest 상세 생성 실패: {e!r})"
        return out
    wrapper.__name__ = orig.__name__; wrapper.__doc__ = orig.__doc__; tool.fn = wrapper


# ---------------------------------------------------------------- VLM 라벨 병기 (NERO_VLM_LABEL=1: phase_c 기존 플래그, 기본 0)
import ast as _ast
def vlm_map(logtxt=None):
    """최신 실행 로그의 'VLM 라벨 모으기 … {obj_N: 라벨}' 줄 -> dict(없으면 {})."""
    L = logtxt if logtxt is not None else _tail_text_full()
    m = _re.findall(r"VLM 라벨 모으기 [^\n]*?: (\{.*\})", L)
    try: return _ast.literal_eval(m[-1]) if m else {}
    except Exception: return {}

def _annotate_vlm(text, V=None, st=None):
    """응답의 '- obj_N: …' 줄에 'VLM라벨 X (CLIP 라벨/점수)' 를 병기. VLM 없으면 'VLM 없음'. 이미 병기된 줄은 건드리지 않음."""
    V = V if V is not None else vlm_map(); st = st if st is not None else R._state(); objs = (st or {}).get("objs") or {}; out = []
    for ln in str(text).split("\n"):
        m = _re.match(r"^(- (obj_\d+): )(.*)$", ln)
        if m and "VLM" not in ln:
            oid = m.group(2); v = V.get(oid); o = objs.get(oid) or {}; clip = f"{o.get('label') or '?'}/{o.get('s')}" if o else None
            tag = f"VLM라벨 '{v}'" if v else "VLM 없음"
            ln = m.group(1) + tag + (f" (CLIP {clip})" if clip else "") + " " + m.group(3)
        out.append(ln)
    return "\n".join(out)

def _wrap_vlm(tool):
    orig = tool.fn
    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        try: return _annotate_vlm(out)
        except Exception as e: return str(out) + f"\n(VLM 병기 실패: {e!r})"
    wrapper.__name__ = orig.__name__; wrapper.__doc__ = orig.__doc__; tool.fn = wrapper


def _scan_json():
    try: return json.load(open(R._run.get("out"))).get("scene", {})
    except Exception: return {}

def _annotate_scan(text):
    """rt_scan 응답: scan JSON 의 scene[oid]['vlm'] 과 CLIP top1 을 'VLM라벨 x (CLIP 라벨/점수)' 로 병기. VLM 없으면 'VLM 없음'."""
    sc = _scan_json(); out = []
    for ln in str(text).split("\n"):
        m = _re.match(r"^(- (obj_\d+): )(.*)$", ln)
        if m and "VLM" not in ln and sc:
            o = sc.get(m.group(2)) or {}; v = o.get("vlm"); top = o.get("top") or []
            clip = f"{top[0][0]}/{top[0][1]:+.1f}" if top else "-"
            ln = m.group(1) + (f"VLM라벨 '{v}'" if v else "VLM 없음") + f" (CLIP top1 {clip}) " + m.group(3)
        out.append(ln)
    return "\n".join(out)

def _wrap_vlm_scan(tool):
    orig = tool.fn
    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        try: return _annotate_scan(out)
        except Exception as e: return str(out) + f"\n(VLM 병기 실패: {e!r})"
    wrapper.__name__ = orig.__name__; wrapper.__doc__ = orig.__doc__; tool.fn = wrapper

def _annotate_decision(text):
    """rt_decision 엔티티 표: 'obj_N' 로 시작하는 줄(앞의 '- ' 선택)에 VLM 라벨 병기(형식 미확인 -- 판단 요청이 실제로 뜰 때 확인 필요). 이미 병기된 줄은 건드리지 않음."""
    V = vlm_map(); out = []
    for ln in str(text).split("\n"):
        m = _re.match(r"^(\s*(?:- )?)(obj_\d+)(\b.*)$", ln)
        if m and "VLM" not in ln:
            v = V.get(m.group(2)); ln = f"{m.group(1)}{m.group(2)} [{'VLM ' + repr(v) if v else 'VLM 없음'}]{m.group(3)}"
        out.append(ln)
    return "\n".join(out)

def _wrap_vlm_decision(tool):
    orig = tool.fn
    def wrapper(*a, **kw):
        out = orig(*a, **kw)
        try: return _annotate_decision(out)
        except Exception as e: return str(out) + f"\n(VLM 병기 실패: {e!r})"
    wrapper.__name__ = orig.__name__; wrapper.__doc__ = orig.__doc__; tool.fn = wrapper

# ---------------------------------------------------------------- 설명·instructions
def _patch_desc(tool, old, new, append=None):
    d = tool.description or ""
    if old and old in d: d = d.replace(old, new, 1)
    elif old: d = d + f"\n[개편 11/3 주의] 원문 문구 '{old[:20]}…' 를 찾지 못해 덧붙임: {new}"
    if append: d = d + "\n" + append
    tool.description = d

INSTRUCTIONS = """nero-rt: 자연어 로봇 제어의 표준 경로(CLIP 접지 -> Contact-GraspNet -> cuRobo -> 실행). 응답은 텍스트·수치뿐이며 이미지는 오지 않는다(라벨은 CLIP/VLM 이 붙인 것이라 틀릴 수 있다).
[어느 도구] 장면 파악 rt_scan(집지 않음) / 집기 rt_pick / 놓기·넣기·꽂기·올리기 rt_place / 진행·결과 rt_status / 중지·홈 rt_stop / 다물체 정리 rt_survey -> rt_sort -> rt_sort_status(-> rt_resume) / 루프가 판단을 물으면 rt_decision -> rt_decide.
[nero-robot(:9000) 은 비권장] 그 서버의 지각 도구는 현재 스택에서 쓸 수 없다(get_system_status 의 vision_objects=0). 자연어 작업은 이 서버(nero-rt)를 쓴다.
[규칙 적용 범위] CLAUDE.md 의 박스 높이·infer_grasp·box_index·place_object 좌표 보정·placement_verified 규칙은 nero-robot 대화형 경로용이며 이 서버에는 적용되지 않는다(CGN 경로는 infer_grasp 불필요, 놓을 곳은 rt_place/rt_pick(place_on) 이 같은 검출기로 스스로 찾는다). 이 서버의 호출 규칙은 각 도구 설명(영어 명사구·'|' 동의어, 외형 묘사, dest_kind, vocab 에 대상 동의어 금지)을 따른다.
[응답 읽는 법] rt_place(wait_s>0)의 첫 줄은 '결과: 성공/실패(사유)/계획 완료(실행 없음)'이다. '프로세스 종료'는 성공을 뜻하지 않는다. 시작만 하는 호출(wait_s=0, rt_pick)은 rt_status 로 결과를 확인한다.
[기록] 모든 호출(도구·인자·응답 요약·모델 ID 추정)은 HDD(~/nero/analysis/nl_runs/)에 기록된다.
[금지·사용자 지시 사항] 장면 초기화·Isaac 재시작·▶ 재생은 사용자가 지시할 때만. 봉인(시험 구간) 장면·시드는 다루지 않는다."""

def install():
    tm = R.mcp._tool_manager; T = tm._tools
    _patch_desc(T["rt_scan"], "(약 20~30초, 블로킹)", "(약 13~30초, 블로킹; 11/3 실측 13초·호출 19초)")
    _patch_desc(T["rt_survey"], "약 30초.", "약 45~55초(전체 스캔·목적지 인식 포함, 11/2 평가 스캔 실측 44~52초).")
    _patch_desc(T["rt_place"], None, None, "[개편 11/3] wait_s>0 로 끝나면 응답 첫 줄은 '결과: 성공 / 실패 — 사유 / 계획 완료(실행 없음)' (상태 파일 verdict 와 실행 로그로 분류). 실패 사유 예: 도착지 없음, 대상 없음, 놓기 계획 실패(NO_PLAN), 파지 실패(SLIPPED/MISSED), 도달 불가(NO_REACH), 놓기 MISSED/TOPPLED/NOT_INSERTED/EXEC_FAIL.")
    _wrap_rt_place(T["rt_place"])      # 안쪽: 응답 첫 줄 보완
    _wrap_vlm_scan(T["rt_scan"]); _wrap_vlm_decision(T["rt_decision"])      # VLM 병기: 스캔·판단 요청 표
    _wrap_vlm(T["rt_status"])      # VLM 라벨 병기(상태 줄)
    _wrap_rt_status(T["rt_status"])      # 완화안 5: 상태=dest 물체의 위치·라벨·점수·종류·개구·근거 덧붙임
    _patch_desc(T["rt_status"], None, None, "[개편 11/3] 끝에 '목적지(dest) 상세' 블록(상태=dest 물체의 월드 xy·CLIP 라벨/점수·종류·개구·선정 근거)이 붙는다.")
    for n, t in T.items(): _wrap_logging(n, t)      # 바깥쪽: 클라이언트가 실제로 받는(보완된) 응답을 기록
    R.mcp._mcp_server.instructions = INSTRUCTIONS
    return sorted(T)

def main():
    os.environ.setdefault("NERO_VLM_MATCH_FIRST", "1")      # 규칙 A(11/3 승인): VLM 라벨이 대상 문구와 일치하는 물체 우선(없으면 CLIP 폴백). 시연 환경만, 러너 환경 미설정
    os.environ.setdefault("NERO_VLM_LABEL", "1")      # VLM(Qwen3-VL) 크롭 라벨 비동기 수집(기본 0 은 phase_c 쪽; 러너 환경엔 설정 안 함)
    os.environ.setdefault("NERO_HOLE_OPEN_RING", "1")      # 완화안 4(11/3 승인): MCP 경로 개구 추정 원형 링 면적 보정 켬(기본 0 은 phase_c 쪽 기본; 러너 환경엔 설정 안 함)
    names = install()
    R.mcp.settings.host = os.environ.get("RT_MCP_HOST", "127.0.0.1"); R.mcp.settings.port = int(os.environ.get("RT_MCP_PORT", "9100"))
    print(f"[rt_plus] nero-rt+ 시작 {R.mcp.settings.host}:{R.mcp.settings.port} 도구 {len(names)}개, 기록 {LOG_DIR}", flush=True)
    R.mcp.run(transport="streamable-http")

def selftest():
    h0 = hashlib.sha256(open(os.path.join(PHASE_C, "rt_mcp_server.py"), "rb").read()).hexdigest()[:16]
    names = install(); T = R.mcp._tool_manager._tools; ok = []
    def chk(c, m): ok.append((bool(c), m)); print(("OK  " if c else "FAIL") + " " + m)
    chk(len(names) == 11, f"도구 11개 등록 유지: {names}")
    chk("13~30초" in T["rt_scan"].description and "45~55초" in T["rt_survey"].description, "rt_scan/rt_survey 설명 갱신")
    chk("결과: 성공" in T["rt_place"].description, "rt_place 설명에 응답 첫 줄 규약")
    chk(R.mcp._mcp_server.instructions and "nero-robot" in R.mcp._mcp_server.instructions, "서버 instructions 설정")
    # 분류기(합성 입력)
    C = classify
    chk(C(dict(picked=[dict(id="obj_2", label="cup", verdict="HELD", place="PLACED")], log=[]), "", 0, False)["ok"] is True, "분류: HELD->PLACED = 성공")
    chk(C(dict(picked=[], log=["도착지 없음 -- 종료"]), "", 0, True)["reason"].startswith("도착지 없음"), "분류: 도착지 없음")
    chk(C(dict(picked=[], log=["대상 없음 -- 종료"]), "", 0, False)["reason"].startswith("대상 없음"), "분류: 대상 없음")
    chk("NO_PLAN" in C(dict(picked=[dict(id="a", verdict="HELD", place="NO_PLAN")], log=[]), "", 0, False)["reason"], "분류: 놓기 계획 실패(NO_PLAN)")
    chk("SLIPPED" in C(dict(picked=[dict(id="a", verdict="SLIPPED", place="")], log=[]), "", 0, False)["reason"], "분류: 파지 실패(SLIPPED)")
    chk("NO_REACH" in C(dict(picked=[dict(id="a", verdict="NO_REACH", place="")], log=[]), "", 0, False)["reason"], "분류: 도달 불가(NO_REACH)")
    chk(C(dict(picked=[], log=[], stage="종료"), "", 0, True)["label"].startswith("계획 완료"), "분류: no_exec 계획 완료")
    chk(C(dict(picked=[], log=[]), "", -15, False)["label"] == "비정상 종료", "분류: 중지(rc=-15)")
    # 로깅(읽기 전용 도구 rt_status 를 실제 호출)
    out = asyncio.run(R.mcp._tool_manager.call_tool("rt_status", {}))
    f = os.path.join(LOG_DIR, time.strftime("%Y%m%d") + ".jsonl"); last = json.loads(open(f).read().splitlines()[-1]) if os.path.exists(f) else {}
    chk(last.get("tool") == "rt_status" and last.get("ok") is True and "result_head" in last, f"호출 기록 생성: {f} (model={last.get('model')!r} src={last.get('model_src')})")
    h1 = hashlib.sha256(open(os.path.join(PHASE_C, "rt_mcp_server.py"), "rb").read()).hexdigest()[:16]
    chk(h0 == h1 == "e24a1513cc579552", f"phase_c/rt_mcp_server.py 해시 불변 {h0} -> {h1}")
    print("selftest:", "PASS" if all(x for x, _ in ok) else "FAIL"); sys.exit(0 if all(x for x, _ in ok) else 1)

if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
