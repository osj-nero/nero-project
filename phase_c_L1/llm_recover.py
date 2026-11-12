#!/usr/bin/env python3
"""llm_recover.py -- C 조건용 LLM 복구 판단 훅(2026-10-22, 기본 꺼짐: NERO_LLM_RECOVER_HOOK=1). 사양: analysis/waste_20261022/LLM_HOOK_SPEC.md.
체인이 한 항목의 계획 실패를 확정한 직후(규칙이 다음 행동을 고르기 전) discover_object.py 가 decide() 를 부른다. 프롬프트는 replay2/prompt.txt(동결, 해시 확인)를 그대로 쓰고
입력은 replay2/replay.py 와 같은 형식("- 이름: JSON 값")이다. 반환 act: None(=규칙 현행 동작으로 폴백) | giveup | defer | alt | view.
기존 NERO_LLM_RECOVER(llm_advisor, E_adv)와 무관하다. 이 모듈은 환경변수를 읽기만 하고 로봇/ROS 를 쓰지 않는다."""
import os, json, time, re, hashlib, subprocess, urllib.request

ENV_SWITCH = "NERO_LLM_RECOVER_HOOK"      # 주 스위치(기존 NERO_LLM_RECOVER 와 구분)
PROMPT_DEFAULT = "~/nero/analysis/waste_20261022/replay2/prompt.txt"
PROMPT_SHA256 = "d63ca22852e4a046146bbe511c26c551172c03cb883eb701852af581051461b9"      # FROZEN_HASHES_replay2.txt
FIELD_ORDER = ("reasons", "no_cgn", "cgn_cands_per_attempt", "collision_objects_per_attempt", "view_retries_done", "planning_fail_count_this_item", "planning_seconds_spent", "blocked_note", "target_xy",
               "target_radius_from_base", "seconds_already_spent_on_this_item", "item_time_cap_seconds", "item_seconds_left_before_cap", "run_elapsed_seconds", "items_in_this_run",
               "planning_failure_events_earlier_in_this_run", "planning_seconds_spent_on_those_events")      # replay_inputs.json 의 필드 순서(rid 제외)
METHODS = ("시점 변경", "다음 후보", "목적지 전환", "순서 미룸")
INFRA_RE = re.compile(r"limit|overloaded|api error|rate|529|503|502|500|usage|quota|capacity|try again|unavailable|timed out|connection", re.I)
QUOTA_RE = re.compile(r"limit|quota|usage|rate|overloaded|capacity|529|429", re.I)
STATE = dict(disabled=None, prompt=None, calls=0)


def enabled():
    return os.environ.get(ENV_SWITCH, "0") == "1"


def backend():
    return os.environ.get("NERO_LLM_RECOVER_BACKEND", "qwen")


def timeout_s():
    return float(os.environ.get("NERO_LLM_RECOVER_TIMEOUT", "20"))


def url():
    return os.environ.get("NERO_LLM_RECOVER_URL", "http://127.0.0.1:18005/v1/chat/completions")


def info_text(ev):
    """replay2/replay.py 의 info() 와 같은 형식."""
    return "\n".join(f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in ev.items() if k != "rid")


def make_event(**kw):
    """입력 필드를 사양서 이름·순서로 정렬(누락/불필요 필드는 오류)."""
    missing = [k for k in FIELD_ORDER if k not in kw]; extra = [k for k in kw if k not in FIELD_ORDER]
    if missing or extra: raise ValueError(f"필드 불일치 missing={missing} extra={extra}")
    return {k: kw[k] for k in FIELD_ORDER}


def load_prompt():
    if STATE["prompt"] is not None: return STATE["prompt"]
    p = os.path.expanduser(os.environ.get("NERO_LLM_RECOVER_PROMPT", PROMPT_DEFAULT))
    txt = open(p, encoding="utf-8").read(); sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if sha != PROMPT_SHA256 and os.environ.get("NERO_LLM_RECOVER_PROMPT_ANY", "0") != "1": raise RuntimeError(f"프롬프트 해시 불일치 {sha[:8]} (동결 {PROMPT_SHA256[:8]})")
    STATE["prompt"] = txt; return txt


def json_objs(o):
    js = []; i = 0; dec = json.JSONDecoder()
    while True:
        i = o.find("{", i)
        if i < 0: break
        try: v, e = dec.raw_decode(o[i:]); js.append(v); i += e
        except Exception: i += 1
    return js


def parse(o):
    """마지막 유효 객체(decision 이 계속/포기)."""
    for j in reversed(json_objs(o or "")):
        if isinstance(j, dict) and j.get("decision") in ("계속", "포기"): return j
    return None


def call_qwen(prompt, tmo):
    t0 = time.time(); body = json.dumps(dict(model="qwen3-vl", messages=[dict(role="user", content=prompt)], temperature=0, max_tokens=200)).encode()
    r = urllib.request.urlopen(urllib.request.Request(url(), body, {"Content-Type": "application/json"}), timeout=tmo)
    return 0, json.load(r)["choices"][0]["message"]["content"], "", time.time() - t0


def call_claude(prompt, tmo):
    t0 = time.time(); p = subprocess.run(["claude", "-p", "--tools", "", "--output-format", "text"], input=prompt, capture_output=True, text=True, timeout=tmo, cwd="/tmp")
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def is_infra(rc, out, err, secs):
    if rc != 0: return True
    if INFRA_RE.search((out or "")[:500] + (err or "")[:500]) and "{" not in (out or ""): return True
    return secs < 3 and not (out or "").strip()


def act_for(res, alt_ok):
    """검증된 판단 -> 코드 동작. 반환 (act, degraded, 폴백사유|None)."""
    if res["decision"] == "포기": return "giveup", False, None
    m = res.get("method")
    if m == "순서 미룸": return "defer", False, None
    if m == "목적지 전환": return ("alt", False, None) if alt_ok else (None, False, "alt_unavailable")
    if m == "시점 변경": return "view", False, None
    if m == "다음 후보": return "view", True, None      # 후보 건너뛰기 인자가 코드에 없어 시점 변경과 같이 처리(degraded)
    return None, False, "method_not_allowed"


def decide(ev, cap_left, alt_ok):
    """ev: make_event() 결과. cap_left: 항목 누적 상한까지 남은 초(0 이하면 LLM 호출 없이 포기 강제). 반환 dict(act, decision, method, reason, fallback, fallback_reason, degraded, source, latency_s, backend, input_hash, raw, disabled)."""
    be = backend(); text = info_text(ev); h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    out = dict(act=None, decision=None, method=None, reason=None, fallback=False, fallback_reason=None, degraded=False, source="llm", latency_s=0.0, backend=be, input_hash=h, raw=None, disabled=STATE["disabled"])
    if cap_left <= 0:
        out.update(act="giveup", decision="포기", method="없음", reason="항목 누적 시간 상한", source="cap_guard"); return out
    if STATE["disabled"]:
        out.update(fallback=True, fallback_reason="llm_disabled=" + STATE["disabled"], source="rule"); return out
    t0 = time.time()
    try:
        prompt = load_prompt().format(info=text)
        rc, raw, err, secs = (call_claude if be == "claude" else call_qwen)(prompt, timeout_s()); STATE["calls"] += 1
    except subprocess.TimeoutExpired: rc, raw, err, secs = None, "", "timeout", time.time() - t0
    except Exception as e:
        rc, raw, err, secs = None, "", repr(e)[:200], time.time() - t0
        if "timed out" in repr(e).lower() or "timeout" in repr(e).lower(): err = "timeout"
    out["latency_s"] = round(secs, 2); out["raw"] = (raw or "")[:600]
    if err == "timeout" or (rc is None and "timed out" in str(err)):
        out.update(fallback=True, fallback_reason="timeout", source="rule"); return out
    if be == "claude" and rc is not None and is_infra(rc, raw, err, secs):
        if QUOTA_RE.search((raw or "")[:500] + (err or "")[:500]): STATE["disabled"] = "quota"; out.update(disabled="quota")
        out.update(fallback=True, fallback_reason="quota" if STATE["disabled"] else "infra", source="rule"); return out
    if rc is None:
        out.update(fallback=True, fallback_reason="error:" + str(err)[:60], source="rule"); return out
    d = parse(raw)
    if d is None: out.update(fallback=True, fallback_reason="parse_failed", source="rule"); return out
    if d["decision"] == "계속" and d.get("method") not in METHODS: out.update(fallback=True, fallback_reason="method_not_allowed", source="rule", decision=d["decision"], method=d.get("method"), reason=str(d.get("reason", ""))[:200]); return out
    act, deg, why = act_for(d, alt_ok)
    out.update(decision=d["decision"], method=d.get("method"), reason=str(d.get("reason", ""))[:200], degraded=deg)
    if act is None: out.update(fallback=True, fallback_reason=why, source="rule"); return out
    out["act"] = act; return out


def jsonl_path(out_arg):
    p = os.environ.get("NERO_LLM_RECOVER_LOG")
    if p: return os.path.expanduser(p)
    d, b = os.path.split(os.path.expanduser(out_arg or "/tmp/discover_object.json")); stem = b[:-5] if b.endswith(".json") else b
    return os.path.join(d, "llm_recover_" + stem + ".jsonl")


def log_record(path, **rec):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception: pass


def log_line(item, r):
    """사양서 §7 의 [noplan-log] 한 줄(문자열). 원문 입력은 jsonl 에 있다."""
    why = str(r.get("reason") or "").replace('"', "'")[:120]
    return (f"llm_recover item={item} backend={r['backend']} latency_s={r['latency_s']} decision={r['decision']} method={r['method']} fallback={'true' if r['fallback'] else 'false'}"
            f" degraded={'true' if r['degraded'] else 'false'} source={r['source']} fallback_reason={r['fallback_reason']} input_hash={r['input_hash']}" + (f" llm_disabled={r['disabled']}" if r.get("disabled") else "") + f' reason="{why}"')
