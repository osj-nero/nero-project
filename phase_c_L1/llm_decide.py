#!/usr/bin/env python3
"""llm_decide.py -- 실행 루프가 판단 지점에서 멈추고 Claude(MCP 클라이언트)에게 선택을 묻는 채널 (2026-10-14, 기본 꺼짐: NERO_LLM_DECIDE=1).
루프(discover_object.py)는 요청 파일을 쓰고 응답 파일을 기다린다. MCP 서버(rt_mcp_server.py)의 rt_decision 이 요청을 텍스트로 보여주고 rt_decide 가 응답을 쓴다.
기다리다 시간이 지나면 규칙이 고른 기본값으로 진행한다(source='rule(timeout)'). Claude 는 이미지를 못 보므로 상황은 전부 텍스트(엔티티 표, 측정 사실, 선택지별 결과 예상)로 준다."""
import os, json, time, uuid
REQ = os.path.expanduser(os.environ.get("NERO_LLM_REQ", "~/grasp/rt_decision_req.json"))
RESP = os.path.expanduser(os.environ.get("NERO_LLM_RESP", "~/grasp/rt_decision_resp.json"))
LOG = os.path.expanduser(os.environ.get("NERO_LLM_LOG", "~/grasp/rt_decision_log.jsonl"))


def enabled():
    return os.environ.get("NERO_LLM_DECIDE", "0") == "1"


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, path)


def ask(kind, question, situation, options, default, timeout=None, log=print):
    """options: [{id, text}] 선택지, default: 규칙이 고른 선택지 id. 반환 (선택 id, 출처 'llm'|'rule(timeout)'|'rule(off)', 이유)."""
    ids = [o["id"] for o in options]
    if not enabled() or len(ids) <= 1:
        return default, "rule(off)" if not enabled() else "rule(single)", ""
    timeout = float(os.environ.get("NERO_LLM_TIMEOUT", "20")) if timeout is None else timeout
    rid = uuid.uuid4().hex[:8]
    for p in (RESP,):
        try: os.remove(p)
        except OSError: pass
    _write(REQ, dict(id=rid, kind=kind, question=question, situation=situation, options=options, default=default, t=time.time(), timeout=timeout))
    log(f"[llm] 판단 요청 {rid} ({kind}): {question} -- 선택지 {ids}, 기본 {default}, 대기 {timeout:.0f}s")
    t_end = time.time() + timeout; choice, src, why = default, "rule(timeout)", ""
    while time.time() < t_end:
        try:
            r = json.load(open(RESP))
            if r.get("id") == rid:
                if r.get("choice") in ids: choice, src, why = r["choice"], "llm", str(r.get("reason", ""))
                else: log(f"[llm] 응답 {r.get('choice')!r} 은 선택지에 없음 -- 기본값 사용"); src = "rule(invalid)"
                break
        except Exception:
            pass
        time.sleep(0.3)
    try: os.remove(REQ)
    except OSError: pass
    try: os.remove(RESP)
    except OSError: pass
    rec = dict(id=rid, kind=kind, question=question, options=ids, default=default, choice=choice, source=src, reason=why, waited=round(time.time() - (t_end - timeout), 1))
    try: open(LOG, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception: pass
    log(f"[llm] 판단 {rid}: {choice} ({src}{', ' + why if why else ''})")
    return choice, src, why


def pending():
    try: return json.load(open(REQ))
    except Exception: return None


def respond(rid, choice, reason=""):
    p = pending()
    if p is None: return False, "대기 중인 판단 요청이 없음"
    if p["id"] != rid: return False, f"요청 id 불일치(현재 {p['id']})"
    if choice not in [o["id"] for o in p["options"]]: return False, f"선택지에 없음: {[o['id'] for o in p['options']]}"
    _write(RESP, dict(id=rid, choice=choice, reason=reason)); return True, "전달함"


def render(p):
    """대기 중인 요청을 Claude 가 읽을 텍스트로."""
    s = p["situation"]
    lines = [f"[판단 요청 {p['id']}] 종류={p['kind']} (남은 시간 {max(0, p['timeout'] - (time.time() - p['t'])):.0f}s, 안 고르면 규칙 기본값 '{p['default']}' 으로 진행)", p["question"], ""]
    if s.get("table"): lines += ["엔티티 표:", s["table"], ""]
    for k, v in s.items():
        if k != "table": lines.append(f"{k}: {v}")
    lines += ["", "선택지:"] + [f"- {o['id']}: {o['text']}" for o in p["options"]]
    lines.append("\nrt_decide(decision_id, choice, reason) 로 답한다. 이미지는 없고 위 측정값만 근거다. 불확실하면 되돌릴 수 있는 쪽을 고른다.")
    return "\n".join(lines)
