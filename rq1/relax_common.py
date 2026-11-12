"""막혔을 때 Gemma 에게 실패를 알리고 풀기/포기를 묻는 부분 -- 판단 시험(blocked_plan_only.py)과 로봇 실행 경로(stage2_fast.py)가 같은 코드를 쓴다(2026-11-09 통일).
- 자세별 실제 실패 이유(서버 why 의 자세 블록별), 시간 상한으로 다 보지 못한 것은 사실대로("불가능이 확인된 것은 아님").
- 덤프 하나(판단 시험) = C 요청 하나(로봇 실행 경로의 시도 하나). 둘 다 최대 3개."""
import re, ast, json
from collections import Counter

# 구체적인 사유를 먼저 맞춘다(전에는 'place 불가'가 먼저 맞아 하강 충돌·이송 실패도 모두 '팔이 닿지 않음'으로 뭉뚱그려짐)
REASON_KO = [("도착지 포함", "내려놓는 경로에서 물체나 손이 받침대·장애물에 부딪힘"), ("transit", "집은 뒤 놓을 곳으로 옮기는 경로가 막힘"), ("손목", "집은 뒤 들어 올리는 동작에서 손목 관절 한계"),
             ("시간 예산", "제한 시간 안에 다 확인하지 못함"), ("접근 자세", "집으러 다가가는 경로가 막힘"), ("pick 불가", "집기 자세에 팔이 닿지 않음"),
             ("place 불가", "놓을 자세에 팔이 닿지 않거나 놓을 자리가 막힘"), ("충돌", "장애물과 부딪힘")]
OCC_MSG = "놓을 자리에 다른 물체가 있어 놓을 수 없음"
NOT_TRIED_MSG = "시간 안에 계획해 보지 못함(불가능이 확인된 것은 아님)"
CUT_SUFFIX = " -- 시간 상한으로 일부만 확인함(불가능이 확인된 것은 아님)"


def reason_summary(whys):
    """서버 후보별 실패 사유 -> 짧은 한국어 빈도 요약(Gemma 에게 줌)."""
    c = Counter()
    for w in whys:
        ko = next((k for key, k in REASON_KO if key in w), w[:40]); c[ko] += 1
    return ", ".join(f"{k} {v}회" for k, v in c.most_common(3)) or "이유 미상"


def per_pose(why):
    """서버 why([[자세(순위 k), 'candN: 사유', ...], ...]) -> {자세: (사유 목록, 상태)}. 상태: 'tried' / 'cut'(일부 파지만 보고 시간 상한) / 'not_tried'(시간 상한으로 계획 못 함)."""
    out = {}
    if not isinstance(why, list): return out
    if why and not isinstance(why[0], list) and any(re.search(r"\(순위 \d+\)$", str(x)) for x in why):      # 2026-11-10: 서버가 자세별 묶음을 한 줄로 펼쳐 보냄 -> '…(순위 k)' 머리로 다시 묶는다(전에는 전부 not_tried 로 읽힘)
        grp = []
        for x in why:
            if re.search(r"\(순위 \d+\)$", str(x)): grp.append([x])
            elif grp: grp[-1].append(x)
        why = grp
    for blk in why:
        if not isinstance(blk, list) or not blk: continue
        name = re.sub(r"\(순위 \d+\)$", "", str(blk[0])); rs = [re.sub(r"^cand\d+: ", "", str(x)) for x in blk[1:]]
        if any("이 자세는 계획하지 못함" in x for x in rs): st = "not_tried"; rs = []
        elif any("시간 예산" in x for x in rs): st = "cut"
        else: st = "tried"
        out[name] = (rs, st)
    return out


def per_pose_from_log(log):
    """로봇 실행 경로: run.log 의 'C 실패(..s) [...] 막힘 ..' 줄(시도마다 하나) -> 시도별 per_pose 결과 [{자세: dict(reasons, state)}].
    자세 목록 하나만 보낸 옛 형식(평평한 목록)은 자세 구분이 없어 '?' 하나로 둔다."""
    res = []
    for m in re.finditer(r"C 실패\([^)]*\) (\[.*\]) 막힘", log):
        try: why = ast.literal_eval(m.group(1))
        except Exception: continue
        if why and (isinstance(why[0], list) or any(re.search(r"\(순위 \d+\)$", str(x)) for x in why)): pp = per_pose(why)      # 2026-11-10: 펼친 형식도 자세별로
        else: pp = {"?": ([re.sub(r"^cand\d+: ", "", str(x)) for x in why], "cut" if any("시간 예산" in str(x) for x in why) else "tried")}
        res.append({k: dict(reasons=v[0], state=v[1]) for k, v in pp.items()})
    return res


def pose_msg(name, per_pose_list, occupied=False):
    """자세 하나에 대한 Gemma 용 한 줄: 여러 시도(덤프/C 요청)의 그 자세 결과를 모아서."""
    if occupied: return OCC_MSG
    ent = [pp.get(name) or pp.get("?") for pp in per_pose_list]
    rs = [w for e in ent if e for w in (e.get("reasons") or [])]
    sts = [(e or {}).get("state", "not_tried") for e in ent]
    if not sts or all(x == "not_tried" for x in sts): return NOT_TRIED_MSG
    msg = reason_summary(rs)
    if any(x != "tried" for x in sts): msg += CUT_SUFFIX
    return msg


def relax_prompt(it, spec, why, tried):
    """tried = [dict(pose, where, why)] -> 풀기/포기 질문."""
    import rq1_decide3_prompt as P3
    t = "\n".join(f"- {c['pose']}" + (f" 위치 {c['where']}" if c.get("where") else "") + f": 계획 실패 ({c['why']})" for c in tried)
    return P3.prompt(it) + f"""

[이전 판단] 너는 다음 조건을 썼다: {json.dumps(spec, ensure_ascii=False)} (근거: {why})
[로봇 시도 결과] 이 조건을 만족하는 놓기 방법을 계획해 보았지만 하나도 서지 않았다(자세별 결과):
{t if t else '- 조건을 만족하는 자세가 하나도 없음'}
명령의 뜻과 물체 안전을 어기지 않는 범위에서만 조건을 풀어 다시 써라. 처음 조건 중 명령이나 물체 안전에 꼭 필요하지 않았던 것이 있으면 뺄 수 있다.
풀 수 있는 조건이 없으면 놓지 않는 것이 맞다: 그때는 {{"abstain": true, "why": "<근거>"}} 만 출력한다."""
