"""RQ1 v3: 로컬 LLM 이 물체 설명 + 명령만 보고 '놓을 때 지켜야 할 제약 명세'(면 조건 + 방식 + 선호 + 근거)를 직접 쓴다.
사용: python3 rq1_decide3.py TAG [--split dev|test] [--style direct|decomp] [--model qwen3-vl|claude] [--temp 0 --n 1] [--mt 1500]
결과: out3/rq1v3_{TAG}.jsonl. 정답은 채점(contplace/constraints.compare)에만 쓰고 프롬프트에 넣지 않는다."""
import json, subprocess, time, sys, argparse, os, hashlib
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "contplace"))
from constraints import compare, GRID_DIMS
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--n", type=int, default=1)
ap.add_argument("--mt", type=int, default=1500); ap.add_argument("--model", default="qwen3-vl"); ap.add_argument("--items", default="")
ap.add_argument("--prereg", default="RQ1_PREREG_v3_draft.json"); ap.add_argument("--split", default="dev"); ap.add_argument("--style", default="direct", choices=["direct", "decomp"])
a = ap.parse_args(); PRE = os.path.join(H, "prereg", a.prereg)
P = json.load(open(PRE)); PH = hashlib.sha256(open(PRE, "rb").read()).hexdigest()[:12]

SCHEMA = '''출력은 JSON 하나만:
{"hard": [조건, ...], "manner": ["place" 그리고/또는 "topple"], "prefer": [선호, ...], "why": "<근거 한두 문장>"}
조건 형식(꼭 필요한 것만, 없으면 빈 목록 = 어떤 자세·위치든 허용). tol 은 "엄격"(10도) / "보통"(20도) / "느슨"(30도) 중 하나:
 {"type":"up", "faces":["면 이름", ...], "tol":"..."}        : 나열한 면 중 하나가 위를 향해야 함
 {"type":"vertical", "face":"면 이름", "tol":"..."}          : 그 면이 옆을 향해야 함(세워진 상태)
 {"type":"not_down", "face":"면 이름"}                        : 그 면이 바닥 쪽을 향하면 안 됨
 {"type":"yaw", "dir":"앞뒤" 또는 "좌우", "tol":"..."}       : 물체의 긴 축이 로봇 기준 그 방향으로 놓여야 함
 {"type":"region", "where":"가까운 쪽"|"먼 쪽"|"왼쪽"|"오른쪽"|"가운데"} : 목표면에서 놓을 구역
방향 기준: 로봇이 목표면을 바라보는 방향. 앞뒤 = 로봇에서 멀어지는 방향, 좌우 = 그에 수직, 왼쪽/오른쪽 = 로봇이 볼 때.
방식: "place" = 목표 자세로 천천히 내려놓기, "topple" = 작업대 위에 기울여 세운 뒤 손을 놓아 넘어지게 해서 눕히기(충격이 있음)
선호 형식: {"type":"keep_current"} 또는 {"type":"prefer_up", "face":"면 이름"}'''


def prompt(it):
    s = f"""너는 로봇이 물체를 놓을 때 지켜야 할 자세 조건을 정하는 판단기다.
물체: {it['obj']}
사람의 명령: "{it['cmd']}"
{P['face_vocab']} 명령이 따로 요구하지 않으면 수직축 둘레 회전과 놓는 위치는 자유다.
명령의 뜻과 물체(내용물, 재질, 표시된 면)의 안전을 지키는 데 꼭 필요한 조건만 하드 조건으로 쓴다. 필요 없는 조건을 넣으면 로봇이 놓을 수 있는 방법이 줄어들고, 필요한 조건을 빼면 물체나 명령을 어기게 된다.
실행하기 쉬운지는 고려하지 말고 의미만 판단한다.
{SCHEMA}"""
    if a.style == "decomp":
        s += '\n먼저 "analysis" 필드에 (1) 명령이 자세에 대해 요구하는 것, (2) 물체에서 방향에 민감한 면이나 내용물, (3) 넘어뜨려도 되는지를 각각 한 문장으로 쓰고, 그다음 조건을 정한다. 출력 JSON 에 "analysis" 필드를 추가한다.'
    return s


def call(p, seed):
    if a.model == "claude":
        t0 = time.time(); r = subprocess.run(["claude", "-p", "--tools", "", "--output-format", "text"], input=p, capture_output=True, text=True, timeout=300, cwd="/tmp")
        return r.stdout, "", len(r.stdout) // 2, round(time.time() - t0, 2), "rc%d" % r.returncode
    req = dict(model=a.model, messages=[{"role": "user", "content": p}], temperature=a.temp, seed=seed, max_tokens=a.mt); t0 = time.time()
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "thor", "curl -s -m 900 http://127.0.0.1:8005/v1/chat/completions -H 'Content-Type: application/json' -d @-"], input=json.dumps(req).encode(), capture_output=True, timeout=940)
    jj = json.loads(r.stdout.decode()); m = jj["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or m.get("reasoning") or ""), jj["usage"]["completion_tokens"], round(time.time() - t0, 2), jj["choices"][0]["finish_reason"]


os.makedirs(os.path.join(H, "out3"), exist_ok=True); fo = open(os.path.join(H, "out3", f"rq1v3_{a.tag}.jsonl"), "a")
items = [it for it in P["items"] if (not a.items or it["id"] in a.items.split(",")) and it["split"] == a.split]
for it in items:
    for k in range(a.n):
        txt, rs, ct, sec, fin = call(prompt(it), k)
        try:
            j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1]); spec = dict(hard=j.get("hard") or [], manner=j.get("manner") or ["place"]); cmp = compare(spec, it["gt"], GRID_DIMS.get(it.get("asset") or "plate")); err = None
        except Exception as e:
            j, spec, cmp, err = {}, None, None, repr(e)[:120]
        rec = dict(prereg=PH, style=a.style, tag=a.tag, model=a.model, temp=a.temp, sample=k, id=it["id"], spec=spec, prefer=j.get("prefer"), why=j.get("why", ""), analysis=j.get("analysis"),
                   cmp=cmp, err=err, ctok=ct, sec=sec, finish=fin, raw=txt, reasoning=rs)
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
        print(f"{it['id']}#{k}: {json.dumps(spec, ensure_ascii=False) if spec else 'PARSE_FAIL ' + str(err)} | 위반 {cmp and cmp['viol']} 놓침 {cmp and cmp['miss']} 일치 {cmp and cmp['exact']} | {ct}tok {sec}s", flush=True)
