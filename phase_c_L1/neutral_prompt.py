#!/usr/bin/env python3
"""neutral_prompt.py -- 중립 프롬프트 조건(2026-10-28, NERO_NEUTRAL_PROMPT=1 일 때만 planner_prompt.py/sort_tools.py 가 사용; 기본 꺼짐=기존 동작).
정답 유출 차단(PREREG v1.3 §3·§7): (L1) 프림/에셋 이름·색·재질·종류 파생 라벨 미사용 (L4) planner_prompt.py:10,17,18 의 범주->목적지 고정문·정답 id 미사용.
목적지는 중립 id(dest_1..)로 바꾸고, 종류 서술은 기하 판정(kind 필드: container/flat/hole)에서만 만든다. 안쪽 개구부는 측정(meas.wall=='measured')이면 측정값, 아니면 '가정값'으로 표기한다."""
import os

RULES = {      # planner_prompt.py 의 원문 줄 앞머리 -> 중립 대체문(정답 id·범주 매핑·용기 높이 수치 제거)
    "1. 물체마다 놓을 곳": "1. 물체마다 놓을 곳(dest)과 대안(dest2)을 정한다: 물체의 종류·크기와 아래 목적지 표(종류 서술·크기·높이·안쪽 개구부)를 보고 상식적으로 가장 어울리는 곳을 정한다. 단 성공률이 낮으면 대안.",
    "- 가늘고 긴 물체(폭 5cm 미만": "- 가늘고 긴 물체(폭 5cm 미만, 길이 14cm 이상)와 곡면 소형 피규어는 이송 중 빠지거나 튕김. 눕혀 놓인 긴 물체를 좁은 입구에 세워 꽂기는 어렵다: 실패하면 dest2 로.",
    "- 바구니(bin2, 높이 10cm)": "- 높이가 낮은 용기(목적지 표의 높이 참고)에 큰/긴 물체를 던져 넣으면 밖으로 쓰러져 빠지는 일이 있음 -> 큰 물체는 높은 용기, 작은 것은 낮은 용기.",
}


def neutralize_line(s):
    for head, new in RULES.items():
        if s.lstrip('"').startswith(head) or s.startswith(head): return new
    return s


def neutralize_dests(D, key=""):
    """-> (D2, idmap{원래 id: 중립 id}). D2 에서 label/prim 을 비우고 id 를 dest_N 으로 바꾼다.
    순서: NERO_NEUTRAL_ORDER=orig(기본: 원래 목록 순서) | shuffle(시드 고정 셔플: random.Random(f"{NERO_NEUTRAL_SEED}:{key}"), id 는 셔플 후 번호). 원래 목록 순서가 정답 id 순서(bin, bin2, tray, cup)라 순서 자체가 종류를 암시할 수 있음(추정)."""
    if os.environ.get("NERO_NEUTRAL_ORDER", "orig") == "shuffle":
        import random; D = list(D); random.Random(f"{os.environ.get('NERO_NEUTRAL_SEED', '0')}:{key}").shuffle(D)
    D2 = []; idmap = {}
    for i, d in enumerate(D, 1):
        e = {k: v for k, v in d.items() if k not in ("prim",)}; e["id"] = f"dest_{i}"; e["label"] = ""; idmap[d["id"]] = e["id"]; D2.append(e)
    return D2, idmap


TAG = dict(measured="측정", inferred="추정", assumed="가정", unmeasured="미측정")
GT_DEFAULT_MEAS = dict(xy="measured", size="measured", rim_z="measured", wall="assumed", open="assumed", depth="assumed")      # 정답 명세(G') 입력의 필드별 출처: 위치·크기는 메시/정답(측정), 벽·개구부·깊이는 명세 상수(가정)


def _wall_default():
    return float(os.environ.get("NERO_PERCEIVE_WALL", "0.02"))      # 인식 모드에서 벽 두께 미측정일 때 쓰는 공통 가정 상수(기본 0.02 = 로더 기존 기본값; R0 는 G' 상수 0.01 로 맞춤)


def _t(d, f):
    """필드 f 의 태그: d['meas'][f] 가 있으면 그것, 없으면 정답 명세 기본(GT_DEFAULT_MEAS), 값이 None 이면 미측정."""
    if f == "rim_z" and d.get("rim_z") is None and (d.get("size") or [None] * 3)[2] is not None: f = "size"      # 높이는 size[2] 가 있으면 그것의 출처 태그를 쓴다(정답 명세 flat 은 rim_z 키가 없음)
    if d.get(f) is None and f not in ("xy", "size"): return TAG["unmeasured"]
    return TAG.get((d.get("meas") or {}).get(f) or GT_DEFAULT_MEAS.get(f, "assumed"), "가정")


def _inner(d):
    w = d.get("wall"); wv = w if w is not None else _wall_default(); a = 100 * (d["size"][0] - 2 * wv); b = 100 * (d["size"][1] - 2 * wv)
    tag = _t(d, "wall") if w is not None else TAG["assumed"]
    return f"안쪽 약 {a:.0f}x{b:.0f}cm[{tag}: 벽 두께 {1000 * wv:.0f}mm]"


def dest_line(d):
    """목적지 한 줄(라벨·이름 없음, 두 모드 같은 필드 목록·서식·태그[측정/추정/가정/미측정]). 종류 서술은 kind(기하 판정)에서만. 값의 출처만 모드별로 다르다(G'=정답 명세, P=인식)."""
    k = d["kind"]; pos = f"위치 ({d['xy'][0]:+.2f},{d['xy'][1]:+.2f})[{_t(d, 'xy')}]"; sz = f"바깥 {100 * d['size'][0]:.0f}x{100 * d['size'][1]:.0f}cm[{_t(d, 'size')}]"
    if k == "container": extra = f"위가 열린 용기 | {sz} | 높이 {100 * d['size'][2]:.0f}cm[{_t(d, 'rim_z')}] | {_inner(d)}"
    elif k == "flat": extra = f"낮은 평면(공동 얕음) | {sz} | 높이 {100 * d['size'][2]:.0f}cm[{_t(d, 'rim_z')}] | 가장자리에서 쓰러지기 쉬움"
    else:
        op = d.get("open"); dp = d.get("depth")
        extra = (f"좁은 원형 입구(세워 꽂기) | 입구 지름 " + (f"약 {100 * op:.0f}cm[{_t(d, 'open')}]" if op is not None else f"[{TAG['unmeasured']}]") + " | 깊이 " + (f"{100 * dp:.0f}cm[{_t(d, 'depth')}]" if dp is not None else f"[{TAG['unmeasured']}]") + " | 긴 물체만, 옆면 파지 필요")
    return f"- {d['id']}: {extra} | {pos}"


def unmap_plan(plan, idmap):
    """플래너가 중립 id 로 낸 계획(order[].dest/dest2)을 원래 id 로 되돌린다(실행기는 원래 id 를 씀). 알 수 없는 id 는 그대로 둔다('leave' 포함)."""
    inv = {v: k for k, v in idmap.items()}
    for it in plan.get("order", []):
        for f in ("dest", "dest2"):
            if it.get(f) in inv: it[f] = inv[it[f]]
    return plan
