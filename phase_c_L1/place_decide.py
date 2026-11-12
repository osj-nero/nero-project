#!/usr/bin/env python3
"""place_decide.py -- /place 가 물체와 목적지의 크기 관계만 보고 놓는 방식을 스스로 정한다(과제 플래그 없음).
  자연어 "~에 ~를 넣어줘" 의 관계(on / in)는 상위(Claude)가 인자로 넘기고, 어떻게 놓을지(그냥 놓기 / 눕혀 넣기 / 세워 꽂기)는 여기서 기하로 판단한다.
물체 기술자 obj:  PCA 가 길쭉하면 kind="elongated"(L=길이, c1<=c2=단면), 아니면 kind="compact"(L=최대변, c1<=c2=나머지 두 변). 모두 m.
목적지 기술자 dest:  kind="surface"(접시/책상: 위에 올림) | "receptacle"(상자/통: 개구부 open=(w,d), 깊이 depth, 축 기울기 tilt_deg).
결정 mode:  "place"(그냥 놓기: 안정 자세 유지) | "insert"(축을 홀 축에 맞춰 꽂기) | None(안 들어감, 사유 반환).
여유: 한 변당 clearance = (개구부 변 - 물체 변) / 2 가 MIN_CLEAR 이상이어야 한다."""
MIN_CLEAR = 0.006       # 한 변당 최소 여유(m). 센서 오차와 파지 오차를 고려한 값(오차 수십 mm 허용 방침 중 보수적인 하한)
ELONG_MIN = 6.0         # axis_pca.ELONG_MIN 과 같은 기준(길쭉함)


def describe_obj(pca=None, dims=None):
    """pca: axis_pca 결과(dict: ok, length, cross, elong) 또는 None. dims: (w,d,h) OBB(m)."""
    if pca and pca.get("ok"):
        c = sorted(float(x) for x in pca["cross"]); return dict(kind="elongated", L=float(pca["length"]), c1=c[0], c2=c[1], elong=pca.get("elong"))
    s = sorted((float(x) for x in dims), reverse=True)
    return dict(kind="compact", L=s[0], c1=s[2], c2=s[1], elong=None)


def _fits(ow, od, a, b, clear):
    """직사각 개구부(ow x od)에 a x b 발자국이 한 변 여유 clear 로 들어가는가(90도 회전 허용). 반환 (들어감, 최소 여유)."""
    best = None
    for (p, q) in ((a, b), (b, a)):
        cl = min((ow - p) / 2.0, (od - q) / 2.0)
        best = cl if best is None else max(best, cl)
    return best >= clear, best


def decide(obj, dest, relation="in"):
    """반환 dict(mode, how, reason, clearance, detail). how: lie(눕혀 넣기) | upright(세워 넣기) | insert(축 정렬 꽂기) | onto(위에 놓기)."""
    if dest.get("kind", "surface") == "surface" or relation == "on":
        return dict(mode="place", how="onto", reason="평면 위에 올림(집은 안정 자세 유지)", clearance=None)
    ow, od = (float(x) for x in dest["open"])
    # 1) 그냥 놓기(눕혀서): 가장 긴 변을 개구부 긴 쪽에 두는 발자국 L x c2
    ok_lie, cl_lie = _fits(ow, od, obj["L"], obj["c2"], MIN_CLEAR)
    if ok_lie:
        return dict(mode="place", how="lie", reason="눕혀서 개구부 안에 들어감(여유 %.0fmm/변)" % (1000 * cl_lie), clearance=cl_lie)
    # 2) 세워서 넣기/꽂기: 발자국이 단면 c1 x c2 인 자세. 높이가 개구부 밖으로 나와도 되는 길쭉한 물체만 꽂는다(공 같은 compact 는 높이=발자국이라 위 1) 로 충분)
    ok_up, cl_up = _fits(ow, od, obj["c1"], obj["c2"], MIN_CLEAR)
    if ok_up and obj["kind"] == "elongated":
        how = "insert"
        return dict(mode="insert", how=how, reason="눕히면 안 들어가고(여유 %.0fmm) 단면은 들어감(여유 %.0fmm/변) -> 세워 꽂기" % (1000 * cl_lie, 1000 * cl_up), clearance=cl_up)
    if ok_up:
        return dict(mode="place", how="upright", reason="세워서 개구부 안에 들어감(여유 %.0fmm/변)" % (1000 * cl_up), clearance=cl_up)
    return dict(mode=None, how=None, reason="안 들어감: 눕힘 여유 %.0fmm, 단면 여유 %.0fmm (< %.0fmm)" % (1000 * cl_lie, 1000 * cl_up, 1000 * MIN_CLEAR), clearance=max(cl_lie, cl_up))


if __name__ == "__main__":
    cases = [
        ("펜 14cm + 연필꽂이(개구부 8x8cm)", dict(kind="elongated", L=.14, c1=.012, c2=.012), dict(kind="receptacle", open=(.08, .08), depth=.10), "insert"),
        ("막대 20x3x3cm + 좁고 긴 박스(5x30cm)", dict(kind="elongated", L=.20, c1=.03, c2=.03), dict(kind="receptacle", open=(.05, .30), depth=.08), "place"),
        ("막대 30x3x3cm + 좁고 긴 박스(5x25cm)", dict(kind="elongated", L=.30, c1=.03, c2=.03), dict(kind="receptacle", open=(.05, .25), depth=.08), "insert"),
        ("공 지름 7cm + 상자(20x20cm)", dict(kind="compact", L=.07, c1=.07, c2=.07), dict(kind="receptacle", open=(.20, .20), depth=.10), "place"),
        ("인형 12x8x6cm + 접시(평면)", dict(kind="compact", L=.12, c1=.06, c2=.08), dict(kind="surface"), "place"),
        ("공 지름 7cm + 연필꽂이(8x8cm, 여유 5mm)", dict(kind="compact", L=.07, c1=.07, c2=.07), dict(kind="receptacle", open=(.08, .08), depth=.10), None),
        ("막대 20x3x3cm + 개구부 3.5x3.5cm", dict(kind="elongated", L=.20, c1=.03, c2=.03), dict(kind="receptacle", open=(.035, .035), depth=.10), None),
    ]
    bad = 0
    for nm, o, d, want in cases:
        r = decide(o, d); ok = r["mode"] == want; bad += not ok
        print(("OK  " if ok else "FAIL"), nm, "->", r["mode"], "|", r["reason"])
    print("모든 케이스 통과" if not bad else f"{bad}건 불일치")
