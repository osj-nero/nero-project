#!/usr/bin/env python3
"""dest_perception_iface.py -- 목적지 인식 모드 인터페이스(뼈대, 2026-10-28). 알고리즘 본체(유형 판정 임계 d_f 등)는 PREREG 확정 후이며 여기에는 없다.
기본값=기존 동작: NERO_DEST_PERCEIVE 가 '1' 이 아니면 이 모듈은 어디서도 호출되지 않는다.
 - dest_mode()                 -> "gt"(기본: 입력 명세 dests.json) | "perceive"
 - validate_dests(D)           -> 스키마 위반 목록(빈 리스트면 통과). 필드별 meas: measured | inferred | assumed | unmeasured
 - registry_to_dests(objs, classify) -> 인식판 dests.json 항목 목록. classify(objs)->[{id_in_registry, kind, ...측정값, meas}] 를 주입(기본은 NotImplementedError)
 - survey_perceived(out, timeout, classify) -> 목적지 제거 survey: --dests 없이 스캔, classify 로 목적지 엔티티를 가려내고 나머지를 물체로 돌려준다(sort_tools.survey 반환 형식과 같음)
 - loader_compatible(D)        -> 기존 로더(discover_object --dests)가 읽는 형식으로 변환(null 값은 로더 기본값으로 채우되 meas 를 assumed 로 표시)
스키마 상세: ~/nero/analysis/code_20261028/DEST_PERCEPTION_DESIGN.md §2.1"""
import os, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
MEAS = ("measured", "inferred", "assumed", "unmeasured")
KINDS = ("container", "flat", "hole")
LOADER_DEFAULTS = dict(wall=0.02, floor_z=0.012)      # discover_object.py 로더(:655-668)의 기본값. 벽 두께는 NERO_PERCEIVE_WALL(기본 0.02=기존)으로 바꿀 수 있다(인식 모드 공통 가정 상수; 프롬프트 동등성 규칙, R0 결정 1)
FORBIDDEN_KEYS = ("prim",)      # 정답 파생 필드(프림 이름)


def dest_mode():
    return "perceive" if os.environ.get("NERO_DEST_PERCEIVE", "0") == "1" else "gt"


def validate_dests(D):
    errs = []; ids = set()
    for i, d in enumerate(D):
        p = f"[{i}]"
        for k in ("id", "kind", "xy", "size"):
            if k not in d: errs.append(f"{p} 필수 필드 없음: {k}")
        if d.get("kind") not in KINDS: errs.append(f"{p} kind 가 {KINDS} 아님: {d.get('kind')!r}")
        if d.get("id") in ids: errs.append(f"{p} id 중복: {d.get('id')}")
        ids.add(d.get("id"))
        for k in FORBIDDEN_KEYS:
            if k in d: errs.append(f"{p} 정답 파생 필드 금지: {k}")
        if d.get("kind") == "hole":
            for k in ("E", "open"):
                if d.get(k) is None: errs.append(f"{p} hole 필수 값 없음: {k}")
        m = d.get("meas")
        if m is None: errs.append(f"{p} meas 없음(인식판은 필드별 측정 여부 필수)")
        else:
            for k, v in m.items():
                if v not in MEAS: errs.append(f"{p} meas.{k} 가 {MEAS} 아님: {v!r}")
            for k, v in d.items():      # 값이 null 이면 meas 는 unmeasured 여야 한다(상수로 채운 값은 assumed)
                if v is None and m.get(k) not in (None, "unmeasured"): errs.append(f"{p} {k}=null 인데 meas.{k}={m.get(k)!r}")
    return errs


def loader_compatible(D):
    """null 인 필드를 로더 기본값으로 채워 반환(원본 불변). 채운 필드는 meas 를 assumed 로 바꾼다."""
    out = []
    for d in D:
        e = json.loads(json.dumps(d)); m = e.setdefault("meas", {})
        for k, v in LOADER_DEFAULTS.items():
            if k == "wall": v = float(os.environ.get("NERO_PERCEIVE_WALL", v))
            if e.get("kind") == "container" and e.get(k) is None: e[k] = v; m[k] = "assumed"
        out.append(e)
    return out


def registry_to_dests(objs, classify=None):
    """등록부 엔티티 dict(id -> 엔티티) -> 인식판 dests.json 목록. 유형 판정·측정은 classify 가 한다(본체 미구현)."""
    if classify is None: raise NotImplementedError("목적지 인식 본체(유형 판정)는 PREREG 확정 후 구현")
    D = []
    for n, c in enumerate(classify(objs), 1):
        e = dict(c); e["id"] = f"dest_{n}"; e["label"] = ""; e.pop("prim", None); D.append(e)
    return D


def survey_perceived(out="/tmp/sort_survey.json", timeout=240, classify=None):
    """sort_tools.survey 의 인식판: --dests 를 주지 않는다(L6). 스캔은 같은 방식(rt_run --scan-only), 목적지 인식은 NERO_DEST_PERCEIVE=1 로 discover_object 가 dest_perceive.perceive 로 수행해
    NERO_PERCEIVED_DESTS(기본 out+'.dests.json')에 저장한 파일을 읽는다(classify 를 주입하면 그것을 씀: 시험용). 파일이 없거나 비면 오류(정답으로 대체하지 않음).
    반환: 물체 dict(sort_tools.survey 와 같은 형식)."""
    import sort_tools as ST
    pp = os.environ.get("NERO_PERCEIVED_DESTS") or (out + ".dests.json"); (os.path.exists(pp) and os.remove(pp))
    env = dict(os.environ, **ST.base_env(), NERO_FOCUS_SCAN="0", NERO_SCAN_FULL="1", NERO_DEST_PERCEIVE="1", NERO_PERCEIVED_DESTS=pp)
    sw = os.environ.get("NERO_PERCEIVE_SWEEP", "")      # 2026-10-28(기본 빈 값 = 기존 7스톱 스윕): 더 촘촘한 스윕 각도 목록(예: 13스톱)
    subprocess.run([os.path.join(HERE, "rt_run.sh"), "--scan-only"] + ([f"--sweep={sw}"] if sw else []) + ["--vocab", ",".join(ST.VOCAB), "--out", out], cwd=HERE, env=env, capture_output=True, timeout=timeout)
    sc = json.load(open(out))["scene"]; ent = {k: v for k, v in sc.items() if v.get("xy") and v.get("size_cm")}
    if classify is not None: D = registry_to_dests(ent, classify); json.dump(D, open(pp, "w"), ensure_ascii=False, indent=1)
    if not os.path.exists(pp): raise RuntimeError("인식판 dests 파일 없음(정답으로 대체하지 않음): " + pp)
    D = json.load(open(pp)); errs = validate_dests(D)
    if errs: raise ValueError("인식판 dests 스키마 위반: " + "; ".join(errs[:5]))
    return ST.objects_only(ent, pp)


def resolve_dests_path(gt_path, layout_asset_dir=None):
    """실행기에 넘길 목적지 파일 경로(L5·L6 배선). 기본(gt)은 gt_path 그대로. 인식 모드는 NERO_PERCEIVED_DESTS 또는 <asset>/dests_perceived.json 이며,
    없으면 None(호출자가 오류 처리) -- 정답 dests.json 으로 조용히 대체하지 않는다."""
    if dest_mode() != "perceive": return gt_path
    p = os.environ.get("NERO_PERCEIVED_DESTS") or (os.path.join(layout_asset_dir, "dests_perceived.json") if layout_asset_dir else None)
    return p if p and os.path.exists(p) and os.path.abspath(p) != os.path.abspath(gt_path) else None


def maybe_unmap_plan(plan, idmap_path):
    """NERO_NEUTRAL_PROMPT=1 이고 정답 명세 입력(gt 모드)일 때, 중립 id(dest_N)로 나온 계획을 원래 id 로 되돌린다. 기본(끔)이면 plan 그대로."""
    if os.environ.get("NERO_NEUTRAL_PROMPT", "0") != "1" or dest_mode() == "perceive" or not os.path.exists(idmap_path): return plan
    import neutral_prompt as NP; return NP.unmap_plan(plan, json.load(open(idmap_path)))
