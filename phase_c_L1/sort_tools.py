#!/usr/bin/env python3
"""sort_tools.py -- 다물체 정리용 공통 도구(2026-10-14): 장면 조사(survey) -> LLM 에 줄 표(table_text), 계획 검증.
 survey(): rt_run.sh --scan-only 로 전체 방위를 훑어 물체 엔티티(id, 위치, 크기, 색, 일반 어휘 라벨 top-3)를 얻는다. 목적지는 인식하지 않고 입력 명세(dests.json)를 쓴다."""
import os, sys, json, subprocess, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = ["a toy", "an animal figurine", "a cartoon character figure", "an action figure", "a bottle", "a pill bottle", "a jar", "a canister", "a box", "a candy box", "a food package", "a flashlight", "a tape measure",
         "a computer mouse", "a clock", "an alarm clock", "a cup", "a cosmetic box", "a book", "a phone", "a remote control", "a can", "a bag", "a tool", "a pair of scissors", "a pen", "a ball", "a kitchen utensil",
         "an electronic device", "a stuffed toy", "a bird figurine", "a dinosaur toy", "a spice container", "a medicine bottle", "a skin care product", "a battery", "a lamp", "a camera", "a toy car", "a toy boat",
         "a toy airplane", "a rattle", "a hard drive", "a camera lens", "a game cartridge", "a film pack", "a coffee jar", "a coffee box", "a box of light bulbs", "a dog chew toy", "a supplement bottle", "a maraca"]
BASE_ENV = dict(ROS_LOCALHOST_ONLY="1", NERO_AUTO_MODE="1", NERO_PLACE_SNAPS="1", NERO_PLACE_AWARE="0", NERO_COND="E", NERO_LEAN="1", NERO_ENT_MERGE_FOOT="1", NERO_RESTILL_FOCUS="1", NERO_CONTAINER_NOSOLID="1",
                NERO_CLEAR_BLOCKER="1", NERO_CONTAINER_INNER="0", NERO_BLOCKED_FREEZE="1", NERO_EMBED="1", NERO_SCAN_FEAT="1", NERO_CLEAR_DEST_PAD="0.05", NERO_DROP_IN="1", NERO_VLM_LABEL="1", NERO_SLIP_STOP="1", NERO_PLACED_SOFT="1")



def base_env():
    """2026-10-14: /tmp/sort_fast 파일이 있으면(기본 꺼짐) 처음 조사를 등록부에 저장하고 이후 물체는 전체 스윕 없이 저장 위치 방위각만 확인한다. 서버를 재시작하지 않고 켜고 끌 수 있게 호출 때마다 본다."""
    e = dict(BASE_ENV); e.setdefault("NERO_RT_C_BUDGET", os.environ.get("NERO_RT_C_BUDGET", "10")); e.setdefault("NERO_FRAME_GATE_MIN", os.environ.get("NERO_FRAME_GATE_MIN", "0.65"))      # 2026-10-15: 회색 손전등은 책상과 색 대비가 낮아 마스크-컬러 겹침이 0.68~0.73 으로 나와 모든 프레임이 게이트(0.75)에서 탈락 -> 식별 불가      # 2026-10-15: C(pick 계획) 예산 2s -> 10s. 후보가 5~8개 있어도 첫 후보 실패가 3초 걸려 1~2개만 시도하고 '시간 예산 초과'로 끝났다
    if os.path.exists("/tmp/sort_fast"): e.update(NERO_REG_PERSIST="1", NERO_DESC_SKIP="1")
    if os.path.exists("/tmp/sort_heldcover"): e["NERO_HELD_COVER"] = "1"; e["NERO_PLACE_MARGIN"] = "0.025"      # 집은 물체 충돌구 외접 방식(기본 꺼짐)
    return e


def survey(dests_path, out="/tmp/sort_survey.json", timeout=240):
    if os.environ.get("NERO_DEST_PERCEIVE", "0") == "1":      # 2026-10-28(기본 꺼짐): 목적지 인식 모드 -- 입력 명세를 쓰지 않는다(dest_perception_iface; 본체는 PREREG 확정 후)
        import dest_perception_iface as DPI; return DPI.survey_perceived(out, timeout)
    env = dict(os.environ, **base_env(), NERO_FOCUS_SCAN="0", NERO_SCAN_FULL="1")
    subprocess.run([os.path.join(HERE, "rt_run.sh"), "--scan-only", "--dests", dests_path, "--vocab", ",".join(VOCAB), "--out", out], cwd=HERE, env=env, capture_output=True, timeout=timeout)
    sc = json.load(open(out))["scene"]; dests = {d["id"] for d in json.load(open(dests_path))}
    return {k: v for k, v in sc.items() if v.get("xy") and v.get("size_cm") and v.get("label") not in dests and not str(v.get("label", "")).startswith("dest")}


def objects_only(scene, dests_path):
    """등록부의 목적지 엔티티(입력 명세의 위치 근처)를 뺀 물체만."""
    D = json.load(open(dests_path)); out = {}
    for k, v in scene.items():
        if any(math.hypot(v["xy"][0] - d["xy"][0], v["xy"][1] - d["xy"][1]) < 0.5 * max(d["size"][0], d["size"][1]) + 0.04 for d in D): continue
        out[k] = v
    return out


def _relations(scene):
    """물체별 기하 관계: 로봇 원점으로부터 거리/방위, 가장 가까운 이웃과 틈, 이 물체 앞(로봇 쪽)을 가로막는 물체. LLM 이 집을 순서를 정할 때 쓴다."""
    rel = {}; P = {k: (v["xy"][0], v["xy"][1], max(v["size_cm"][0], v["size_cm"][1]) / 200.0) for k, v in scene.items()}
    for k, (x, y, r) in P.items():
        d = math.hypot(x, y); b = math.degrees(math.atan2(y, x))
        nb = sorted(((math.hypot(x - x2, y - y2) - r - r2, k2) for k2, (x2, y2, r2) in P.items() if k2 != k))
        blk = []
        for k2, (x2, y2, r2) in P.items():
            if k2 == k: continue
            d2 = math.hypot(x2, y2)
            if d2 < d and abs(math.sin(math.atan2(y2, x2) - math.atan2(y, x))) * d2 < (r + r2 + 0.04) and math.cos(math.atan2(y2, x2) - math.atan2(y, x)) > 0.5: blk.append(k2)      # 로봇 쪽에서 이 물체로 가는 길을 가로막는 물체
        rel[k] = dict(dist=d, bearing=b, gap=(nb[0][0] if nb else None), near=(nb[0][1] if nb else None), blocked_by=blk)
    return rel


def table_text(scene, dests_path, dests=None):
    D = dests if dests is not None else json.load(open(dests_path)); lines = ["목적지(입력값):"]
    for d in D:
        if os.environ.get("NERO_NEUTRAL_PROMPT", "0") == "1":      # 2026-10-28(기본 꺼짐): 라벨·이름 없는 중립 서술(neutral_prompt.dest_line)
            import neutral_prompt as NP; lines.append(NP.dest_line(d)); continue
        extra = {"container": f"container, 안쪽 약 {100 * (d['size'][0] - 2 * d.get('wall', 0.02)):.0f}x{100 * (d['size'][1] - 2 * d.get('wall', 0.02)):.0f}cm, 높이 {100 * d['size'][2]:.0f}cm", "flat": f"flat(평평한 접시), 지름 약 {100 * d['size'][0]:.0f}cm, 가장자리에서 쓰러지기 쉬움",
                 "hole": f"hole(좁은 입구에 세워 꽂기), 입구 지름 약 {100 * d.get('open', 0.1):.0f}cm, 깊이 {100 * d.get('depth', 0.14):.0f}cm, 긴 물체만, 옆면 파지 필요"}[d["kind"]]
        lines.append(f"- {d['id']}: {d['label']}({extra}) 위치 ({d['xy'][0]:+.2f},{d['xy'][1]:+.2f})")
    lines.append("물체(라벨은 VLM·CLIP 추정이라 틀릴 수 있고 크기도 측정 오차가 있음. 거리/방위는 로봇 원점 기준, 방위각 |b|가 크면 손목 한계로 도달이 어려울 수 있음, 틈은 가장 가까운 이웃과의 표면 간격):")
    R = _relations(scene)
    for k, v in sorted(scene.items()):
        top = ", ".join(f"{t[0]}" for t in v.get("top", [])[:3]); s = v["size_cm"]; r = R[k]
        if v.get("vlm") or v.get("general"): top = f"VLM 추정 '{v.get('vlm') or '?'}' / 일반 어휘 CLIP 후보 {', '.join(v.get('general', [])[:4]) or '?'}"      # 2026-10-19 승격: 고정 사전 라벨 대신 VLM 자유 서술 + 공개 범주 목록 CLIP 후보
        extra = f" | 거리 {100 * r['dist']:.0f}cm 방위 {r['bearing']:+.0f}° | 이웃 {r['near']} 틈 {100 * r['gap']:.0f}cm" if r["near"] else f" | 거리 {100 * r['dist']:.0f}cm 방위 {r['bearing']:+.0f}°"
        if r["blocked_by"]: extra += f" | 앞을 {','.join(r['blocked_by'])} 가 가로막음(먼저 치워야 할 수 있음)"
        lines.append(f"- {k}: 라벨 {top} | 색 {v.get('color') or '?'} | 크기 {s[0]}x{s[1]}x{s[2]}cm | 위치 ({v['xy'][0]:+.2f},{v['xy'][1]:+.2f}){extra}")
    return "\n".join(lines)


def gt_snapshot():
    """Isaac 의 gso_* 물체 위치(진단용): 한 물체를 옮기는 동안 다른 물체가 얼마나 밀렸는지 보려고 전후로 읽는다. 실패하면 빈 dict."""
    if os.environ.get("NERO_NO_GT", "0") == "1": return {}      # 2026-10-16: GT 없이 실행(현실 이식 시험) -- 시뮬 정답을 읽지 않는다
    try:
        from isaac_exec import run_code
        code = ("import json, omni.usd\nfrom pxr import UsdGeom, Usd\nst = omni.usd.get_context().get_stage(); out = {}\n"
                "for p in st.GetPrimAtPath('/World').GetChildren():\n    n = p.GetName()\n    if n.startswith('gso_'):\n"
                "        t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(); out[n] = [float(t[0]), float(t[1]), float(t[2])]\n"
                "json.dump(out, open('/tmp/gt_snap.json', 'w'))\n")
        run_code(code, timeout=20); return json.load(open("/tmp/gt_snap.json"))
    except Exception:
        return {}


LLM_LOG = "/tmp/rt_llm_log.jsonl"
def llm_log(direction, kind, text, **kw):
    """2026-10-15: 로봇 <-> LLM 주고받은 내용을 RViz 가 보여 주도록 한 줄씩 남긴다. direction: 'robot>llm'(로봇이 LLM 에 넘긴 것) / 'llm>robot'(LLM 이 로봇에 지시한 것)."""
    import json as _j, time as _t
    try: open(LLM_LOG, "a").write(_j.dumps(dict(t=_t.time(), dir=direction, kind=kind, text=str(text)[:600], **kw), ensure_ascii=False) + "\n")
    except Exception: pass
