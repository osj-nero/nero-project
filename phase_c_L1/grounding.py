#!/usr/bin/env python3
"""grounding.py -- 이름 없는 마스크(위치) + CLIP 크롭 채점(이름)으로 자연어 라벨을 붙이는 접지 클라이언트.
눈은 검출기(깊이 기반 마스크 + Thor CLIP 서버 :8030)이고, 언어 해석/어휘 확장은 상위(Claude/MCP)가 문자열로만 넘긴다.
  ground_masks(rgb, masks, queries, competitors) -> [ {phrase: p} ... ]   (마스크 순서와 같음, softmax 확률)
  select_target(probs, queries, thr) -> (ok, p_target, best_phrase)"""
import os, json, base64, urllib.request
from thor_net import thor_ip
import numpy as np
import cv2

CLIP_URL = os.environ.get("NERO_CLIP_URL", f"http://{thor_ip()}:8030")

# 경쟁 어휘(상시): 탁상 위 흔한 물체. 상위가 --vocab 로 추가/대체한다. 대상 문구와 같은 뜻의 문구(동의어)는 넣지 말 것.
DEFAULT_VOCAB = ["a cup", "a bottle", "a box", "a toy", "a tool", "a can", "a book", "a roll of tape", "a pair of scissors",
                 "a phone", "a remote control", "a hard drive", "a game cartridge", "a figurine", "a sponge", "a bag"]


def norm_phrase(p):
    p = p.strip()
    low = p.lower()
    return p if low.startswith(("a ", "an ", "the ")) else f"a {p}"


def split_queries(text):
    """'flashlight|torch' -> ['a flashlight', 'a torch'] (동의어는 | 로 구분, 확률은 합산 취급)"""
    return [norm_phrase(t) for t in text.split("|") if t.strip()]


def ground_masks(rgb, masks, queries, competitors=None, timeout=20, url=None):
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    if not boxes:
        return []
    if os.environ.get("NERO_CROP_EXTEND", "1") == "1":      # 2026-10-12(2026-10-13 기본 켜짐, =0 으로 끔): 마스크가 화면 아래 28%(그리퍼 영역, agn_detect.GRIPPER_ROW)에서 잘린 물체는 CLIP 크롭을 아래로 늘린다(파지 점군/마스크는 그대로)
        _H = rgb.shape[0]; _cap = int(0.72 * _H)
        boxes = [[b[0], b[1], b[2], (min(_H - 1.0, b[3] + max(1.2 * (b[3] - b[1]), 0.12 * _H)) if b[3] >= _cap - 3 else b[3])] for b in boxes]
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., :3].astype(np.uint8))[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    body = json.dumps(dict(image_b64=base64.b64encode(buf.tobytes()).decode(), boxes=boxes, queries=list(queries),
                           competitors=list(competitors or []), return_feats=(os.environ.get("NERO_EMBED", "0") == "1"))).encode()      # 2026-10-14: NERO_EMBED=1 이면 크롭별 CLIP 이미지 임베딩도 받는다
    rq = urllib.request.Request((url or CLIP_URL) + "/ground", data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(rq, timeout=timeout).read())
    ground_masks.last_logits = r.get("logits_all", []); ground_masks.last_feats = r.get("img_feats")      # 원시 로짓(장면 내 검색식 판정/분석용)
    _d = os.environ.get("NERO_GROUND_DUMP")      # 2026-10-12 분석용(2026-10-13 기본 켜짐, =0 으로 끔): 접지 호출마다 이미지/박스/문구/응답을 저장한다
    if _d:
        try:
            import time as _t; os.makedirs(_d, exist_ok=True); _k = f"{_d}/g_{int(_t.time() * 1000)}"
            open(_k + ".jpg", "wb").write(buf.tobytes()); _dz = getattr(ground_masks, "depth_dump", None)
            if _dz is not None: np.save(_k + "_z.npy", np.asarray(_dz, dtype=np.float16))
            np.save(_k + "_m.npy", np.stack([m for m in masks]).astype(np.uint8)); json.dump(dict(boxes=boxes, queries=list(queries), competitors=list(competitors or []), probs_all=r.get("probs_all"), logits_all=r.get("logits_all")), open(_k + ".json", "w"))
        except Exception as _e: print("접지 덤프 실패", repr(_e))
    return r["probs_all"]


def select_target(probs, queries, thr=0.35):
    """probs: {문구: p}. 대상(동의어 합산) 확률이 thr 이상이고 가장 높은 단일 문구가 대상 문구 중 하나일 때만 대상.
    -> (ok, p_target, best_phrase)"""
    pt = sum(probs.get(q, 0.0) for q in queries)
    best = max(probs, key=probs.get)
    return (best in queries and pt >= thr), float(pt), best


# ── 2026-10-14: 저장된 임베딩으로 새 문구를 점수 매기기(NERO_SWEEP_SKIP) ──
GENERIC = ["an object", "a table", "the floor", "a wall", "a robot arm", "a robot gripper", "a shadow", "background", "a black table surface"]      # 서버(clip_ground_server.py)와 같아야 한다
_TF_CACHE = {}


def text_feats(phrases, url=None, timeout=20):
    """문구 -> 정규화 텍스트 임베딩(P,D). 서버 /text_feats(\"a photo of {문구}\" 템플릿)를 쓰고 캐시한다."""
    miss = [p for p in dict.fromkeys(phrases) if p not in _TF_CACHE]
    if miss:
        rq = urllib.request.Request((url or CLIP_URL) + "/text_feats", data=json.dumps(dict(phrases=miss)).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(rq, timeout=timeout).read())
        for ph, f in zip(r["phrases"], r["feats"]): _TF_CACHE[ph] = np.asarray(f, np.float32)
    return np.stack([_TF_CACHE[p] for p in phrases])


# ── 2026-10-19: 일반 어휘 라벨 + VLM 자유 서술 라벨(승격). 사전이 아니라 공개 범주 목록(LVIS/ImageNet) 임베딩 + 비동기 VLM 이라 추론 시간은 스캔 뒤에 숨는다. ──
_GV = None


def general_labels(feat, k_each=2):
    """크롭 CLIP 이미지 임베딩 -> 일반 어휘(LVIS + ImageNet) 상위 라벨. 캐시 npz 만 읽는다(서버 호출 없음)."""
    global _GV
    try:
        if _GV is None:
            z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_assets", "general_vocab.npz")); _GV = dict(F=z["feats"].astype(np.float32), own=z["own"], src=z["src"], names=z["names"])
        f = np.asarray(feat, np.float32); f = f / (np.linalg.norm(f) + 1e-9); s = _GV["F"] @ f; cs = np.full(len(_GV["names"]), -9.0, np.float32)
        np.maximum.at(cs, _GV["own"], s); out = []
        for tag in ("l", "i"):
            idx = np.where(_GV["src"] == tag)[0]; top = idx[np.argsort(-cs[idx])[:k_each]]
            out += [str(_GV["names"][j]) for j in top if str(_GV["names"][j]) not in out]
        return out
    except Exception:
        return []


VLM_URL = os.environ.get("NERO_VLM_URL", "http://127.0.0.1:18005/v1/chat/completions")
_VLM_POOL = None


def _vlm_tunnel():
    """Thor 의 vLLM 은 127.0.0.1:8005 에만 열려 있어 PC 에서 SSH 터널(18005)로 부른다. 없으면 한 번 만든다(실패해도 라벨만 없을 뿐 동작은 그대로)."""
    import socket, subprocess
    try:
        socket.create_connection(("127.0.0.1", 18005), timeout=0.3).close(); return True
    except OSError:
        pass
    try:
        subprocess.run(["ssh", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=4", "-f", "-N", "-L", "18005:127.0.0.1:8005", "thor"], timeout=8, capture_output=True)
        socket.create_connection(("127.0.0.1", 18005), timeout=0.5).close(); return True
    except Exception:
        return False


def vlm_label(jpeg_bytes, timeout=30):
    """크롭 JPEG -> 짧은 명사구(예: 'coffee mug'). 실패하면 None."""
    try:
        b = base64.b64encode(jpeg_bytes).decode()
        body = dict(model="qwen3-vl", max_tokens=24, temperature=0, messages=[dict(role="user", content=[dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + b)),
                    dict(type="text", text="What is this object? Answer with one short common noun phrase only (e.g. 'coffee mug'), no extra words.")])])
        r = json.loads(urllib.request.urlopen(urllib.request.Request(VLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout).read())
        return r["choices"][0]["message"]["content"].strip().strip(".").lower() or None
    except Exception:
        return None


def vlm_submit(jpeg_bytes):
    """비동기 제출 -> Future (스캔을 막지 않는다). 터널/서버가 없으면 None."""
    global _VLM_POOL
    from concurrent.futures import ThreadPoolExecutor
    if _VLM_POOL is None:
        if not _vlm_tunnel(): return None
        _VLM_POOL = ThreadPoolExecutor(max_workers=3)
    return _VLM_POOL.submit(vlm_label, jpeg_bytes)


def vlm_resolve(futs, wait=4.0):
    """제출한 Future 들의 결과 중 가장 많이 나온 라벨. 남은 대기는 합계 wait 초."""
    import time as _t
    t0 = _t.time(); got = []
    for f in futs or []:
        try:
            r = f.result(timeout=max(0.05, wait - (_t.time() - t0)))
            if r: got.append(r)
        except Exception:
            pass
    return max(set(got), key=got.count) if got else None
