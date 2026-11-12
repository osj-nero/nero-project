#!/usr/bin/env python3
"""clip_ground_server.py -- (Thor :8030) 이름 없는 탐지가 만든 물체 크롭을 CLIP 으로 채점해 자연어 라벨을 붙이는 접지 서버.
이름 기반 검출(YOLO-World)은 비COCO 시뮬 물체에서 재현율이 낮아(구체 이름 10%) 위치는 깊이 기반 마스크가 맡고, 이 서버는 "이 크롭이 어느 문구인가"만 판정한다.
POST /ground  {image_b64, boxes:[[x0,y0,x1,y1] px,...], queries:[...], competitors:[...](선택), pad:0.15}
  -> {scores:[{query: p, ...} 박스별], probs_all:[{문구: p}], ms}
확률 = [queries + competitors + 일반 방해어] 위 softmax(온도 100). 방해어는 서버 내장(배경/바닥/로봇 등)."""
import os, io, sys, time, json, base64, argparse
import numpy as np
from PIL import Image
import torch, clip
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

WEIGHT = os.environ.get("CLIP_WEIGHT", os.path.expanduser("~/weights/clip/ViT-L-14.pt"))     # 비교 결과 L/14 가 B/32 보다 정확(검색 top1 8/9 대 6~7/9)
GENERIC = ["an object", "a table", "the floor", "a wall", "a robot arm", "a robot gripper", "a shadow", "background", "a black table surface"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, pre = clip.load(WEIGHT, device=dev)
model.eval()
_cache = {}


def text_feats(phrases):
    miss = [p for p in phrases if p not in _cache]
    if miss:
        with torch.no_grad():
            f = model.encode_text(clip.tokenize([f"a photo of {p}" for p in miss]).to(dev)).float()
            f = f / f.norm(dim=-1, keepdim=True)
        for p, v in zip(miss, f):
            _cache[p] = v
    return torch.stack([_cache[p] for p in phrases])


class Req(BaseModel):
    image_b64: str
    boxes: List[List[float]]
    queries: List[str]
    competitors: Optional[List[str]] = None
    pad: float = 0.15
    return_feats: bool = False      # 2026-10-14: True 면 크롭별 정규화 이미지 임베딩(img_feats)도 돌려준다(엔티티 기억/문구 점수 재계산용)


class TReq(BaseModel):
    phrases: List[str]


app = FastAPI()


@app.get("/health")
def health():
    return dict(status="ok", model="clip", device=dev)


@app.post("/ground")
def ground(r: Req):
    t0 = time.time()
    img = Image.open(io.BytesIO(base64.b64decode(r.image_b64))).convert("RGB")
    W, H = img.size
    crops = []
    for x0, y0, x1, y1 in r.boxes:
        w, h = x1 - x0, y1 - y0
        s = max(w, h) * (1 + 2 * r.pad)                     # 정사각 + 여백(종횡비 왜곡 방지)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2)))
        crops.append(pre(img.crop(box)))
    if not crops:
        return dict(scores=[], probs_all=[], ms=0.0)
    phrases = list(dict.fromkeys(list(r.queries) + list(r.competitors or []) + GENERIC))
    tf = text_feats(phrases)
    with torch.no_grad():
        imf = model.encode_image(torch.stack(crops).to(dev)).float()
        imf = imf / imf.norm(dim=-1, keepdim=True)
        lg = 100.0 * imf @ tf.T
        p = lg.softmax(dim=-1).cpu().numpy(); lg = lg.cpu().numpy()
    probs_all = [{ph: round(float(p[i, j]), 4) for j, ph in enumerate(phrases)} for i in range(len(crops))]
    logits_all = [{ph: round(float(lg[i, j]), 2) for j, ph in enumerate(phrases)} for i in range(len(crops))]
    scores = [{q: pa[q] for q in r.queries} for pa in probs_all]
    out = dict(scores=scores, probs_all=probs_all, logits_all=logits_all, ms=round((time.time() - t0) * 1000, 1))
    if r.return_feats: out["img_feats"] = [[round(float(x), 4) for x in row] for row in imf.cpu().numpy()]
    return out


@app.post("/text_feats")
def text_feats_ep(r: TReq):
    tf = text_feats(list(dict.fromkeys(r.phrases)))      # 서버가 쓰는 "a photo of {문구}" 템플릿과 같은 정규화 텍스트 임베딩. 점수 = scale * (img_feat @ text_feat)
    ph = list(dict.fromkeys(r.phrases))
    return dict(phrases=ph, feats=[[round(float(x), 4) for x in row] for row in tf.cpu().numpy()], scale=100.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8030)
    a = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=a.port)
