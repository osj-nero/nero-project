#!/usr/bin/env python3
"""place_rank_worker.py -- (개발 사본, 2026-11-04) 확정 모델(H1 F2 gbm, p(NO_PLAN)) 추론 워커. sklearn 이 있는 venv 파이썬으로 실행되는 상주 프로세스.
시작 시 모델 5개를 모두 로드하며 각 sha256 을 동결 목록과 대조(하나라도 불일치면 종료코드 3): 'all'=FROZEN_MODELS.json 의 H1|F2|gbm(전체 ABC), 'fold0..3'=hardcase_v3_4/oof/FROZEN_OOF_v3_4.json 의 제외 폴드 모델.
추가로 NERO_PLACE_RANK_MODEL(경로, +NERO_PLACE_RANK_MODEL_SHA 또는 같은 폴더 FROZEN_OOF_v3_4.json)을 'env' 로 로드 가능.
요청 한 줄 JSON {"rows": [[F2 28개]...], "model": "all|fold0..3|env"(선택)} -> {"p": [...]}. model 생략 시 기본 = NERO_PLACE_RANK_FOLD(0~3)/NERO_PLACE_RANK_MODEL 지정이 있으면 그것, 없으면 'all'."""
import os as _nm_os; NERO_MODEL_DIR = _nm_os.environ.get("NERO_MODEL_DIR") or _nm_os.path.expanduser("~/nero_models")  # optional learned-ranker files (override: NERO_MODEL_DIR)
import sys, os, json, hashlib
D = os.environ.get("NERO_PLACE_RANK_DIR", "" + NERO_MODEL_DIR + "/exp_pro_ml_20261104")
KEY = os.environ.get("NERO_PLACE_RANK_KEY", "H1|F2|gbm")
OOFD = D + "/hardcase_v3_4/oof"
import joblib, numpy as np
M = {}
def load(tag, path, want):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != want: print(json.dumps({"err": "sha 불일치 %s %s" % (tag, h[:12])}), flush=True); sys.exit(3)
    M[tag] = joblib.load(path)
fz = json.load(open(D + "/FROZEN_MODELS.json"))[KEY]; load("all", D + "/" + fz["file"], fz["sha256"])
oo = json.load(open(OOFD + "/FROZEN_OOF_v3_4.json"))
for k, v in oo.items(): load("fold" + k.rsplit("fold", 1)[1], OOFD + "/" + v["file"], v["sha256"])
mp = os.environ.get("NERO_PLACE_RANK_MODEL", ""); default = "all"
if mp:
    sha = os.environ.get("NERO_PLACE_RANK_MODEL_SHA", "")
    reg = os.path.join(os.path.dirname(mp), "FROZEN_OOF_v3_4.json")
    if not sha and os.path.exists(reg):
        for k, v in json.load(open(reg)).items():
            if v.get("file") == os.path.basename(mp): sha = v["sha256"]; break
    if not sha: print(json.dumps({"err": "모델 sha 기준 없음: " + mp}), flush=True); sys.exit(3)
    load("env", mp, sha); default = "env"
elif os.environ.get("NERO_PLACE_RANK_FOLD", "") != "": default = "fold%d" % int(os.environ["NERO_PLACE_RANK_FOLD"])
nf = len(json.load(open(D + "/features.json"))["F2"])
print(json.dumps({"ready": True, "nf": nf, "models": sorted(M), "default": default}), flush=True)
for line in sys.stdin:
    try:
        q = json.loads(line); k = q.get("model") or default
        if k not in M: raise KeyError("모델 없음 " + k)
        X = np.array(q["rows"], float).reshape(-1, nf)
        print(json.dumps({"p": [float(v) for v in M[k].predict_proba(X)[:, 1]], "model": k}), flush=True)
    except Exception as e: print(json.dumps({"err": repr(e)}), flush=True)
