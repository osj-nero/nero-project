"""rq1_decide3.py 의 프롬프트·호출을 그대로 가져온다(원본 수정 없이, 실험 루프 전까지만 실행). Gemma 4 31B, temp 0, test 정답표 v3."""
import os, sys
_H = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_H, "rq1_decide3.py")).read().split("os.makedirs(os.path.join(H, \"out3\")")[0]
_argv = sys.argv; sys.argv = ["rq1_decide3.py", "stage2", "--prereg", "RQ1_PREREG_v3.json", "--split", "test"]
_g = {"__file__": os.path.join(_H, "rq1_decide3.py"), "__name__": "rq1_decide3_lib"}
exec(compile(_src, "rq1_decide3.py", "exec"), _g)
sys.argv = _argv
prompt = _g["prompt"]; call = _g["call"]
