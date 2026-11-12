"""합성 아님, CGN 전체 후보(B 상한 해제, NERO_C_ALL_CGN=1)로 판 3·막대 1 장면 계획 요청(creq)을 새로 받는 실행 (2026-11-12).
흐름 = stage2_rules.episode_rule(정책 F: 아무 자세, 풀기 없음) 그대로 + 로봇 env 에 NERO_C_ALL_CGN=1. 실행 결과는 기둥·뚜껑 계획 시험용 요청 수집이 목적(성공률 집계에 안 씀)."""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import os, sys, json
os.environ.setdefault("S2V_OUT", "" + NERO_ROOT + "/rq1/capture_cgnall_20261112")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_rules as SR
SR.SV.EM.ENV["NERO_C_ALL_CGN"] = "1"
cases = [dict(scene="R_plate_y29", item="t3_wet_paint"), dict(scene="R_plate_y36.9", item="t3_wet_paint"), dict(scene="R_plate_y38.5", item="t3_cutting_board"), dict(scene="R_bar_y117.3", item="y_lr_pref")]
fo = open(SR.SV.OUT + "/episodes.jsonl", "a"); n = [len(os.listdir(SR.SV.OUT + "/runs"))]
for c in cases:
    ep = SR.episode_rule(c, "F", n); fo.write(json.dumps(ep, ensure_ascii=False) + "\n"); fo.flush()
    print(c["scene"], ep["outcome"], ep["sec"], flush=True)
SR.S.clean_scene(); print("CAPTURE_DONE", flush=True)
