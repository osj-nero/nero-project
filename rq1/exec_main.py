"""RQ1 메인 실험 1단계(2026-11-08): Isaac 실행 표.
장면 = 30cm 작업대(-0.46,0) + 물체 1개(-0.30,0.30, 시작 yaw). 각 장면에서 놓기 자세 6개(그대로/뒤집기/앞·뒤·왼·오른면이 위)를
각각 시스템 그대로(rt_run, 실행 포함) 한 번씩 돌려 '계획이 섰나' + 'Isaac 최종 자세'를 기록한다. (넘어뜨리기는 주장 범위 밖 -> 제외)
정책 판정(2단계, exec_policies.py)은 이 표의 실제 실행 결과만 쓴다: 정책이 고른 자세의 실행 결과 = 그 정책의 결과.
결과: exec_main/memo.json (키 scene|자세), exec_main/<scene>/<자세>/{run.log,run.json,final.png}
사용: python3 exec_main.py plate_y0,box_y60,...   (기본: plate/box/bar x 0/60/120)"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import os, sys, json, re, time, math, subprocess
import numpy as np
from scipy.spatial.transform import Rotation as Rot
R = "" + NERO_ROOT + "/rq1"; PC = "" + NERO_ROOT + "/phase_c_L1"
sys.path.insert(0, PC); sys.path.insert(0, R)
from isaac_exec import run_code
SRC = open(R + "/step0_v2.py").read()
PLATFORM_CODE = SRC.split("PLATFORM = '''")[1].split("'''")[0]; CUBE_CODE = SRC.split("CUBE = '''")[1].split("'''")[0]
OX, OY = -0.30, 0.30; PX, PY, TOP = -0.46, 0.0, 0.30
DIMS = {"plate": (0.12, 0.015, 0.10), "box": (0.07, 0.10, 0.15), "bar": (0.20, 0.03, 0.03), "block": (0.05, 0.05, 0.10), "eggbox": (0.15, 0.10, 0.07)}
# 자세 이름(지금 자세 기준) -> 시작 yaw 좌표계에서의 회전. constraints.box_stable 과 같은 정의(물체 로컬 축 기준)
from place_spec_core import POSES      # 2026-11-10 공통 모듈로


from place_spec_core import world_rot      # 2026-11-10 공통 모듈로(같은 코드)


def spawn(obj, yaw):
    run_code(PLATFORM_CODE, timeout=40)
    sx, sy, sz = DIMS[obj]
    return run_code(CUBE_CODE.format(name=obj, x=OX, y=OY, z=sz / 2 + 0.002, yaw=float(yaw), sx=sx, sy=sy, sz=sz), timeout=40)["status"]


POSE_Q = ("import json, omni.usd\nfrom pxr import UsdGeom, Usd\nst=omni.usd.get_context().get_stage(); out={}\n"
          "for p in st.GetPrimAtPath('/World').GetChildren():\n    n=p.GetName()\n    if n.startswith('gso_'):\n"
          "        m=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()); t=m.ExtractTranslation()\n"
          "        out[n]=dict(p=[float(t[0]),float(t[1]),float(t[2])], rows=[[float(m[i][j]) for j in range(3)] for i in range(3)])\n"
          "json.dump(out, open('/tmp/rq1_exec_pose.json','w'))\n")


def final_state():
    run_code(POSE_Q, timeout=20); d = json.load(open("/tmp/rq1_exec_pose.json"))
    if not d: return None
    v = list(d.values())[0]; rows = np.array(v["rows"])
    cols = np.stack([rows[i] / np.linalg.norm(rows[i]) for i in range(3)], axis=1)      # 열 = 물체 로컬 x,y,z 축의 월드 방향(스케일 제거)
    x, y, z = v["p"]
    return dict(xyz=[round(x, 4), round(y, 4), round(z, 4)], R=cols.round(4).tolist(), on_platform=bool(abs(x - PX) < 0.12 and abs(y - PY) < 0.12 and z > TOP))


def shot(path):
    try: run_code(f"from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file\ncapture_viewport_to_file(get_active_viewport(), '{path}')\n", timeout=20)
    except Exception: pass


ENV = dict(os.environ, ROS_LOCALHOST_ONLY="1", NERO_RT_IKURL="http://127.0.0.1:8020", NERO_RT_CUROBO="http://127.0.0.1:8021", NERO_RT_THOR="http://" + os.environ.get("NERO_THOR_IP", "127.0.0.1") + ":8012",
           NERO_PLACE_MODE="same", NERO_C_MAX_CANDS="40", NERO_RT_C_BUDGET="120", NERO_DEST_MASS="2.0", NERO_RT_PLACE_BUDGET="16.0", NERO_PLACE_AWARE="1", NERO_DESC_HINT_W="10", NERO_PLACE_FINGER_REJECT="report", NERO_PLACE_SNAPS="1")
for k in ("NERO_PLACE_TOPFIRST", "NERO_PLACE_TILT_FREE", "NERO_PLACE_YAWS", "NERO_PLACE_YAW0", "NERO_PLACE_TOPZ", "NERO_PLACE_WORLD_YAW", "NERO_PLACE_OBJ_ROT", "NERO_LOG_TRIAL", "NERO_C_REQ_DUMP"):
    ENV.pop(k, None)
VOC = subprocess.run(["python3", "-c", "import sort_tools as ST;print(','.join(ST.VOCAB))"], cwd=PC, capture_output=True, text=True).stdout.strip()
DESTS = R + "/dests_platform30.json"


def run_pose(scene, obj, yaw, pose):
    D = f"{R}/exec_main/{scene}/{pose.replace(' ', '_')}"; os.makedirs(D, exist_ok=True)
    st = spawn(obj, yaw); time.sleep(5)
    e = dict(ENV); Rw = world_rot(pose, yaw)
    if Rw is not None: e["NERO_PLACE_OBJ_ROT"] = json.dumps([float(v) for v in Rw.as_quat()])
    t0 = time.time()
    try:
        subprocess.run(["./rt_run.sh", f"--target-at={OX},{OY}", "--dests", DESTS, "--dest-id", "platform", "--dest-kind", "flat", "--vocab", VOC, "--max-picks", "1",
                        "--place-clear", "0.015", "--out", D + "/run.json"], cwd=PC, env=e, stdout=open(D + "/run.log", "w"), stderr=subprocess.STDOUT, timeout=600)
    except subprocess.TimeoutExpired:
        pass
    sec = round(time.time() - t0); time.sleep(5)
    # 2026-11-08 v2: 판정은 홈 복귀 '전' 상태(놓고 5초 뒤). 홈 복귀는 충돌 검사 없이 팔을 옮겨 잘 놓인 물체도 쳐서 떨어뜨렸다(plate_y0 윗면, 진단 실행).
    # (pilot1 의 21도 기울어짐은 ㄷ자 놓기로 판이 손가락에 걸린 경우였고, 그 놓기는 이제 서버가 거른다 -> NERO_PLACE_FINGER_REJECT)
    s_pre = final_state(); shot(D + "/before_home.png")
    subprocess.run(["python3", "home_and_get_armq.py"], cwd=PC, env=dict(os.environ, ROS_LOCALHOST_ONLY="1"), capture_output=True, timeout=90); time.sleep(3)
    log = open(D + "/run.log").read()
    planned = "pick 과 place 모두 계획됨" in log
    mv = re.search(r"place verdict=(\w+)", log)
    mg = re.search(r"pre_open 그리퍼 관절 ([0-9.]+)", log); mgr = re.search(r"파지 기록: 접촉 ([0-9.]+)m / 물체높이 ([0-9.]+)m = (\d+)%, 접근각 (-?\d+)도", log)
    s_post = final_state(); shot(D + "/after_home.png"); s = s_pre
    mfd = re.search(r"놓기 손 진단 (\{.*\})", log)
    return dict(scene=scene, pose=pose, spawn=st, planned=planned, verdict=(mv.group(1) if mv else None), state=s, sec=sec,
                pre_open_grip=(float(mg.group(1)) if mg else None), state_after_home=s_post, fdiag=(json.loads(mfd.group(1)) if mfd else None), grasp=(dict(height_pct=int(mgr.group(3)), approach_deg=int(mgr.group(4))) if mgr else None))


if __name__ == "__main__":
    scenes = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"{o}_y{y}" for o in ("plate", "box", "bar") for y in (0, 60, 120)]
    os.makedirs(f"{R}/exec_main", exist_ok=True); mf = f"{R}/exec_main/memo.json"
    memo = json.load(open(mf)) if os.path.exists(mf) else {}
    for scene in scenes:
        obj, yaw = scene.split("_y")[0], int(scene.split("_y")[1])
        for pose in POSES:
            key = f"{scene}|{pose}"
            if key in memo: continue
            memo[key] = run_pose(scene, obj, yaw, pose); json.dump(memo, open(mf, "w"), indent=1, ensure_ascii=False)
            r = memo[key]; print(f"{scene} {pose}: 계획 {'O' if r['planned'] else 'X'} verdict={r['verdict']} 작업대위={r['state'] and r['state']['on_platform']} {r['sec']}s", flush=True)
    run_code("import omni.usd\nst=omni.usd.get_context().get_stage()\nfor p in list(st.GetPrimAtPath('/World').GetChildren()):\n    if p.GetName().startswith('gso_') or p.GetName().startswith('Platform_'): st.RemovePrim(p.GetPath())\n", timeout=40)
    print("EXEC_MAIN_DONE", flush=True)
