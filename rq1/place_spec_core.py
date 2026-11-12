"""놓기 조건(Gemma/VLM 이 쓴 hard spec) -> 플래너 후보(자세 이름, 월드 회전, yaw 목록, 놓을 점). 2026-11-10 PC(stage2_loop/exec_main)에서 옮김.
한 파일을 PC 와 Thor(플래너 서버 8021 폴더)에 같은 내용으로 두고 둘 다 이것만 쓴다(sha256 으로 같은지 확인). 의존: constraints.py(같이 배포), numpy, scipy.
면 이름은 '지금 자세 기준' 물체 로컬 축: 윗면 +z, 바닥면 -z, 앞면 -y, 뒷면 +y, 왼면 -x, 오른면 +x. yaw0 = 물체 지금 yaw(도; 인식 모드면 perc_frame 의 인식 yaw)."""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
_h = os.path.dirname(os.path.abspath(__file__))
for _d in (_h, os.path.join(_h, "..", "contplace")):      # Thor: 같은 폴더의 constraints.py, PC: ../contplace/constraints.py
    if os.path.exists(os.path.join(_d, "constraints.py")) and _d not in sys.path: sys.path.insert(0, _d)
from constraints import check, GRID_DIMS, CENTER
POSES = {"윗면이 위": None, "바닥면이 위": ("x", 180), "뒷면이 위": ("x", 90), "앞면이 위": ("x", -90), "왼면이 위": ("y", 90), "오른면이 위": ("y", -90)}
ORDER = ["윗면이 위", "바닥면이 위", "앞면이 위", "뒷면이 위", "왼면이 위", "오른면이 위"]
YAWS = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]
SPOTS = {"가운데": [-0.46, 0.0], "가까운 쪽": [-0.40, 0.0], "먼 쪽": [-0.52, 0.0], "왼쪽": [-0.46, -0.06], "오른쪽": [-0.46, 0.06]}   # 왼쪽 = 로봇 기준(월드 -y)


def world_rot(pose, yaw_deg):
    """물체 목표 회전(월드, '잡을 때 -> 놓을 때') = Rz(yaw0) S Rz(yaw0)^-1. 그대로면 None."""
    if POSES[pose] is None: return None
    ax, deg = POSES[pose]; Rz = Rot.from_euler("z", yaw_deg, degrees=True)
    return Rz * Rot.from_euler(ax, deg, degrees=True) * Rz.inv()


def final_R(pose, yaw0, yw):
    """잡기 전 물체 = Rz(yaw0). 서버: obj_rot(월드) 먼저, 그다음 z 로 yw -> 최종 = Rz(yw) Rz(yaw0) S."""
    S = Rot.identity() if POSES[pose] is None else Rot.from_euler(POSES[pose][0], POSES[pose][1], degrees=True)
    return Rot.from_euler("z", yw + yaw0, degrees=True) * S


def candidates(spec, asset, yaw0, prefer=None):
    """조건을 만족하는 (자세, yaw 목록, 놓을 점) 목록. 면 조건만 보고 자세를 거르고, yaw/region 은 실제 최종 상태로 판정."""
    dims = GRID_DIMS[asset]; regs = [c["where"] for c in spec["hard"] if c["type"] == "region"]
    xy = SPOTS[regs[0]] if regs else None; pos = xy or CENTER
    has_yaw = any(c["type"] == "yaw" for c in spec["hard"])
    out = []
    for pose in ORDER:
        ok_y = [yw for yw in YAWS if check(final_R(pose, yaw0, yw), spec, "place", pos=pos, dims=dims, center=CENTER, half=0.12)[0]]
        if has_yaw:      # 1도 간격으로 조건 만족 구간을 찾아 각 구간 가운데(목표 방향과 정확히 맞는 회전량)를 맨 앞에(사용자 지적: 35.2도 같은 중간 각도)
            fine = [yw for yw in range(-180, 180) if check(final_R(pose, yaw0, yw), spec, "place", pos=pos, dims=dims, center=CENTER, half=0.12)[0]]
            mids, run = [], []
            for yw in fine + [None]:
                if run and (yw is None or yw != run[-1] + 1):
                    mids.append(float(run[len(run) // 2])); run = []
                if yw is not None: run.append(yw)
            ok_y = mids + [y for y in ok_y if all(abs(((y - m + 180) % 360) - 180) > 0.5 for m in mids)]
        if not ok_y: continue
        out.append(dict(pose=pose, yaws=(ok_y if has_yaw else None), xy=xy))
    return out


def pose_entries(spec, asset, yaw0, order=None):
    """서버용 목록 [{name, rot(xyzw|None), yaws, slots}]. order = Gemma 가 정한 자세 이름 순서(빠진 자세는 기본 순서로 뒤에)."""
    cs = candidates(spec, asset, yaw0); by = {c["pose"]: c for c in cs}
    names = [n for n in (order or []) if n in by]; names = list(dict.fromkeys(names)) + [c["pose"] for c in cs if c["pose"] not in names]
    out = []
    for n in names:
        c = by[n]; Rw = world_rot(n, yaw0)
        out.append(dict(name=n, rot=(None if Rw is None else [float(v) for v in Rw.as_quat()]), yaws=c["yaws"], slots=([c["xy"]] if c["xy"] else None)))
    return out
