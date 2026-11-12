#!/usr/bin/env python3
"""
randomize_box_positions.py
-----------------------------
메인 씬 usd 안의 박스 prim들(TestBox, TestBox1, TestBox2)을, 저희가
데이터로 확인한 "안전 영역" 안의 랜덤한 (x,y)로 재배치한다.
z는 그대로 두고(테이블 위에 놓인 높이 유지), x,y와 함께 z축 회전(yaw)도
랜덤화한다.

[2026-08 수정 — yaw 랜덤화 + ground truth 로그 추가]
align 단계(approach 후 재측정하는 박스 각도)의 인식 오차를 시뮬에서
정량 측정하기 위해, 이 스크립트가 쓰는 (x, y, yaw)를 곧 ground truth로
쓸 수 있게 CSV로 같이 기록한다. 기존엔 x,y만 랜덤화해서 박스 방향이
한 번도 안 바뀌었는데, 그러면 각도 인식 정확도를 애초에 검증할 방법이
없었다(항상 원본 USD에 저장된 고정 각도만 보게 됨).

★ 순수 pxr(OpenUSD)만 쓰므로 Kit/SimulationApp 없이 실행됨 (usd-core venv).
★ 라이브로 떠 있는 Isaac Sim 세션에는 즉시 반영 안 됨 -- 이 스크립트로
  파일을 먼저 바꿔두고, 그 다음 Isaac Sim을 (재)실행해야 반영된다.
  (start_nero_isaac_all.sh가 매번 --open으로 새로 여니까, 그 앞에 이
  스크립트를 끼워넣으면 실행할 때마다 자동으로 새 배치가 됨)

안전 영역 기준 (find_z_ceiling.py / find_z_ceiling_semitopdown.py 결과 반영):
  - r < 0.25m: 도넛 구멍(최소 도달거리 미만) -> 제외
  - r > 0.40m: 바깥쪽은 z 여유가 줄어들어 스택 3단째(끝단 z=0.125,
    flange z≈0.26)에서 마진이 빡빡해짐 -> 제외 (0.40~0.45는 방향에
    따라 아슬아슬해서 보수적으로 뺌)
  - theta가 ±90도(정측면) 근방 ±25도: 이 방향만 유독 천장이 낮고
    일부는 전멸이었음 -> 제외
  => 결과: r ∈ [0.25, 0.40], theta ∈ 전체에서 [65,115]도, [-115,-65]도
     구간을 뺀 나머지

사용:
    python3 randomize_box_positions.py \
        --main-usd "$HOME/nero/nero_simul.usd" \
        --seed 42   # (선택) 재현 가능한 랜덤이 필요하면

    --gt-csv 를 생략하면 ~/nero_runs/box_gt_<timestamp>.csv 에
    자동 기록된다 (align_angle_measure.py가 이 파일을 읽어 GT와 비교함).
"""
import argparse
import csv
import math
import os
import random
import time

from pxr import Usd, UsdGeom, Gf

BOX_PRIM_PATHS = ['/World/TestBox', '/World/TestBox1', '/World/TestBox2']

R_MIN, R_MAX = 0.25, 0.40
EXCLUDE_THETA_RANGES_DEG = [(65, 115), (-115, -65)]  # ±90도 근방 제외
MIN_SEPARATION_M = 0.12  # 박스끼리 너무 가깝게 겹쳐 스폰되지 않게 최소 간격

DEFAULT_GT_CSV_DIR = os.path.expanduser('~/nero_runs')


def theta_excluded(theta_deg):
    for lo, hi in EXCLUDE_THETA_RANGES_DEG:
        if lo <= theta_deg <= hi:
            return True
    return False


def sample_safe_xy(existing_points, rng):
    for _ in range(200):  # 최대 200번 재시도 (겹침 방지)
        r = rng.uniform(R_MIN, R_MAX)
        theta_deg = rng.uniform(-180, 180)
        if theta_excluded(theta_deg):
            continue
        theta = math.radians(theta_deg)
        x, y = r * math.cos(theta), r * math.sin(theta)
        if all(math.hypot(x - ex, y - ey) >= MIN_SEPARATION_M for ex, ey in existing_points):
            return round(x, 4), round(y, 4)
    raise RuntimeError('안전 영역 안에서 겹치지 않는 위치를 못 찾음 (박스 개수/간격 조정 필요)')


def _get_or_add_rotate_z_op(xformable):
    """기존 rotateZ(또는 rotateXYZ) op가 있으면 그걸 재사용하고, 없으면
    새로 추가한다. 기존 op를 재사용해야 xform op 순서가 매번 늘어나지
    않는다(재실행할 때마다 AddRotateZOp을 새로 부르면 op가 중복 누적됨)."""
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() in (UsdGeom.XformOp.TypeRotateZ, UsdGeom.XformOp.TypeRotateXYZ):
            return op, op.GetOpType()
    new_op = xformable.AddRotateZOp()
    return new_op, UsdGeom.XformOp.TypeRotateZ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-usd', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--prims', nargs='*', default=BOX_PRIM_PATHS,
                         help='재배치할 prim 경로들 (기본: TestBox/TestBox1/TestBox2)')
    parser.add_argument('--no-yaw', action='store_true',
                         help='yaw는 건드리지 않고 x,y만 랜덤화 (기존 동작으로 되돌림)')
    parser.add_argument('--gt-csv', default=None,
                         help='ground truth (prim,x,y,yaw_deg) 기록할 CSV 경로. '
                              '생략하면 analysis/box_gt_<timestamp>.csv 에 자동 생성')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    stage = Usd.Stage.Open(args.main_usd)
    if stage is None:
        raise RuntimeError(f'usd를 열 수 없습니다: {args.main_usd}')

    placed = []
    gt_rows = []
    for prim_path in args.prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f'[randomize_box_positions] 건너뜀 (prim 없음): {prim_path}')
            continue

        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        # 기존 z는 그대로 유지 (테이블 위 높이), x,y만 새로 뽑음
        if translate_op is not None:
            cur = translate_op.Get()
            cur_z = cur[2] if cur is not None else 0.0
        else:
            cur_z = 0.0
            translate_op = xformable.AddTranslateOp()

        x, y = sample_safe_xy(placed, rng)
        placed.append((x, y))
        translate_op.Set(Gf.Vec3d(x, y, cur_z))

        yaw_deg = 0.0
        if not args.no_yaw:
            yaw_deg = round(rng.uniform(0.0, 360.0), 2)
            rotate_op, op_type = _get_or_add_rotate_z_op(xformable)
            if op_type == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op.Set(Gf.Vec3f(0.0, 0.0, yaw_deg))
            else:
                rotate_op.Set(yaw_deg)

        print(f'[randomize_box_positions] {prim_path} -> '
              f'({x}, {y}, {cur_z}), yaw={yaw_deg}\xb0')
        gt_rows.append({'prim': prim_path, 'x': x, 'y': y, 'z': cur_z, 'yaw_deg': yaw_deg})

    stage.GetRootLayer().Save()
    print(f'[randomize_box_positions] 저장 완료: {args.main_usd} '
          f'({len(placed)}개 박스 재배치)')

    gt_csv = args.gt_csv
    if gt_csv is None:
        os.makedirs(DEFAULT_GT_CSV_DIR, exist_ok=True)
        gt_csv = os.path.join(
            DEFAULT_GT_CSV_DIR, f'box_gt_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    with open(gt_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['prim', 'x', 'y', 'z', 'yaw_deg'])
        w.writeheader()
        w.writerows(gt_rows)
    print(f'[randomize_box_positions] ground truth 기록: {gt_csv}')


if __name__ == '__main__':
    main()
