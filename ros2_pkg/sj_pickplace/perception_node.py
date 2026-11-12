#!/usr/bin/env python3
"""
perception_node.py  (박스 전용 단순화 버전 v3 — 각도보정 복원판)
=================================================================================
[변경 이력 vs 기존 nero_ai perception_node]
- 멀티서버 클러스터(CLUSTER_N, 라운드로빈) 제거 → 박스 서버 하나에만 디스패치
- RealSense 노출값 수동 설정 추가 (어두운 환경 대응)
- 이미지 원본 크기(640x480)로 서버에 전송 (다운스케일 시 탐지 누락 방지)
- /detected_objects 퍼블리시 포맷 기존과 동일 유지 (planning_node 변경 불필요)

[2026-07 eye-in-hand 캘리브레이션 통합]
- pixel_to_robot_xyz(호모그래피 기반 평면 가정) 제거
- pixel_to_camera_xyz(표준 핀홀 역투영) + tf2(camera_color_optical_frame
  -> base_link)로 교체. 카메라가 tcp_link에 붙어 팔과 함께 움직이는
  eye-in-hand 구조이므로, 매 detection마다 그 순간의 실제 팔 자세를
  반영한 tf 변환이 필요함 (고정 오프셋 계산으로는 안 됨).

[2026-07 각도보정 제거 -> 재도입]
- 1차 시도: joint5/joint7 값을 직접 조작하는 방식의 조인트 제어 기반
  각도보정은 기구학적으로 성립 불가(직렬 링크 특정 조인트만 돌리는 게
  불가능)하여 실패, 관련 코드 전부 제거했었음.
- 2차 시도: 이 파일의 angle_base_deg 계산(본 함수)은 유지한 채,
  planning_node 쪽에서 "그리퍼가 top-down을 유지하면서 접근축 기준으로만
  회전하는" 자세를 approach 단계 진입 전에 한 번만 계산해서 시퀀스 내내
  고정 사용하는 방식으로 재설계함 (중간에 자세를 바꾸는 로직 없음).
  실제 하드웨어(agx_arm_ctrl_single_node)에서 pitch=90°, yaw=0° 고정,
  roll=-각도 로 15° 간격 스윕 검증 완료 (roll 0~-90° 전 구간 IK 성공,
  실측 tf 결과가 항상 top-down 자세(roll≈180°,pitch≈0°) + 원하는 yaw
  회전으로 정확히 일치함을 확인, 2026-07).
- 이 재도입 버전에서 perception_node가 할 일은 변경 없음: 박스의
  base_link 기준 yaw 각도(0~90도 정규화)를 계산해서 angle_base_deg로
  실어 보내는 것까지가 이 노드의 책임.

[환경변수]
  BOX_SERVER_URL  : 박스 서버 주소 (기본: http://127.0.0.1:8002/detect)
  BOX_HEALTH_URL  : 헬스체크 주소 (기본: http://127.0.0.1:8002/health)
  CAM_EXPOSURE    : RealSense 노출값 (기본: 500, 0이면 자동노출)
  DISPATCH_RATE_HZ: 추론 요청 주기 (기본: 10Hz)
  TARGET_LABEL    : 탐지 대상 클래스 (기본: box). 쉼표로 복수 지정 가능: "box,bottle"
  BASE_FRAME      : tf 변환 목표 프레임 (기본: base_link)
  CAMERA_OPTICAL_FRAME : 카메라 광학 프레임 (기본: camera_color_optical_frame)
"""

import os
import json
import base64
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3Stamped

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (PointStamped/Vector3Stamped 변환 등록용)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from sj_pickplace.camera_calibration import (
    CameraIntrinsics, set_intrinsics, pixel_to_camera_xyz,
)


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
_raw_labels      = os.environ.get("TARGET_LABEL", "box")
TARGET_LABELS    = [l.strip() for l in _raw_labels.split(",") if l.strip()]
TARGET_LABEL     = TARGET_LABELS[0]  # 단일 라벨 기대 코드와의 하위 호환용
BOX_SERVER_URL   = os.environ.get("BOX_SERVER_URL", "http://127.0.0.1:8002/detect")
BOX_HEALTH_URL   = os.environ.get("BOX_HEALTH_URL", "http://127.0.0.1:8002/health")
REQUEST_TIMEOUT  = float(os.environ.get("REQUEST_TIMEOUT", "3.0"))
DISPATCH_RATE_HZ = float(os.environ.get("DISPATCH_RATE_HZ", "10.0"))
# 이미지 하단 N% 마스킹 (그리퍼가 카메라에 잡히는 경우 사용, 기본 0=비활성)
GRIPPER_MASK_BOTTOM = float(os.environ.get("GRIPPER_MASK_BOTTOM", "0.0"))


# 카메라 해상도
CAM_W   = int(os.environ.get("CAM_W", "640"))
CAM_H   = int(os.environ.get("CAM_H", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))

# 노출값: 0이면 자동노출, 그 외 수동값 (500 권장)
CAM_EXPOSURE = int(os.environ.get("CAM_EXPOSURE", "500"))

# 오탐 필터
MIN_BBOX_SIZE = 0.02

# [2026-08 포팅] 기존엔 2D bbox 중심좌표만으로 dedup(DEDUP_THRESH=0.08)했는데,
# 쌓여있는 두 박스는 top-down 근처 각도에서 x,y footprint가 거의 겹쳐서
# "같은 박스"로 오인되어 위/아래 중 하나가 통째로 사라지는 문제가 실측
# 확인됨(placement_verification이 새로 놓인 박스를 못 찾아 verified=false/null로
# 오판하는 원인). perception_node_sim.py가 동일 문제를 먼저 겪고 depth까지
# 포함한 3D 위치 기준 dedup(_dedup_3d)으로 고쳐뒀던 걸 여기로 포팅한다 --
# x,y가 가깝더라도 z(depth)가 이 임계값보다 멀면 쌓인 별개 박스로 보고 유지.
DEDUP_XY_THRESH_M = float(os.environ.get("DEDUP_XY_THRESH_M", "0.03"))
DEDUP_Z_THRESH_M  = float(os.environ.get("DEDUP_Z_THRESH_M", "0.025"))

# tf 변환 대상 프레임 (URDF에서 tcp_link 하위에 붙인 광학 프레임과 일치해야 함)
BASE_FRAME = os.environ.get("BASE_FRAME", "base_link")
CAMERA_OPTICAL_FRAME = os.environ.get("CAMERA_OPTICAL_FRAME", "camera_color_optical_frame")
TF_TIMEOUT_SEC = float(os.environ.get("TF_TIMEOUT_SEC", "0.2"))

# [2026-08 추가, 프로토타입 -- 실기 미검증] 평면적합(point cloud plane fit)
# 기반 face normal yaw 계산용 파라미터. _compute_box_angle_base(2D Hough
# edge 각도)와 다른 신호 -- depth로 3D 평면을 직접 맞춰서 그 평면이 실제로
# 향하는 방향(normal)을 구한다. side/pinch 그립처럼 물체의 "옆면"을 볼 때만
# 의미 있는 신호이고(카메라가 top-down으로 물체 윗면을 볼 때는 normal이
# 거의 수직이라 방위각 정보가 없음 -- 아래 verticality 게이트가 이 경우를
# 걸러낸다), 아직 planning_node.py 소비 경로에는 연결하지 않았다(설계
# 배경은 docs/wiki 참고 예정).
FACE_NORMAL_GRID_STEP      = int(os.environ.get("FACE_NORMAL_GRID_STEP", "8"))
FACE_NORMAL_MIN_POINTS     = int(os.environ.get("FACE_NORMAL_MIN_POINTS", "25"))
FACE_NORMAL_MAX_POINTS     = int(os.environ.get("FACE_NORMAL_MAX_POINTS", "400"))
# planarity: 점군이 평면 하나에 얼마나 잘 맞는지(1=완벽한 평면, 0=평면
# 아님). surface-variation(최소고유값/전체합) 기반 -- 포인트클라우드
# 처리에서 흔히 쓰는 "flatness/curvature" 지표.
FACE_NORMAL_PLANARITY_MIN    = float(os.environ.get("FACE_NORMAL_PLANARITY_MIN", "0.7"))
# verticality: normal의 수평성분 크기(단위벡터라 0~1, 완전 수직면=1,
# 완전 수평면=0). 수평면(물체 윗면)을 보고 있으면 방위각이 노이즈에
# 불과하므로 이 임계값 밑이면 아예 값을 안 낸다.
FACE_NORMAL_VERTICALITY_MIN  = float(os.environ.get("FACE_NORMAL_VERTICALITY_MIN", "0.5"))


def _transform_with_fallback(tf_buffer, msg, target_frame: str, timeout_sec: float,
                              logger=None):
    """지정된 stamp(msg.header.stamp)로 tf 조회를 먼저 시도한다.

    Isaac Sim(sim-time) 이미지 stamp와 tf 트리(wall-clock)의 시간 도메인이
    어긋나 있으면(use_sim_time 미설정 등) 'extrapolation into the past'
    예외가 항상 발생할 수 있다. 이 경우 최신 tf(=stamp를 '지금'으로 재설정)
    로 한 번 더 시도해서, 프레임-tf 시점 불일치는 감수하더라도 파이프라인
    전체가 죽지 않게 한다. 두 시도 모두 실패하면 예외를 그대로 올린다.

    근본 해결책은 아님 -- clock 도메인 불일치 자체는 use_sim_time을 관련
    노드 전부에 일관되게 설정하고 Isaac Sim이 /clock을 발행하도록 해야
    고쳐진다 (CLAUDE.md/런치 스크립트 쪽에서 조치할 사항).
    """
    try:
        return tf_buffer.transform(msg, target_frame, timeout=Duration(seconds=timeout_sec))
    except Exception as e:
        if logger is not None:
            logger.warn(
                f'stamp 기준 tf 조회 실패({e}) -> 최신 tf로 폴백 '
                f'(sim-time/wall-clock 불일치 의심, use_sim_time 확인 필요)',
                throttle_duration_sec=5.0)
        msg.header.stamp = rclpy.time.Time().to_msg()
        return tf_buffer.transform(msg, target_frame, timeout=Duration(seconds=timeout_sec))


def _compute_box_angle_base(color: np.ndarray, d: dict,
                             tf_buffer, cam_frame: str, base_frame: str,
                             timeout_sec: float = 0.2,
                             depth: Optional[np.ndarray] = None,
                             stamp=None,
                             logger=None):
    """
    bbox ROI에서 박스의 base_link 기준 yaw 각도(도) 계산.

    [2026-07 수정] 기존엔 검출된 모든 선분(윗면/옆면/그림자/노이즈 구분 없이)의
    각도를 단순 median 처리해서, 뷰(top/side/oblique) 전반에서 프레임마다
    값이 크게 요동치는 문제가 있었음 (같은 박스, 같은 자세인데도 5~88도까지
    널뛰기하는 것을 반복 측정으로 확인).

    수정: median 대신, 선분들을 길이로 가중한 방향 히스토그램(0~90도, mod 90)을
    만들어 가장 표를 많이 받은 "지배적 방향"을 찾고, 그 방향 근처(±5도) 선분들만
    모아 가중 원형평균(circular mean)으로 대표각을 계산. 박스는 직육면체라 어느
    뷰에서 봐도 모서리가 두 개의 주 방향(90도 관계)으로 모이므로, 다수결 방식이
    노이즈/그림자/타면 모서리에 훨씬 덜 흔들림.
    """
    # stamp: 이 프레임이 실제로 캡처된 시각 (rclpy Time msg). None이면
    # 기존처럼 "지금"을 씀 (하위호환용, 렉 있을 땐 쓰지 말 것).
    _stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()

    H, W = color.shape[:2]
    x1 = int(d["x_min"] * W); y1 = int(d["y_min"] * H)
    x2 = int(d["x_max"] * W); y2 = int(d["y_max"] * H)
    roi = color[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # [2026-07 수정 v2] depth로 gray를 0으로 지우니, 그 마스크 경계선
    # 자체가 Canny에 새 인공 edge로 잡혀서 매번 똑같이 13~15도로
    # 수렴하는 '안정적으로 틀린' 결과가 나옴(마스크 경계는 프레임마다
    # 거의 안 변해서 진짜 박스 edge의 노이즈보다 더 강한 가짜 신호가
    # 됨). gray/edges는 원본 그대로 두고, depth는 검출된 선분을 사후
    # 필터링하는 데만 쓴다(아래 border 필터 루프에서 사용).
    depth_roi = None
    center_depth = None
    if depth is not None:
        depth_roi = depth[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if depth_roi.shape != gray.shape:
            depth_roi = None
        else:
            valid = depth_roi[(depth_roi > 0.05) & (depth_roi < 5.0)]
            if valid.size > 0:
                center_depth = float(np.median(valid))
            else:
                depth_roi = None

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                             minLineLength=roi.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return None, None

    # (0~90도로 정규화된 각도, 선분 길이) 쌍 모으기
    weighted = []
    # [2026-07 수정] bbox crop 경계선(ROI 테두리) 자체가 이미지 좌표축에
    # 정확히 평행한(0/90도) 가짜 직선으로 Hough에 잡혀서, 히스토그램에서
    # 압도적으로 큰 표를 받아 진짜 박스 모서리를 이기는 현상이 확인됨
    # (GT 90도/10도에서 심하게 틀리고 45도는 멀쩡했던 원인). ROI 가장자리
    # margin 픽셀 이내에 끝점이 걸친 선분은 crop 아티팩트로 간주해 제외.
    h_roi, w_roi = roi.shape[:2]
    # [2026-07 수정] margin이 고정 3px이면, 카메라가 박스에 가까이
    # 붙어 bbox가 박스를 거의 딱 맞게 감쌀 때(실측: depth_m=0.209,
    # top-down 근접) 박스의 진짜 모서리 선분들까지 전부 "가짜 테두리"로
    # 오인돼 걸러지고, weighted가 통째로 비어 angle_base_deg=None이
    # 되는 문제가 확인됨. ROI 크기에 비례한(작은 변 기준 2%) margin으로
    # 바꿔 근접/원거리 어느 쪽에서도 일관된 여유를 갖게 한다.
    margin = max(3, int(min(h_roi, w_roi) * 0.02))
    def _is_border_artifact(xa, ya, xb, yb):
        for x, y in [(xa, ya), (xb, yb)]:
            if x <= margin or x >= w_roi - margin or y <= margin or y >= h_roi - margin:
                return True
        return False

    # [2026-07 수정] 박스 옆면의 수직(높이) 모서리가 depth 필터를 못
    # 뚫고(옆면도 박스 표면이라 depth가 비슷함) 검출되어, 윗면 모서리와
    # 경쟁하며 dominant bin을 가끔 뺏는 현상이 roi_debug.jpg로 실측
    # 확인됨(GT 90/10도에서 유독 심함). world Z축(수직)이 지금 카메라
    # 각도에서 이미지상 몇 도로 투영되는지 미리 계산해, 그 근처 각도의
    # 선분은 '옆면 모서리 후보'로 보고 아예 배제한다.
    vertical_img_angle = None
    try:
        vz = Vector3Stamped()
        vz.header.frame_id = base_frame
        vz.header.stamp = _stamp
        vz.vector.x = 0.0
        vz.vector.y = 0.0
        vz.vector.z = 1.0
        vz_cam = _transform_with_fallback(tf_buffer, vz, cam_frame, timeout_sec, logger=logger)
        _raw_v = math.degrees(math.atan2(vz_cam.vector.y, vz_cam.vector.x)) % 180
        if _raw_v > 90:
            _raw_v -= 90
        vertical_img_angle = _raw_v
    except Exception:
        vertical_img_angle = None
    VERTICAL_EXCLUDE_DEG = 15.0

    def _collect_weighted(use_depth, use_vertical):
        out = []
        for line in lines:
            x_a, y_a, x_b, y_b = line[0]
            if _is_border_artifact(x_a, y_a, x_b, y_b):
                continue
            if use_depth and depth_roi is not None and center_depth is not None:
                mx, my = (x_a + x_b) // 2, (y_a + y_b) // 2
                my = min(max(my, 0), depth_roi.shape[0] - 1)
                mx = min(max(mx, 0), depth_roi.shape[1] - 1)
                mid_depth = depth_roi[my, mx]
                if mid_depth <= 0.05 or abs(mid_depth - center_depth) > 0.03:
                    continue
            length = math.hypot(x_b - x_a, y_b - y_a)
            angle = math.degrees(math.atan2(y_b - y_a, x_b - x_a)) % 180
            if angle > 90:
                angle -= 90
            if use_vertical and vertical_img_angle is not None:
                _diff = abs(angle - vertical_img_angle) % 90
                _diff = min(_diff, 90 - _diff)
                if _diff <= VERTICAL_EXCLUDE_DEG:
                    continue
            out.append((angle, length))
        return out

    # [2026-07 수정] strict 필터(테두리+깊이+수직배제)로 아무것도 안 남으면
    # (실측: 진짜 박스 각도가 vertical_img_angle 배제범위와 겹칠 때
    # 63~90도 근처에서 weighted가 통째로 비어 None 반환) 단계적으로
    # 필터를 완화해 재시도한다.
    weighted = _collect_weighted(use_depth=True, use_vertical=True)
    if not weighted:
        weighted = _collect_weighted(use_depth=True, use_vertical=False)
    if not weighted:
        weighted = _collect_weighted(use_depth=False, use_vertical=False)

    if not weighted:
        return None, None

    # 1도 단위 히스토그램(0~89), 선분 길이로 가중 → 길고 뚜렷한 모서리일수록
    # 더 큰 표를 받게 함 (짧은 노이즈성 edge는 자연히 묻힘)
    def _circ_dist90(a, b):
        diff = abs(a - b) % 90
        return min(diff, 90 - diff)

    def _circular_estimate(pairs, offset):
        shifted = [((a + offset) % 90, l) for a, l in pairs]
        bins = np.zeros(90)
        for angle, length in shifted:
            bins[int(angle) % 90] += length
        kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
        padded = np.concatenate([bins[-2:], bins, bins[:2]])
        bins_smooth = np.convolve(padded, kernel, mode='same')[2:-2]
        dominant_bin = int(np.argmax(bins_smooth))
        sel = [(a, l) for a, l in shifted if _circ_dist90(a, dominant_bin) <= 5]
        if not sel:
            sel = shifted
        total_w = sum(l for _, l in sel)
        sin_sum = sum(l * math.sin(math.radians(a * 4)) for a, l in sel)
        cos_sum = sum(l * math.cos(math.radians(a * 4)) for a, l in sel)
        R = math.hypot(sin_sum, cos_sum) / total_w if total_w > 0 else 0.0
        angle_deg = (math.degrees(math.atan2(sin_sum, cos_sum)) / 4.0 - offset) % 90
        return angle_deg, R

    angle_a, R_a = _circular_estimate(weighted, 0.0)
    angle_b, R_b = _circular_estimate(weighted, 45.0)
    cam_angle_deg = angle_a if R_a >= R_b else angle_b

    # [2026-07 추가] 박스가 회전(yaw!=0,90)해 있으면 축정렬 bbox의
    # 중심이 실제 박스 중심에서 어긋나는 문제가 실측으로 확인됨
    # (calib_debug: rel_bearing이 물체마다 랜덤한 방향/크기로 흔들림).
    # 이미 필터를 통과한(=진짜 박스 모서리로 판단된) 선분들의 끝점을
    # 모아 minAreaRect로 회전된 사각형의 중심을 구해, 축정렬 bbox
    # 중심보다 정확한 픽셀 좌표를 함께 반환한다.
    refined_center_px = None
    try:
        pts = []
        for line in lines:
            xa, ya, xb, yb = line[0]
            if not _is_border_artifact(xa, ya, xb, yb):
                pts.append((xa, ya))
                pts.append((xb, yb))
        if len(pts) >= 4:
            pts_arr = np.array(pts, dtype=np.float32)
            rect = cv2.minAreaRect(pts_arr)
            (rcx, rcy), (rw, rh), _rang = rect
            # ROI 로컬 좌표 -> 원본 이미지 절대 픽셀 좌표
            # [2026-07 수정] 선분이 한쪽 모서리에만 몰려있으면(예: 위쪽 변
            # 하나만 검출되고 나머지 세 변은 놓침) minAreaRect가 그 선분
            # 자체를 감싸는 얇은 사각형을 만들어, 중심이 박스 중앙이 아니라
            # 한쪽 끝으로 쏠리는 문제가 roi_debug.jpg로 실측 확인됨. 폭/높이
            # 둘 다 최소 크기 이상이어야(=실제로 사각형 형태를 이룸) 신뢰.
            MIN_RECT_SIDE_PX = 10
            if min(rw, rh) >= MIN_RECT_SIDE_PX:
                refined_center_px = (int(rcx) + x1, int(rcy) + y1)
    except Exception:
        refined_center_px = None

    if os.environ.get("DEBUG_SAVE_ROI", "0") == "1":
        try:
            dbg = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if roi.ndim == 2 else roi.copy()
            for line in lines:
                xa, ya, xb, yb = line[0]
                cv2.line(dbg, (xa, ya), (xb, yb), (0, 0, 255), 1)
            cv2.imwrite('/tmp/roi_debug.jpg', dbg)
            cv2.imwrite('/tmp/edges_debug.jpg', edges)
        except Exception:
            pass

    # 카메라 이미지 각도 → base_link yaw 변환 (기존과 동일)
    try:
        v = Vector3Stamped()
        v.header.frame_id = cam_frame
        v.header.stamp = _stamp
        v.vector.x = math.cos(math.radians(cam_angle_deg))
        v.vector.y = math.sin(math.radians(cam_angle_deg))
        v.vector.z = 0.0
        v_base = _transform_with_fallback(tf_buffer, v, base_frame, timeout_sec, logger=logger)
        base_angle_deg = math.degrees(math.atan2(v_base.vector.y, v_base.vector.x))
        base_angle_deg = base_angle_deg % 90
        return round(base_angle_deg, 1), refined_center_px
    except Exception:
        return None, refined_center_px


# ═══════════════════════════════════════════════════════════════════════
# [2026-08 추가, 프로토타입 -- 실기 미검증] 평면적합 기반 face normal yaw
# ═══════════════════════════════════════════════════════════════════════
# 배경: _compute_box_angle_base는 2D 이미지의 Hough edge 각도만 본다 --
# 카메라가 물체 "윗면"을 top-down에 가깝게 볼 때는 이게 실제 회전과 잘
# 맞지만, 원근왜곡에 취약하다(perception_calibration.md의 "사선각도
# Y축 25~35mm 계통오차"와 동일 계열 문제). 이 함수들은 depth를 직접 써서
# 물체 "표면이 실제로 향하는 3D 방향"(평면 normal)을 구한다 -- 단,
# 이 신호는 카메라가 그 면을 옆에서(수직에 가깝게) 보고 있을 때만
# 의미가 있다. 물체 윗면(수평면)을 보고 있으면 normal이 거의 수직이라
# 방위각 성분이 노이즈에 불과하므로, verticality 게이트로 이 경우를
# 걸러내고 None을 반환한다(호출부는 기존 angle_base_deg로 폴백해야 함).
#
# _compute_box_angle_base의 mod-90(변 대칭) 각도와 달리, 이 함수는
# mod-180(평면의 앞/뒤 부호만 모호, "어느 변인지"의 모호함은 없음)
# 각도를 반환한다 -- grasp_kinematics.side_face_candidates_from_normal_deg
# 가 이 mod-180 성질을 전제로 2후보(mod-90의 절반)만 생성한다.


def _fit_plane_normal(points):
    """3D 점들(N,3 array-like)에 평면을 맞춰 (normal_unit_vector, planarity)
    반환. ROS 의존성 없는 순수 함수 -- 단위테스트 가능.

    planarity(0~1, 1=완벽한 평면)는 surface variation(최소 고유값 / 전체
    고유값 합) 기반 -- 점군 표면이 평평할수록 최소 고유값(평면에 수직인
    방향의 분산)이 다른 두 고유값(평면 내 분산)보다 훨씬 작아진다.
    점이 3개 미만이거나 SVD가 실패하면 (None, 0.0).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return None, 0.0
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, 0.0
    eigvals = (s ** 2) / max(1, len(pts))  # 내림차순 (SVD 관례)
    l_min = eigvals[-1]
    total = float(eigvals.sum())
    if total <= 1e-12:
        return None, 0.0
    surface_variation = l_min / total
    planarity = max(0.0, 1.0 - 3.0 * surface_variation)
    normal = vt[-1]  # 최소 고유값에 대응하는 축 = 평면 normal
    # 부호를 카메라(원점) 쪽으로 통일 -- 완전한 "바깥쪽" 해석은 아니고
    # (그런 의미는 애초에 단일 시점 depth만으로는 확정 불가), 최소한
    # 재현 가능하게 한쪽으로 고정해서 mod-180 후보 생성이 일관되게 한다.
    if np.dot(normal, centroid) > 0:
        normal = -normal
    return normal, planarity


def _compute_face_normal_yaw(d: dict, depth: Optional[np.ndarray],
                              tf_buffer, cam_frame: str, base_frame: str,
                              timeout_sec: float = 0.2, stamp=None, logger=None):
    """bbox ROI의 depth를 점군으로 역투영해 평면을 맞추고, 그 평면 normal의
    수평(azimuth) 성분을 base_link 절대각(도, mod 180)으로 반환한다.

    Returns:
        (yaw_deg, confidence, n_points) 튜플.
        yaw_deg: base_link 기준 mod-180 각도. 신뢰 불가(점 부족/평면 아님/
            수평면이라 방위각 무의미)면 None.
        confidence: planarity * verticality (0~1). yaw_deg가 None이어도
            디버깅/로그용으로 반환(왜 걸러졌는지 참고).
        n_points: 실제 역투영에 쓰인 유효 점 개수.
    """
    if depth is None:
        return None, 0.0, 0
    _stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()

    H, W = depth.shape[:2]
    x1 = max(0, int(d["x_min"] * W)); y1 = max(0, int(d["y_min"] * H))
    x2 = min(W, int(d["x_max"] * W)); y2 = min(H, int(d["y_max"] * H))
    if x2 - x1 < FACE_NORMAL_GRID_STEP or y2 - y1 < FACE_NORMAL_GRID_STEP:
        return None, 0.0, 0

    points_cam = []
    for py in range(y1, y2, FACE_NORMAL_GRID_STEP):
        for px in range(x1, x2, FACE_NORMAL_GRID_STEP):
            depth_m = float(depth[py, px])
            pt = pixel_to_camera_xyz(float(px), float(py), depth_m)
            if pt is not None:
                points_cam.append((pt['x'], pt['y'], pt['z']))
            if len(points_cam) >= FACE_NORMAL_MAX_POINTS:
                break
        if len(points_cam) >= FACE_NORMAL_MAX_POINTS:
            break

    if len(points_cam) < FACE_NORMAL_MIN_POINTS:
        return None, 0.0, len(points_cam)

    normal_cam, planarity = _fit_plane_normal(points_cam)
    if normal_cam is None or planarity < FACE_NORMAL_PLANARITY_MIN:
        return None, round(planarity, 3), len(points_cam)

    # 카메라 좌표계 normal → base_link로 "회전만" 변환 (평행이동 없음).
    # _compute_box_angle_base가 이미 쓰는 Vector3Stamped 패턴과 동일.
    try:
        v = Vector3Stamped()
        v.header.frame_id = cam_frame
        v.header.stamp = _stamp
        v.vector.x, v.vector.y, v.vector.z = [float(c) for c in normal_cam]
        v_base = _transform_with_fallback(tf_buffer, v, base_frame, timeout_sec, logger=logger)
        nx, ny, nz = v_base.vector.x, v_base.vector.y, v_base.vector.z
    except Exception:
        return None, round(planarity, 3), len(points_cam)

    verticality = math.hypot(nx, ny)  # 단위벡터이므로 수평성분 크기 = sin(면 기울기각)
    confidence = round(planarity * verticality, 3)
    if verticality < FACE_NORMAL_VERTICALITY_MIN:
        return None, confidence, len(points_cam)

    yaw_deg = math.degrees(math.atan2(ny, nx)) % 180.0
    return round(yaw_deg, 1), confidence, len(points_cam)


def filter_detections(dets):
    """크기 필터만 적용한다 (중복 제거는 depth 조회 이후 _dedup_3d로 미룸 --
    2D 단계에서는 쌓인 박스와 진짜 중복을 구분할 depth 정보가 없기 때문).
    """
    return [
        d for d in dets
        if (d["x_max"] - d["x_min"]) >= MIN_BBOX_SIZE
        and (d["y_max"] - d["y_min"]) >= MIN_BBOX_SIZE
    ]


def _dedup_3d(objs, xy_thresh=DEDUP_XY_THRESH_M, z_thresh=DEDUP_Z_THRESH_M):
    """3D 위치(x, y, z) 기준으로 진짜 중복 검출만 제거한다.

    2D bbox 겹침만으로 판단하면, 쌓여있는 박스(화면상 x,y는 거의 겹치지만
    depth/z는 서로 다른 별개 물체)까지 하나로 합쳐버리는 문제가 있어서,
    반드시 z(depth)까지 같이 비교해야 한다.
    - x,y,z가 전부 임계값 안으로 가까우면 → 같은 박스의 중복 검출로 보고 병합
      (이 경우 confidence가 더 높은 detection을 남긴다)
    - x,y는 가깝지만 z가 임계값보다 멀면 → 쌓여있는 별개 박스로 보고 둘 다 유지
    """
    objs_sorted = sorted(objs, key=lambda o: -o.get("confidence", 0.0))
    filtered = []
    for o in objs_sorted:
        p = o["center_3d"]
        is_dup = False
        for f in filtered:
            fp = f["center_3d"]
            if (abs(p["x"] - fp["x"]) < xy_thresh
                    and abs(p["y"] - fp["y"]) < xy_thresh
                    and abs(p["z"] - fp["z"]) < z_thresh
                    and f["label"] == o["label"]):
                is_dup = True
                break
        if not is_dup:
            filtered.append(o)
    return filtered



class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        if rs is None:
            self.get_logger().error('pyrealsense2 미설치')
            raise RuntimeError("pyrealsense2 not installed")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, "/detected_objects", qos)
        self.pub_image = self.create_publisher(Image, "/camera/color/image_raw", qos)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", qos)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", qos)

        # ── RealSense 초기화 (이 안에서 intrinsics도 등록됨) ──
        self._init_camera()

        # ── tf2: eye-in-hand라 매 순간 camera->base_link 변환이 바뀜 ──
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.frame_lock = threading.Lock()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_stamp = None
        self._last_dispatched_stamp = None
        # [2026-08 추가] 박스서버 요청이 밀릴 때(GPU 경합 등) 이전 요청이
        # 안 끝났는데 새 요청을 또 스레드로 띄우던 게 /detected_objects
        # 0.5~0.6Hz까지 떨어뜨린 원인으로 실측 확인됨(perception_node_sim.py와
        # 동일 원인/동일 패치). 이전 요청 처리 중이면 dispatch를 건너뛴다.
        self._inflight = False
        self._inflight_lock = threading.Lock()


        if not self._wait_for_box_server():
            self.get_logger().error(
                f'박스 서버 응답 없음 ({BOX_SERVER_URL}). vlm_boxyolo.py 켜져있는지 확인.')
            raise RuntimeError("box detection server not available")

        threading.Thread(target=self._capture_loop, daemon=True).start()
        self.timer = self.create_timer(1.0 / DISPATCH_RATE_HZ, self._dispatch_inference)

        self.get_logger().info(
            f'PerceptionNode 시작 | target={TARGET_LABELS} | '
            f'server={BOX_SERVER_URL} | exposure={CAM_EXPOSURE} | '
            f'tf: {CAMERA_OPTICAL_FRAME} -> {BASE_FRAME}')

    def _init_camera(self):
        self.rs_pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
        cfg.enable_stream(rs.stream.depth, CAM_W, CAM_H, rs.format.z16, CAM_FPS)
        profile = self.rs_pipe.start(cfg)

        self.rs_align = rs.align(rs.stream.color)

        # depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 노출값 설정 (0이면 자동노출 유지)
        if CAM_EXPOSURE > 0:
            try:
                color_sensor = profile.get_device().query_sensors()[1]
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, CAM_EXPOSURE)
                self.get_logger().info(f'RealSense 노출값 수동 설정: {CAM_EXPOSURE}')
            except Exception as e:
                self.get_logger().warn(f'노출값 설정 실패 (자동노출 유지): {e}')
        else:
            self.get_logger().info('RealSense 자동노출 사용')

        self.get_logger().info(
            f'RealSense: {CAM_W}x{CAM_H}@{CAM_FPS}fps | depth_scale={self.depth_scale}')

        # ── intrinsics 등록: 반드시 color 스트림 기준 (depth/IR 아님) ──
        # align(rs.stream.color) 를 썼으므로 depth도 color 픽셀 좌표계에 맞춰져
        # 있음. 따라서 역투영도 color intrinsics로 해야 함 — 다른 렌즈(좌측 IR)
        # intrinsics를 쓰면 렌즈 baseline만큼(15~25mm) 어긋난다.
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        set_intrinsics(CameraIntrinsics.from_realsense_profile(color_profile))
        intr = color_profile.get_intrinsics()
        self._cam_info = CameraInfo()
        self._cam_info.width = intr.width
        self._cam_info.height = intr.height
        self._cam_info.k = [intr.fx, 0.0, intr.ppx, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 1.0]
        self._cam_info.distortion_model = "plumb_bob"

    def _capture_loop(self):
        while rclpy.ok():
            try:
                frames = self.rs_pipe.wait_for_frames(timeout_ms=1000)
                aligned = self.rs_align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue
                color_np = np.asanyarray(color.get_data())
                depth_np = np.asanyarray(depth.get_data()).astype(np.float32)
                depth_np *= self.depth_scale
                with self.frame_lock:
                    img_msg = Image()
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.header.frame_id = "camera_color_optical_frame"
                    img_msg.height = color_np.shape[0]
                    img_msg.width = color_np.shape[1]
                    img_msg.encoding = "bgr8"
                    img_msg.step = color_np.shape[1] * 3
                    img_msg.data = color_np.tobytes()
                    self.pub_image.publish(img_msg)
                    self._cam_info.header = img_msg.header
                    self.pub_info.publish(self._cam_info)

                    depth_msg = Image()
                    depth_msg.header = img_msg.header
                    depth_msg.height = depth_np.shape[0]
                    depth_msg.width  = depth_np.shape[1]
                    depth_msg.encoding = "32FC1"
                    depth_msg.step = depth_np.shape[1] * 4
                    depth_msg.data = depth_np.astype(np.float32).tobytes()
                    self.pub_depth.publish(depth_msg)

                    self.latest_color = color_np
                    self.latest_depth = depth_np
                    self.latest_stamp = img_msg.header.stamp   # 방금 publish한 stamp 재사용
            except Exception as e:
                self.get_logger().warn(f'프레임 수신 실패: {e}')
                time.sleep(0.05)

    def _wait_for_box_server(self, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(BOX_HEALTH_URL, timeout=1.0)
                if r.status_code == 200:
                    self.get_logger().info(f'박스 서버 ready: {r.json()}')
                    return True
            except Exception:
                pass
            self.get_logger().info('박스 서버 대기 중...')
            time.sleep(2.0)
        return False

    def _dispatch_inference(self):
        with self.frame_lock:
            if self.latest_color is None or self.latest_depth is None:
                return
            color = self.latest_color.copy()
            depth = self.latest_depth.copy()
            stamp = self.latest_stamp

        # 카메라가 dispatch(DISPATCH_RATE_HZ)보다 느리면 같은 프레임이
        # latest_color/depth에 그대로 남아있음 -> 중복 처리/전송 방지.
        if stamp is not None and self._last_dispatched_stamp is not None \
                and stamp.sec == self._last_dispatched_stamp.sec \
                and stamp.nanosec == self._last_dispatched_stamp.nanosec:
            return
        with self._inflight_lock:
            if self._inflight:
                # 이전 박스서버 요청이 아직 처리 중 -> 이번 프레임은 건너뛴다.
                return
            self._inflight = True

        self._last_dispatched_stamp = stamp

        threading.Thread(
            target=self._send_and_publish, args=(color, depth, stamp), daemon=True
        ).start()

    def _send_and_publish(self, color: np.ndarray, depth: np.ndarray, stamp=None):
        _stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()
        try:
            self._send_and_publish_impl(color, depth, _stamp)
        finally:
            with self._inflight_lock:
                self._inflight = False

    def _send_and_publish_impl(self, color: np.ndarray, depth: np.ndarray, _stamp):
        try:
            # 원본 크기(640x480)로 전송 — 다운스케일 시 탐지 누락 방지
            if GRIPPER_MASK_BOTTOM > 0.0:
                h = color.shape[0]
                cutoff = int(h * (1.0 - GRIPPER_MASK_BOTTOM))
                color = color.copy()
                color[cutoff:, :] = 0
            ok, buf = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return
            img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            payload = {"image_b64": img_b64, "labels": TARGET_LABELS}
            r = requests.post(BOX_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return
            data = r.json()
        except requests.exceptions.RequestException:
            return
        except Exception as e:
            self.get_logger().warn(f'요청 처리 오류: {e}')
            return

        raw_dets = data.get("detections", [])
        filtered = filter_detections(raw_dets)

        ch, cw = color.shape[:2]
        objs = []
        for d in filtered:
            cx_norm = (d["x_min"] + d["x_max"]) / 2
            cy_norm = (d["y_min"] + d["y_max"]) / 2
            cx_px = int(cx_norm * cw)
            cy_px = int(cy_norm * ch)

            # [2026-07 수정] 회전된 박스 중심 보정 (축정렬 bbox 중심 대신
            # minAreaRect 중심 사용). depth 샘플링보다 먼저 실행.
            angle_deg, refined_center_px = _compute_box_angle_base(
                color, d, self.tf_buffer,
                CAMERA_OPTICAL_FRAME, BASE_FRAME, TF_TIMEOUT_SEC, depth=depth,
                stamp=_stamp, logger=self.get_logger())
            # [2026-08 추가, 프로토타입 -- 실기 미검증] depth 평면적합 기반
            # face normal yaw. angle_base_deg(2D Hough, mod-90)와 별개 필드로
            # 추가만 하고 기존 필드/소비 경로는 그대로 둔다 -- planning_node.py는
            # 아직 이 필드를 읽지 않으므로 기존 pick 동작에 영향 없음.
            face_yaw_deg, face_conf, _face_n = _compute_face_normal_yaw(
                d, depth, self.tf_buffer,
                CAMERA_OPTICAL_FRAME, BASE_FRAME, TF_TIMEOUT_SEC,
                stamp=_stamp, logger=self.get_logger())
            if refined_center_px is not None:
                rx, ry = refined_center_px
                if 0 <= rx < cw and 0 <= ry < ch:
                    cx_px, cy_px = rx, ry
                    cx_norm, cy_norm = rx / cw, ry / ch

            # [2026-07 수정] 단일 중심픽셀+depth 1개 방식은 카메라가 비스듬한
            # 각도로 볼 때 원근왜곡으로 2D 중심이 실제 3D 중심과 어긋나는(Y축
            # -25~-35mm 계통오차 실측 확인) 문제가 있어, bbox 영역 안 여러
            # 점의 depth를 읽어 3D로 역투영한 뒤 평균(centroid)을 쓴다.
            _bx1, _by1 = int(d['x_min'] * cw), int(d['y_min'] * ch)
            _bx2, _by2 = int(d['x_max'] * cw), int(d['y_max'] * ch)
            depth_m, pt_cam = self._multi_point_3d_centroid(depth, _bx1, _by1, _bx2, _by2)
            if pt_cam is None:
                continue  # depth 무효하거나 intrinsics 미설정 → 이 물체는 스킵

            # ── 카메라 좌표계 점 → base_link (eye-in-hand이므로 매번 tf 조회) ──
            pt_stamped = PointStamped()
            pt_stamped.header.frame_id = CAMERA_OPTICAL_FRAME
            pt_stamped.header.stamp = _stamp
            pt_stamped.point.x = pt_cam["x"]
            pt_stamped.point.y = pt_cam["y"]
            pt_stamped.point.z = pt_cam["z"]

            try:
                pt_base = _transform_with_fallback(
                    self.tf_buffer, pt_stamped, BASE_FRAME, TF_TIMEOUT_SEC,
                    logger=self.get_logger())
                xyz = {
                    "x": round(pt_base.point.x, 3),
                    "y": round(pt_base.point.y, 3),
                    "z": round(pt_base.point.z, 3),
                }
            except Exception as e:
                self.get_logger().warn(
                    f'tf 변환 실패 ({CAMERA_OPTICAL_FRAME} -> {BASE_FRAME}), 폴백도 실패: {e}',
                    throttle_duration_sec=5.0)
                continue


            objs.append({
                "label": d["label"],
                "bbox": [d["x_min"], d["y_min"], d["x_max"], d["y_max"]],
                "center_2d": {"x": cx_norm, "y": cy_norm},
                "center_3d": xyz,
                "depth_m": round(float(depth_m), 3) if depth_m else None,
                "confidence": round(float(d.get("confidence", 0.0)), 3),
                "angle_base_deg": angle_deg,
                # [2026-08 추가, 프로토타입] mod-180 각도(각도가 없으면 None).
                # face_normal_confidence는 yaw가 None이어도 참고용으로 채워짐
                # (예: 0.3이면 "평면성/수직성이 낮아서 걸렀다"는 뜻).
                "face_normal_yaw_deg": face_yaw_deg,
                "face_normal_confidence": face_conf,
            })

        # ── 3D 위치 기준 최종 dedup ──────────────────────────────────
        # x,y,z 전부 가까운 것만 같은 박스의 중복 검출로 보고 병합한다.
        # 쌓여있는 박스처럼 z(depth)가 다르면 그대로 유지된다.
        objs = _dedup_3d(objs)

        # ── YOLO 결과 발행 ───────────────────────────────────────────
        msg = String()
        msg.data = json.dumps({"objects": objs}, ensure_ascii=False)
        self.pub.publish(msg)



    @staticmethod
    @staticmethod
    def _multi_point_3d_centroid(depth, x1, y1, x2, y2):
        """[2026-07 추가] 단일 중심픽셀+depth 1개 샘플 방식은, 카메라가
        비스듬한 각도로 내려다볼 때 원근왜곡 때문에 2D 중심이 실제 3D
        박스 중심과 어긋나는(계통 오차 Y축 -25~-35mm 실측 확인) 문제가
        있었음. bbox 영역 안을 격자로 나눠 각 점의 depth를 읽고 각각을
        카메라 좌표계 3D 점으로 역투영한 뒤, 그 점들의 실제 3D 평균
        (centroid)을 구한다 -- 원근왜곡에 안 흔들리는 진짜 3D 중심.
        depth_m(대표값, median)과 카메라좌표 3D dict를 반환. 실패 시
        (None, None).
        """
        GRID = 5  # 5x5 = 25개 샘플 지점
        h, w = depth.shape
        x1c, x2c = max(0, x1), min(w, x2)
        y1c, y2c = max(0, y1), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return None, None

        pts_cam = []
        depths = []
        for gi in range(GRID):
            for gj in range(GRID):
                # 가장자리(원근왜곡/노이즈가 심한 경계)를 피해 안쪽 80% 영역만 샘플링
                fx_ratio = 0.1 + 0.8 * gi / (GRID - 1)
                fy_ratio = 0.1 + 0.8 * gj / (GRID - 1)
                px = int(x1c + fx_ratio * (x2c - x1c))
                py = int(y1c + fy_ratio * (y2c - y1c))
                px = min(max(px, 0), w - 1)
                py = min(max(py, 0), h - 1)
                d = depth[py, px]
                if not (0.05 < d < 2.0):
                    continue
                pt = pixel_to_camera_xyz(px, py, float(d))
                if pt is None:
                    continue
                pts_cam.append(pt)
                depths.append(float(d))

        if not pts_cam:
            return None, None

        mean_x = sum(p["x"] for p in pts_cam) / len(pts_cam)
        mean_y = sum(p["y"] for p in pts_cam) / len(pts_cam)
        mean_z = sum(p["z"] for p in pts_cam) / len(pts_cam)
        depth_repr = float(np.median(depths))
        return depth_repr, {"x": mean_x, "y": mean_y, "z": mean_z}

    def _sample_depth(depth: np.ndarray, cx: int, cy: int) -> Optional[float]:
        h, w = depth.shape
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & (patch < 2.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def destroy_node(self):
        try:
            self.rs_pipe.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
