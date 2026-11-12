#!/usr/bin/env python3
"""
camera_calibration.py
Pinhole intrinsics 기반 픽셀→카메라 좌표 역투영 + (선택) 저장된 extrinsic 적용.

[아키텍처 변경 이유]
기존 버전은 "카메라가 테이블 위에 고정되어 아래를 내려다본다"는
단안 평면 가정(호모그래피 + TABLE_TO_CAMERA_DIST)을 사용했다.

현재 시스템은 eye-in-hand (카메라가 로봇 팔 tcp_link 부근에 부착되어
팔과 함께 움직임) 구조이기 때문에 이 가정이 더 이상 성립하지 않는다:
  - 팔이 움직이면 카메라 각도/거리가 계속 바뀜 → 고정 호모그래피 무효
  - "테이블-카메라 거리 - depth" 식의 z 계산도 카메라가 기울어지면 틀림
  - 카메라가 정확히 렌즈 중앙이 아니라 다른 지점을 보는 것처럼 어긋나는
    증상도, 애초에 실제 카메라 기하(intrinsics/extrinsics)를 전혀 반영
    안 하니 생기는 문제였을 가능성이 크다.

대신 표준 핀홀 카메라 모델로 픽셀+depth를 카메라 광학 좌표계
(REP-103: Z forward, X right, Y down)의 3D 점으로 역투영한다.
base_link 로의 변환은:
  (1) ROS 환경(권장): perception_node.py에서 tf2로 처리.
      robot_state_publisher가 tcp_link→camera 고정 변환(URDF) +
      실시간 관절각을 조합해 자동으로 계산해준다. 카메라가 움직여도
      매 순간 정확한 base_link 좌표가 나온다.
  (2) 오프라인/비-ROS 검증용: 이 파일에 저장된 고정 4x4 extrinsic으로
      camera_point_to_flange() 사용 (근사치, 팔이 매 순간 다른 자세라는
      점은 반영 못 함 — 정식 파이프라인에서는 쓰지 말 것).
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


CALIB_DIR = os.path.expanduser(os.environ.get("NERO_CALIB_DIR", "~/.nero_calib"))
EXTRINSIC_PATH = os.path.join(CALIB_DIR, "flange_to_camera.json")


# ──────────────────────────────────────────────
# Intrinsics
# ──────────────────────────────────────────────
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0

    @classmethod
    def from_realsense_profile(cls, color_stream_profile) -> "CameraIntrinsics":
        """pyrealsense2 VideoStreamProfile → CameraIntrinsics.

        반드시 rs.align(rs.stream.color) 이후, **color** 스트림 profile에서
        뽑을 것. depth(좌측 IR) intrinsics를 쓰면 렌즈 baseline만큼
        (보통 15~25mm) 투영 위치가 어긋난다 — "다른 렌즈로 보는 것 같은"
        증상의 가장 흔한 원인.
        """
        intr = color_stream_profile.get_intrinsics()
        return cls(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                    width=intr.width, height=intr.height)


_intrinsics: Optional[CameraIntrinsics] = None


def set_intrinsics(intr: CameraIntrinsics) -> None:
    """perception_node 초기화 시 (카메라 스트림 시작 직후) 1회 호출."""
    global _intrinsics
    _intrinsics = intr
    print(f"[calib] intrinsics 설정: fx={intr.fx:.2f} fy={intr.fy:.2f} "
          f"cx={intr.cx:.2f} cy={intr.cy:.2f}")


def is_intrinsics_ready() -> bool:
    return _intrinsics is not None


# ──────────────────────────────────────────────
# 픽셀 → 카메라 광학 좌표계 3D 역투영
# ──────────────────────────────────────────────
def pixel_to_camera_xyz(px: float, py: float,
                         depth_m: Optional[float]) -> Optional[dict]:
    """
    픽셀 좌표(px, py: 정규화 아님, 실제 픽셀) + depth(미터)
    → 카메라 광학 좌표계 3D 점 {"x","y","z"}.

    intrinsics 가 설정 안 됐거나 depth 가 유효하지 않으면 None 반환.
    (예전처럼 임의 더미값을 반환하지 않는다 — 잘못된 좌표를 로봇에
    넘기는 것보다 "모름"을 명시적으로 알리는 게 안전하다.)
    """
    if _intrinsics is None:
        print("[calib] ⚠️ intrinsics 미설정 — set_intrinsics() 먼저 호출")
        return None
    if depth_m is None or not (0.05 < depth_m < 3.0):
        return None

    x_cam = (px - _intrinsics.cx) * depth_m / _intrinsics.fx
    y_cam = (py - _intrinsics.cy) * depth_m / _intrinsics.fy
    z_cam = depth_m

    return {"x": round(float(x_cam), 4),
            "y": round(float(y_cam), 4),
            "z": round(float(z_cam), 4)}


# ──────────────────────────────────────────────
# (선택) 저장된 고정 extrinsic 적용 — tf2 없는 환경/오프라인 검증용
# ──────────────────────────────────────────────
_T_ref_cam: Optional[np.ndarray] = None  # 4x4, 기준 링크(tcp_link) → camera


def _load_extrinsic() -> bool:
    global _T_ref_cam
    if not os.path.exists(EXTRINSIC_PATH):
        return False
    try:
        with open(EXTRINSIC_PATH, "r") as f:
            data = json.load(f)
        _T_ref_cam = np.array(data["T_flange_camera"], dtype=np.float64)
        if _T_ref_cam.shape != (4, 4):
            print(f"[calib] ⚠️ extrinsic shape 이상: {_T_ref_cam.shape}")
            _T_ref_cam = None
            return False
        print(f"[calib] ✅ extrinsic(reference→camera) 로드: {EXTRINSIC_PATH}")
        return True
    except Exception as e:
        print(f"[calib] ⚠️ extrinsic 로드 실패: {e}")
        _T_ref_cam = None
        return False


_load_extrinsic()


def camera_point_to_reference(pt_cam: dict) -> Optional[dict]:
    """카메라 광학 좌표계 점 → 기준 링크(hand_eye_calibration.py 기준 tcp_link)
    좌표계 점.

    실제 로봇 파이프라인에서는 이 함수 대신 tf2 (perception_node)로
    base_link까지 한 번에 변환하는 걸 권장한다. 이 함수는 팔이 캘리브레이션
    당시와 "같은 자세"라는 가정이 없으면 의미가 없으므로, 오프라인 검증
    스크립트(예: 캘리브레이션 결과 재투영 오차 확인)에서만 사용할 것.
    """
    if _T_ref_cam is None:
        return None
    p = np.array([pt_cam["x"], pt_cam["y"], pt_cam["z"], 1.0])
    p_ref = _T_ref_cam @ p
    return {"x": float(p_ref[0]), "y": float(p_ref[1]), "z": float(p_ref[2])}


def reload_extrinsic():
    return _load_extrinsic()


def is_extrinsic_ready() -> bool:
    return _T_ref_cam is not None


# ──────────────────────────────────────────────
# 하위호환 스텁 — 옛 호출부가 남아있으면 조용히 틀린 값을 주는 대신
# 명확한 에러로 마이그레이션을 강제한다.
# ──────────────────────────────────────────────
def pixel_to_robot_xyz(*args, **kwargs):
    raise RuntimeError(
        "pixel_to_robot_xyz()는 제거되었습니다. eye-in-hand 구조로 전환하면서 "
        "호모그래피 기반 평면 가정을 더 이상 쓰지 않습니다. "
        "perception_node.py에서 pixel_to_camera_xyz() + tf2("
        "camera_color_optical_frame → base_link) 조합으로 교체하세요."
    )
