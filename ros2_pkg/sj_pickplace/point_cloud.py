#!/usr/bin/env python3
"""
point_cloud.py

[신설, 2026-08] Segmentation mask + depth + camera intrinsics → point cloud
(카메라 좌표계) 변환.

camera_calibration.pixel_to_camera_xyz와 동일한 핀홀 역투영 공식을 쓰되,
마스크 전체 픽셀을 numpy로 벡터화해서 한 번에 처리한다.
perception_node._compute_face_normal_yaw는 격자 샘플이라 최대 400점이라
파이썬 루프로도 충분했지만, 이 함수는 마스크 전체(수만 픽셀일 수 있음)를
다루므로 벡터화가 필요하다.

깊이 단위는 NERO 전체 관례와 동일하게 **미터**다 — perception_node.py/
mcp_robot_server.py의 depth_m 필드가 전부 미터 단위이고,
camera_calibration.pixel_to_camera_xyz의 유효범위 체크(0.05 < depth_m < 3.0)가
미터 기준으로 실측 확인됐으므로 이 모듈도 그 관례를 그대로 따른다(추측
아님 — 기존 코드 값 그대로 재사용).

좌표계는 camera_color_optical_frame(REP-103: Z forward, X right, Y down,
camera_calibration.py 모듈 docstring과 동일). 픽셀 인덱싱은 NumPy 관례대로
depth_image[v, u](row=v=세로, col=u=가로) — perception_node.py 전체가 이
순서를 따른다.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


DEFAULT_DEPTH_MIN_M = 0.05
DEFAULT_DEPTH_MAX_M = 3.0


def mask_depth_to_pointcloud(depth_image: np.ndarray, mask: np.ndarray,
                              intrinsics: Intrinsics,
                              depth_min_m: float = DEFAULT_DEPTH_MIN_M,
                              depth_max_m: float = DEFAULT_DEPTH_MAX_M,
                              max_points: Optional[int] = None,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """mask=True인 픽셀의 depth를 카메라 좌표계 3D 점으로 역투영한다.

    NaN/Inf/음수/범위 밖 depth는 자동 제외한다 — RealSense/Isaac Sim 등에서
    무효 depth가 0 또는 NaN으로 오는 경우가 흔하다(perception_node.py의
    valid depth range 관례를 그대로 재사용, 새로 발명한 임계값 아님).

    Args:
        depth_image: (H, W) 미터 단위 depth 배열.
        mask: (H, W) bool 배열 (SegmentationResult.mask).
        intrinsics: fx, fy, cx, cy (camera_color_optical_frame 기준,
            RealSense color stream profile에서 뽑은 값 — camera_calibration.py
            CameraIntrinsics.from_realsense_profile 참고, depth/IR intrinsics
            섞으면 15~25mm 어긋난다는 그 파일의 경고와 동일하게 적용됨).
        max_points: 지정하면 그 개수로 무작위 다운샘플(RANSAC/PCA엔 전체
            픽셀이 필요 없고, 점이 너무 많으면 SVD/RANSAC 비용만 커짐).

    Returns:
        (N, 3) float64 array, 카메라 좌표계. 유효 점이 없으면 (0, 3) 빈
        배열(예외 아님 — 호출부가 len()으로 체크해야 함, "물체가 없다"와
        "이 프레임에서 유효 depth가 하나도 없다"를 구분해 로그로 남길 수
        있도록 조용히 실패하지 않게 설계).
    """
    if depth_image is None or mask is None:
        return np.zeros((0, 3), dtype=np.float64)
    if depth_image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"depth_image shape {depth_image.shape[:2]} != mask shape {mask.shape[:2]}")

    depth = depth_image.astype(np.float64)
    finite = np.isfinite(depth)
    valid_range = (depth > depth_min_m) & (depth < depth_max_m)
    valid = mask.astype(bool) & finite & valid_range

    vs, us = np.nonzero(valid)  # row(v), col(u)
    if len(us) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if max_points is not None and len(us) > max_points:
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(us), size=max_points, replace=False)
        us, vs = us[idx], vs[idx]

    z = depth[vs, us]
    x = (us.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (vs.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=1)
