#!/usr/bin/env python3
"""
geometry_3d.py

[신설, 2026-08] Point cloud → RANSAC(평면+normal) + PCA/OBB(주축) →
GeometryResult. ROS 비의존 순수 함수 모듈(grasp_kinematics.py와 동일한
모듈 경계 원칙 — numpy만 있으면 pytest로 단위테스트 가능, 실제로 synthetic
point cloud로 검증 완료).

이 모듈이 대체하는 것: 기존 SIDE grasp의 유일한 orientation source였던
`yaw = atan2(y, x)`(물체 위치 방향일 뿐, 물체 자체의 회전과 무관)와,
perception_node.py의 2D Hough 기반 angle_base_deg(원근왜곡에 취약, 이미
`_fit_plane_normal`로 단일 평면 normal까지는 확장했었음). 이 모듈은 그
연장선에서 RANSAC(여러 평면 지원) + PCA(주축/OBB)까지 정식으로 구현한다.

perception_node._fit_plane_normal과의 관계: 그 함수(SVD 최소자승 평면
적합)는 이 모듈의 `_fit_plane_svd`와 알고리즘이 동일하다 — RANSAC의 "최종
정제(refit)" 단계로 재사용한다. 중복 유지하지 않기 위해, 장기적으로는
perception_node.py 쪽이 이 모듈을 import하는 방향으로 정리하는 게
맞지만(이 리팩터는 이번 라운드 범위 밖 — perception_node.py는 이미 여러
곳에서 그 함수를 쓰고 있어 안전하게 옮기려면 별도 검증이 필요), 지금은
이 모듈 자체가 완결적으로 동작하도록 독립 구현했다.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .grasp_types import GeometryResult


# ═══════════════════════════════════════════════════════════════════════
# RANSAC 평면 적합
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PlaneResult:
    normal: Tuple[float, float, float]     # 단위벡터
    centroid: Tuple[float, float, float]
    inlier_count: int
    inlier_ratio: float                     # inlier_count / 입력 전체 점 개수
    inlier_mask: np.ndarray                 # bool 배열, RANSAC에 넘긴 points 기준


def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """3점 이상에서 최소자승 평면 적합. (centroid, unit_normal) 반환.
    perception_node._fit_plane_normal과 동일 알고리즘(SVD 최소 특이값
    방향) — RANSAC 최종 정제 단계로 재사용."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return centroid, normal


def _ransac_single_plane(points: np.ndarray, distance_threshold: float,
                          min_inliers: int, max_iterations: int,
                          rng: np.random.Generator) -> Optional[PlaneResult]:
    n = len(points)
    if n < 3:
        return None
    best_mask = None
    best_count = -1
    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue  # 3점이 거의 일직선 -- 무효 샘플, 재시도
        normal = normal / norm
        dist = np.abs((points - p0) @ normal)
        mask = dist < distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < min_inliers:
        return None

    inlier_pts = points[best_mask]
    centroid, normal = _fit_plane_svd(inlier_pts)
    # normal 부호를 관측 원점(카메라) 방향으로 통일 -- perception_node.
    # _fit_plane_normal과 동일 관례(재현 가능한 후보 생성을 위함, "완전한
    # 바깥쪽" 해석은 아님 -- 단일 시점 depth만으로는 확정 불가).
    if np.dot(normal, centroid) > 0:
        normal = -normal
    return PlaneResult(
        normal=tuple(normal), centroid=tuple(centroid),
        inlier_count=best_count, inlier_ratio=best_count / n,
        inlier_mask=best_mask,
    )


def fit_dominant_planes(points: np.ndarray, distance_threshold: float = 0.005,
                         min_inliers: int = 30, max_planes: int = 2,
                         max_iterations: int = 200,
                         rng: Optional[np.random.Generator] = None) -> List[PlaneResult]:
    """RANSAC으로 우세 평면(들)을 순차적으로 찾는다 — sequential RANSAC:
    평면을 찾을 때마다 그 inlier를 제거하고 남은 점에서 다음 평면을 다시
    찾는다. 물체 표면이 여러 면으로 이뤄진 경우(책의 표지+옆면 등)
    복수 후보 면을 얻기 위함.

    threshold/min_inliers는 호출부가 config로 관리한다(하드코딩 금지 원칙)
    — 기본값은 실측 튜닝값이 아니라 합리적 시작값이므로, 실기 검증 후
    조정 필요.

    Returns:
        list[PlaneResult], 최대 max_planes개, inlier_count 내림차순
        (sequential이라 자연히 내림차순).
    """
    rng = rng or np.random.default_rng()
    points = np.asarray(points, dtype=np.float64)
    remaining = points
    planes: List[PlaneResult] = []
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        result = _ransac_single_plane(remaining, distance_threshold, min_inliers,
                                      max_iterations, rng)
        if result is None:
            break
        # inlier_ratio는 이번 RANSAC 호출의 전체 points 대비 비율로 재계산
        # (sequential 진행에 따라 remaining이 줄어들므로, "전체 대비"가
        # "이번 라운드 remaining 대비"보다 호출부가 이해하기 쉬움).
        recomputed_ratio = result.inlier_count / len(points)
        planes.append(PlaneResult(
            normal=result.normal, centroid=result.centroid,
            inlier_count=result.inlier_count, inlier_ratio=recomputed_ratio,
            inlier_mask=result.inlier_mask,
        ))
        remaining = remaining[~result.inlier_mask]
    return planes


# ═══════════════════════════════════════════════════════════════════════
# PCA / OBB
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PCAResult:
    centroid: np.ndarray
    major_axis: np.ndarray     # 최대 분산 방향
    minor_axis: np.ndarray
    third_axis: np.ndarray
    eigenvalues: Tuple[float, float, float]   # 내림차순 (major, minor, third)
    extents: Tuple[float, float, float]        # 각 축으로 투영한 점들의 범위(min-max)


def compute_principal_axes(points: np.ndarray) -> Optional[PCAResult]:
    """PCA로 major/minor/third axis + eigenvalues + OBB extents 계산.

    [중요, 사용자 지시 9번] 반환된 axis는 항상 부호가 정해지지 않은
    "line"이다(+axis와 -axis는 같은 principal axis) — 이 함수는 sign
    ambiguity를 해소하지 않는다. candidate generation에서 양방향을 모두
    후보로 고려해야 한다(grasp_pose_generator.py가 담당).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (3, 3):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)   # eigh: 오름차순 반환(대칭행렬 전용, svd보다 안정적)
    order = np.argsort(eigvals)[::-1]         # 내림차순으로 재정렬
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major, minor, third = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]

    proj = centered @ eigvecs   # (N,3), 각 열이 해당 축 위 투영 스칼라
    extents = proj.max(axis=0) - proj.min(axis=0)

    return PCAResult(
        centroid=centroid, major_axis=major, minor_axis=minor, third_axis=third,
        eigenvalues=(float(eigvals[0]), float(eigvals[1]), float(eigvals[2])),
        extents=(float(extents[0]), float(extents[1]), float(extents[2])),
    )


# ═══════════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════════

def geometry_confidence(point_count: int, plane_inlier_ratio: float,
                         eigenvalues: Tuple[float, float, float],
                         depth_quality: float = 1.0) -> float:
    """여러 지표를 가중합해 0~1 신뢰도로 압축한다.

    weighting 근거:
      - point_count(가중치 0.25): 점이 너무 적으면(대략 30점 미만)
        통계적으로 불안정 -- 30점 근처를 변곡점으로 하는 sigmoid로 완만하게
        낮아지게 한다(경계에서 급격한 on/off보다 안전).
      - plane_inlier_ratio(0.35, 가장 큰 비중): RANSAC이 찾은 평면이 점군의
        몇 %를 설명하는지 -- 낮으면 물체가 여러 조각/곡면이라 단일 평면
        가정 자체가 안 맞는다는 강한 신호라 가중치를 가장 높게 뒀다.
      - eigenvalue separation(0.25): PCA major/minor 고유값 차이가 작으면
        (원형/정사각형에 가까운 단면) principal axis 방향이 노이즈에 취약
        (고유벡터가 축퇴돼 회전축 주위로 불안정하게 흔들림) --
        (l_max-l_mid)/l_max로 분리도를 반영.
      - depth_quality(0.15): 호출부가 넘기는 외부 신호(예: 평균 depth,
        grazing angle 여부 등) -- 이 함수는 계산하지 않고 곱셈 아닌 가중
        합산 인자로만 반영한다(호출부가 아직 이 신호를 안 줄 수도 있으므로
        기본값 1.0 = "모른다"가 아니라 "문제없다고 가정" -- 호출부가 실제
        추정치를 넘기지 않으면 이 항은 사실상 상수로 작동함에 유의).

    이 weighting은 실측 데이터로 캘리브레이션된 값이 아니라 설계 시점의
    합리적 추정이다 -- 실기 검증 후 조정 필요(perception_node.py의 다른
    threshold들과 마찬가지로 회귀테스트 없이 실측 튜닝 대상).
    """
    point_score = 1.0 / (1.0 + math.exp(-(point_count - 30) / 10.0))
    plane_score = max(0.0, min(1.0, plane_inlier_ratio))

    l = sorted(eigenvalues)  # 오름차순: l[0]<=l[1]<=l[2]
    l_max, l_mid = l[2], l[1]
    axis_stability = 0.0 if l_max <= 1e-12 else max(0.0, min(1.0, (l_max - l_mid) / l_max))

    depth_score = max(0.0, min(1.0, depth_quality))

    weights = {"point": 0.25, "plane": 0.35, "axis": 0.25, "depth": 0.15}
    score = (weights["point"] * point_score
             + weights["plane"] * plane_score
             + weights["axis"] * axis_stability
             + weights["depth"] * depth_score)
    return round(max(0.0, min(1.0, score)), 3)


# ═══════════════════════════════════════════════════════════════════════
# Top-level: point cloud → GeometryResult
# ═══════════════════════════════════════════════════════════════════════

def _axis_to_yaw_deg(vec: Optional[np.ndarray], verticality_min: float = 0.5,
                      mod: float = 180.0) -> Optional[float]:
    """3D 단위벡터를 수평면(x,y) 방위각(도)으로 투영. verticality(수평성분
    크기)가 낮으면(벡터가 수직에 가까움) None -- perception_node.
    _compute_face_normal_yaw의 verticality 게이트와 동일 원칙(수직 방향은
    방위각이 노이즈에 불과함)."""
    if vec is None:
        return None
    x, y = float(vec[0]), float(vec[1])
    verticality = math.hypot(x, y)
    if verticality < verticality_min:
        return None
    return round(math.degrees(math.atan2(y, x)) % mod, 1)


def compute_geometry(points: np.ndarray,
                      distance_threshold: float = 0.005,
                      min_inliers: int = 30,
                      depth_quality: float = 1.0) -> GeometryResult:
    """point cloud(카메라 또는 base_link 좌표계, 호출부 책임) 하나에서
    RANSAC(1개 평면) + PCA를 계산해 GeometryResult로 묶는다.

    좌표계 변환(카메라→base_link)은 이 함수 책임이 아니다 -- 축/normal은
    "방향 벡터"라 Vector3Stamped 회전 변환이 필요한데, 그건 ROS(tf2)
    의존적이라 이 순수 함수 모듈에 넣을 수 없다. 호출부(ROS 노드)가
    좌표계를 맞춘 뒤 넘기거나, 결과의 axis/normal을 받아 별도로 회전
    변환해야 한다.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < min_inliers:
        return GeometryResult(valid=False, point_count=n, filtered_point_count=n,
                               geometry_confidence=0.0, source="ransac_pca")

    pca = compute_principal_axes(points)
    if pca is None:
        return GeometryResult(valid=False, point_count=n, filtered_point_count=n,
                               geometry_confidence=0.0, source="ransac_pca")

    planes = fit_dominant_planes(points, distance_threshold=distance_threshold,
                                  min_inliers=min_inliers, max_planes=1)
    plane = planes[0] if planes else None

    conf = geometry_confidence(
        point_count=n,
        plane_inlier_ratio=(plane.inlier_ratio if plane else 0.0),
        eigenvalues=pca.eigenvalues,
        depth_quality=depth_quality,
    )

    return GeometryResult(
        valid=True,
        centroid=tuple(pca.centroid),
        major_axis=tuple(pca.major_axis),
        minor_axis=tuple(pca.minor_axis),
        third_axis=tuple(pca.third_axis),
        eigenvalues=pca.eigenvalues,
        extents=pca.extents,
        plane_normal=(plane.normal if plane else None),
        plane_inlier_count=(plane.inlier_count if plane else 0),
        plane_inlier_ratio=(plane.inlier_ratio if plane else 0.0),
        point_count=n,
        filtered_point_count=n,
        geometry_confidence=conf,
        major_axis_yaw_deg=_axis_to_yaw_deg(pca.major_axis),
        normal_yaw_deg=(_axis_to_yaw_deg(np.array(plane.normal)) if plane else None),
        source="ransac_pca",
    )
