#!/usr/bin/env python3
"""
learned_grasp_backend.py

[신설, 2026-08] Learned 6-DoF Grasp Detector backend abstraction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
중요 — 이 세션에서 실제로 한 일과 안 한 일을 명확히 구분한다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이 세션(코드 분석/작성 환경)에는 GPU/CUDA/torch가 전혀 없다(env_check.py를
직접 실행해 확인할 방법도 없음 — pip으로 numpy 하나 설치하는 데도 별도
조치가 필요했던 환경). 따라서:
  - AnyGrasp/GraspNet-baseline/Contact-GraspNet 등 실제 학습된 6-DoF grasp
    모델을 이 세션에서 설치/실행/검증하는 것은 불가능하다.
  - 이 파일은 backend interface(LearnedGraspBackend)와, 실제 모델 없이도
    파이프라인이 끝까지 동작하도록 하는 FallbackGeometryBackend만 제공한다.
  - 실제 모델(AnyGraspBackend 등)을 연결하려면 GPU가 있는 머신(4090
    랩탑)에서: (1) 모델 설치, (2) 현재 PyTorch/CUDA 버전과의 호환성 확인,
    (3) 이 인터페이스(predict())를 구현하는 새 클래스 작성이 필요하다 —
    이건 이 세션에서 완료할 수 없는 작업이라 사용자에게 그대로 남긴다.

FallbackGeometryBackend가 하는 일: geometry_3d.GeometryResult(RANSAC+PCA)를
그대로 "generic grasp candidate 1개"로 감싸서 반환한다 — 즉 "학습된
모델이 없으면 geometry만으로 후보를 만든다"는 뜻으로, 사용자가 지정한
fallback hierarchy(Level 1 Learned → Level 2 RANSAC+PCA → ...)에서 Level 1이
사실상 Level 2로 즉시 위임되는 경우다. 이건 실패가 아니라 설계대로 동작하는
것 — candidate.source 필드로 어느 backend가 실제로 후보를 냈는지 항상
구분 가능하다(실험 비교 요구사항, architecture 문서 29번).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .grasp_types import GeometryResult, GraspCandidate, GraspIntent


@dataclass
class LearnedGraspOutput:
    """backend.predict()의 원시 출력. GraspCandidate로 변환하기 전 단계 --
    학습된 모델마다 좌표계/그리퍼 폭 표현이 다를 수 있어 이 중간 타입을
    둔다(예: GraspNet은 approach+binormal+width, AnyGrasp는 다른 파라미터화).
    변환(→GraspCandidate)은 grasp_pose_generator.py가 backend별로 담당."""
    position: np.ndarray            # (3,) 카메라 또는 base_link 좌표계(backend 명시 책임)
    approach_vector: np.ndarray     # (3,) 단위벡터
    grasp_axis: Optional[np.ndarray] = None   # (3,) 그리퍼 폐쇄 축, 없으면 None
    score: float = 0.0              # backend 자체 신뢰도(0~1)
    width: Optional[float] = None   # 그리퍼 개방 폭 추정치(미터), 없으면 None


class LearnedGraspBackend(ABC):
    @abstractmethod
    def predict(self, point_cloud: np.ndarray,
                geometry_features: Optional[GeometryResult] = None,
                grasp_intent: Optional[GraspIntent] = None) -> List[LearnedGraspOutput]:
        """point_cloud(N,3)에서 6-DoF grasp 후보들을 생성한다.

        [Approach A, 이번 세션 확정] grasp_intent는 backend에 강제로 넣지
        않는다 — 대부분의 사전학습 6-DoF grasp 모델(GraspNet-baseline,
        AnyGrasp 등)이 language/task conditioning을 지원하지 않으므로,
        범용 후보를 낸 뒤 grasp_pose_generator.py의 semantic filtering
        단계에서 VLM intent와의 일치도로 점수를 매겨 랭킹한다(Approach B —
        intent를 detector 입력으로 직접 쓰는 방식 — 로 갈아탈 backend가
        생기면, 그 backend의 predict()가 grasp_intent를 실제로 활용하도록
        구현하면 되고 이 인터페이스 자체는 안 바뀐다).

        구현체가 실패(모델 로드 안 됨, 추론 에러 등)하면 빈 리스트를
        반환한다 — 예외를 던지지 않는다(호출부가 다음 fallback 레벨로
        넘어가야 하므로, 여기서 예외가 새면 그 판단을 backend가 가로채는
        셈이 됨).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """candidate.source에 기록할 backend 식별자(예: 'anygrasp', 'fallback_geometry')."""
        raise NotImplementedError


class FallbackGeometryBackend(LearnedGraspBackend):
    """실제 학습 모델이 없을 때(또는 미검증 상태일 때) geometry_3d 결과를
    generic candidate 1개로 감싸는 backend. 이 세션 기준 **유일하게 실제로
    동작하는 backend**다.

    후보 1개만 낸다는 게 "학습 모델처럼 여러 후보 중 순위를 매기는" 것과
    다르다 -- geometry_3d가 이미 dominant plane/PCA로 "가장 그럴듯한 면
    하나"를 골라놓은 상태라, 여기서 추가로 할 수 있는 랭킹이 없다.
    candidate 자체는 major_axis/plane_normal 방향을 approach_vector 후보로
    변환해서 만든다(양쪽 부호 모두 -- PCA sign ambiguity, 9번 원칙).
    """

    @property
    def name(self) -> str:
        return "fallback_geometry"

    def predict(self, point_cloud: np.ndarray,
                geometry_features: Optional[GeometryResult] = None,
                grasp_intent: Optional[GraspIntent] = None) -> List[LearnedGraspOutput]:
        if geometry_features is None or not geometry_features.valid:
            return []

        outputs = []
        centroid = np.array(geometry_features.centroid)

        # major_axis 기반 후보 -- SIDE류(perpendicular_to_long_axis) grasp가
        # 필요로 하는 정보. sign ambiguity 때문에 +축/-축에 수직인 두 방향
        # 모두 approach 후보로 낸다(major_axis 자체가 아니라 그 수직 방향이
        # side grasp의 접근 방향임에 주의 -- 긴 축을 따라 접근하면 잡을 수
        # 없고, 긴 축에 수직으로 접근해야 옆에서 감싸쥘 수 있음).
        if geometry_features.major_axis is not None:
            major = np.array(geometry_features.major_axis)
            major = major / (np.linalg.norm(major) + 1e-12)
            # major_axis에 수직이면서 대략 수평인 방향 하나를 구성 (world Z와
            # 외적 -> major에 수직인 수평 벡터, world Z에 평행하면 X축 사용)
            world_z = np.array([0.0, 0.0, 1.0])
            perp = np.cross(major, world_z)
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(major, np.array([1.0, 0.0, 0.0]))
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            for sign in (1.0, -1.0):
                outputs.append(LearnedGraspOutput(
                    position=centroid,
                    approach_vector=sign * perp,
                    grasp_axis=major,
                    score=geometry_features.geometry_confidence,
                ))

        # plane_normal 기반 후보 -- TOP류(perpendicular_to_surface/
        # along_surface_normal) grasp에 필요.
        if geometry_features.plane_normal is not None:
            normal = np.array(geometry_features.plane_normal)
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            outputs.append(LearnedGraspOutput(
                position=centroid,
                approach_vector=-normal,   # 표면 안쪽으로 파고드는 게 아니라
                                            # 표면에서 바깥으로 나가는(= 접근은
                                            # 그 반대인 -normal) 방향
                grasp_axis=None,
                score=geometry_features.geometry_confidence,
            ))

        return outputs


def get_default_backend() -> LearnedGraspBackend:
    """현재 기본 backend. 실제 학습 모델이 연결되기 전까지는 항상
    FallbackGeometryBackend -- 이 함수 하나만 바꾸면 실제 모델로 교체된다
    (backend abstraction 원칙, segmentation_backend.get_default_backend와
    동일 패턴)."""
    return FallbackGeometryBackend()
