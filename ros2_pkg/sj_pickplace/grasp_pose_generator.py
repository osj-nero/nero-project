#!/usr/bin/env python3
"""
grasp_pose_generator.py

[신설, 2026-08] GraspIntent + GeometryResult + Learned Grasp 후보 →
최종 GraspCandidate 리스트. ROS 비의존 순수 함수 모듈(grasp_kinematics.py와
동일한 경계 원칙).

책임: candidate 생성 + semantic filtering(랭킹) + post-grasp action vector
계산. **여기서 motion execution을 하지 않는다** — IK/충돌검사(reachability_score
채우기)와 실제 실행은 planning_node.py 책임으로 남긴다.

Approach A(2026-08 확정, 사용자 지시): point cloud → Learned Grasp Detector가
먼저 generic 후보를 내고, 그 다음에 VLM grasp_intent와의 일치도로 점수를
매겨 랭킹한다. Learned detector에 intent를 직접 넣는 방식(Approach B)이
필요한 backend가 생기면 backend.predict()가 grasp_intent를 활용하도록
구현하면 되고, 이 모듈의 구조는 바뀌지 않는다(semantic_score는 항상 사후
채점으로 남아있어도 무해함 -- intent를 이미 반영한 backend라면 대체로 높은
semantic_score를 받을 뿐).
"""
import math
from typing import List, Optional, Tuple

from .grasp_types import GeometryResult, GraspCandidate, GraspIntent
from .grasp_kinematics import candidate_quat_for
from .learned_grasp_backend import LearnedGraspOutput

# total_score 가중치. reachability_score는 이 모듈 단계에서는 항상 0(IK를
# 아직 안 돌렸으므로) -- planning_node.py가 IK 체크 후 reachability_score를
# 채우고 total_score를 재계산해야 진짜 최종 순위가 나온다(rescan_total_score
# 참고). 이 가중치는 geometry_confidence의 실측 캘리브레이션과 마찬가지로
# 설계 시점 추정값이며 실기 튜닝 대상이다.
SCORE_WEIGHTS = {"geometry": 0.3, "semantic": 0.5, "reachability": 0.2}

# semantic_score 계산 시 "충분히 수직/수평"으로 볼 내적 임계값.
_ALIGN_DOT_THRESHOLD = 0.7   # acos(0.7) ~= 45.6도 이내면 "대략 평행/수직"으로 인정


def _unit(v) -> Tuple[float, float, float]:
    import numpy as np
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return tuple(arr / n)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def semantic_score(intent: Optional[GraspIntent], geometry: Optional[GeometryResult],
                    approach_vector, grasp_axis=None) -> float:
    """VLM grasp_relation과 candidate의 approach_vector/grasp_axis가 실제
    geometry(major_axis, plane_normal) 기준으로 얼마나 일치하는지 0~1로
    채점한다.

    intent가 없거나 grasp_relation이 'unknown'이면 중립값 0.5를 반환한다
    (모르는 걸 벌점 주지 않는다 -- VLM 확장 전 구버전 호환 경로에서 이
    점수가 항상 낮게 나와 후보가 부당하게 밀리면 안 됨).
    """
    if intent is None or geometry is None or not geometry.valid:
        return 0.5
    relation = intent.grasp_relation
    if relation == "unknown":
        return 0.5

    approach_u = _unit(approach_vector)
    if approach_u == (0.0, 0.0, 0.0):
        return 0.5

    if relation == "perpendicular_to_long_axis":
        if geometry.major_axis is None:
            return 0.5
        cos_a = abs(_dot(approach_u, _unit(geometry.major_axis)))
        return round(max(0.0, 1.0 - cos_a), 3)   # cos~0(수직) -> 1.0점

    if relation == "along_long_axis":
        if geometry.major_axis is None:
            return 0.5
        cos_a = abs(_dot(approach_u, _unit(geometry.major_axis)))
        return round(max(0.0, cos_a), 3)          # cos~1(평행) -> 1.0점

    if relation in ("perpendicular_to_surface", "along_surface_normal"):
        if geometry.plane_normal is None:
            return 0.5
        cos_a = abs(_dot(approach_u, _unit(geometry.plane_normal)))
        return round(max(0.0, cos_a), 3)          # 표면에 수직으로 접근 = normal과 평행

    if relation == "from_top":
        # approach_vector가 world -Z에 가까울수록(아래로 내려가는 방향) 고득점.
        return round(max(0.0, -approach_u[2]), 3)

    return 0.5


def _approach_dir_hint(intent: Optional[GraspIntent], approach_vector) -> str:
    """candidate_quat_for에 넘길 grasp_dir_hint('top'|'side'|'pinch') 결정.
    VLM grasp_type이 있으면 그걸 최우선으로 쓴다(TOP/SIDE/PINCH를 geometry가
    다시 판단하지 않는다 -- "잡는 형태"는 VLM의 semantic 판단 영역).
    intent가 없거나 grasp_type이 무효면 approach_vector의 수직성으로
    top/side를 휴리스틱 구분한다(pinch는 geometry만으로 side와 구분 불가 --
    CLAUDE.md에 이미 문서화된 한계와 동일, intent 없이는 side로 취급)."""
    if intent is not None and intent.grasp_type in ("TOP", "SIDE", "PINCH"):
        return intent.grasp_type.lower()
    au = _unit(approach_vector)
    if au[2] < -0.7:   # 충분히 아래를 향함
        return "top"
    return "side"


def generate_candidates(learned_outputs: List[LearnedGraspOutput],
                        geometry: Optional[GeometryResult],
                        intent: Optional[GraspIntent],
                        pos: dict, use_moveit2: bool = False,
                        angle_deg: Optional[float] = None) -> List[GraspCandidate]:
    """LearnedGraspOutput 리스트(FallbackGeometryBackend 또는 실제 학습
    모델의 출력) → GraspCandidate 리스트로 변환하고 semantic_score까지
    채운다. reachability_score/total_score의 reachability 성분은 아직
    0(IK 미실행) -- planning_node.py가 rescan_total_score로 채운다.

    angle_deg: top-down 후보의 edge 정렬각. None이면 geometry.major_axis_yaw_deg
        를 쓰고, 그것도 없으면 0.0(자유 twist)로 candidate_quat_for가 처리한다.
    """
    if not learned_outputs:
        return []

    _angle = angle_deg
    if _angle is None and geometry is not None:
        _angle = geometry.major_axis_yaw_deg

    candidates = []
    for out in learned_outputs:
        dir_hint = _approach_dir_hint(intent, out.approach_vector)
        quat, is_side = candidate_quat_for(
            pos, out.approach_vector, grasp_dir_hint=dir_hint,
            angle_deg=_angle, use_moveit2=use_moveit2,
        )
        sem_score = semantic_score(intent, geometry, out.approach_vector, out.grasp_axis)
        geo_score = float(out.score)
        total = (SCORE_WEIGHTS["geometry"] * geo_score
                 + SCORE_WEIGHTS["semantic"] * sem_score)  # reachability 아직 0

        candidates.append(GraspCandidate(
            position=(pos['x'], pos['y'], pos['z']),
            quaternion=list(quat),
            approach_vector=tuple(_unit(out.approach_vector)),
            grasp_axis=(tuple(_unit(out.grasp_axis)) if out.grasp_axis is not None else None),
            geometry_score=round(geo_score, 3),
            semantic_score=sem_score,
            learned_grasp_score=round(geo_score, 3),
            reachability_score=0.0,
            total_score=round(total, 3),
            is_side=is_side,
            grasp_dir_hint=dir_hint,
            source=("fallback_geometry" if geometry is not None else "learned_grasp"),
            debug_reason=(
                f"intent_relation={intent.grasp_relation if intent else 'none'} "
                f"semantic_score={sem_score}"
            ),
        ))

    candidates.sort(key=lambda c: c.total_score, reverse=True)
    return candidates


def rescan_total_score(candidate: GraspCandidate, reachability_score: float) -> GraspCandidate:
    """IK/충돌검사 후(planning_node.py) reachability_score를 반영해
    total_score를 재계산한 **새** GraspCandidate를 반환한다(원본 불변 --
    호출부가 여러 후보를 순회하며 원본 리스트를 안전하게 재사용 가능하게
    하기 위함)."""
    from dataclasses import replace
    total = (SCORE_WEIGHTS["geometry"] * candidate.geometry_score
             + SCORE_WEIGHTS["semantic"] * candidate.semantic_score
             + SCORE_WEIGHTS["reachability"] * reachability_score)
    return replace(candidate, reachability_score=round(reachability_score, 3),
                   total_score=round(total, 3))


# ═══════════════════════════════════════════════════════════════════════
# Post-grasp Action Vector (pull/slide/push 방향 -- grasp 방향과 별개)
# ═══════════════════════════════════════════════════════════════════════

def generate_action_vector(intent: Optional[GraspIntent],
                           geometry: Optional[GeometryResult],
                           candidate: GraspCandidate) -> Optional[Tuple[float, float, float]]:
    """action_direction enum -> 실제 3D 방향 벡터로 변환한다(사용자 지시
    23번의 정확한 매핑):
        along_long_axis            -> major_axis
        perpendicular_to_long_axis -> major_axis에 수직인 벡터
        along_surface_normal       -> plane_normal
        opposite_approach          -> -approach_vector (기존 _do_slide의
                                       "접근방향 반대로 후퇴"와 정확히 동일 --
                                       회귀 없음)

    [부호 모호성 처리] major_axis/plane_normal은 PCA/RANSAC에서 나온
    "line"이라 방향(+/-)이 임의로 정해져 있다(9번 원칙). opposite_approach가
    아닌 세 경우는, candidate.approach_vector의 반대 방향(-approach_vector,
    즉 "방금 접근한 반대쪽으로 계속 진행")과 내적이 양수인 쪽 부호를
    택한다 -- "접근 동작을 자연스럽게 이어서 후퇴/당김 동작으로 연결한다"는
    휴리스틱이며, 실측 검증된 물리 법칙이 아니다. 손잡이가 실제로 어느
    쪽으로 당겨야 열리는지는 이 함수가 알 수 없는 정보이므로, 이 부호가
    틀리면 planning_node.py 쪽에서 재시도/반대 부호 시도 로직을 추가하는
    게 근본 해결책이다(이번 라운드 범위 밖 -- 기존 _do_slide도 원래
    단일 후퇴 방향만 시도했으므로 이 한계는 새로 생긴 게 아님).

    intent가 없거나 action_direction이 'unknown'이면 None -- 호출부는
    반드시 기존 동작(예: _do_slide의 접근방향 반대 고정 후퇴)으로
    폴백해야 한다.
    """
    if intent is None or intent.action_direction == "unknown":
        return None

    ad = intent.action_direction
    approach = candidate.approach_vector

    if ad == "opposite_approach":
        return tuple(-c for c in approach)

    if ad == "along_surface_normal":
        if geometry is None or geometry.plane_normal is None:
            return None
        axis = _unit(geometry.plane_normal)
    elif ad == "along_long_axis":
        if geometry is None or geometry.major_axis is None:
            return None
        axis = _unit(geometry.major_axis)
    elif ad == "perpendicular_to_long_axis":
        if geometry is None or geometry.major_axis is None:
            return None
        major = _unit(geometry.major_axis)
        world_z = (0.0, 0.0, 1.0)
        # major x world_z (외적) -- major_axis가 world_z에 거의 평행하면
        # (수직으로 선 물체) 대신 world_x로 외적
        cx = major[1] * world_z[2] - major[2] * world_z[1]
        cy = major[2] * world_z[0] - major[0] * world_z[2]
        cz = major[0] * world_z[1] - major[1] * world_z[0]
        if math.hypot(cx, cy, cz) < 1e-6:
            wx = (1.0, 0.0, 0.0)
            cx = major[1] * wx[2] - major[2] * wx[1]
            cy = major[2] * wx[0] - major[0] * wx[2]
            cz = major[0] * wx[1] - major[1] * wx[0]
        axis = _unit((cx, cy, cz))
    else:
        return None

    neg_approach = tuple(-c for c in approach)
    if _dot(axis, neg_approach) < 0:
        axis = tuple(-c for c in axis)
    return axis
