#!/usr/bin/env python3
"""
grasp_types.py

[신설, 2026-08] NERO grasp pose pipeline 재구조화 — VLM(semantic) / Geometry(metric)
/ Learned Grasp Detector / IK 각 단계가 주고받는 공용 데이터 구조.

배경: 기존 파이프라인은 VLM이 grasp_type(TOP/SIDE/PINCH)만 판단하고, 실제 접근각은
grasp_kinematics.py의 atan2(y,x) 기반 공식 + planning_node.py의 블라인드/Hough 후보
탐색이 담당했다(자세한 문제의식은 docs/wiki 또는 세션 기록 참고). 이 모듈은 그 사이에
"물체의 실제 3D 형상에서 유도한 6-DoF grasp 후보"라는 중간 표현을 추가하기 위한
타입 정의다.

ROS 비의존 순수 dataclass만 정의한다(grasp_kinematics.py와 동일한 모듈 경계 원칙 --
ROS 서비스 호출/실행은 이 모듈 책임이 아니다. pytest로 단위테스트 가능해야 한다).
"""
from dataclasses import dataclass
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# VLM Semantic Grasp Intent
# ═══════════════════════════════════════════════════════════════════════

# VLM에게 자유 텍스트를 그대로 downstream geometry에 넣지 않기 위한 허용 enum.
# vlm_grasp_server.py의 응답이 이 집합 밖의 값을 내면 안전값/unknown으로
# 강등하고, downstream은 unknown을 "구조화된 힌트 없음"으로 취급해 기존
# fallback 경로(grasp_type만으로 판단)로 넘어가야 한다.
GRASP_TYPES = {"TOP", "SIDE", "PINCH"}

GRASP_RELATIONS = {
    "perpendicular_to_long_axis",
    "along_long_axis",
    "perpendicular_to_surface",
    "along_surface_normal",
    "from_top",
    "unknown",
}

ACTIONS = {"PICK", "PULL", "PUSH", "SLIDE"}

ACTION_DIRECTIONS = {
    "along_long_axis",
    "perpendicular_to_long_axis",
    "along_surface_normal",
    "opposite_approach",
    "unknown",
}


@dataclass
class GraspIntent:
    """VLM의 semantic 판단 결과. metric 각도/좌표를 절대 담지 않는다 --
    이 dataclass에 도(degree)나 quaternion 필드를 추가하면 그건 설계
    위반이다(VLM은 "무엇을 어떻게"만 말하고 "몇 도"는 절대 말하지 않는다).

    target_part/grasp_relation/action/action_direction은 [2026-08 확장]
    기존 grasp_type/orientation/confidence(하위호환 유지)에 추가된 필드다.
    구버전 VLM 서버(vlm_grasp_server.py 확장 전 배포본)는 이 필드들을 안
    주므로 전부 기본값(None/unknown)으로 채워지고, downstream(특히
    grasp_pose_generator.generate_action_vector)은 grasp_relation/
    action_direction이 "unknown"이면 반드시 안전한 기존 동작(action=PICK
    가정, geometry 후보만으로 진행)으로 폴백해야 한다 -- 이 필드가
    unknown인데 임의로 방향을 추측하지 말 것.
    """
    grasp_type: str                      # TOP | SIDE | PINCH
    orientation: str = "HORIZONTAL"      # HORIZONTAL | VERTICAL (기존 필드, 유지)
    confidence: float = 0.0
    target_part: Optional[str] = None    # 예: "handle". None이면 object 전체
    grasp_relation: str = "unknown"      # GRASP_RELATIONS 중 하나
    action: str = "PICK"                 # ACTIONS 중 하나
    action_direction: str = "unknown"    # ACTION_DIRECTIONS 중 하나
    reason: str = ""
    source: str = "vlm"                  # vlm | label_fallback | unknown

    @classmethod
    def from_vlm_response(cls, data: dict) -> "GraspIntent":
        """vlm_grasp_server.py의 /infer_grasp 원본 JSON(dict)에서 생성.
        지원 안 하는 값은 자유 텍스트를 그대로 믿지 않고 안전값으로 강등한다
        (모듈 docstring "자유 텍스트를 그대로 downstream에 넣지 않는다" 원칙,
        기존 mcp_robot_server.infer_grasp()의 grasp_type/orientation 검증
        로직과 동일한 패턴 -- 중복 유지 대신 여기로 통합해도 됨).
        """
        grasp_type = str(data.get("grasp_type", "")).upper()
        if grasp_type not in GRASP_TYPES:
            grasp_type = "TOP"  # 안전 기본값 -- 호출부가 별도로 실패 처리해야 함

        orientation = str(data.get("orientation", "HORIZONTAL")).upper()
        if orientation not in ("HORIZONTAL", "VERTICAL"):
            orientation = "VERTICAL" if grasp_type == "PINCH" else "HORIZONTAL"

        grasp_relation = str(data.get("grasp_relation", "unknown")).strip()
        if grasp_relation not in GRASP_RELATIONS:
            grasp_relation = "unknown"

        action = str(data.get("action", "PICK")).upper()
        if action not in ACTIONS:
            action = "PICK"

        action_direction = str(data.get("action_direction", "unknown")).strip()
        if action_direction not in ACTION_DIRECTIONS:
            action_direction = "unknown"

        target_part = data.get("target_part")
        if target_part is not None:
            target_part = str(target_part).strip() or None

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return cls(
            grasp_type=grasp_type,
            orientation=orientation,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            target_part=target_part,
            grasp_relation=grasp_relation,
            action=action,
            action_direction=action_direction,
            reason=str(data.get("reason", "")),
            source="vlm",
        )


# ═══════════════════════════════════════════════════════════════════════
# VLM grasp_type/orientation → planning_node grasp_dir 변환 (정책)
# ═══════════════════════════════════════════════════════════════════════
#
# [2026-09-02 추가] CLAUDE.md "그립 형태가 애매할 때" 절의 변환표를 코드로
# 옮긴 것. 지금까지는 Claude가 매 호출마다 그 표를 손으로 적용했는데,
# 이건 규칙(정책)이지 매번 사람이 판단할 일이 아니다 -- 룰베이스 제어와
# 같은 레이어(코드)에 정책을 둔다.
#
# ⚠️ VLM 응답의 grasp_type 을 grasp_dir 에 그대로 문자열 매칭하면 안 된다
#    (개념 불일치): VLM의 PINCH 는 "손끝으로 얇은 단면" 이라는 분류학 용어,
#    이 프로젝트의 grasp_dir='pinch' 는 특정 실측 각도(가로로 놓인 얇은
#    물체용). 병처럼 세로로 긴 물체를 VLM이 PINCH로 분류해도 side 로 보내야
#    기하학적으로 맞는다.

# z(base_link, 미터) 대역: 로봇 "허리 높이" 근처면 top-down 접근이 IK
# 실패하기 쉬워 side 로 강등한다. 이 경계는 2026-08~09 세션에서 반복
# 관찰된 경험적 범위이지 체계적 실측값이 아니다 -- 확정값 아님, 실기
# 검증 후 조정.
WAIST_Z_MIN_M = 0.30
WAIST_Z_MAX_M = 0.50

# VLM approach_direction(FRONT/LEFT/RIGHT/BACK, 카메라 시점) → side_approach_deg
# (position_yaw 기준 상대각). mcp_robot_server.infer_grasp 의 _APPROACH_DEG_MAP
# 과 동일 -- 한쪽만 바꾸면 어긋나니 같이 확인할 것.
_APPROACH_DIRECTION_DEG = {"FRONT": 0.0, "LEFT": 90.0, "RIGHT": -90.0, "BACK": 180.0}


@dataclass
class ResolvedGrasp:
    grasp_dir: str                      # top | side | pinch  (planning_node 인자)
    side_approach_deg: float = 0.0       # grasp_dir in (side,pinch)일 때만 의미
    top_downgraded_to_side: bool = False # z 대역 때문에 top→side 로 바꿨나
    reason: str = ""


def resolve_grasp_dir(vlm_response: dict,
                       object_z: Optional[float] = None) -> ResolvedGrasp:
    """VLM /infer_grasp 응답(dict 또는 mcp_robot_server.infer_grasp 반환 dict)과
    물체 z좌표로부터 planning_node 의 grasp_dir / side_approach_deg 를 결정한다.

    변환표(CLAUDE.md):
      grasp_type=TOP                              → top
      grasp_type=SIDE                             → side
      grasp_type=PINCH, orientation=HORIZONTAL    → pinch
      grasp_type=PINCH, orientation=VERTICAL      → side
    추가 보정:
      grasp_type=TOP 인데 object_z 가 WAIST 대역(0.30~0.50m)이면 → side
      (top-down IK 실패가 잦은 구간, top_downgraded_to_side=True 로 표시).
      호출부는 이 경우 top 을 먼저 시도하고 실패 시 즉시 side 로 전환해도 됨.

    side_approach_deg 는 응답의 suggested_side_approach_deg 를 그대로,
    없으면 approach_direction 을 매핑해서 채운다(둘 다 없으면 0.0=FRONT).
    정밀값이 아니라 planning_node 접근각 스윕의 1순위 추측일 뿐.
    """
    grasp_type = str(vlm_response.get("grasp_type", "")).upper()
    orientation = str(vlm_response.get("orientation", "")).upper()

    if "suggested_side_approach_deg" in vlm_response:
        try:
            approach_deg = float(vlm_response["suggested_side_approach_deg"])
        except (TypeError, ValueError):
            approach_deg = 0.0
    else:
        ad = str(vlm_response.get("approach_direction", "FRONT")).upper()
        approach_deg = _APPROACH_DIRECTION_DEG.get(ad, 0.0)

    if grasp_type == "PINCH":
        if orientation == "VERTICAL":
            return ResolvedGrasp("side", approach_deg, False,
                                 "VLM PINCH+VERTICAL → side (세로 원통 감싸쥐기)")
        return ResolvedGrasp("pinch", approach_deg, False,
                             "VLM PINCH+HORIZONTAL → pinch")

    if grasp_type == "SIDE":
        return ResolvedGrasp("side", approach_deg, False, "VLM SIDE → side")

    if grasp_type == "TOP":
        if object_z is not None and WAIST_Z_MIN_M <= object_z <= WAIST_Z_MAX_M:
            return ResolvedGrasp(
                "side", approach_deg, True,
                f"VLM TOP 이지만 z={object_z:.3f}m 가 허리 대역"
                f"({WAIST_Z_MIN_M}~{WAIST_Z_MAX_M}) → side 로 시도 "
                f"(top-down IK 실패 잦음, 미검증 경계)")
        return ResolvedGrasp("top", 0.0, False, "VLM TOP → top")

    # grasp_type 이 비었거나 이상값 -- 안전하게 top 으로(호출부가 별도 처리)
    return ResolvedGrasp("top", 0.0, False,
                         f"grasp_type={grasp_type!r} 불명 → top 기본값")


# ═══════════════════════════════════════════════════════════════════════
# 3D Geometry Result (RANSAC + PCA/OBB) — geometry_3d.py가 채움
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GeometryResult:
    """geometry_3d.py가 point cloud에서 추출한 형상 정보.
    좌표계는 입력 point cloud와 동일(호출부 책임 -- 보통 카메라 좌표계 입력,
    필요시 base_link로 변환은 벡터 회전으로 별도 처리, geometry_3d.py 참고).
    """
    valid: bool = False
    centroid: Optional[Tuple[float, float, float]] = None

    # PCA
    major_axis: Optional[Tuple[float, float, float]] = None   # 최대 분산 방향(단위벡터)
    minor_axis: Optional[Tuple[float, float, float]] = None
    third_axis: Optional[Tuple[float, float, float]] = None
    eigenvalues: Optional[Tuple[float, float, float]] = None  # 내림차순
    extents: Optional[Tuple[float, float, float]] = None      # OBB 각 축 길이(입력 단위, 보통 미터)

    # RANSAC (dominant plane 1순위 -- 여러 평면 지원은 fit_dominant_planes가
    # list[PlaneResult]로 반환, GeometryResult엔 1순위만 요약)
    plane_normal: Optional[Tuple[float, float, float]] = None
    plane_inlier_count: int = 0
    plane_inlier_ratio: float = 0.0

    # 품질 지표
    point_count: int = 0
    filtered_point_count: int = 0
    geometry_confidence: float = 0.0

    # 파생값(로깅/디버깅 편의) -- major_axis/plane_normal을 수평면에 투영한
    # mod-180 방위각(도). None이면 수직성 부족 등으로 방위각이 무의미하다는 뜻
    # (perception_node._compute_face_normal_yaw의 verticality 게이트와 동일 원칙).
    major_axis_yaw_deg: Optional[float] = None
    normal_yaw_deg: Optional[float] = None

    source: str = "ransac_pca"   # 실험 비교용 -- source 기록 원칙(비교실험 A/B/C/D)


# ═══════════════════════════════════════════════════════════════════════
# Grasp Candidate
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GraspCandidate:
    """6-DoF grasp 후보 하나. Learned detector 출력이든 geometry fallback
    이든 동일한 타입으로 표현해서 candidate generator/semantic filtering이
    출처를 몰라도 동작하게 한다.

    position/quaternion은 base_link 프레임 기준(planning_node.py 기존 관례와
    동일). quaternion은 스펙 예시의 Tuple[float,float,float,float] 대신
    **list[4] xyzw**를 쓴다 -- grasp_kinematics.py 전체(euler_to_quat 등)가
    이미 이 표현을 쓰고 있어서, 여기서만 다른 타입을 쓰면 변환 지점마다
    실수 유발 위험이 커진다(기존 코드베이스 관례 우선).
    """
    position: Tuple[float, float, float]
    quaternion: list                              # [x,y,z,w], base_link 기준
    approach_vector: Tuple[float, float, float]    # 접근 방향(단위벡터, base_link)
    grasp_axis: Optional[Tuple[float, float, float]] = None  # 그리퍼 손끝이 닫히는 축

    geometry_score: float = 0.0
    semantic_score: float = 0.0
    learned_grasp_score: float = 0.0
    reachability_score: float = 0.0
    total_score: float = 0.0

    is_side: bool = False           # grasp_kinematics.py의 is_side 관례와 연결(TCP offset 분기용)
    grasp_dir_hint: str = "auto"    # resolve_grasp_quat 호환용 문자열(top/side/pinch)
    source: str = "unknown"         # learned_grasp | ransac_pca | face_normal | hough | blind_sweep
    debug_reason: str = ""
