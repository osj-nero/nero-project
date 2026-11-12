#!/usr/bin/env python3
# 서브에이전트로 수정함
"""
planning_node.py  (IK 안정화 패치 + 박스 각도보정 재도입판)
[변경점 vs 이전 버전]
1. _move_joints_sequence 중복 정의 제거
2. _move() 재시도 로직 추가 (max_attempts=3)
   - OMPL 랜덤 샘플링 특성상 같은 목표라도 재시도 시 성공률 크게 향상
3. 이동 단계별 tolerance 분리
   - approach/lift/transit: tolerance_orientation=0.15 (느슨하게 → 플래너 성공률 ↑)
   - descend/ascend: tolerance_orientation=0.05 (파지 직전/직후는 정밀하게)
4. POSES_FILE 경로를 XDG_DATA_HOME 기반으로 변경 (Jetson/PC 모두 호환)
5. force_reset 타이밍 개선: move_to_pose 직전에만 호출
[2026-07 각도보정 — 1차 시도 실패 -> 제거 -> 2차 시도로 재도입]
- 1차 시도(실패, 제거됨): approach 이후 joint5 값을 읽어 joint7 자리에
  넣는 방식. 직렬 링크 로봇은 특정 조인트 하나만 움직여도 그 이후 링크
  전체가 같이 움직이므로 "그리퍼 yaw만 살짝 돌리기"가 조인트 스페이스
  제어로는 애초에 성립 불가능했음. 게다가 approach 중간에 자세를 바꾸는
  방식이라, MoveIt2/펌웨어가 매번 새로 IK를 풀면서 조인트1~4가 20도
  이상 튀는 위험한 동작(엘보 플립)까지 관찰되어 폐기.
- 2차 시도(현재, 채택): 실물 컨트롤러(agx_arm_ctrl_single_node,
  pyAgxArm 펌웨어)를 대상으로 직접 orientation sweep 실험을 수행한 결과:
    * roll=180°,pitch=0°(우리가 "top-down"이라 가정했던 자세)는 여러
      위치에서 광범위하게 NO_SOLUTION.
    * 반대로 pitch=90°,yaw=0° 고정, roll을 0°~-90°로 스윕한 전 구간은
      전부 IK 성공. 실측 tf 결과는 항상 roll≈180°,pitch≈0°(즉 실제로는
      정확히 top-down 자세)이면서, yaw만 "실측yaw = 요청roll - 90°"
      관계로 정확히 선형 대응함을 확인 (2026-07 검증, 15° 간격 스윕).
    * 즉 이 로봇 펌웨어에서 "그리퍼를 top-down으로 유지한 채 접근축
      기준 yaw만 돌리는" 자세를 안정적으로 얻으려면, 요청 쿼터니언을
      (roll=-angle_deg, pitch=90°, yaw=0)로 구성해야 한다.
  이 자세는 approach 진입 전에 딱 한 번만 계산해서 pick 시퀀스
  (approach→descend→lift) 내내 동일하게 유지한다. 시퀀스 중간에 자세를
  바꾸는 로직이 전혀 없으므로, 1차 시도 때 있었던 "중간 전환 -> IK 분기
  튐" 문제가 구조적으로 재발하지 않는다.
  perception_node가 계산해 보내는 angle_base_deg(0~90도 정규화)를
  그대로 사용한다.
[2026-07 복원] 별도 세션에서 project knowledge의 예전 아키텍처를 참고해
_seeded_ik/_best_yaw_solution/_move_joint_target(관절한계 게이트 +
joint-goal 직접이동) 및 _register_floor_collision(PlanningScene ACM
직접 조작)을 새로 도입했으나, approach 자체가 전혀 안 움직이는 회귀가
발생해(게이트가 후보 전부를 거부, ACM 조작이 SRDF 기반 자기충돌 허용
목록을 깨뜨림) 실측 검증된 이전 상태로 전량 복원함. 복원된 구조:
pre-rotate(joint1 사전회전) + _pick_best_yaw_candidate(2후보 IK 이동량
비교) + _move(pose-goal, planner_chain=['PTP','LIN']→OMPL 안전폴백).
바닥/자기충돌은 SRDF(nero.srdf의 disable_collisions)와 joint_limits.yaml
레벨에서 처리하며, 런타임에 PlanningScene ACM을 직접 재작성하지 않는다.
"""
import json
import math
import os
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from std_srvs.srv import Empty as EmptySrv
import tf2_ros
from geometry_msgs.msg import Vector3Stamped
from rclpy.duration import Duration as RclpyDuration
try:
    from agx_arm_msgs.msg import AgxArmStatus
    HAVE_ARM_STATUS_MSG = True
except ImportError:
    HAVE_ARM_STATUS_MSG = False
# [신설] 그립 각도/자세 선택 로직 분리 모듈 (ROS 비의존, 순수 함수).
# 상세: grasp_kinematics.py 모듈 docstring 참고 (폐기된 접근/채택 이유/한계).
from .grasp_kinematics import (
    SIDE_MIN_DIST, SIDE_PITCH_DEG, SIDE_TCP_OFFSET, TOP_ANGLE_PITCH_DEG,
    QUAT_TOP_DOWN, QUAT_TOP_DOWN_R, QUAT_TOP_DOWN_L, QUAT_TOP_DOWN_RR,
    QUAT_HOME, QUAT_SIDE_FRONT, SIDE_TAG, GRASP_DIR_MAP, LABEL_GRASP_HINT,
    quat_angle_diff, top_down_angle_quat, sim_top_down_angle_quat,
    sim_box_aligned_quat,
    side_quat_for, sim_side_quat_for, pinch_quat_for, sim_pinch_quat_for,
    side_reachability_check, auto_grasp_quat,
    resolve_grasp_quat, YawCandidateSelector, SequenceRejected,
    SIDE_BLIND_SWEEP_DEG, side_face_candidates_deg,
)
# [신설] 폐루프(closed-loop) place 검증 -- 그리퍼를 열고 물러난 뒤 실제로
# 목표 위치에 물체가 안착했는지 perception으로 재확인한다.
# 상세: placement_verification.py 모듈 docstring 참고.
from .placement_verification import verify_placement
from .pick_verification import verify_pick
# arm_status/motion_status 값 -> 사람이 읽을 이름 (로그용)
ARM_STATUS_NAMES = {
    0: "NORMAL", 1: "EMERGENCY_STOP", 2: "NO_SOLUTION",
    3: "SINGULARITY_POINT", 4: "TARGET_POS_EXCEEDS_LIMIT",
    5: "JOINT_COMMUNICATION_ERR", 6: "JOINT_BRAKE_NOT_RELEASED",
    7: "COLLISION_OCCURRED",
}
MOTION_STATUS_NAMES = {0: "SUCCESS", 1: "FAILED", 255: "UNKNOWN"}
# 실물 모드에서 _move()가 도착 확인을 기다리는 최대 시간(초)
REAL_MOVE_TIMEOUT = 120.0
REAL_MOVE_POLL_SEC = 0.1
REAL_MOVE_SETTLE_SEC = 0.3  # 퍼블리시 직후 arm_status가 갱신되기까지 최소 대기
# ── 안전장치: 조인트 급회전("빙글 도는 문제") 감지 ────────────────────────────
JOINT_JUMP_MAX_DEG_PER_SEC = 150.0
JOINT_JUMP_MIN_DT_SEC = 0.02  # 이보다 짧은 간격의 샘플은 노이즈로 보고 무시
# ── 그리퍼 상수 ───────────────────────────────────────────────────────────────
GRIPPER_OPEN  = 0.08
GRIPPER_CLOSE = 0.01
GRIPPER_FORCE = 1.5
SIM_GRIPPER_OPEN  = [0.08]
SIM_GRIPPER_CLOSE = [0.01]
# [2026-08-31 추가] _gripper_holding_object 판정 허용오차(미터) -- SIM 그리퍼
# 조인트가 완전 닫힘(0.0)에서 이보다 더 벌어져 있으면 "물체를 물어서 못
# 닫힘"으로 본다. 실측 관찰: 빈 채로 닫히면 정확히 0.0/0.0, 뭔가 물면
# 0.028~0.048 범위로 멈추는 걸로 확인됨(2026-08-31) -- 둘 사이 여유있게 설정.
GRIPPER_HOLDING_TOL_M = 0.01
# ── 시퀀스 오프셋 (미터) ──────────────────────────────────────────────────────
APPROACH_Z = 0.10
DESCEND_Z  = 0.0
LIFT_Z     = 0.10
# [2026-08-26 추가] side 그립 approach standoff (미터, 접근방향 반대쪽으로
# 후퇴한 거리). top이 APPROACH_Z만큼 위에서 내려오는 것과 동일한 목적 --
# 이전엔 side가 approach에서 바로 최종 목표점(px,py)으로 이동해서, PTP
# 구간의 곡선 경로 때문에 물체를 툭 치는 문제가 실측 확인됐다(2026-08-26).
# 이제 approach는 이 standoff만큼 물러난 지점으로 가고, descend에서
# 접근방향을 따라 LIN(직선)으로 최종 지점까지 들어간다.
SIDE_APPROACH_STANDOFF_M = 0.10
# [2026-08-31 추가] place retreat_mode='lift'일 때 side/pinch approach가
# 올라가는 z 오프셋. top-down용 APPROACH_Z(0.10m)와 공유하지 않고 별도
# 상수로 분리 -- side/pinch는 물체 옆을 감싸쥔 채로 위에서 내려오다보니
# top-down보다 더 여유 있게 올라가야 주변(예: 바구니 벽)을 안 치는 걸로
# 실측 확인됨(사용자 지시로 0.10 -> 0.30 상향).
SIDE_LIFT_APPROACH_Z = 0.25
# [2026-08-26 추가] slide 액션(문고리/서랍 손잡이) 기본 후퇴 거리(미터) --
# 손잡이를 쥔 채로 접근각 반대방향으로 이만큼 직선 이동한 뒤 놓는다.
DEFAULT_SLIDE_DISTANCE_M = 0.08
# [2026-08-31 추가] side/pinch 접근 스윕이 목표 물체 주변의 다른 감지된
# 물체를 치는 현상이 실측 확인됨(면-정렬/블라인드 스윕 후보 각도가 넓게
# 도는데, MoveIt2 planning scene엔 테이블/바닥 정도만 등록돼 있고 YOLO가
# 인식한 물체들은 충돌 회피 대상이 아니었음). _sync_scene_collision_objects
# 가 /detected_objects의 각 물체를 keep-out 존으로 planning scene에
# 등록해 OMPL이 우회 경로를 찾게 한다.
#
# [2026-08-31 수정, 실측 지적] 처음엔 고정 10cm 정육면체를 썼는데, 실제로
# side 그립 접근 중 근처 물체(컵)를 스치는 사고가 재현됐다 -- 물체 실제
# 크기 + 그리퍼 손끝이 벌어지는 폭(TCP 오프셋 13.6cm)까지 감안하면 10cm
# 박스가 실제 여유 공간보다 작았던 것으로 보임. perception_node_sim.py가
# 이제 bbox를 depth에서 역투영해 물체별 실측 폭/높이(bbox_size_m, 카메라
# 광학축 기준 투영 크기)를 /detected_objects에 같이 보낸다 -- 있으면
# 그걸 우선 쓰고(최소크기+패딩 적용), 없으면(구버전 perception 등) 이
# 고정값으로 폴백한다.
COLLISION_OBJECT_SIZE_M = (0.10, 0.10, 0.10)
# bbox_size_m 기반 동적 크기 계산용 파라미터
COLLISION_BOX_MIN_XY_M = 0.06    # bbox_size_m가 너무 작게 나와도(원거리 등) 보장할 최소 크기
COLLISION_BOX_PADDING_M = 0.05   # 실측 오차 + 그리퍼 자체 폭 여유
COLLISION_BOX_Z_SIZE_M = 0.15    # bbox_size_m엔 높이(카메라 depth축) 정보가 없어 고정값 유지(기존보다 상향)
# [이관] SIDE_MIN_DIST, SIDE_PITCH_DEG, TOP_ANGLE_PITCH_DEG -> grasp_kinematics.py (import 참고)
# [이관] top_down_angle_quat, sim_top_down_angle_quat ->
# grasp_kinematics.top_down_angle_quat / sim_top_down_angle_quat
# ── 그리퍼 TCP 오프셋 (gripper_flange → 손가락 중간 지점까지 거리) ───────────
TOP_TCP_OFFSET  = 0.1358  # 미터 (GRIPPER_FLANGE_TO_FINGERTIP과 동일 실측값, 2026-07 확정)
# ── descend z 안전 하한 (그리퍼 끝단 기준) ─────────────────────────────
GRIPPER_FLANGE_TO_FINGERTIP = 0.1358
DESCEND_MIN_FINGERTIP_Z = 0.03
DESCEND_MIN_FLANGE_Z = DESCEND_MIN_FINGERTIP_Z + GRIPPER_FLANGE_TO_FINGERTIP
# [이관] SIDE_TCP_OFFSET -> grasp_kinematics.SIDE_TCP_OFFSET (import 참고)
# place(내려놓기) 시 그리퍼 손가락 끝이 바닥에서 떨어진 높이.
PLACE_DROP_Z = 0.01   # 미터, place_pos.z 기준 손가락 끝 절대 높이
# ── 홈 자세 ───────────────────────────────────────────────────────────────────
HOME_X, HOME_Y, HOME_Z = 0.0, 0.0, 0.75
# ── 타이밍 ────────────────────────────────────────────────────────────────────
MOVE_DELAY    = 6.0
GRIPPER_DELAY = 2.0  # [2026-07 수정] 4.0->2.0, 체감 속도 개선
# [신설] place 후 폐루프 검증 전 perception이 새 프레임을 관측하도록
# 주는 여유 시간. 그리퍼가 lift로 물러난 직후라 시야는 이미 열려있지만,
# 토픽 발행 주기/처리 지연을 감안해 약간의 여유를 둔다.
VERIFY_SETTLE_SEC = 0.8
# [신설] pre-rotate와 verify-vantage 재사용 -- joint1 명령각과 실측 물체
# 방향(atan2) 사이 오프셋. 스캔 실측 2개 지점(-155°→46°, 145°→-39°)의
# 평균 약 188°(범위 176~201°)로 확인됨. 카메라가 팔 끝에 top-down으로
# 장착되어 "정면"의 의미가 base 기준과 반대에 가깝게 어긋나는 것으로 추정.
BEARING_OFFSET_DEG = 188.0
# ── MoveIt2 플래너 설정 ────────────────────────────────────────────────────────
TOL_POS_LOOSE   = 0.01
TOL_ORI_LOOSE   = 0.15   # 약 8.6도 — 수평 이동 단계
TOL_POS_TIGHT   = 0.003
TOL_ORI_TIGHT   = 0.03   # 약 1.7도 (0.05->0.03, 실측 6~7도 편차 확인되어 완만히 강화)
MOVE_MAX_ATTEMPTS = 3
MOVE_TIMEOUT      = 30.0  # 단일 시도 타임아웃
# ── 반경 기반 워크스페이스 필터 (2026-07 추가) ────────────────────────────
MIN_REACH_R_M = 0.20   # 이보다 가까우면 top-down으로는 물리적으로 안 닿는 "도넛 구멍" -> 사전배제
BOUNDARY_R_M  = 0.30   # 이 반경 안쪽이면 tolerance 완화 폴백 필요할 수 있는 "경계 링"
# descend용 tolerance 폴백 단계 (x/y축=top-down 유지정도만 완화, z축=그립각도는 항상 고정)
DESCEND_TOL_ORI_FALLBACK = [
    (TOL_ORI_TIGHT, TOL_ORI_TIGHT, TOL_ORI_TIGHT),   # 1차: 원래대로 (완전 top-down)
    (0.10, 0.10, TOL_ORI_TIGHT),                      # 2차: 앞뒤/틸트만 살짝 완화 (~5.7도)
    (0.20, 0.20, TOL_ORI_TIGHT),                      # 3차: 더 완화 (~11.5도) -- 그립각도는 여전히 고정
]
# ── (A) 조인트1 스윕 탐색 기본값 (나중에 실제 값으로 교체 가능) ─────────────
JOINT1_LOWER = -2.70526
JOINT1_UPPER = 2.70526
DEFAULT_SCAN_BASE_JOINTS = {
    'joint1': 0.0, 'joint2': -0.8026, 'joint3': 0.0,
    'joint4': 1.8273, 'joint5': 0.0, 'joint6': 0.0, 'joint7': 1.3,
}
# [2026-08-31 추가] mcp_robot_server.py의 save_pose/move_to_saved_pose가 쓰는
# 것과 동일한 저장 파일 -- planning_node도 여기서 "observation" 자세를 직접
# 읽어서 place 폐루프 검증 전 복귀용으로 쓴다(사용자 지시: 카메라로 판정할
# 땐 반드시 옵저베이션 자세로 돌아가서 볼 것). 하드코딩 상수 대신 파일을
# 읽는 이유: 사용자가 옵저베이션 자세를 다시 저장(save_pose)해도 자동으로
# 최신값을 따라가게 하기 위함(이 세션에서 실제로 한 번 재저장됨).
_DEFAULT_POSES_DIR = os.path.join(
    os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share')),
    'nero_robot'
)
POSES_FILE = os.environ.get(
    'NERO_POSES_FILE',
    os.path.join(_DEFAULT_POSES_DIR, 'saved_poses.json')
)
DEFAULT_SCAN_STEP_DEG = 60.0
# [2026-08-26 추가, 2026-09-24 재수정] 스캔 스윕의 매 joint1 스텝마다 joint7을
# 추가로 틀어서 프레임 위/아래로 잘리는 물체까지 잡기 위한 값. camera_stand_link가
# gripper_flange(= joint7 하류)에 고정 마운트돼 있어 joint7을 돌리면 카메라 시야각이
# 바뀐다(Isaac Sim USD 트리 확인).
# 1차 시도(+/-0.15 대칭 델타)는 joint7 상한(1.5708)에 거의 붙어있는 observation
# 값(1.5491) 때문에 + 방향이 거의 즉시 클램프돼 효과가 없었음(실측 스크린샷 비교로
# 확인). 대신 **joint4를 1.399->1.8로 올려 기본 시야 자체를 근처 워크스페이스로
# 낮추고**(사용자 지시, saved_poses.json observation 갱신), joint7은 절대값 기준
# [SCAN_JOINT7_SWEEP_LOWER, observation 값(상한)] 사이를 SCAN_JOINT7_SWEEP_STEPS개로
# 나눠 스윕한다(사용자 지시: "지금을 상한, 1.1을 하한으로 3번").
# discover_boxes.py(tools/phase_c, "ours" 조건)의 J7_SWEEP_LOWER/STEPS와 반드시
# 동일하게 유지할 것 -- 두 조건의 씬 커버리지가 달라지면 비교실험이 "탐색 방식"
# 차이가 아니라 우연한 시야 차이로 왜곡된다.
SCAN_JOINT7_SWEEP_LOWER = 1.1
SCAN_JOINT7_SWEEP_STEPS = 3
SCAN_STEP_SETTLE_SEC = 2.0
SCAN_DEDUP_DIST_M = 0.05
# [이관] QUAT_TOP_DOWN 계열, GRASP_DIR_MAP, SIDE_TAG, LABEL_GRASP_HINT,
# euler_to_quat, quat_angle_diff, side_quat_for, side_reachability_check,
# auto_grasp_quat -> 전부 grasp_kinematics.py로 이관 (파일 상단 import 참고).
# 호출부는 grasp_kinematics.resolve_grasp_quat() 하나로 통합됨
# (기존 on_command 3곳의 거의 동일한 중복 분기 로직 대체).
# [이관] SequenceRejected -> grasp_kinematics.SequenceRejected (import 참고,
# task_planner.py가 이 예외를 잡아야 해서 순환참조 방지 목적으로 이동함).
class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        # [2026-07 추가] pick 시점 카메라 orientation 로그용
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        from moveit_msgs.srv import GetPositionFK
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk')
        from moveit_msgs.srv import GetStateValidity
        self._state_validity_client = self.create_client(
            GetStateValidity, '/check_state_validity')
        self.use_moveit2 = self.declare_parameter('use_moveit2', True).value
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self._cb = ReentrantCallbackGroup()
        self.latest_joint_state = None
        self._prev_joint_snapshot = None   # (dict{name:pos}, monotonic_time)
        self._safety_tripped = False
        self._emergency_stop_client = self.create_client(EmptySrv, 'emergency_stop')
        self.create_subscription(
            JointState, '/feedback/joint_states', self._on_joint_state_safety_check, 10)
        self.latest_joint_state_sim = None
        self.create_subscription(
            JointState, '/joint_states',
            lambda msg: setattr(self, 'latest_joint_state_sim', msg), 10)
        self.latest_arm_status = None
        self.latest_arm_status_recv_time = 0.0
        self._arm_status_available = False
        if HAVE_ARM_STATUS_MSG:
            try:
                def _on_arm_status(msg):
                    self.latest_arm_status = msg
                    self.latest_arm_status_recv_time = time.time()
                self.create_subscription(
                    AgxArmStatus, '/feedback/arm_status', _on_arm_status, 10)
                self._arm_status_available = True
            except Exception as e:
                self.get_logger().warn(
                    f'/feedback/arm_status 구독 생성 실패 (typesupport 문제로 추정): {e} | '
                    f'실물 모드 _move()가 도착 확인 없이 예전처럼 즉시 성공 처리됩니다.')
        else:
            self.get_logger().warn(
                'agx_arm_msgs.msg.AgxArmStatus import 실패 — 실물 모드 _move()가 '
                '도착 확인 없이 예전처럼 즉시 성공 처리됩니다. '
                '(ros2 interface show agx_arm_msgs/msg/AgxArmStatus 로 실제 타입 확인 필요)')
        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self.on_objects,
            qos_best_effort, callback_group=self._cb)
        self.sub_cmd = self.create_subscription(
            String, '/arm_command', self.on_command,
            qos_best_effort, callback_group=self._cb)
        self.pub_move    = self.create_publisher(PoseStamped, '/control/move_p',      qos_reliable)
        self.pub_gripper = self.create_publisher(JointState,  '/control/joint_states', qos_reliable)
        self.pub_move_j  = self.create_publisher(JointState,  '/control/move_j',       qos_reliable)
        self.pub_result  = self.create_publisher(String,      '/pick_result',           qos_reliable)
        self.moveit2 = self.moveit2_gripper = None
        if self.use_moveit2:
            from pymoveit2 import MoveIt2
            cb_arm     = ReentrantCallbackGroup()
            cb_gripper = ReentrantCallbackGroup()
            self.moveit2 = MoveIt2(
                node=self,
                joint_names=['joint1','joint2','joint3','joint4','joint5','joint6','joint7'],
                base_link_name='base_link',
                end_effector_name='gripper_flange',
                group_name='arm',
                callback_group=cb_arm,
                use_move_group_action=True,
                ignore_new_calls_while_executing=True,
            )
            self.moveit2.allowed_planning_time = 5.0
            self.moveit2.num_planning_attempts = 20
            self.get_logger().info(
                'MoveIt2 ENABLED (allowed_planning_time=5.0s, num_planning_attempts=20)')
        self._gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
        )
        self.latest_objects = []
        self._latest_objects_stamp = 0.0
        self._scanned_boxes = []  # [2026-07 추가] scan_box 결과 저장
        self._box_angle_deg = None
        self.busy = False
        self.lock = threading.Lock()
        if self.use_moveit2:
            time.sleep(3.0)
        self.get_logger().info('PlanningNode 준비 완료')
    # ── 토픽 콜백 ─────────────────────────────────────────────────────────────
    def on_objects(self, msg):
        try:
            self.latest_objects = json.loads(msg.data).get('objects', [])
            self._latest_objects_stamp = time.time()
        except Exception:
            pass
    def on_command(self, msg):
        self.get_logger().info(f'[CMD] 수신: {msg.data[:80]}')
        if self._safety_tripped:
            try:
                cmd_peek = json.loads(msg.data)
            except Exception:
                cmd_peek = {}
            if cmd_peek.get('action') != 'clear_safety':
                self.get_logger().error(
                    '🚨 안전 잠금 상태라 명령을 거부합니다. '
                    '로봇 상태 육안 확인 후 {"action":"clear_safety"} 로 해제하세요.')
                return
            else:
                self._safety_tripped = False
                with self.lock:
                    self.busy = False
                self.get_logger().warn('✅ 안전 잠금 해제됨 (clear_safety).')
                self._publish_result('success', 'safety_cleared')
                return
        with self.lock:
            if self.busy:
                self.get_logger().warn('작업 중. 명령 무시.')
                return
            self.busy = True
        try:
            cmd = json.loads(msg.data)
        except Exception:
            with self.lock:
                self.busy = False
            return
        action = cmd.get('action', 'pick')
        if action == 'pick':
            label = cmd.get('target_label', '')
            use_from_scan = cmd.get('from_scan', False)
            override_pos = cmd.get('override_pos')
            override_angle_deg = cmd.get('override_angle_deg')
            skip_reacquire = override_pos is not None
            if skip_reacquire:
                pos = override_pos
                angle_deg = override_angle_deg if override_angle_deg is not None else 0.0
                self.get_logger().info(
                    f'[override] 좌표/각도 직접 지정 -> pos={pos} angle_deg={angle_deg} '
                    f'(perception 재조회 전부 스킵)')
            elif use_from_scan:
                box_index = cmd.get('box_index')
                _label_matches = [b for b in self._scanned_boxes if b.get('label') == label]
                if box_index is not None:
                    # [신규] 명시적 인덱스 지정 -- 큐 순서(맨 앞)가 아니라
                    # 사용자가(Claude가) remaining_scanned_boxes에서 본
                    # 특정 항목을 정확히 집는다. 이미 다른 박스 위에
                    # 쌓인 좌표를 "다음 큐 항목"으로 착각해 잘못 집는
                    # 사고(2026-07-22 실측)가 구조적으로 발생하지 않는다.
                    if 0 <= box_index < len(_label_matches):
                        _scan_match = _label_matches[box_index]
                    else:
                        _scan_match = None
                        self.get_logger().warn(
                            f"[from_scan] box_index={box_index} 범위 초과 "
                            f"('{label}' 매칭 {len(_label_matches)}개)")
                else:
                    # 하위호환: 인덱스 미지정 시 기존처럼 큐 맨 앞
                    _scan_match = _label_matches[0] if _label_matches else None
                if _scan_match is not None:
                    pos = {'x': _scan_match['x'], 'y': _scan_match['y'], 'z': _scan_match['z']}
                    # [2026-08 수정] 기존엔 `or 0.0`으로 None(각도 인식 실패/
                    # 원형 물체)을 실측 0.0도와 구분 못 하게 뭉갰다 --
                    # side/pinch의 면-정렬 후보 탐색(side_face_candidates_deg)이
                    # 이 None 여부로 "물체 각도를 아는지"를 판단하므로, 여기서
                    # 뭉개면 from_scan 경로에서만 그 판단이 항상 무력화된다.
                    # _find_object_with_angle(비-scan 경로)은 원래도 None을
                    # 그대로 보존하고 있었음 -- 여기도 동일하게 맞춘다.
                    # top-down 쪽 numeric 사용처는 이미 None-safe(아래 align
                    # 재조회 폴백, _approach_angle_cands 계산부 참고).
                    angle_deg = _scan_match.get('angle_base_deg')
                    skip_reacquire = False
                    self.get_logger().info(
                        f'[from_scan] 저장된 좌표 사용 -> pos={pos} angle_deg={angle_deg} '
                        f'(align 시 각도 재조회, 실패 시 스캔값 fallback)')
                    try:
                        self._scanned_boxes.remove(_scan_match)
                    except ValueError:
                        pass
                else:
                    pos, angle_deg = None, None
                    self.get_logger().warn(f"[from_scan] 저장된 '{label}' 박스 없음.")
            else:
                pos, angle_deg = self._find_object_with_angle(label)
            self._box_angle_deg = angle_deg
            if pos is None:
                self.get_logger().warn(f"'{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                # [2026-08-31 수정] 여기서 결과를 발행 안 하면 MCP 클라이언트가
                # 75초 타임아웃까지 그냥 대기하게 되는 버그가 실측 확인됨
                # (낮은 confidence로 필터링돼 candidates가 비는 경우 반복 재현).
                self._publish_result('rejected', f"'{label}' 인식 안 됨(신뢰도 부족 또는 미검출)")
                return
            grasp_dir = cmd.get('grasp_dir', None)
            # [2026-08-26 수정] side_approach_deg의 기준을 world 절대각에서
            # 물체 방위각(position_yaw) 기준 상대각(+-)으로 바꿈 -- top-down의
            # angle_rel과 동일한 개념. 0(기본)이면 물체 정면(자연스러운 방향)
            # 에서 접근, +/-로 그 방위각 대비 회전한 방향에서 접근한다.
            # (이전엔 world 절대각을 받아 position_yaw를 빼서 상대각으로
            # 역산했는데, 절대각 기준이라 물체 위치가 바뀔 때마다 "정면"에
            # 해당하는 값을 사용자가 매번 다시 계산해야 했다.)
            _side_approach_offset_deg = cmd.get('side_approach_deg') or 0.0
            # [2026-08 추가, 프로토타입] grasp_pose_generator.py가 만든 후보를
            # 그대로 실행하고 싶은 호출부를 위한 옵션 경로. payload에
            # {'grasp_candidate': {'quaternion':[x,y,z,w], 'is_side':bool, ...}}
            # 가 있으면 resolve_grasp_quat() 계산을 건너뛰고 그 quat/is_side를
            # 그대로 쓴다 -- _do_pick 이후의 approach/descend/lift 시퀀스는
            # 전혀 안 바뀌므로(quat/is_side는 원래도 이 시퀀스에 전달되는
            # 파라미터일 뿐) 회귀 위험이 없다. 필드가 없으면(기존 모든
            # 호출부) 100% 기존 동작과 동일 -- 이 분기 자체가 새 로직을
            # 추가한 게 아니라 quat/is_side를 "어디서 계산했는지"만 바뀐다.
            _grasp_candidate = cmd.get('grasp_candidate')
            if (isinstance(_grasp_candidate, dict)
                    and isinstance(_grasp_candidate.get('quaternion'), list)
                    and len(_grasp_candidate['quaternion']) == 4):
                quat = [float(v) for v in _grasp_candidate['quaternion']]
                is_side = bool(_grasp_candidate.get('is_side', False))
                self.get_logger().info(
                    f'[grasp_candidate] 외부 제공 후보 사용 -- '
                    f'source={_grasp_candidate.get("source", "?")} '
                    f'total_score={_grasp_candidate.get("total_score", "?")}')
            else:
                # [단순화] GRASP_DIR_MAP 조회 + SIDE_TAG 체크 + 라벨/위치
                # 휴리스틱을 grasp_kinematics.resolve_grasp_quat() 하나로
                # 통합 (동작 동일, 기존 경로).
                quat, is_side = resolve_grasp_quat(grasp_dir, pos, label,
                                                   side_approach_offset_deg=_side_approach_offset_deg,
                                                   use_moveit2=self.use_moveit2)
            self.get_logger().info(
                f'PICK 시작: {label} @ {pos} | 자세: {grasp_dir or "auto"} '
                f'({"side" if is_side else "top"}) {[round(v,3) for v in quat]} | '
                f'박스각도={angle_deg}')
            t = threading.Thread(target=self._pick_sequence,
                                  args=(pos, quat, is_side, label, skip_reacquire, angle_deg),
                                  kwargs={'side_approach_offset_deg': _side_approach_offset_deg,
                                          'is_pinch': grasp_dir == 'pinch'})
            t.daemon = False
            t.start()
        elif action == 'slide':
            # [2026-08-26 추가] 문고리/서랍 손잡이 등 -- 잡아서 들어올리는
            # pick이 아니라, 잡은 채로 접근방향 반대로 당겨서 여는 동작.
            # side/pinch가 이미 공유하는 접근 시퀀스(TCP오프셋/standoff/
            # 각도후보탐색/사전회전없음)를 그대로 재사용하고 마지막
            # 단계(lift)만 slide(직선 후퇴+release)로 바꾼다. top-down은
            # 이 시퀀스 자체에 안 걸리므로 자연스럽게 미지원.
            label = cmd.get('target_label', '')
            use_from_scan = cmd.get('from_scan', False)
            override_pos = cmd.get('override_pos')
            # [2026-08 추가] 손잡이 각도(angle_base_deg) -- pick과 동일하게
            # side/pinch 접근각 후보를 물체(손잡이) 면-법선 방향으로 한정하는
            # 데 쓴다. 기존엔 여기서 버려져서 slide가 pick과 동일한 "아무
            # 각도로나 접근" 버그를 그대로 갖고 있었다.
            angle_deg = None
            if override_pos is not None:
                pos = override_pos
                self.get_logger().info(f'[slide][override] pos={pos}')
            elif use_from_scan:
                box_index = cmd.get('box_index')
                _label_matches = [b for b in self._scanned_boxes if b.get('label') == label]
                if box_index is not None and 0 <= box_index < len(_label_matches):
                    _scan_match = _label_matches[box_index]
                else:
                    _scan_match = _label_matches[0] if _label_matches else None
                if _scan_match is not None:
                    pos = {'x': _scan_match['x'], 'y': _scan_match['y'], 'z': _scan_match['z']}
                    angle_deg = _scan_match.get('angle_base_deg')
                    try:
                        self._scanned_boxes.remove(_scan_match)
                    except ValueError:
                        pass
                else:
                    pos = None
                    self.get_logger().warn(f"[slide][from_scan] 저장된 '{label}' 없음.")
            else:
                pos, angle_deg = self._find_object_with_angle(label)
            if pos is None:
                self.get_logger().warn(f"[slide] '{label}' 못 찾음.")
                with self.lock:
                    self.busy = False
                self._publish_result('rejected', f"'{label}' 인식 안 됨(신뢰도 부족 또는 미검출)")
                return
            grasp_dir = cmd.get('grasp_dir', None)
            if grasp_dir not in ('side', 'pinch'):
                self.get_logger().warn(
                    f"[slide] grasp_dir='{grasp_dir}' 미지원 -- 'side' 또는 'pinch'만 가능"
                    f"(top-down은 slide 대상이 아님).")
                with self.lock:
                    self.busy = False
                self._publish_result('rejected', "slide는 grasp_dir='side'|'pinch'만 지원합니다.")
                return
            _side_approach_offset_deg = cmd.get('side_approach_deg') or 0.0
            slide_distance_m = cmd.get('slide_distance_m') or DEFAULT_SLIDE_DISTANCE_M
            quat, is_side = resolve_grasp_quat(grasp_dir, pos, label,
                                               side_approach_offset_deg=_side_approach_offset_deg,
                                               use_moveit2=self.use_moveit2)
            self.get_logger().info(
                f'SLIDE 시작: {label} @ {pos} | grasp_dir={grasp_dir} | '
                f'slide_distance={slide_distance_m}m')
            t = threading.Thread(target=self._slide_sequence,
                                  args=(pos, quat, label),
                                  kwargs={'side_approach_offset_deg': _side_approach_offset_deg,
                                          'is_pinch': grasp_dir == 'pinch',
                                          'slide_distance_m': slide_distance_m,
                                          'angle_deg': angle_deg})
            t.daemon = False
            t.start()
        elif action == 'place':
            pos = cmd.get('place_pos')
            if pos is None:
                with self.lock:
                    self.busy = False
                return
            # [2026-07 추가] 반경 사전 필터 -- 도넛 구멍(r < MIN_REACH_R_M)은
            # top-down을 완화해도 물리적으로 안 닿는 구간이라, OMPL/IK를
            # 시도하기 전에 즉시 거부한다. (경계 링은 여기서 거르지 않고
            # descend 단계의 tolerance 폴백 체인이 처리하도록 넘긴다.)
            _r = math.hypot(pos.get('x', 0.0), pos.get('y', 0.0))
            if _r < MIN_REACH_R_M:
                self.get_logger().warn(
                    f'PLACE 거부: 목표 반경 r={_r:.3f}m < {MIN_REACH_R_M}m '
                    f'(물리적 도달 불가 구간 -- top-down 완화로도 해결 안 됨)')
                self._publish_result('rejected', f'place_pos too close to base (r={_r:.3f}m)')
                with self.lock:
                    self.busy = False
                return
            grasp_dir = cmd.get('grasp_dir', None)
            # [2026-08-26 추가] pick과 동일 -- side_approach_deg는 물체
            # 방위각(position_yaw) 기준 상대각(top-down의 angle_rel과 동일
            # 개념). place에도 이제 side/pinch 접근각 후보 탐색이 있어서
            # 필요해짐 (_try_place_at 참고).
            _place_side_approach_offset_deg = cmd.get('side_approach_deg') or 0.0
            # [단순화] place는 grasp_dir 미지정 시 pick과 달리 라벨 휴리스틱이
            # 아니라 "스캔된 박스 각도(top-down)"가 기본값이라, 이 분기만은
            # resolve_grasp_quat의 auto 경로를 그대로 쓸 수 없다 (의도된
            # 차이 — nero_robot_place_reachability memory: 명시적 grasp_dir
            # 없이는 top-down 유지가 기본 안전값). explicit 지정 시에만
            # resolve_grasp_quat로 통합.
            if grasp_dir:
                quat, is_side = resolve_grasp_quat(
                    grasp_dir, pos, label='',
                    side_approach_offset_deg=_place_side_approach_offset_deg,
                    use_moveit2=self.use_moveit2)
            else:
                is_side = False
                angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                quat = self._box_aligned_quat_for(pos, angle_deg)
            self.get_logger().info(
                f'PLACE 시작 @ {pos} | 자세: {"side" if is_side else "top"} '
                f'{[round(v,3) for v in quat]}')
            # [2026-08-31 추가] side/pinch place 직후 후퇴 방식 -- 'slide'(수평,
            # xy만 변화) | 'lift'(수직, z만 변화, 기본값). top-down은 원래부터
            # lift뿐이라 이 값 무관.
            _place_retreat_mode = cmd.get('retreat_mode') or 'lift'
            # 2026-09-18 추가(사용자 지시): 다중물체 스태킹 루프처럼 place 직후 바로
            # 다음 pick으로 넘어가는 자동화 시나리오에서, 폐루프 검증(observation/
            # verify-vantage 왕복 + 카메라 판정, 경우에 따라 VLM 재검증까지)이 매
            # 물체마다 수 초~수십 초를 태운다. skip_verify=True 로 이 전체 블록을
            # 건너뛰면 place는 물리적으로 그대로 수행되고 verified 필드만 None(측정
            # 안 함)으로 온다 -- 대신 그 시간만큼 다음 pick이 더 빨리 시작된다.
            _skip_verify = bool(cmd.get('skip_verify', False))
            # 2026-09-29: 호출자(파이프라인)가 아는 물체 박스 [{id,xyz,dims}] -- 이번 place 동안
            # /detected_objects 와 함께 충돌 오브젝트로 등록 (파이프라인 중엔 인식 토픽이 비어 있음).
            self._extra_obstacles = list(cmd['obstacles']) if 'obstacles' in cmd else None
            threading.Thread(
                target=self._place_sequence, args=(pos, quat, is_side),
                kwargs={'side_approach_offset_deg': _place_side_approach_offset_deg,
                        'is_pinch': grasp_dir == 'pinch',
                        'retreat_mode': _place_retreat_mode,
                        'skip_verify': _skip_verify},
                daemon=True).start()
        elif action == 'move':
            pos = cmd.get('target_pos')
            if pos is None:
                self.get_logger().warn("'move' 명령에 target_pos 없음.")
                with self.lock:
                    self.busy = False
                return
            quat_override = cmd.get('quat_override')
            if quat_override is not None:
                # [2026-08-26 추가] side 그립 등 RPY 후보를 grasp_dir 경유 없이
                # 직접 실험하기 위한 우회 경로 -- move_to_position(quat_override=)
                # 전용. resolve_grasp_quat/_box_aligned_quat_for를 전부 건너뛴다.
                quat = list(quat_override)
                self.get_logger().info(f'MOVE 시작 @ {pos} | 자세(override): {quat}')
                threading.Thread(
                    target=self._move_sequence, args=(pos, quat), daemon=True).start()
            else:
                grasp_dir = cmd.get('grasp_dir', None)
                # [2026-08-26 수정] pick과 동일 -- side_approach_deg는 이제
                # 물체 방위각(position_yaw) 기준 상대각(top-down의 angle_rel과
                # 동일 개념). world 절대각 역산 로직 제거.
                _side_approach_offset_deg = cmd.get('side_approach_deg') or 0.0
                # [단순화] move도 place와 동일 이유로 auto 기본값은 top-down 유지.
                # (원본 버그 후보였던 기본값 불일치 — move만 QUAT_HOME을 fallback으로
                # 썼던 것 — 는 resolve_grasp_quat 도입으로 place와 동일하게 통일됨.
                # 실질 동작은 원래도 is_side=False 분기에서 top_down으로 덮어써져
                # 차이가 없었으므로 회귀 아님.)
                if grasp_dir:
                    quat, is_side = resolve_grasp_quat(
                        grasp_dir, pos, label='',
                        side_approach_offset_deg=_side_approach_offset_deg,
                        use_moveit2=self.use_moveit2)
                else:
                    is_side = False
                    quat = None
                if not is_side:
                    angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
                    quat = self._box_aligned_quat_for(pos, angle_deg)
                elif cmd.get('apply_side_tcp_offset'):
                    # [2026-08-26 추가] pick_object의 _do_pick과 동일한 오프셋
                    # 계산(접근각 기준, position_yaw 고정 아님)을 미리보기에도
                    # 적용 -- 그리퍼를 안 닫고도 실제 pick 최종 도달 지점을
                    # 확인하고 싶을 때 씀 (move_to_position apply_side_tcp_offset=True).
                    dx, dy = pos['x'], pos['y']
                    approach_yaw = math.atan2(dy, dx) + math.radians(_side_approach_offset_deg)
                    pos = dict(pos)
                    pos['x'] = dx - math.cos(approach_yaw) * SIDE_TCP_OFFSET
                    pos['y'] = dy - math.sin(approach_yaw) * SIDE_TCP_OFFSET
                self.get_logger().info(f'MOVE 시작 @ {pos} | 자세: {quat}')
                threading.Thread(
                    target=self._move_sequence, args=(pos, quat),
                    kwargs={'is_side': is_side}, daemon=True).start()
        elif action == 'move_joints':
            joints = cmd.get('joints', {})
            if not joints:
                self.get_logger().warn("'move_joints' 명령에 joints 없음.")
                with self.lock:
                    self.busy = False
                return
            self.get_logger().info(f'MOVE_JOINTS 시작: {joints}')
            t = threading.Thread(target=self._move_joints_sequence, args=(joints,))
            t.daemon = False
            t.start()
        elif action == 'scan_box':
            base_joints = cmd.get('base_joints', DEFAULT_SCAN_BASE_JOINTS)
            step_deg = cmd.get('step_deg', DEFAULT_SCAN_STEP_DEG)
            scan_label = cmd.get('target_label', 'box')
            self.get_logger().info(
                f'SCAN_BOX 시작: base_joints={base_joints} step_deg={step_deg} label={scan_label}')
            t = threading.Thread(
                target=self._scan_box_sequence, args=(base_joints, step_deg, scan_label))
            t.daemon = False
            t.start()
        elif action == 'home':
            self.get_logger().info('HOME 시작')
            threading.Thread(target=self._home_sequence, daemon=True).start()
        else:
            self.get_logger().warn(f"알 수 없는 action: '{action}'")
            with self.lock:
                self.busy = False
    # ── 안전장치: 조인트 급회전 감지 ──────────────────────────────────────────
    def _on_joint_state_safety_check(self, msg: JointState):
        self.latest_joint_state = msg
        now = time.monotonic()
        current = dict(zip(msg.name, msg.position))
        if self._prev_joint_snapshot is not None:
            prev, prev_time = self._prev_joint_snapshot
            dt = now - prev_time
            if dt >= JOINT_JUMP_MIN_DT_SEC:
                for jname, pos in current.items():
                    if jname not in prev or jname == 'gripper':
                        continue
                    delta_deg = abs(math.degrees(pos - prev[jname]))
                    speed_deg_s = delta_deg / dt
                    # [2026-08-14 임시 비활성화, 원격 real 작업] 스캔 중 joint1이
                    # 빠르게 스윕할 때 이 임계값을 넘겨 emergency stop이 계속
                    # 트리거됨(실물, /feedback/joint_states 전용 경로라 sim은
                    # 원래도 이 콜백 자체가 안 불림). 미검증 상태로 남겨둠 --
                    # 실물 운영 시 주의 필요. 장기적으로는 스캔 경로를 느리게
                    # 하거나 임계값을 구분해서 재활성화해야 함.
                    # if speed_deg_s > JOINT_JUMP_MAX_DEG_PER_SEC and not self._safety_tripped:
                    #     self.get_logger().error(
                    #         f'🚨 [안전정지] {jname} 급회전 감지: '
                    #         f'{delta_deg:.1f}° in {dt*1000:.0f}ms '
                    #         f'({speed_deg_s:.0f}°/s > 임계값 {JOINT_JUMP_MAX_DEG_PER_SEC}°/s)')
                    #     self._trigger_emergency_stop(
                    #         reason=f'{jname} {speed_deg_s:.0f}°/s 급회전 감지')
        self._prev_joint_snapshot = (current, now)
    def _trigger_emergency_stop(self, reason: str):
        """emergency_stop 서비스 호출 + 이후 모든 이동 명령 거부하는 안전 잠금."""
        self._safety_tripped = True
        with self.lock:
            self.busy = True  # 이후 on_command가 새 명령을 아예 안 받도록 잠금
        self.get_logger().error(f'🚨🚨🚨 EMERGENCY STOP 발동: {reason}')
        try:
            if self._emergency_stop_client.wait_for_service(timeout_sec=1.0):
                self._emergency_stop_client.call_async(EmptySrv.Request())
                self.get_logger().error('emergency_stop 서비스 호출 완료.')
            else:
                self.get_logger().error(
                    'emergency_stop 서비스 응답 없음! agx_arm_ctrl_single_node 상태 즉시 확인 필요.')
        except Exception as e:
            self.get_logger().error(f'emergency_stop 서비스 호출 중 예외: {e}')
        self.get_logger().error(
            '⚠ 안전 잠금 상태입니다. 로봇 상태를 육안으로 확인한 뒤, '
            '"clear_safety" 액션으로 잠금을 해제해야 다음 명령을 받습니다.')
        self._publish_result('emergency_stop', reason)
    # ── 유틸 ──────────────────────────────────────────────────────────────────
    def _find_object(self, label):
        for obj in self.latest_objects:
            if obj.get('label') == label:
                return obj.get('center_3d')
        return None
    OBJECT_CONF_MIN = 0.35  # 이 아래는 그림자/오검출로 간주해 후보에서 제외
    def _find_object_with_angle(self, label):
        BOX_MIN_Z = 0.015
        candidates = [
            obj for obj in self.latest_objects
            if obj.get('label') == label
            and obj.get('confidence', 0.0) >= self.OBJECT_CONF_MIN
            and obj.get('center_3d', {}).get('z', 0.0) >= BOX_MIN_Z
        ]
        if not candidates:
            all_matching = [o for o in self.latest_objects if o.get('label') == label]
            self.get_logger().warn(
                f'[find_object] candidates 없음. label={label} '
                f'전체 매칭={len(all_matching)}개, '
                f'confidences={[round(o.get("confidence",0),3) for o in all_matching]}')
            return None, None
        def _dist(obj):
            c = obj.get('center_3d', {})
            return math.hypot(c.get('x', 0.0), c.get('y', 0.0))
        best = min(candidates, key=_dist)
        return best.get('center_3d'), best.get('angle_base_deg', None)

    # ── 충돌 회피(감지 물체 keep-out zone) ──────────────────────────────────
    def _sync_scene_collision_objects(self, exclude_pos=None, exclude_radius_m=0.08, extra=None):
        """[2026-08-31 추가] /detected_objects의 각 물체를 MoveIt2 planning
        scene에 박스형 충돌 오브젝트로 등록한다 -- side/pinch 접근 스윕이
        목표 물체 주변의 다른 물체를 치는 사고(실측 확인) 방지.

        exclude_pos가 주어지면 그 좌표에서 exclude_radius_m 이내의 물체는
        등록하지 않는다 -- 지금 집으려는/막 내려놓은 목표 물체 자신을
        장애물로 등록하면 접근 자체가 막히기 때문. 매 호출마다 기존
        등록을 전부 지우고 최신 /detected_objects로 다시 채운다(물체가
        옮겨졌을 수 있어 상태를 따로 추적하지 않고 매번 새로 씀 -- 단순함
        우선).

        [2026-08-31 수정, 실측 지적] 처음엔 고정 10cm 정육면체로 등록했는데
        side 그립 접근 중 근처 물체를 스치는 사고가 재현됐다 -- 물체마다
        실제 크기가 다른데 고정값이 실제 여유보다 작았던 것으로 보임.
        perception_node_sim이 이제 bbox_size_m(depth에서 역투영한 물체별
        실측 폭/높이)를 같이 보내면 그걸 우선 쓰고(최소크기+패딩 적용),
        없으면(구버전 perception, VLM-only 등) 기존 고정 크기로 폴백한다.

        use_moveit2=False(실물 로봇 경로)면 self.moveit2가 None이라
        아무것도 안 함 -- 이 기능은 현재 sim(MoveIt2 OMPL 플래닝) 경로
        전용이다.
        """
        if self.moveit2 is None:
            return
        try:
            self.moveit2.clear_all_collision_objects()
        except Exception as e:
            self.get_logger().warn(f'[충돌회피] 오브젝트 초기화 실패: {e}')
            return
        n_added = 0
        # 파이프라인이 물체 목록을 직접 준 경우(extra 가 None 이 아님)엔 /detected_objects 를 무시한다 --
        # 손에 든 물체나 오래된 검출이 장애물로 잡혀 place 자세를 막았다(2026-09-29 nbr9: 컵을 든 채
        # place 5후보 전부 collision_blocked).
        for i, obj in enumerate(self.latest_objects if extra is None else []):
            c = obj.get('center_3d')
            if not c:
                continue
            if exclude_pos is not None:
                dx = c.get('x', 0.0) - exclude_pos.get('x', 0.0)
                dy = c.get('y', 0.0) - exclude_pos.get('y', 0.0)
                if math.hypot(dx, dy) < exclude_radius_m:
                    continue
            _bsize = obj.get('bbox_size_m')
            if _bsize and len(_bsize) == 2 and _bsize[0] and _bsize[1]:
                box_size = (
                    max(_bsize[0], COLLISION_BOX_MIN_XY_M) + COLLISION_BOX_PADDING_M,
                    max(_bsize[1], COLLISION_BOX_MIN_XY_M) + COLLISION_BOX_PADDING_M,
                    COLLISION_BOX_Z_SIZE_M,
                )
            else:
                box_size = COLLISION_OBJECT_SIZE_M
            try:
                self.moveit2.add_collision_box(
                    id=f'nero_obs_{i}',
                    size=box_size,
                    position=(c.get('x', 0.0), c.get('y', 0.0), c.get('z', 0.0)),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id='base_link',
                )
                n_added += 1
            except Exception as e:
                self.get_logger().warn(f'[충돌회피] 오브젝트 추가 실패(label={obj.get("label")}): {e}')
        for ob in (extra or []):
            try:
                self.moveit2.add_collision_box(
                    id=f"nero_ext_{ob.get('id', n_added)}",
                    size=tuple(float(v) for v in ob['dims']),
                    position=tuple(float(v) for v in ob['xyz']),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    frame_id='base_link',
                )
                n_added += 1
            except Exception as e:
                self.get_logger().warn(f'[충돌회피] 외부 장애물 추가 실패({ob}): {e}')
        if n_added:
            self.get_logger().info(f'[충돌회피] 감지+외부 물체 {n_added}개를 충돌 오브젝트로 등록')

    def _clear_scene_collision_objects(self):
        """[2026-08-31 추가] _sync_scene_collision_objects로 등록한 충돌
        오브젝트를 전부 제거한다. pick/slide/place 시퀀스 종료 직후
        (finally 블록)에서 호출해, 이 시퀀스 전용으로 등록한 keep-out
        존이 이후의 무관한 이동(go_home, move_joints 등)까지 막지 않게
        한다."""
        if self.moveit2 is None:
            return
        try:
            self.moveit2.clear_all_collision_objects()
        except Exception as e:
            self.get_logger().warn(f'[충돌회피] 오브젝트 정리 실패: {e}')

    # ── top-down 쿼터니언 백엔드 분기 ────────────────────────────────────────
    def _top_down_quat_for(self, pos, angle_deg):
        """
        top 그립(side 아님)용 쿼터니언을 self.use_moveit2 여부로 분기해서 반환.
        angle_deg=0(자유 twist, approach/entry처럼 자세 최적화만 필요하고
        박스 각도 정렬은 상관없는 곳) 전용 -- 실제 박스 edge 각도로
        정렬해야 하는 곳(align, place/move의 self._box_angle_deg 사용)은
        이 메서드가 아니라 _box_aligned_quat_for를 써야 한다(아래 참고).
        """
        if self.use_moveit2:
            return sim_top_down_angle_quat(pos['x'], pos['y'], angle_deg)
        return top_down_angle_quat(pos['x'], pos['y'], angle_deg)

    def _box_aligned_quat_for(self, pos, angle_deg):
        """
        [2026-08 추가, 근본원인 수정] 실제 박스 edge 각도(perception이
        base_link 기준 절대각으로 tf 변환해 보내는 값)로 그리퍼를
        정렬해야 하는 곳 전용. _top_down_quat_for와 차이는 sim에서
        position_yaw(물체 방향 bearing)를 더하지 않는다는 것 -- angle_deg가
        이미 절대각인데 position_yaw를 더하면 그만큼 그리퍼가 박스 edge와
        어긋난다. 실측 확진: 박스(0.233,0.250), 인식각도=12.9도인데
        position_yaw=bearing=47.0도만큼 어긋나서 모서리를 잡는 현상 재현
        (2026-08-15, grasp_kinematics.sim_box_aligned_quat docstring 참고).

        approach(angle_deg=0 고정, "yaw 자유" — 자세 최적화만 필요하고
        박스 정렬은 무관)처럼 position_yaw가 있어야 IK branch가 안 꼬이는
        곳은 이 메서드가 아니라 _top_down_quat_for를 계속 써야 한다 --
        이 둘을 혼동하면 안 됨.

        [2026-08-17 추가, real 분기 버그 수정] real 백엔드의
        top_down_angle_quat(x,y,angle_deg)는 angle_deg를 "상대각"(roll
        입력)으로 기대하는데, 이 메서드는 절대각을 그대로 넘기고 있었다
        -- position_yaw만큼 빼서 상대각으로 변환한 뒤 넘겨야 한다
        (angle_rel = angle_abs - position_yaw, world-frame jaw 방향 =
        position_yaw + angle_rel = angle_abs가 되도록). sim만 테스트하는
        동안 real 쪽에 이 변환 누락 버그가 있는 채로 있었다 (merge
        검토 중 발견, 상세 근거는 [[grasp_kinematics_ik]] "top_down_angle_quat
        공식 & 절대각 변환" 참고).
        """
        if self.use_moveit2:
            return sim_box_aligned_quat(angle_deg)
        position_yaw_deg = math.degrees(math.atan2(pos.get('y', 0.0), pos.get('x', 0.0)))
        angle_rel = (angle_deg - position_yaw_deg) % 180.0
        return top_down_angle_quat(pos['x'], pos['y'], angle_rel)

    def _abs_angle_to_quat(self, pos: dict, angle_deg_abs: float) -> list:
        """perception angle_base_deg(base_link 절대각) → top-down quat.
        _box_aligned_quat_for의 별칭. YawCandidateSelector 콜백 등 절대각을
        직접 넘기는 경로에서 사용."""
        return self._box_aligned_quat_for(pos, angle_deg_abs)
    # ── 시퀀스 ────────────────────────────────────────────────────────────────
    def _do_pick(self, pos, quat, is_side=False, label='', skip_reacquire=False, angle_deg=0.0,
                 side_approach_offset_deg=0.0, is_pinch=False):
        # [2026-08 추가] side/pinch 그립의 면-정렬 신뢰도를 결과에 실어보내기
        # 위한 진단값. top-down이면 계속 None(해당 없음). is_side면 아래에서
        # 'unknown'(각도 모름, 원형 물체이거나 인식 실패 -- 기존 동작과 동일)
        # / 'face_aligned'(물체 면-법선 후보로 성공) / 'blind_fallback'(면
        # 후보 전부 실패해서 각도 무관 블라인드 스윕으로 폴백, 그립 불안정
        # 가능성 있음) 중 하나로 채워진다.
        side_grasp_alignment = None
        # [2026-08-31 추가] 목표 물체 자신은 제외하고, 주변 다른 감지 물체를
        # 충돌 오브젝트로 등록 -- side/pinch 접근 스윕이 옆 물체를 치는
        # 사고 방지(실측 확인, _sync_scene_collision_objects docstring 참고).
        self._sync_scene_collision_objects(exclude_pos=pos)
        if is_side:
            ok, reason = side_reachability_check(pos)
            if not ok:
                self.get_logger().warn(f'PICK 거부: {reason}')
                raise SequenceRejected(reason)
        if is_side:
            # [2026-08-26 수정] 오프셋 방향을 position_yaw(베이스→물체 반경
            # 방향) 고정이 아니라, 실제 그리퍼 접근 방향
            # (position_yaw + side_approach_offset_deg, sim_side_quat_for의
            # yaw 계산과 동일 기준)으로 바꿈.
            # [2026-08-26 추가] 접근각에 따라 IK 해 유무가 크게 갈리는 것이
            # 실측 확인됨 -- 요청받은 각도를 최우선 후보로 하고, 그 주변을
            # 스윕한 대체 후보들을 순서대로 시도해서 처음 IK가 풀리는
            # 각도를 채택한다 (아래 "1/6: 접근" 루프 참고). 여기서는 후보
            # 목록만 만들고, 실제 px/py/standoff/quat 확정은 그 루프에서 함.
            dx, dy = pos['x'], pos['y']
            position_yaw = math.atan2(dy, dx)
            # [2026-08 수정] 사각/각진 물체(책, 각진 물병 등)는 접근각이
            # 아무거나 되는 게 아니라 물체의 면-법선 방향이어야 두 손끝이
            # 평행면에 밀착한다 -- perception이 준 angle_base_deg(물체
            # 실측 각도)가 있으면 그 방향 4후보로 먼저 한정해서 시도하고,
            # 전부 IK 실패할 때만 기존 블라인드 스윕으로 폴백한다(never
            # 완전히 새 실패를 만들지 않도록, 대신 폴백 시 로그로 경고).
            # angle_base_deg=None(원형 물체이거나 인식 실패)이면 기존과
            # 동일하게 처음부터 블라인드 스윕만 쓴다.
            # 상세: grasp_kinematics.side_face_candidates_deg docstring 참고.
            _face_offset_candidates = side_face_candidates_deg(
                math.degrees(position_yaw), angle_deg, side_approach_offset_deg)
            _num_face_candidates = len(_face_offset_candidates)
            if _face_offset_candidates:
                _side_offset_candidates = list(_face_offset_candidates)
            else:
                _side_offset_candidates = [side_approach_offset_deg]
            for _d in SIDE_BLIND_SWEEP_DEG:
                _cand = side_approach_offset_deg + _d
                if _cand not in _side_offset_candidates:
                    _side_offset_candidates.append(_cand)
            # [2026-08-26 추가] pinch는 side와 접근 시퀀스는 동일하지만
            # 자세 공식(pitch)이 달라서 후보 재계산에 쓸 quat 함수도
            # pinch 전용으로 골라야 한다 -- 안 그러면 pinch로 요청했는데
            # candidate loop가 side 공식으로 재계산해버리는 버그가 생김.
            if is_pinch:
                _side_quat_fn = sim_pinch_quat_for if self.use_moveit2 else pinch_quat_for
                _grip_name = 'pinch'
            else:
                _side_quat_fn = sim_side_quat_for if self.use_moveit2 else side_quat_for
                _grip_name = 'side'
            approach_yaw = position_yaw + math.radians(side_approach_offset_deg)
            offset_desc = f'{_grip_name}(TCP오프셋 {SIDE_TCP_OFFSET}m 적용, 접근각 후보 {len(_side_offset_candidates)}개 탐색)'
        else:
            px = pos['x']
            py = pos['y']
            offset_desc = 'top(z)'
        if is_side:
            approach_z = pos['z']
            descend_z  = pos['z']
            lift_z     = pos['z'] + LIFT_Z      # top과 동일한 상승폭
        else:
            pz = pos['z']
            approach_z = pz + APPROACH_Z + TOP_TCP_OFFSET
            descend_z  = pz + DESCEND_Z + TOP_TCP_OFFSET
            descend_z  = max(descend_z, DESCEND_MIN_FLANGE_Z)
            lift_z     = pz + LIFT_Z + TOP_TCP_OFFSET
        # [2026-08-26 추가] 그리퍼를 approach 이동 전에 미리 연다. 기존엔
        # approach 완료 후(구 "2/6: 그리퍼 열기")에 열었는데, 직전 pick에서
        # 이미 뭔가 쥔 채로 새 pick을 시작하면 닫힌 그리퍼가 approach 경로
        # 내내 그 상태로 새 목표 쪽으로 이동해서 물체를 밀어내는 문제가
        # 실측 확인됐다(2026-08-26). 이동 시작 전에 열어서 근본적으로
        # 막는다 -- 어차피 기존 코드도 이전에 쥐고 있던 물체를 무조건
        # 놓는 동작이었으므로(선행 조건 확인 없음), 순서만 앞당긴 것이지
        # 새로운 부작용은 없다.
        self.get_logger().info('0/6: 그리퍼 열기 (접근 전)')
        self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
        time.sleep(GRIPPER_DELAY)
        # [2026-08-17] 한 번 완전 삭제했다가 복원함. 삭제 근거였던 "잘못된
        # IK branch 선택"은 그립각도 공식 수정(position_yaw 반영+180도
        # axis-flip 보정, docs/wiki/grasp_kinematics_ik.md 참고)으로
        # 해결됐지만, pre-rotate에는 **별개의 두 번째 역할**이 있었다는 게
        # 재현 테스트로 드러남: 스캔 직후 자세(예: joint1=145°)에서
        # approach로 바로 가면, joint1의 큰 변화량을 top-down 자세 유지와
        # 동시에 풀어야 해서 IK/모션플래너 부담이 커져 초반 approach가
        # 불안정해짐(사용자 실측 확인, yaw 공식 수정 이후에도 재현). 이
        # 두 번째 역할은 yaw 공식과 무관하므로 pre-rotate를 복원한다.
        # [2026-08-26 수정, 사용자 요청] side/pinch(is_side=True)는 joint1
        # 사전회전을 완전히 건너뛴다. 아래 pre-rotate는 top-down 전용
        # 최적화로 남긴다 -- side/pinch는 standoff+접근각 후보 탐색
        # (아래 "1/6: 접근" 루프)이 이미 자체적으로 여러 각도를 시도하므로
        # 사전회전의 원래 목적(단일 top-down IK 부담 경감)과 안 맞고,
        # bearing 계산이 side_approach_deg/후보각을 전혀 모르는 채
        # 원 위치 기준으로만 도는 게 오히려 불필요한 왕복이 될 수 있다.
        if not is_side:
            # [2026-07 추가, 원본] approach 진입 전 joint1만 먼저 물체 방향으로
            # 회전. 스캔이 끝난 자세(예: joint1=145°)에서 바로 approach를
            # 시도하면, 물체 방향까지의 큰 차이를 approach의 나머지
            # 조인트(top-down 자세 유지)와 동시에 풀어야 해서 IK가 훨씬
            # 어려워지는 것이 실측+로그 분석으로 확인됨. joint1만 먼저
            # 물체 쪽으로 돌려 "쉬운 시작 상태"를 만든다.
            # BEARING_OFFSET_DEG: joint1 명령각과 실측 물체 방향(atan2)
            # 사이 오프셋. 스캔 실측 2개 지점(-155°→46°, 145°→-39°)의
            # 평균 약 188°(범위 176~201°)로 확인됨. 카메라가 팔 끝에
            # top-down으로 장착되어 "정면"의 의미가 base 기준과 반대에
            # 가깝게 어긋나는 것으로 추정 — 완벽한 정렬이 목적이 아니라
            # IK 부담을 줄이는 사전 회전이므로 이 정도 오차는 허용.
            target_bearing = math.atan2(pos['y'], pos['x']) - math.radians(BEARING_OFFSET_DEG)
            target_bearing = math.atan2(math.sin(target_bearing), math.cos(target_bearing))
            js = self.latest_joint_state_sim
            if js is not None:
                joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
                current = dict(zip(js.name, js.position))
                pre_target = [current.get(jn, 0.0) for jn in joint_names]
                pre_target[0] = target_bearing
                # home(all-zero)에서 joint1만 돌리면 특이점(SINGULARITY_POINT) 발생.
                # 팔이 거의 뻗은 상태(joint2≈0, joint4≈0)이면 전형적 작업 자세로
                # 미리 구부려 특이점을 회피한다.
                if abs(pre_target[1]) < 0.15 and abs(pre_target[3]) < 0.15:
                    pre_target[1] = 0.6   # ~34°
                    pre_target[3] = 1.0   # ~57°
                self.get_logger().info(
                    f'[pre-rotate] joint1을 물체 방향 근처'
                    f'({math.degrees(target_bearing):.1f}\xb0)로 사전 회전...')
                if self.use_moveit2:
                    self.moveit2.force_reset_executing_state()
                    self.moveit2.max_velocity = 0.15
                    self.moveit2.max_acceleration = 0.3
                    self.moveit2.pipeline_id = ''
                    self.moveit2.planner_id = ''
                    self.moveit2.move_to_configuration(pre_target)
                    time.sleep(0.5)
                    deadline = time.time() + MOVE_TIMEOUT
                    while time.time() < deadline:
                        if (not self.moveit2._MoveIt2__is_motion_requested and
                                not self.moveit2._MoveIt2__is_executing):
                            break
                        time.sleep(0.1)
                    # [2026-08 수정, 코드리뷰 발견] 이전엔 motion_suceeded를
                    # 확인 안 하고 무조건 approach로 넘어갔음 -- pre-rotate가
                    # 조용히 실패해도 원인 불명의 "approach가 왜 이렇게 크게
                    # 도나"로만 보였음. 하드게이트(중단)는 걸지 않음 --
                    # pre-rotate는 순수 최적화라 실패해도 approach 자체의
                    # 재시도/OMPL 폴백이 이어받을 수 있음(handoff 6절 원칙).
                    if not self.moveit2.motion_suceeded:
                        self.get_logger().warn(
                            '[pre-rotate] 이동 실패/중단 (motion_suceeded=False) -- '
                            '사전회전 안 된 자세에서 approach 시작함. '
                            'approach 자체의 재시도/OMPL 폴백에 맡기고 계속 진행.')
                else:
                    self._move_joints_real_wait(pre_target)
            else:
                self.get_logger().warn('[pre-rotate] latest_joint_state_sim=None, 사전 회전 스킵')
        else:
            self.get_logger().info('[pre-rotate] side/pinch 그립 -- 사전회전 스킵(설계상)')
        self.get_logger().info(
            f'1/6: 접근 (approach) | {offset_desc}')
        # [2026-07 수정] yaw tolerance=3.14(완전 자유)가 너무 넓어서, IK가
        # gripper_flange와 link6이 충돌하는 극단적인 yaw 해를 고르는 경우가
        # 실측 확인됨. 1.6(±91.7도)으로 좁혀 자기충돌 유발 여지를 줄인다.
        approach_tol_ori = (TOL_ORI_TIGHT, TOL_ORI_TIGHT, 1.6)
        _align_quat_override = None
        if not is_side:
            if self.use_moveit2:
                # sim: 기존 그대로 유지(단일 quat, yaw 자유) -- 절대각 후보
                # 시도로 바꾸면 position_yaw 기반 자세 최적화가 깨져서
                # IK branch가 꼬이는 회귀가 실측 확인됐다(2026-08-17,
                # docs/wiki/grasp_kinematics_ik.md 참고). sim은 절대 이
                # 방식을 건드리지 말 것.
                _approach_angle_deg = 0.0
                quat = self._top_down_quat_for(pos, _approach_angle_deg)
                self.get_logger().info(
                    f'[top:sim(pitch180)] approach는 yaw 자유(위치만 정렬) -> '
                    f'quat={[round(v,3) for v in quat]}')
                ok = self._move(px, py, approach_z, quat, tol_ori=approach_tol_ori, planner_chain=['PTP', 'LIN'])
            else:
                # [2026-08-14 추가, 원격 real 작업] 절대각 두 후보
                # (angle_deg, +90°)를 직접 시도 -- 성공하면 align에서 다시
                # 정렬할 필요가 없어 중복 move를 줄인다(아래
                # _align_quat_override). _box_aligned_quat_for 내부에서
                # position_yaw 기준 상대각으로 변환됨.
                # [2026-08 수정] angle_deg가 None일 수 있음(아래 side_face_
                # candidates_deg 도입으로 from_scan 경로가 더 이상 None을
                # 0.0으로 뭉개지 않게 됨) -- top-down은 각도 불명이면 0.0
                # (축정렬) 취급이 기존과 동일하게 안전한 기본값.
                _angle_deg_num = angle_deg if angle_deg is not None else 0.0
                _approach_angle_cands = [_angle_deg_num % 180.0, (_angle_deg_num + 90.0) % 180.0]
                ok = False
                quat = None
                _approach_angle_deg = 0.0
                for _cand_deg in _approach_angle_cands:
                    _cand_quat = self._box_aligned_quat_for(pos, _cand_deg)
                    self.get_logger().info(
                        f'[top:real(top_down angle={_cand_deg:.0f}\xb0)] approach 시도 -> '
                        f'quat={[round(v,3) for v in _cand_quat]}')
                    ok = self._move(px, py, approach_z, _cand_quat,
                                    tol_ori=approach_tol_ori, planner_chain=['PTP', 'LIN'])
                    if ok:
                        quat = _cand_quat
                        _approach_angle_deg = _cand_deg
                        break
                # 실물: QUAT_TOP_DOWN 최후 수단
                if not ok:
                    _fallback_quat = list(QUAT_TOP_DOWN)
                    self.get_logger().warn(
                        f'[top:real] approach angle=0/90 모두 NO_SOLUTION → QUAT_TOP_DOWN 폴백 '
                        f'(시각적 방향 부정확) quat={[round(v,3) for v in _fallback_quat]}')
                    ok = self._move(px, py, approach_z, _fallback_quat,
                                    tol_ori=approach_tol_ori, planner_chain=['PTP', 'LIN'])
                    if ok:
                        quat = _fallback_quat
                        _approach_angle_deg = 0.0  # QUAT_TOP_DOWN은 각도 무관; 0으로 마킹
                # 실물 모드: IK 사전검사 불가 → approach에서 성공한 quat을 align에 그대로 전달.
                # 이렇게 하지 않으면 _pick_best_yaw_candidate가 항상 angle=0°를 반환해
                # approach가 angle=90°로 성공한 위치에서 align이 FAIL한다.
                _align_quat_override = quat if ok else None
        else:
            # [2026-08-26 추가] 접근각 후보 탐색 -- 요청 각도부터 순서대로
            # 시도, 첫 성공(standoff 지점 IK 존재)한 각도를 채택한다.
            # _move는 실패 시 IK 사전조회 단계에서 조기 종료하므로(fail-fast)
            # 여러 후보를 시도해도 비용이 크지 않다.
            # [2026-08 수정] _num_face_candidates > 0이면 앞쪽 후보들은
            # "면-법선 정렬" 후보(물체가 각진 경우 그립 안정성 보장), 뒤쪽은
            # 기존 블라인드 스윕 폴백 -- 어느 쪽에서 성공했는지 로그로 구분한다.
            if _num_face_candidates:
                self.get_logger().info(
                    f'  [{_grip_name} approach] 물체 각도 인식됨(angle_base_deg='
                    f'{angle_deg:.1f}°) -- 면 정렬 후보 {_num_face_candidates}개 우선 시도, '
                    f'실패 시 블라인드 스윕 폴백')
            ok = False
            _last_cand_target = _last_cand_quat = None
            for _i, _cand_offset in enumerate(_side_offset_candidates):
                _cand_yaw = position_yaw + math.radians(_cand_offset)
                _cux, _cuy = math.cos(_cand_yaw), math.sin(_cand_yaw)
                _cand_px = dx - _cux * SIDE_TCP_OFFSET
                _cand_py = dy - _cuy * SIDE_TCP_OFFSET
                _cand_standoff_px = _cand_px - _cux * SIDE_APPROACH_STANDOFF_M
                _cand_standoff_py = _cand_py - _cuy * SIDE_APPROACH_STANDOFF_M
                _cand_quat = _side_quat_fn(pos, approach_offset_deg=_cand_offset)
                self.get_logger().info(
                    f'  [{_grip_name} approach 후보 {_i+1}/{len(_side_offset_candidates)}] '
                    f'접근각={math.degrees(_cand_yaw):.1f}° (offset={_cand_offset:+.1f}°) 시도...')
                _last_cand_target = [_cand_standoff_px, _cand_standoff_py, approach_z]
                _last_cand_quat = _cand_quat
                # [2026-08-31 추가] 실제 모션(PTP/LIN, 각 0.3~4초+)을 시도하기
                # 전에 IK만 먼저 빠르게(ms 단위) 확인 -- 사용자 지적: 도달
                # 불가 후보들을 하나씩 실제로 움직여보며 거르는 게 너무
                # 느림. IK가 아예 안 풀리는 후보는 여기서 즉시 건너뛴다.
                if self._compute_ik_joints([_cand_standoff_px, _cand_standoff_py, approach_z],
                                           _cand_quat, quiet=True) is None:
                    continue
                ok = self._move(_cand_standoff_px, _cand_standoff_py, approach_z, _cand_quat,
                                tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
                if ok:
                    quat = _cand_quat
                    px, py = _cand_px, _cand_py
                    side_standoff_px, side_standoff_py = _cand_standoff_px, _cand_standoff_py
                    approach_yaw = _cand_yaw
                    _used_blind_fallback = _num_face_candidates and _i >= _num_face_candidates
                    if _used_blind_fallback:
                        side_grasp_alignment = 'blind_fallback'
                        self.get_logger().warn(
                            f'  [{_grip_name} approach] ⚠ 면 정렬 후보 {_num_face_candidates}개 '
                            f'전부 IK 실패 -- 블라인드 폴백 각도(offset={_cand_offset:+.1f}°)로 '
                            f'진행. 물체가 각진 형태라면 모서리를 무는 불안정한 그립일 수 있음.')
                    elif _num_face_candidates:
                        side_grasp_alignment = 'face_aligned'
                        if _cand_offset != side_approach_offset_deg:
                            self.get_logger().warn(
                                f'  [{_grip_name} approach] 요청 접근각(offset={side_approach_offset_deg:+.1f}°)은 '
                                f'IK 실패 -- 면 정렬 대체 각도(offset={_cand_offset:+.1f}°, '
                                f'접근각={math.degrees(approach_yaw):.1f}°)로 진행')
                        else:
                            self.get_logger().info(f'  [{_grip_name} approach] 요청 접근각 그대로 성공 (면 정렬)')
                    else:
                        side_grasp_alignment = 'unknown'
                        if _cand_offset != side_approach_offset_deg:
                            self.get_logger().warn(
                                f'  [{_grip_name} approach] 요청 접근각(offset={side_approach_offset_deg:+.1f}°)은 '
                                f'IK 실패 -- 대체 각도(offset={_cand_offset:+.1f}°, '
                                f'접근각={math.degrees(approach_yaw):.1f}°)로 진행')
                        else:
                            self.get_logger().info(f'  [{_grip_name} approach] 요청 접근각 그대로 성공')
                    break
        if not ok:
            if is_side:
                # [2026-08-31 추가] 후보 스윕을 IK 사전필터로 빠르게 걸렀기
                # 때문에(위 continue), _move() 내부의 fail-fast 진단이 한
                # 번도 안 불렸을 수 있다 -- 전부 실패했을 때 마지막 후보
                # 하나로 딱 1회만 "충돌 때문인지 진짜 도달불가인지" 진단한다.
                _diag = (self._diagnose_ik_failure(_last_cand_target, _last_cand_quat)
                        if _last_cand_target is not None else '진단 불가(후보 없음)')
                raise RuntimeError(
                    f'approach 이동 실패 (재시도 초과, {_grip_name} 접근각 후보 전부 실패) '
                    f'[원인진단(마지막 후보 기준): {_diag}]')
            raise RuntimeError('approach 이동 실패 (재시도 초과)')
        # [2026-08-26] 그리퍼 열기는 0/6(approach 전)으로 이동함 -- 여기서는
        # 더 이상 열지 않는다 (물체 밀어내기 방지, 위 주석 참고).
        _current_top_angle_deg = None  # side 그립이면 None 유지 (align 단계 자체가 없음)
        if not is_side:
            if skip_reacquire:
                angle_deg_refined = angle_deg
                self.get_logger().info(
                    f'[override] align 각도 재조회 스킵, override 값 사용: {angle_deg_refined}\xb0')
            else:
                angle_deg_refined = None
                for _retry in range(3):
                    _, angle_deg_refined = self._find_object_with_angle(label)
                    if angle_deg_refined is not None:
                        break
                    time.sleep(0.3)
                if angle_deg_refined is None and angle_deg is not None:
                    self.get_logger().warn(
                        f'[top] align 각도 재조회 실패, 스캔 각도 fallback: {angle_deg}°')
                    angle_deg_refined = angle_deg
            if angle_deg_refined is not None:
                if _align_quat_override is not None:
                    # [2026-08-14 추가, 원격 real 작업] real 전용 -- approach와
                    # 동일한 quat 사용 → 팔이 이미 올바른 위치·방향에 있으므로
                    # move 명령 없이 스킵. 중복 move는 불필요한 궤적(z 살짝
                    # 내려갔다 복귀 등)을 유발하고 순서가 틀려 보이는 문제의
                    # 원인이었음. sim은 _align_quat_override가 항상 None이라
                    # 이 분기를 절대 안 탄다.
                    # _current_top_angle_deg를 _approach_angle_deg로 세팅해야
                    # descend 단계의 quat 재계산도 approach와 일치한다.
                    best_deg, quat = _approach_angle_deg, _align_quat_override
                    self.get_logger().info(
                        f'[top] 3/6: 정렬(align) 스킵 — approach quat 유지 '
                        f'angle={_approach_angle_deg:.0f}° '
                        f'quat={[round(v,3) for v in quat]}')
                else:
                    best_deg, quat = self._pick_best_yaw_candidate(
                        pos, angle_deg_refined, ik_check_z=approach_z)
                    self.get_logger().info(
                        f'[top] 3/6: 정렬(align) 재조회각도={angle_deg_refined}\xb0 -> '
                        f'선택각도={best_deg:.1f}\xb0 quat={[round(v,3) for v in quat]}')
                    ok = self._move(px, py, approach_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
                    if not ok:
                        raise RuntimeError('align 이동 실패 (재시도 초과)')
                _current_top_angle_deg = best_deg
            else:
                self.get_logger().warn('[top] 각도 재조회 실패, yaw=0 유지')
                _current_top_angle_deg = _approach_angle_deg
            time.sleep(0.3)
            try:
                _refreshed_pos = pos if skip_reacquire else \
                    self._find_object_with_angle(label)[0]
                _cam_tf = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_color_optical_frame',
                    rclpy.time.Time(), timeout=RclpyDuration(seconds=0.2))
                _cq = _cam_tf.transform.rotation
                _cam_yaw = math.degrees(math.atan2(
                    2*(_cq.w*_cq.z + _cq.x*_cq.y), 1 - 2*(_cq.y*_cq.y + _cq.z*_cq.z)))
                _p = _refreshed_pos if _refreshed_pos is not None else pos
                _dist = math.hypot(_p.get('x', 0.0), _p.get('y', 0.0))
                _bearing = math.degrees(math.atan2(_p.get('y', 0.0), _p.get('x', 0.0)))
                self.get_logger().info(
                    f'[calib_debug] align완료후 cam_yaw={_cam_yaw:.1f} | '
                    f'obj_pos=({_p.get("x",0):.3f},{_p.get("y",0):.3f},{_p.get("z",0):.3f}) | '
                    f'obj_dist={_dist:.3f}m obj_bearing={_bearing:.1f} | '
                    f'rel_bearing={_bearing - _cam_yaw:.1f}')
            except Exception as e:
                self.get_logger().warn(f'[calib_debug] align완료후 tf 조회 실패: {e}')
            if _refreshed_pos is not None:
                px = _refreshed_pos.get('x', px)
                py = _refreshed_pos.get('y', py)
                self.get_logger().info(
                    f'[top] descend 목표 위치 갱신: ({px:.4f}, {py:.4f})')
                # [2026-08 수정, 코드리뷰 발견] real 백엔드는
                # top_down_angle_quat(x,y,angle_deg)가 yaw=atan2(y,x)로
                # 위치에 의존하는데, 위에서 px,py만 갱신하고 quat은
                # 갱신 전 pos 기준 값이 그대로 남아있었음. sim은 quat이
                # 위치와 무관해 문제가 안 드러났지만, real에서는 descend
                # 커맨드의 위치와 orientation이 서로 다른 시점 값으로
                # 섞여서 나가는 버그였음. 갱신된 위치로 quat도 재계산.
                # [2026-08 추가 수정] _current_top_angle_deg는 align이 고른
                # 실제 박스 정렬각이므로 _top_down_quat_for(자유 twist용,
                # sim에서 position_yaw를 더함)가 아니라 _box_aligned_quat_for
                # 를 써야 한다 -- 안 그러면 align에서 맞게 정렬해놓고
                # descend(실제로 집는 순간)에서 다시 position_yaw만큼
                # 어긋난 각도로 내려가는 회귀가 생긴다.
                # [2026-08-17 추가 수정, real 전용] real에서 approach가
                # QUAT_TOP_DOWN 폴백을 썼을 경우(_align_quat_override가
                # 그 폴백 quat, _current_top_angle_deg는 의미 없는 0.0
                # 마커) 여기서 재계산하면 각도 공식이 QUAT_TOP_DOWN을
                # 재현 못 해 오히려 틀린 quat으로 덮어씀 -- 그래서 real은
                # approach/align에서 이미 검증된 quat을 그대로 유지하고,
                # sim만(원래 동작 그대로) 위치가 갱신됐을 때 재계산한다.
                if self.use_moveit2:
                    quat = self._box_aligned_quat_for(
                        {'x': px, 'y': py, 'z': _refreshed_pos.get('z', pos.get('z', 0.0))},
                        _current_top_angle_deg)

        if is_side:
            self.get_logger().info(
                f'4/6: 내려가기 (descend) — side는 standoff에서 접근각 방향으로 '
                f'{SIDE_APPROACH_STANDOFF_M}m 직선(LIN) 진입')
        else:
            self.get_logger().info('4/6: 내려가기 (descend) — tight tolerance (+ 실패 시 완화 폴백)')
        if not is_side:
            # [2026-08-17 추가, 사용자 실측 발견] tolerance 완화 폴백까지
            # 다 실패해도 그립각도(±90°, 박스 mod-90 대칭 — 짧은변/긴변
            # 둘 중 하나를 잘못 골랐을 때 반대쪽을 시도)는 전혀 재시도
            # 안 했음. 90° 회전한 후보를 추가로 준비해 descend가 진짜
            # 막판까지 실패했을 때만 시도하도록 한다.
            _descend_quats = [quat]
            if _current_top_angle_deg is not None:
                _alt_deg = _current_top_angle_deg + 90.0
                _alt_pos = {'x': px, 'y': py, 'z': pos.get('z', 0.0)}
                _descend_quats.append(self._box_aligned_quat_for(_alt_pos, _alt_deg))
            ok = self._move_descend_with_tilt_fallback(px, py, descend_z, _descend_quats)
        else:
            ok = self._move(px, py, descend_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['LIN'])
        if not ok:
            raise RuntimeError('descend 이동 실패 (재시도 초과, tolerance 완화 폴백까지 실패)')
        self.get_logger().info('5/6: 그리퍼 닫기')
        self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
        time.sleep(GRIPPER_DELAY)
        self.get_logger().info('6/6: 들어올리기 (lift) — joint-space (빙글 방지)')
        if not is_side:
            lift_quat = quat
        else:
            lift_quat = quat
        lift_candidates = [lift_z, pos['z'] + 0.06 + (0 if is_side else TOP_TCP_OFFSET),
               pos['z'] + 0.03 + (0 if is_side else TOP_TCP_OFFSET)]
        ok = False
        for i, lz in enumerate(lift_candidates):
            if i > 0:
                self.get_logger().warn(
                    f'  lift 높이 낮춰서 재시도: {lz:.3f}m (원래 목표 {lift_z:.3f}m)')
            ok = self._move(px, py, lz, lift_quat, tol_ori=TOL_ORI_TIGHT, planner_chain=['LIN'])
            if ok:
                break
        if not ok:
            raise RuntimeError('lift 이동 실패 (모든 높이 재시도 초과)')
        return _current_top_angle_deg, side_grasp_alignment
    def _pick_sequence(self, pos, quat, is_side=False, label='', skip_reacquire=False, angle_deg=0.0,
                        side_approach_offset_deg=0.0, is_pinch=False):
        try:
            align_angle_deg, side_grasp_alignment = self._do_pick(
                pos, quat, is_side, label, skip_reacquire, angle_deg,
                side_approach_offset_deg=side_approach_offset_deg,
                is_pinch=is_pinch)
            self.get_logger().info('✅ PICK 완료')
            # [2026-08-31 수정, 사용자 지시] pick 성공/실패의 주 판정 신호를
            # 카메라(verify_pick, 원래 자리에서 사라졌나) 대신 그리퍼
            # 조인트 값으로 바꾼다 -- 카메라는 lift 직후 인식 사각지대에
            # 자주 걸려 null이 되는 경우가 많았고, 그리퍼 조인트는 카메라
            # 시야와 무관하게 즉시/항상 판정 가능하다. verify_pick(카메라)
            # 결과는 참고용으로 별도 필드에 같이 남긴다.
            holding, holding_reason = self._gripper_holding_object()
            self.get_logger().info(f'  [pick verify/그리퍼] {holding_reason}')
            time.sleep(VERIFY_SETTLE_SEC)
            _pick_verify_age = (time.time() - self._latest_objects_stamp
                                if self._latest_objects_stamp > 0 else None)
            pick_verification = verify_pick(pos, self.latest_objects, label=label,
                                            obj_age_sec=_pick_verify_age)
            self.get_logger().info(f'  [pick verify/카메라, 참고용] {pick_verification["reason"]}')
            # busy 해제를 발행보다 먼저 (레이스 방지 -- _scan_box_sequence 참고)
            with self.lock:
                self.busy = False
            self._publish_result(
                'success', 'pick_complete',
                extra={'remaining_scanned_boxes': self._scanned_boxes,
                       'align_angle_deg': align_angle_deg,
                       # [2026-08 추가] side/pinch 그립 면-정렬 신뢰도.
                       # top-down이면 항상 None. side/pinch면:
                       #   'face_aligned'    -- 물체의 실측 면-법선 방향으로 정상 접근
                       #   'blind_fallback'  -- 면 정렬 후보 전부 IK 실패, 각도 무관
                       #                        폴백 -- 각진 물체면 불안정한 그립일 수 있음
                       #   'unknown'         -- perception이 물체 각도를 못 줌(원형
                       #                        물체이거나 인식 실패) -- 기존 동작과 동일
                       'side_grasp_alignment': side_grasp_alignment,
                       # [2026-08-31 수정] pick 폐루프 검증의 주 신호 -- 이제
                       # 그리퍼 조인트 기반(_gripper_holding_object). true/false/
                       # null(판정 불가, 실물 로봇 경로 등) 의미는 기존과 동일.
                       'pick_verified': holding,
                       'pick_verification_reason': holding_reason,
                       # [2026-08-31 추가] 카메라 기반(verify_pick) 결과는 참고용
                       # 보조 필드로 유지 -- 그리퍼 판정과 교차검증하고 싶을 때 사용.
                       'pick_verified_camera': pick_verification['verified'],
                       'pick_verified_camera_reason': pick_verification['reason']})
        except SequenceRejected as e:
            with self.lock:
                self.busy = False
            self._publish_result('rejected', str(e))
        except Exception as e:
            self.get_logger().error(f'PICK 오류: {e}')
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
            # [2026-08-31 추가] 이 시퀀스 전용 keep-out zone이 이후 무관한
            # 이동(go_home 등)까지 막지 않도록 정리.
            self._clear_scene_collision_objects()

    def _slide_sequence(self, pos, quat, label='', side_approach_offset_deg=0.0,
                         is_pinch=False, slide_distance_m=DEFAULT_SLIDE_DISTANCE_M,
                         angle_deg=None):
        try:
            self._do_slide(pos, quat, label, side_approach_offset_deg=side_approach_offset_deg,
                            is_pinch=is_pinch, slide_distance_m=slide_distance_m,
                            angle_deg=angle_deg)
            self.get_logger().info('✅ SLIDE 완료')
            with self.lock:
                self.busy = False
            self._publish_result('success', 'slide_complete')
        except SequenceRejected as e:
            with self.lock:
                self.busy = False
            self._publish_result('rejected', str(e))
        except Exception as e:
            self.get_logger().error(f'SLIDE 오류: {e}')
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
            self._clear_scene_collision_objects()

    def _do_slide(self, pos, quat, label='', side_approach_offset_deg=0.0,
                  is_pinch=False, slide_distance_m=DEFAULT_SLIDE_DISTANCE_M,
                  angle_deg=None):
        """문고리/서랍 손잡이 등 -- 잡아서 들어올리는 pick이 아니라, 잡은
        채로 접근방향 반대로 당겨서 여는 동작. _do_pick의 side/pinch
        approach 루프(TCP오프셋/standoff/각도후보탐색)를 그대로 재사용하고
        마지막 단계만 lift 대신 slide+release로 바꾼다.
        """
        # [2026-08-31 추가] _do_pick과 동일한 이유로 주변 물체를 충돌
        # 오브젝트로 등록 (_sync_scene_collision_objects docstring 참고).
        self._sync_scene_collision_objects(exclude_pos=pos)
        ok, reason = side_reachability_check(pos)
        if not ok:
            self.get_logger().warn(f'SLIDE 거부: {reason}')
            raise SequenceRejected(reason)

        dx, dy = pos['x'], pos['y']
        position_yaw = math.atan2(dy, dx)
        # [2026-08 수정] _do_pick과 동일한 면-정렬 후보 우선 탐색 -- 손잡이도
        # 대개 각진 바 형태라 아무 각도에서나 핀치하면 안정적으로 안 물린다.
        # 상세 원리는 grasp_kinematics.side_face_candidates_deg docstring 참고.
        _face_offset_candidates = side_face_candidates_deg(
            math.degrees(position_yaw), angle_deg, side_approach_offset_deg)
        _num_face_candidates = len(_face_offset_candidates)
        if _face_offset_candidates:
            _offset_candidates = list(_face_offset_candidates)
        else:
            _offset_candidates = [side_approach_offset_deg]
        for _d in SIDE_BLIND_SWEEP_DEG:
            _cand = side_approach_offset_deg + _d
            if _cand not in _offset_candidates:
                _offset_candidates.append(_cand)
        if is_pinch:
            _quat_fn = sim_pinch_quat_for if self.use_moveit2 else pinch_quat_for
            _grip_name = 'pinch_slide'
        else:
            _quat_fn = sim_side_quat_for if self.use_moveit2 else side_quat_for
            _grip_name = 'side_slide'
        grip_z = pos['z']

        self.get_logger().info('0/5: 그리퍼 열기 (접근 전)')
        self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
        time.sleep(GRIPPER_DELAY)

        if _num_face_candidates:
            self.get_logger().info(
                f'1/5: 접근 (approach) | {_grip_name}(TCP오프셋 {SIDE_TCP_OFFSET}m 적용, '
                f'물체 각도 인식됨(angle_base_deg={angle_deg:.1f}°) -- 면 정렬 후보 '
                f'{_num_face_candidates}개 우선, 총 {len(_offset_candidates)}개 탐색)')
        else:
            self.get_logger().info(
                f'1/5: 접근 (approach) | {_grip_name}(TCP오프셋 {SIDE_TCP_OFFSET}m 적용, '
                f'접근각 후보 {len(_offset_candidates)}개 탐색)')
        ok = False
        px = py = standoff_px = standoff_py = approach_yaw = ux = uy = None
        _last_cand_target = _last_cand_quat = None
        for _i, _cand_offset in enumerate(_offset_candidates):
            _cand_yaw = position_yaw + math.radians(_cand_offset)
            _cux, _cuy = math.cos(_cand_yaw), math.sin(_cand_yaw)
            _cand_px = dx - _cux * SIDE_TCP_OFFSET
            _cand_py = dy - _cuy * SIDE_TCP_OFFSET
            _cand_standoff_px = _cand_px - _cux * SIDE_APPROACH_STANDOFF_M
            _cand_standoff_py = _cand_py - _cuy * SIDE_APPROACH_STANDOFF_M
            _cand_quat = _quat_fn(pos, approach_offset_deg=_cand_offset)
            self.get_logger().info(
                f'  [{_grip_name} approach 후보 {_i+1}/{len(_offset_candidates)}] '
                f'접근각={math.degrees(_cand_yaw):.1f}° (offset={_cand_offset:+.1f}°) 시도...')
            _last_cand_target = [_cand_standoff_px, _cand_standoff_py, grip_z]
            _last_cand_quat = _cand_quat
            # [2026-08-31 추가] 실제 모션 전 IK 사전 필터 (_do_pick과 동일 이유)
            if self._compute_ik_joints([_cand_standoff_px, _cand_standoff_py, grip_z],
                                       _cand_quat, quiet=True) is None:
                continue
            ok = self._move(_cand_standoff_px, _cand_standoff_py, grip_z, _cand_quat,
                            tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
            if ok:
                quat = _cand_quat
                px, py = _cand_px, _cand_py
                standoff_px, standoff_py = _cand_standoff_px, _cand_standoff_py
                approach_yaw = _cand_yaw
                ux, uy = _cux, _cuy
                if _num_face_candidates and _i >= _num_face_candidates:
                    self.get_logger().warn(
                        f'  [{_grip_name} approach] ⚠ 면 정렬 후보 {_num_face_candidates}개 '
                        f'전부 IK 실패 -- 블라인드 폴백 각도(offset={_cand_offset:+.1f}°)로 진행. '
                        f'손잡이가 각진 형태라면 불안정하게 물릴 수 있음.')
                break
        if not ok:
            _diag = (self._diagnose_ik_failure(_last_cand_target, _last_cand_quat)
                    if _last_cand_target is not None else '진단 불가(후보 없음)')
            raise RuntimeError(
                f'{_grip_name} approach 이동 실패 (재시도 초과, 접근각 후보 전부 실패) '
                f'[원인진단(마지막 후보 기준): {_diag}]')

        self.get_logger().info('2/5: 손잡이까지 직선(LIN) 진입')
        ok = self._move(px, py, grip_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['LIN'])
        if not ok:
            raise RuntimeError(f'{_grip_name} 진입 실패 (재시도 초과)')

        self.get_logger().info('3/5: 그리퍼 닫기 (손잡이 파지)')
        self._gripper(SIM_GRIPPER_CLOSE, GRIPPER_CLOSE)
        time.sleep(GRIPPER_DELAY)

        # 접근방향(ux,uy) 반대로 후퇴 -- 손잡이를 쥔 채로 당겨서 연다.
        slide_px = px - ux * slide_distance_m
        slide_py = py - uy * slide_distance_m
        self.get_logger().info(
            f'4/5: 슬라이드 ({slide_distance_m}m, 접근각 {math.degrees(approach_yaw):.1f}° 반대 방향)')
        ok = self._move(slide_px, slide_py, grip_z, quat, tol_ori=TOL_ORI_LOOSE, planner_chain=['LIN'])
        if not ok:
            raise RuntimeError(f'{_grip_name} 슬라이드 이동 실패 (재시도 초과)')

        self.get_logger().info('5/5: 그리퍼 열기 (놓기)')
        self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
        time.sleep(GRIPPER_DELAY)

    def _do_place(self, pos, quat, is_side=False, side_approach_offset_deg=0.0,
                  is_pinch=False, retreat_mode='lift', skip_verify=False) -> dict:
        """[2026-08 재설계] 실패 시 좌표를 자동으로 시프트해가며 재시도한다.
        근거: nero_robot_place_reachability memory (2026-07-30~08-03 실측).
          - approach/descend 실패는 특정 (x,y) 국소 도달불가 영역(singularity)일
            수 있다. 같은 좌표를 그냥 재시도해봐야 소용없다.
          - +y 먼저 시도, 실패하면 -y도 시도 (한 방향만 믿으면 안 됨 -- 08-03
            사례는 +y도 실패, -y만 성공했음).
          - z가 비정상적으로 높으면(≳0.2m) x/y가 아니라 z를 0.05~0.1m씩
            낮춰야 하는 별도의 height ceiling 문제였음.
        이 재시도는 이전에는 Claude가 place_object 실패 응답을 보고 매번
        직접 판단해서 순차 호출하던 것 -- 서버가 대신 처리하므로 Claude는
        place_object를 한 번만 호출하면 된다 (토큰/지연시간 절감).
        """
        # [신설] 폐루프 검증용 -- place 시작 전(그리퍼가 아직 아무것도
        # 안 놓은 상태) 장면을 스냅샷. 쌓기 시나리오에서 베이스 박스를
        # "새로 놓인 물체"로 착각하지 않으려면 이 스냅샷이 필수다
        # (placement_verification.py 모듈 docstring "쌓기 시 오탐 방지"
        # 참고). approach/descend 재시도 중에는 장면이 안 바뀌므로
        # 후보 좌표를 여러 번 시도해도 이 스냅샷 하나를 계속 재사용한다.
        before_objects = list(self.latest_objects)
        # [2026-08-31 추가] _do_pick과 동일한 이유로 주변 물체를 충돌
        # 오브젝트로 등록(_sync_scene_collision_objects docstring 참고).
        # place는 목표 좌표에 아직 물체가 없는 게 보통이지만, 쌓기처럼
        # 근처에 이미 다른 물체가 있는 경우까지 대비해 동일하게 적용.
        self._sync_scene_collision_objects(exclude_pos=pos, extra=getattr(self, '_extra_obstacles', None))
        self._extra_obstacles = None
        if is_side:
            # side는 좌표 시프트 재시도 대상이 아님(원본과 동일) --
            # 도달불가 판정은 여기서 즉시 'rejected'로 보고하고 종료.
            ok, reason = side_reachability_check(pos)
            if not ok:
                self.get_logger().warn(f'PLACE 거부: {reason}')
                raise SequenceRejected(reason)
        candidates = [dict(pos)]
        if not is_side:
            shifted_plus = dict(pos); shifted_plus['y'] = pos['y'] + 0.05
            shifted_minus = dict(pos); shifted_minus['y'] = pos['y'] - 0.05
            candidates += [shifted_plus, shifted_minus]
            if pos['z'] > 0.20:
                for dz in (0.05, 0.10, 0.15):
                    lowered = dict(pos); lowered['z'] = pos['z'] - dz
                    candidates.append(lowered)

        # [2026-09-02 추가, 사용자 지시 #3] top-down place는 후보 좌표를
        # 하나씩 실제로 approach 이동시켜보며 거르면 너무 느리다(후보당
        # Pilz 2회+OMPL 재시도 = 수십 초, 실측). 실제 모션 전에 각 후보의
        # approach pose IK만 ms 단위로 먼저 확인해서, IK가 아예 없는 후보는
        # _try_place_at 루프에 넣지 않는다. 전 후보가 IK 없으면 이 좌표는
        # top-down으로 원천 불가이므로(y시프트해도 마찬가지) 즉시 실패로
        # 반환해서 상위(mcp_robot_server/Claude)가 곧바로 grasp_dir='side'로
        # 재시도하게 한다 -- OMPL 재시도 예산을 후보 3개에 다 태우지 않는다.
        # sim(MoveIt) 모드 전용 -- 실물 모드는 _compute_ik_joints가 항상
        # None을 주므로(서비스 없음) 이 사전필터를 건너뛴다.
        if not is_side and self.use_moveit2:
            _reachable = []
            for _c in candidates:
                _probe_z = _c['z'] + APPROACH_Z + TOP_TCP_OFFSET
                _probe_quat = self._top_down_quat_for(_c, 0.0)
                if self._compute_ik_joints([_c['x'], _c['y'], _probe_z],
                                           _probe_quat, quiet=True) is not None:
                    _reachable.append(_c)
            if not _reachable:
                self.get_logger().error(
                    f'PLACE 사전필터: top-down approach IK 해가 후보 '
                    f'{len(candidates)}개 전부 없음 ({pos}) -- OMPL 재시도 생략, '
                    f"grasp_dir='side'로 재시도 권장")
                raise RuntimeError(
                    "top-down place approach IK 해 없음 (y시프트 후보 포함 전부) "
                    "-- 이 좌표는 top-down으로 도달 불가. grasp_dir='side'로 재시도하라.")
            if len(_reachable) < len(candidates):
                self.get_logger().info(
                    f'  PLACE 사전필터: 후보 {len(candidates)}개 중 IK 있는 '
                    f'{len(_reachable)}개만 시도')
            candidates = _reachable

        last_err = None
        for i, cand in enumerate(candidates):
            try:
                verification = self._try_place_at(
                    cand, quat, is_side, before_objects,
                    side_approach_offset_deg=side_approach_offset_deg,
                    is_pinch=is_pinch, retreat_mode=retreat_mode,
                    skip_verify=skip_verify)
                if i > 0:
                    self.get_logger().warn(
                        f'  PLACE: 원래 목표 {pos} 실패 -> 대체 좌표 {cand}로 성공')
                self.get_logger().info('✅ PLACE 완료')
                # [신설] verified=False여도 여기서 재시도하지 않는다 --
                # 그리퍼는 이미 물체를 놓고 물러난 상태라, 좌표를 바꿔서
                # 다시 움직여봐야 애초에 잡고 있는 물체가 없어 의미가
                # 없다. 검증 결과는 그대로 Claude에게 보고해서 판단을
                # 맡긴다 (재감지 후 다시 pick&place 하거나, 사용자에게
                # 알리거나).
                return {
                    'actual_place_pos': cand,
                    'placement_verified': verification['verified'],
                    'verification_reason': verification['reason'],
                    # [2026-09-02 #1] mcp_robot_server가 VLM 재검증을 걸지
                    # 말지 판단할 수 있게 수치 오차도 같이 넘긴다 (넓은
                    # 컨테이너면 xy 40~100mm 벗어남은 안착으로 봐도 됨).
                    'xy_error_m': verification.get('xy_error_m'),
                    'z_error_m': verification.get('z_error_m'),
                }
            except RuntimeError as e:
                last_err = e
                self.get_logger().warn(
                    f'  PLACE 후보 {i+1}/{len(candidates)} {cand} 실패: {e}')
                continue
        self.get_logger().error(f'PLACE 오류: 모든 후보 좌표 소진 ({last_err})')
        raise RuntimeError(f'모든 대체 좌표 실패: {last_err}')

    def _place_sequence(self, pos, quat, is_side=False, side_approach_offset_deg=0.0,
                         is_pinch=False, retreat_mode='lift', skip_verify=False):
        try:
            result_extra = self._do_place(pos, quat, is_side,
                                          side_approach_offset_deg=side_approach_offset_deg,
                                          is_pinch=is_pinch, retreat_mode=retreat_mode,
                                          skip_verify=skip_verify)
            # busy 해제를 발행보다 먼저 (레이스 방지 -- _scan_box_sequence 참고)
            with self.lock:
                self.busy = False
            self._publish_result('success', 'place_complete', extra=result_extra)
        except SequenceRejected as e:
            with self.lock:
                self.busy = False
            self._publish_result('rejected', str(e))
        except RuntimeError as e:
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
            self._clear_scene_collision_objects()

    def _get_observation_pose_joints(self):
        """[2026-08-31 추가] POSES_FILE에서 "observation" 저장 자세의 joint
        값을 읽어 _move_to_joint_config가 기대하는 순서(joint1~joint7)의
        리스트로 반환한다. 파일이 없거나 포즈가 없으면 None."""
        try:
            if not os.path.exists(POSES_FILE):
                return None
            with open(POSES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            joints = data.get('observation', {}).get('joints')
            if not joints:
                return None
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
            return [joints.get(jn, 0.0) for jn in joint_names]
        except Exception as e:
            self.get_logger().warn(f'[verify] observation 자세 로드 실패: {e}')
            return None

    def _move_to_joint_config(self, target_positions, label=''):
        """[신설] joint-space 목표로 이동하고 완료까지 대기, 성공 시 True.
        scan_box/pre-rotate/home과 동일 패턴(pipeline_id/planner_id 리셋
        포함 -- 이거 빠뜨리면 직전 Pilz 이동 설정이 남아 매번 즉시
        ABORTED되는 버그가 재발한다, 2026-08 scan_box에서 실측 확인된
        바로 그 문제). pose 기반 _move()와 달리 IK를 풀 필요가 없어서
        NO_IK_SOLUTION으로 실패할 위험이 없다 -- verify-vantage처럼
        "반드시 도달 가능해야 하는" 이동에 적합하다.
        """
        if not self.use_moveit2:
            return self._move_joints_real_wait(target_positions)
        self.moveit2.force_reset_executing_state()
        self.moveit2.pipeline_id = ''
        self.moveit2.planner_id = ''
        self.moveit2.max_velocity = 0.15
        self.moveit2.max_acceleration = 0.3
        self.moveit2.move_to_configuration(target_positions)
        time.sleep(0.5)
        deadline = time.time() + MOVE_TIMEOUT
        while time.time() < deadline:
            if (not self.moveit2._MoveIt2__is_motion_requested and
                    not self.moveit2._MoveIt2__is_executing):
                break
            time.sleep(0.1)
        ok = self.moveit2.motion_suceeded
        if not ok:
            self.get_logger().warn(f'  [{label}] joint-space 이동 실패/중단')
        return ok

    def _try_place_at(self, pos, quat, is_side=False, before_objects=None,
                      side_approach_offset_deg=0.0, is_pinch=False, retreat_mode='lift',
                      skip_verify=False):
        """실제 approach->descend->open->lift 시퀀스 1회 시도.
        기존 _place_sequence 본문과 100% 동일 (분리만 함). 실패 시
        self._publish_result를 직접 부르지 않고 RuntimeError만 던진다 --
        결과 보고는 바깥의 _place_sequence 재시도 루프가 전담한다.

        before_objects: place 시작 전 장면 스냅샷 (폐루프 검증용, 쌓기
        시나리오에서 베이스 박스 오탐 방지 -- placement_verification.py
        모듈 docstring 참고).

        [2026-08-26 추가] side/pinch 분기를 _do_pick과 동일한 방식으로
        재작성 -- 접근각 후보 탐색(offset 스윕) + standoff에서 직선 진입.
        이전엔 position_yaw 고정 반경 방향 TCP오프셋 + 단일 시도뿐이라
        pick에서 발견됐던 것과 동일한 버그(접근각 무시, 실패시 대안 없음,
        standoff 없이 바로 최종점行)를 그대로 갖고 있었다.
        """
        if not is_side:
            place_angle_deg = self._box_angle_deg if self._box_angle_deg is not None else 0.0
            px = pos['x']
            py = pos['y']
            approach_z = pos['z'] + APPROACH_Z + TOP_TCP_OFFSET
            descend_z = pos['z'] + PLACE_DROP_Z + TOP_TCP_OFFSET
            descend_z = max(descend_z, DESCEND_MIN_FLANGE_Z)
            lift_z = pos['z'] + LIFT_Z + TOP_TCP_OFFSET
        else:
            ok, reason = side_reachability_check(pos)
            if not ok:
                raise RuntimeError(reason)
            dx, dy = pos['x'], pos['y']
            position_yaw = math.atan2(dy, dx)
            _offset_candidates = [side_approach_offset_deg]
            for _d in (15.0, -15.0, 30.0, -30.0, 45.0, -45.0,
                       60.0, -60.0, 75.0, -75.0, 90.0, -90.0):
                _cand = side_approach_offset_deg + _d
                if _cand not in _offset_candidates:
                    _offset_candidates.append(_cand)
            if is_pinch:
                _quat_fn = sim_pinch_quat_for if self.use_moveit2 else pinch_quat_for
                _grip_name = 'pinch'
            else:
                _quat_fn = sim_side_quat_for if self.use_moveit2 else side_quat_for
                _grip_name = 'side'
            # [2026-08-31 추가] retreat_mode='lift'면 approach도 top-down처럼
            # 최종 xy 그대로 z만 올린 지점으로 간 뒤 LIN으로 수직 하강해야
            # 한다 -- retreat_mode='slide'(기존 동작)처럼 standoff에서
            # 수평으로 밀고 들어가면, 바구니/박스 벽처럼 옆으로 막힌 물체
            # 안에 놓을 때 접근 중에 그 벽을 치는 문제가 실측 확인됨.
            # descend_z는 그대로 두고(px,py도 아래에서 standoff 대신 최종
            # 지점 그대로 씀) approach_z만 올려서, "내려가기" 단계가 자동으로
            # 순수 수직 LIN이 되게 한다.
            approach_z = pos['z'] + (SIDE_LIFT_APPROACH_Z if retreat_mode == 'lift' else 0.0)
            descend_z = pos['z']
            lift_z = pos['z'] + LIFT_Z
        # [2026-07 삭제] place pre-rotate 제거. approach가 이미
        # yaw 자유(±91.7°)로 위치만 맞추면 되는 이동이라, 그 전에
        # joint1을 미리 돌려놓는 별도 스텝이 굳이 필요 없다고 판단됨
        # (pick 쪽 pre-rotate는 그대로 유지 -- approach의 목표
        # orientation이 top-down 고정이라 사정이 다름).
        self.get_logger().info('1/4: 접근 (approach)')
        if not is_side:
            entry_quat = self._top_down_quat_for(pos, 0.0)
            approach_tol_ori = (TOL_ORI_TIGHT, TOL_ORI_TIGHT, 1.6)
            ok = self._move(px, py, approach_z, entry_quat, tol_ori=approach_tol_ori, planner_chain=['PTP', 'LIN'])
            if not ok:
                raise RuntimeError('place approach 이동 실패 (재시도 초과)')
        else:
            ok = False
            entry_quat = quat
            ux = uy = None  # [2026-08-31 추가] 후퇴(slide-out) 방향 계산용, _do_slide와 동일 패턴
            _last_cand_target = _last_cand_quat = None
            for _i, _cand_offset in enumerate(_offset_candidates):
                _cand_yaw = position_yaw + math.radians(_cand_offset)
                _cux, _cuy = math.cos(_cand_yaw), math.sin(_cand_yaw)
                _cand_px = dx - _cux * SIDE_TCP_OFFSET
                _cand_py = dy - _cuy * SIDE_TCP_OFFSET
                if retreat_mode == 'lift':
                    # 최종 xy 그대로, z만 위(approach_z)에서 접근 -- standoff
                    # (수평 오프셋) 없음. descend가 그대로 수직 LIN이 됨.
                    _cand_wp_px, _cand_wp_py = _cand_px, _cand_py
                else:
                    _cand_wp_px = _cand_px - _cux * SIDE_APPROACH_STANDOFF_M
                    _cand_wp_py = _cand_py - _cuy * SIDE_APPROACH_STANDOFF_M
                _cand_quat = _quat_fn(pos, approach_offset_deg=_cand_offset)
                self.get_logger().info(
                    f'  [place {_grip_name} approach 후보 {_i+1}/{len(_offset_candidates)}] '
                    f'접근각={math.degrees(_cand_yaw):.1f}° (offset={_cand_offset:+.1f}°, '
                    f'retreat_mode={retreat_mode}) 시도...')
                _last_cand_target = [_cand_wp_px, _cand_wp_py, approach_z]
                _last_cand_quat = _cand_quat
                # [2026-08-31 추가] 실제 모션 전 IK 사전 필터 (_do_pick과 동일 이유)
                if self._compute_ik_joints([_cand_wp_px, _cand_wp_py, approach_z],
                                           _cand_quat, quiet=True) is None:
                    continue
                ok = self._move(_cand_wp_px, _cand_wp_py, approach_z, _cand_quat,
                                tol_ori=TOL_ORI_LOOSE, planner_chain=['PTP', 'LIN'])
                if ok:
                    entry_quat = _cand_quat
                    px, py = _cand_px, _cand_py
                    ux, uy = _cux, _cuy
                    if _cand_offset != side_approach_offset_deg:
                        self.get_logger().warn(
                            f'  [place {_grip_name} approach] 요청 접근각(offset='
                            f'{side_approach_offset_deg:+.1f}°)은 IK 실패 -- 대체 각도'
                            f'(offset={_cand_offset:+.1f}°)로 진행')
                    break
            if not ok:
                _diag = (self._diagnose_ik_failure(_last_cand_target, _last_cand_quat)
                        if _last_cand_target is not None else '진단 불가(후보 없음)')
                raise RuntimeError(
                    f'place approach 이동 실패 (재시도 초과, {_grip_name} 접근각 후보 전부 실패) '
                    f'[원인진단(마지막 후보 기준): {_diag}]')
        # align 단계(재이동)는 여전히 없음 -- approach 도착 orientation과
        # descend orientation을 어차피 동일하게 맞출 거라 별도 이동이
        # 불필요했고, 오히려 재시도로 인한 지연의 주범이었음.
        if not is_side:
            place_quat = entry_quat  # [2026-08 수정] descend를 approach와 동일 각도(0°)로 통일
                           # — 회전+하강 동시요구로 인한 덤블링/IK실패 방지.
                           # trade-off: 박스 각도 정렬 없이 항상 축정렬로 place됨
        else:
            place_quat = entry_quat  # [2026-08-26 수정] quat(요청값) 아니라 후보탐색에서
                                      # 실제로 성공한 entry_quat을 써야 함
        self.get_logger().info('3/4: 내려가기 (descend)')
        if not is_side:
            ok = self._move_descend_with_tilt_fallback(px, py, descend_z, place_quat)
        else:
            ok = self._move(px, py, descend_z, place_quat, tol_ori=TOL_ORI_TIGHT, planner_chain=['LIN'])
        if not ok:
            raise RuntimeError('place descend 이동 실패 (재시도 초과, tolerance 완화 폴백까지 실패)')
        self.get_logger().info('4/4: 그리퍼 열기')
        self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
        time.sleep(GRIPPER_DELAY)
        # [2026-08-31 추가] side/pinch는 place 직후 후퇴 방식을 두 가지로
        # 고를 수 있다 -- 하나로 무조건 고정하면 안 됨(실측 지적):
        #   'slide' — standoff에서 접근방향 직선(slide-in)으로 들어왔던 것과
        #             대칭으로, 접근방향(ux,uy) 반대로 xy만 바뀌고 z는 고정한
        #             채 수평 후퇴(_do_slide 후퇴 로직과 동일 원리). 위쪽에
        #             장애물이 있어 수직으로 못 빠질 때 적합.
        #   'lift'   — xy는 고정하고 z만 올리는, top-down의 lift와 동일 원리
        #             (기본값). 위쪽이 트인 경우 자연스러움.
        # top-down(not is_side)은 원래부터 lift 개념밖에 없어 이 선택지가
        # 없다 -- retreat_mode는 is_side일 때만 의미가 있다.
        if is_side and retreat_mode == 'slide':
            self.get_logger().info(
                f'  후퇴 (slide-out) — 접근방향({math.degrees(math.atan2(uy, ux)):.1f}°) '
                f'반대로 {SIDE_APPROACH_STANDOFF_M}m 직선 후퇴')
            retreat_px = px - ux * SIDE_APPROACH_STANDOFF_M
            retreat_py = py - uy * SIDE_APPROACH_STANDOFF_M
            ok = self._move(retreat_px, retreat_py, descend_z, place_quat,
                            tol_ori=TOL_ORI_LOOSE, planner_chain=['LIN'])
            if not ok:
                self.get_logger().warn('  place 후 slide-out 후퇴 실패 (place 자체는 성공, 무시하고 진행)')
        else:
            self.get_logger().info('  들어올리기 (lift) — 다음 동작 여유 확보')
            lift_candidates = [lift_z, pos['z'] + 0.06 + (0 if is_side else TOP_TCP_OFFSET),
                               pos['z'] + 0.03 + (0 if is_side else TOP_TCP_OFFSET)]
            ok = False
            achieved_lift_z = lift_z
            for i, lz in enumerate(lift_candidates):
                if i > 0:
                    self.get_logger().warn(
                        f'  place lift 높이 낮춰서 재시도: {lz:.3f}m (원래 목표 {lift_z:.3f}m)')
                ok = self._move(px, py, lz, place_quat, tol_ori=TOL_ORI_TIGHT, planner_chain=['LIN'])
                if ok:
                    achieved_lift_z = lz
                    break
            if not ok:
                self.get_logger().warn('  place 후 lift 모든 높이 재시도 실패 (place 자체는 성공, 무시하고 진행)')

        # 2026-09-18 추가(사용자 지시): 다중물체 스태킹처럼 place 직후 바로 다음
        # pick으로 넘어가는 자동화 루프에서는 이 폐루프 검증(observation 왕복 +
        # 카메라판정 + 애매하면 verify-vantage 왕복까지)이 매물체 수 초~수십 초를
        # 태우고, 게다가 verify-vantage 이후엔 observation으로 복귀도 안 해서
        # 다음 홈 복귀가 항상 풀 경로를 타게 만든다(재홈 스킵 무력화, 실측 확인).
        # skip_verify=True 면 이 블록 전체를 건너뛰고 place를 그 자리(lift 직후
        # 자세)에서 바로 끝낸다 -- verified 필드는 None(측정 안 함)으로 정직하게
        # 표시한다.
        if skip_verify:
            return {'verified': None, 'reason': 'skip_verify=True -- 검증 생략',
                    'xy_error_m': None, 'z_error_m': None}

        # [신설] 폐루프 검증 -- 그리퍼가 물러났으니 이제 카메라 시야가
        # 안 가려진 상태. perception이 새로 관측할 시간을 잠깐 준 뒤,
        # 목표 위치에 실제로 물체가 있는지 확인한다. 이 검사는 place
        # 시퀀스를 실패시키지 않는다(로봇 동작 자체는 이미 물리적으로
        # 완료됨) -- 대신 결과에 verified 여부를 실어서 Claude가
        # 다음 판단(재확인/재조정/사용자 보고)을 할 수 있게 한다.
        #
        # [2026-08-31 추가, 사용자 지시] 카메라로 판정하기 전에 항상
        # "observation" 저장 자세로 먼저 복귀한다 -- lift/후퇴 직후
        # 그 자리에서 바로 확인하면 근접 각도 인식 사각지대에 자주 걸려서
        # (실측 반복 확인) None 판정이 너무 잦았다. 이동 자체가 실패해도
        # (예: 옵저베이션 자세 파일 없음) 기존처럼 현재 위치에서 그냥
        # 확인을 진행한다 -- 이 이동은 검증 품질을 높이기 위한 것이지
        # place 시퀀스의 필수 단계가 아니다.
        _obs_joints = self._get_observation_pose_joints()
        if _obs_joints is not None:
            _moved_to_obs = self._move_to_joint_config(_obs_joints, label='verify-observation')
            if not _moved_to_obs:
                self.get_logger().warn('  [verify] 옵저베이션 자세 복귀 실패 -- 현재 위치에서 확인 진행')
        time.sleep(VERIFY_SETTLE_SEC)
        age = (time.time() - self._latest_objects_stamp
               if self._latest_objects_stamp > 0 else None)
        verification = verify_placement(
            pos, self.latest_objects, before_objects=before_objects, obj_age_sec=age)
        self.get_logger().info(f'  [verify] {verification["reason"]}')

        # [2026-08 재설계] 첫 검증이 애매하면(True가 아니면) 스캔에서 이미
        # 검증된 자세(DEFAULT_SCAN_BASE_JOINTS, conf 0.85~0.95로 안정적으로
        # 박스를 잡아온 실측 근거 있음)를 재사용해서 다시 본다. joint1만
        # 이번 물체 방향으로 역산해서 바꾸고 나머지 관절은 스캔값 그대로
        # 유지한다.
        #
        # [1차 시도, 폐기] pose 기반 _move()로 반경 바깥쪽 xy 오프셋을 주는
        # 방식을 먼저 시도했으나, 실측 결과 두 후보 다 NO_IK_SOLUTION
        # (error_code=-31)으로 즉시 실패함 (2026-08). pose 기반 이동은
        # IK를 풀어야 하는데, 그 근처 좌표에 항상 해가 있다는 보장이 없다.
        # joint-space 이동(_move_to_joint_config)은 애초에 IK를 풀 필요가
        # 없어서 이 문제가 원천적으로 발생하지 않는다 -- 도달 가능성이
        # 훨씬 확실한 접근으로 교체함.
        #
        # Claude Code hook으로는 이걸 할 수 없다 -- PostToolUse는 exit
        # code로 이미 반환된 결과를 바꿀 방법이 없고(도구 차단은
        # PreToolUse만 가능), 애초에 hook은 로봇을 직접 움직일 수도 없다.
        # 반드시 서버(planning_node) 안에서, Claude가 다시 판단할 필요
        # 없이 결정론적으로 처리해야 한다.
        if verification['verified'] is not True:
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4',
                           'joint5', 'joint6', 'joint7']
            target_bearing = math.atan2(pos['y'], pos['x']) - math.radians(BEARING_OFFSET_DEG)
            target_bearing = math.atan2(math.sin(target_bearing), math.cos(target_bearing))
            vantage_cfg = dict(DEFAULT_SCAN_BASE_JOINTS)
            vantage_cfg['joint1'] = target_bearing
            target_list = [vantage_cfg.get(jn, 0.0) for jn in joint_names]
            self.get_logger().info(
                f'  [verify] 1차 결과 애매함({verification["verified"]}) -> '
                f'스캔 자세 재사용, joint1={math.degrees(target_bearing):.1f}\xb0로 이동')
            moved = self._move_to_joint_config(target_list, label='verify-vantage')
            if moved:
                time.sleep(VERIFY_SETTLE_SEC)
                age = (time.time() - self._latest_objects_stamp
                       if self._latest_objects_stamp > 0 else None)
                retry_verification = verify_placement(
                    pos, self.latest_objects, before_objects=before_objects, obj_age_sec=age)
                self.get_logger().info(
                    f'  [verify] 재확인 결과: {retry_verification["reason"]}')
                verification = retry_verification
            else:
                self.get_logger().warn(
                    '  [verify] 재확인 자세 이동 실패, 1차 검증 결과 그대로 유지')
        return verification

    def _move_sequence(self, pos, quat, is_side=False):
        try:
            # [2026-08-26 수정, 사용자 요청] side/pinch는 joint1 사전회전
            # 완전 삭제 (_do_pick과 동일 이유 -- 위 주석 참고).
            if is_side:
                self.get_logger().info('[move pre-rotate] side/pinch 그립 -- 사전회전 스킵(설계상)')
            else:
                target_bearing = math.atan2(pos['y'], pos['x']) - math.radians(BEARING_OFFSET_DEG)
                target_bearing = math.atan2(math.sin(target_bearing), math.cos(target_bearing))
                js = self.latest_joint_state_sim
                if js is not None:
                    joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
                    current = dict(zip(js.name, js.position))
                    pre_target = [current.get(jn, 0.0) for jn in joint_names]
                    pre_target[0] = target_bearing
                    if abs(pre_target[1]) < 0.15 and abs(pre_target[3]) < 0.15:
                        pre_target[1] = 0.6
                        pre_target[3] = 1.0
                    self.get_logger().info(
                        f'[move pre-rotate] joint1 → {math.degrees(target_bearing):.1f}°')
                    if self.use_moveit2:
                        self.moveit2.force_reset_executing_state()
                        self.moveit2.max_velocity = 0.15
                        self.moveit2.max_acceleration = 0.3
                        self.moveit2.pipeline_id = ''
                        self.moveit2.planner_id = ''
                        self.moveit2.move_to_configuration(pre_target)
                        time.sleep(0.5)
                        deadline = time.time() + MOVE_TIMEOUT
                        while time.time() < deadline:
                            if (not self.moveit2._MoveIt2__is_motion_requested and
                                    not self.moveit2._MoveIt2__is_executing):
                                break
                            time.sleep(0.1)
                    else:
                        self._move_joints_real_wait(pre_target)
                else:
                    self.get_logger().warn('[move pre-rotate] joint_state 없음, 사전 회전 스킵')
            self.get_logger().info(
                f"1/1: 이동 → ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            ok = self._move(pos['x'], pos['y'], pos['z'], quat,
                            tol_ori=TOL_ORI_LOOSE)
            if not ok:
                raise RuntimeError('move 이동 실패 (재시도 초과)')
            self.get_logger().info('✅ MOVE 완료')
            # busy 해제를 발행보다 먼저 (레이스 방지 -- _scan_box_sequence 참고)
            with self.lock:
                self.busy = False
            self._publish_result('success', 'move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE 오류: {e}')
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
    def _publish_joint_positions(self, positions, joint_names=None):
        """조인트 이동 명령. /control/move_j 로 발행해 speed_percent와 모션플래닝을 따른다.
        (예전 /control/joint_states 는 직접 위치 쓰기라 속도제어/특이점 처리 없음.)"""
        joint_names = joint_names or ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = joint_names
        js.position = [float(p) for p in positions]
        self.pub_move_j.publish(js)
    def _move_joints_real_wait(self, positions, joint_names=None) -> bool:
        if not self._arm_status_available:
            self._publish_joint_positions(positions, joint_names)
            time.sleep(3.0)  # 확인 불가 시 예전 방식(고정 대기)으로 폴백
            return True
        for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
            self.get_logger().info(
                f'  → [실물] joint 이동 시도 {attempt}/{MOVE_MAX_ATTEMPTS}: '
                f'{[round(p,3) for p in positions]}')
            self.latest_arm_status = None
            publish_time = time.time()
            self._publish_joint_positions(positions, joint_names)
            time.sleep(REAL_MOVE_SETTLE_SEC)
            deadline = time.time() + REAL_MOVE_TIMEOUT
            last_status = None
            saw_in_progress = False
            while time.time() < deadline:
                status = self.latest_arm_status
                # publish 후 settle 기간 이내 수신된 상태는 이전 상태값 — 무시
                if status is not None and self.latest_arm_status_recv_time > publish_time + REAL_MOVE_SETTLE_SEC:
                    last_status = status
                    elapsed = time.time() - publish_time
                    if status.motion_status != 0:  # 비-SUCCESS = 이동 중
                        saw_in_progress = True
                    if status.motion_status == 0 and (saw_in_progress or elapsed > 2.0):
                        # 이동 완료: in_progress 후 SUCCESS, 또는 2s 경과 후 SUCCESS (빠른/가까운 이동 폴백)
                        suffix = '' if saw_in_progress else ' [폴백: 2s 경과]'
                        self.get_logger().info(
                            f'  ✓ [실물] joint 이동 완료 ({elapsed*1000:.0f}ms){suffix}')
                        return True
                    if saw_in_progress and status.motion_status == 1:  # FAILED
                        if status.arm_status != 0:
                            break
                time.sleep(REAL_MOVE_POLL_SEC)
            if last_status is not None:
                arm_s = ARM_STATUS_NAMES.get(last_status.arm_status, str(last_status.arm_status))
                motion_s = MOTION_STATUS_NAMES.get(last_status.motion_status, str(last_status.motion_status))
                self.get_logger().warn(
                    f'  [실물] joint 이동 실패/타임아웃: arm_status={arm_s} motion_status={motion_s}')
            else:
                self.get_logger().warn('  [실물] arm_status 수신 안 됨 (타임아웃)')
            if attempt < MOVE_MAX_ATTEMPTS:
                time.sleep(0.5)
        self.get_logger().error(
            f'[실물] joint 이동 최종 실패: {[round(p,3) for p in positions]} '
            f'[{MOVE_MAX_ATTEMPTS}회 모두 실패]')
        return False
    def _merge_scanned_box(self, obj):
        c3d = obj.get('center_3d', {})
        x, y, z = c3d.get('x'), c3d.get('y'), c3d.get('z')
        if x is None or y is None or z is None:
            return
        conf = obj.get('confidence', 0.0)
        entry = {
            'x': x, 'y': y, 'z': z,
            'confidence': conf,
            'angle_base_deg': obj.get('angle_base_deg'),
            'label': obj.get('label'),
        }
        for i, existing in enumerate(self._scanned_boxes):
            ex, ey, ez = existing['x'], existing['y'], existing['z']
            dist = math.sqrt((x-ex)**2 + (y-ey)**2 + (z-ez)**2)
            if dist < SCAN_DEDUP_DIST_M:
                if conf > existing.get('confidence', 0.0):
                    self._scanned_boxes[i] = entry
                    self.get_logger().info(
                        f'[스캔] 기존 박스 갱신(더 높은 conf): ({x:.3f},{y:.3f},{z:.3f}) conf={conf:.2f}')
                return
        self._scanned_boxes.append(entry)
        self.get_logger().info(
            f'[스캔] 새 박스 추가: ({x:.3f},{y:.3f},{z:.3f}) conf={conf:.2f}')
    def _scan_box_sequence(self, base_joints, step_deg, label):
        try:
            self._scanned_boxes = []
            joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
            step_rad = math.radians(step_deg)
            j1 = JOINT1_LOWER
            n_steps = 0
            # [2026-09-24 재수정] joint7을 observation 값(상한)부터
            # SCAN_JOINT7_SWEEP_LOWER(하한)까지 SCAN_JOINT7_SWEEP_STEPS개 절대값으로
            # 스윕(사용자 지시, 위 상수 주석 참고).
            j7_upper = float(base_joints.get('joint7', 0.0))
            n_j7 = max(SCAN_JOINT7_SWEEP_STEPS, 1)
            if n_j7 == 1:
                j7_sweep_values = [j7_upper]
            else:
                j7_sweep_values = [
                    SCAN_JOINT7_SWEEP_LOWER + i * (j7_upper - SCAN_JOINT7_SWEEP_LOWER) / (n_j7 - 1)
                    for i in range(n_j7)
                ]
            while j1 <= JOINT1_UPPER + 1e-6:
                for j7v in j7_sweep_values:
                    target = [base_joints.get(jn, 0.0) for jn in joint_names]
                    target[0] = j1
                    target[6] = j7v
                    self.get_logger().info(
                        f'[스캔] joint1={math.degrees(j1):.1f}\xb0 '
                        f'(joint7={math.degrees(target[6]):.1f}\xb0) 로 이동 중...')
                    if self.use_moveit2:
                        self.moveit2.force_reset_executing_state()
                        # [2026-08 수정] _home_sequence에서 이미 겪었던 것과 동일한
                        # 버그: 직전 명령(pick/place)이 Pilz(PTP/LIN)로 성공하면
                        # self.moveit2.pipeline_id/planner_id에 그 설정이 그대로
                        # 남는다. scan_box는 joint-space move_to_configuration을
                        # 쓰는데, 여기서 리셋을 안 하면 Pilz 설정이 그대로 재사용돼
                        # 매 스텝 100% 즉시 ABORTED되는 것이 실측 확인됨(2026-08,
                        # place 직후 scan_box 재호출 시 6스텝 전부 실패). 원인/수정
                        # 방식은 _home_sequence와 완전히 동일.
                        self.moveit2.pipeline_id = ''
                        self.moveit2.planner_id = ''
                        self.moveit2.max_velocity = 0.15
                        self.moveit2.max_acceleration = 0.3
                        self.moveit2.move_to_configuration(target)
                        time.sleep(0.5)
                        deadline = time.time() + MOVE_TIMEOUT
                        while time.time() < deadline:
                            if (not self.moveit2._MoveIt2__is_motion_requested and
                                    not self.moveit2._MoveIt2__is_executing):
                                break
                            time.sleep(0.1)
                        # [2026-08 수정, 코드리뷰 발견] 이전엔 이동 성공 여부를
                        # 확인 안 하고 바로 SCAN_STEP_SETTLE_SEC만큼 대기 후
                        # 검출 결과를 저장했음 -- 이동이 실패/중단됐으면 이
                        # 스텝에서 저장되는 박스는 의도한 joint1 각도가 아닌
                        # 다른(어중간한) 위치에서 관측된 것일 수 있음. TF 기반
                        # 좌표 계산이라 좌표 자체가 틀리진 않겠지만, 그 각도
                        # 구간의 커버리지가 비어있는 걸 놓치기 쉬워 경고만 남김
                        # (하드게이트 없이 계속 진행 -- handoff 6절 원칙).
                        if not self.moveit2.motion_suceeded:
                            self.get_logger().warn(
                                f'[스캔] joint1={math.degrees(j1):.1f}\xb0 이동 실패/중단 '
                                f'(motion_suceeded=False) -- 이 스텝의 검출 결과는 '
                                f'의도한 각도가 아닌 다른 위치에서 관측된 것일 수 있음.')
                    else:
                        self._move_joints_real_wait(target)
                    time.sleep(SCAN_STEP_SETTLE_SEC)
                    # [2026-08-26 수정, 사용자 요청] 이전엔 target_label(기본
                    # 'box')과 정확히 일치하는 물체만 기억하고 나머지는
                    # 스쳐지나가며 버렸다. YOLO dual 백엔드가 box 외에 COCO
                    # 클래스(cup, bottle 등)도 같이 인식하는데 스캔 중 놓치는
                    # 게 아까워서, 라벨 무관하게 이번 스윕에서 보인 물체를
                    # 전부 _scanned_boxes에 기억하도록 확장한다. target_label은
                    # 더 이상 필터링에 안 쓰이고 로그/응답 메시지용으로만 남음
                    # (하위호환 -- pick_object(from_scan=True)는 어차피 자기
                    # target_label로 다시 필터링해서 씀).
                    for obj in self.latest_objects:
                        self._merge_scanned_box(obj)
                    n_steps += 1
                j1 += step_rad
            _label_counts = {}
            for b in self._scanned_boxes:
                _label_counts[b.get('label')] = _label_counts.get(b.get('label'), 0) + 1
            self.get_logger().info(
                f'[스캔] 완료 ({n_steps}스텝), 누적 물체 {len(self._scanned_boxes)}개 '
                f'(라벨별: {_label_counts}): '
                f'{[(round(b["x"],3), round(b["y"],3), round(b["z"],3)) for b in self._scanned_boxes]}')
            # [2026-08 수정] busy 해제를 결과 발행보다 먼저 한다. 발행이
            # 먼저 나가면, 그 결과를 받은 클라이언트가 곧바로 다음 명령
            # (예: stack_boxes)을 보낼 때 busy=True가 아직 안 풀린 좁은
            # 시간창에 걸려 on_command가 새 명령을 조용히 버리는(결과 미발행)
            # 레이스가 있었다. 이 경우 클라이언트는 원인 불명의 "timeout"만
            # 받는다 -- "스캔 → 즉시 stack" 조합에서만 재현되고 턴을 나눠
            # 부르면(자연스러운 지연으로 레이스를 피해서) 안 재현되는 것으로
            # 실측 확인됨(2026-08).
            with self.lock:
                self.busy = False
            self._publish_result(
                'success', f'scan_complete:{len(self._scanned_boxes)}',
                extra={'boxes': self._scanned_boxes})
        except Exception as e:
            self.get_logger().error(f'SCAN 오류: {e}')
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
    def _move_joints_sequence(self, joints: dict):
        """joint space 직접 이동. _move_joints_sequence 중복 정의 제거됨."""
        try:
            joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
            key_map = {'j1':'joint1','j2':'joint2','j3':'joint3','j4':'joint4',
                       'j5':'joint5','j6':'joint6','j7':'joint7'}
            try:
                # [2026-08 수정] self.latest_joint_state는 실물 전용
                # /feedback/joint_states에서만 채워져 sim에서는 항상 None ->
                # 아래 except로 떨어져 미지정 관절이 전부 0으로 리셋되는 버그
                # 실측 확인됨. 코드베이스 다른 곳과 동일하게 latest_joint_state_sim
                # (/joint_states, 실물에서도 relay로 채워짐) 사용으로 통일.
                js_msg = self.latest_joint_state_sim
                name_to_pos = dict(zip(js_msg.name, js_msg.position))
                positions = [float(name_to_pos.get(jn, 0.0)) for jn in joint_names]
            except Exception:
                positions = [0.0] * 7
            for k, v in joints.items():
                jname = key_map.get(k, k)
                if jname in joint_names:
                    idx = joint_names.index(jname)
                    positions[idx] = float(v)
            self.get_logger().info(f'joint positions: {[round(p,3) for p in positions]}')
            if self.use_moveit2:
                # [2026-08 수정] scan_box/home과 동일한 버그 -- 직전 명령이
                # Pilz(PTP/LIN)로 성공하면 pipeline_id/planner_id가 남아서
                # 이 joint-space 이동이 항상 즉시 ABORTED될 수 있음.
                self.moveit2.force_reset_executing_state()
                self.moveit2.pipeline_id = ''
                self.moveit2.planner_id = ''
                self.moveit2.max_velocity = 0.15
                self.moveit2.max_acceleration = 0.3
                self.moveit2.move_to_configuration(positions)
                time.sleep(0.5)
                deadline = time.time() + MOVE_TIMEOUT
                while time.time() < deadline:
                    if (not self.moveit2._MoveIt2__is_motion_requested and
                            not self.moveit2._MoveIt2__is_executing):
                        break
                    time.sleep(0.1)
                # [2026-08 수정, 코드리뷰 발견] 이전엔 motion_suceeded를
                # 확인 안 하고 무조건 'success'를 퍼블리시했음 -- 이동이
                # 실제로 ABORTED돼도 MCP/사용자에게는 성공으로 보고돼서,
                # 그 뒤 명령들이 로봇이 도달 못 한 잘못된 전제(joint값)
                # 위에서 계산되는 위험이 있었음. 다른 액션들(home 등)과
                # 동일하게 명시적으로 확인.
                if not self.moveit2.motion_suceeded:
                    raise RuntimeError('move_joints 이동 실패 (motion_suceeded=False)')
            else:
                ok = self._move_joints_real_wait(positions)
                if not ok:
                    raise RuntimeError('move_joints 이동 실패 (재시도 초과)')
            # busy 해제를 발행보다 먼저 (레이스 방지 -- _scan_box_sequence 참고)
            with self.lock:
                self.busy = False
            self._publish_result('success', 'joint_move_complete')
        except Exception as e:
            self.get_logger().error(f'MOVE_JOINTS 오류: {e}')
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
    def _home_sequence(self):
        try:
            # [2026-09-02 변경, 사용자 지시 #4/#5] go_home의 복귀 목표를
            # all-zero 홈이 아니라 저장된 "observation"(지정) 자세로 바꾼다.
            # all-zero 홈은 카메라가 장면을 안 보는 자세라, go_home 직후
            # list_detected_objects/analyze_scene가 빈 결과를 주고(이번
            # 세션 실측) 매번 observation 자세로 다시 이동해야 했다.
            # observation을 기본 복귀 자세로 삼으면 그 왕복이 없어진다.
            # 저장 포즈가 없으면 안전하게 all-zero로 폴백한다.
            _obs = self._get_observation_pose_joints()
            if _obs is not None:
                target_cfg = _obs
                _pose_desc = 'observation(지정 자세)'
            else:
                target_cfg = [0.0] * 7
                _pose_desc = 'all-zero 홈(observation 저장 포즈 없음 -- 폴백)'
            self.get_logger().info(f'1/2: 복귀 이동 (joint space) -> {_pose_desc}')
            if self.use_moveit2:
                ok = False
                for attempt in range(1, 4):
                    self.moveit2.force_reset_executing_state()
                    # [2026-07 수정] pipeline_id/planner_id를 명시적으로
                    # OMPL로 리셋하지 않으면, 직전 이동이 Pilz(PTP/LIN)로
                    # 성공했을 경우 그 설정이 그대로 남아 조인트 스페이스
                    # 홈 이동에 Pilz LIN이 그대로 쓰이면서 매번 동일하게
                    # 즉시 ABORTED되는 현상이 실측 확인됨(handoff 3.2절과
                    # 같은 "공유 상태, 세팅 안 하면 이전 값 잔류" 패턴).
                    self.moveit2.pipeline_id = ''
                    self.moveit2.planner_id = ''
                    self.moveit2.max_velocity = 0.15
                    self.moveit2.max_acceleration = 0.3
                    self.moveit2.move_to_configuration(target_cfg)
                    time.sleep(0.5)
                    deadline = time.time() + MOVE_TIMEOUT
                    while time.time() < deadline:
                        if (not self.moveit2._MoveIt2__is_motion_requested and
                                not self.moveit2._MoveIt2__is_executing):
                            break
                        time.sleep(0.1)
                    ok = self.moveit2.motion_suceeded
                    if ok:
                        break
                    self.get_logger().warn(
                        f'  복귀 이동 실패/중단(시도 {attempt}/3), 재시도...')
                    time.sleep(0.5)
                if not ok:
                    raise RuntimeError('복귀 이동 실패 (재시도 초과)')
            else:
                while True:
                    ok = self._move_joints_real_wait(target_cfg)
                    if ok:
                        break
                    self.get_logger().warn('  복귀 조인트 이동 실패, 2초 후 재시도 (성공까지 무한 반복)...')
                    time.sleep(2.0)
            self._gripper(SIM_GRIPPER_OPEN, GRIPPER_OPEN)
            time.sleep(GRIPPER_DELAY)
            # [신규, 2026-08] settle delay -- 홈 직후 첫 pick_object가 매번
            # 타임아웃나던 문제의 근본 처방 (nero_robot_pick_after_home_timeout
            # memory, 2026-07-03 실측: get_system_status는 정상인데 첫 시도만
            # 타임아웃, 즉시 재시도하면 100% 성공 -- planning_node/IK서비스가
            # 아직 정착 중인 것으로 추정). busy 락이 풀리기 전에 넣어서, 다음
            # 명령이 이 시간만큼 자연스럽게 대기하도록 한다.
            time.sleep(1.5)
            self.get_logger().info(f'✅ 복귀 완료 ({_pose_desc})')
            # busy 해제를 발행보다 먼저 (레이스 방지 -- _scan_box_sequence 참고)
            with self.lock:
                self.busy = False
            self._publish_result('success', 'home_complete')
        except Exception as e:
            self.get_logger().error(f'HOME 오류: {e}')
            with self.lock:
                self.busy = False
            self._publish_result('failed', str(e))
        finally:
            with self.lock:
                self.busy = False
    # ── 핵심: 재시도 포함 _move() ─────────────────────────────────────────────
    def _compute_ik_joints(self, position, quat_xyzw, avoid_collisions: bool = True,
                            quiet: bool = False):
        """IK만 조회(실행 없음). 성공 시 dict{joint_name: pos}, 실패 시 None.

        avoid_collisions: [2026-08-31 추가] False로 주면 충돌 무시하고 순수
        기구학적 도달 가능성만 본다 -- error_code=-31(NO_IK_SOLUTION)이
        "진짜 도달 불가"인지 "충돌회피 때문에 막힘"인지 구분하는 진단용
        (_diagnose_ik_failure 참고). 실제 이동 전 안전 검사(기존 fail-fast)
        에는 반드시 True로 써야 한다 -- False는 진단 목적 외에 쓰지 말 것.
        quiet: True면 실패해도 warn 로그를 안 남긴다 -- 후보 스윕 중
        매 후보마다 찍히면 로그가 지저분해지므로 사전 필터링 용도로 씀.
        """
        try:
            from moveit_msgs.srv import GetPositionIK
            if not hasattr(self.moveit2, '_MoveIt2__compute_ik_client'):
                self.moveit2._MoveIt2__init_compute_ik()
            client = self.moveit2._MoveIt2__compute_ik_client
            if not client.service_is_ready():
                return None
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.moveit2._MoveIt2__group_name
            req.ik_request.pose_stamped.header.frame_id = self.moveit2._MoveIt2__base_link_name
            req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
            req.ik_request.pose_stamped.pose.position.x = float(position[0])
            req.ik_request.pose_stamped.pose.position.y = float(position[1])
            req.ik_request.pose_stamped.pose.position.z = float(position[2])
            req.ik_request.pose_stamped.pose.orientation.x = float(quat_xyzw[0])
            req.ik_request.pose_stamped.pose.orientation.y = float(quat_xyzw[1])
            req.ik_request.pose_stamped.pose.orientation.z = float(quat_xyzw[2])
            req.ik_request.pose_stamped.pose.orientation.w = float(quat_xyzw[3])
            req.ik_request.ik_link_name = 'gripper_flange'
            req.ik_request.avoid_collisions = avoid_collisions
            js = self.latest_joint_state_sim
            if js is not None:
                req.ik_request.robot_state.joint_state = js
            future = client.call_async(req)
            _deadline = time.time() + 5.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            if not future.done():
                if not quiet:
                    self.get_logger().warn(
                        '[ik_check] ⚠ compute_ik 타임아웃(1.0s 초과) — '
                        '바닥 콜리전 체크로 느려졌을 가능성')
                return None
            res = future.result()
            from moveit_msgs.msg import MoveItErrorCodes
            if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                if not quiet:
                    err_val = res.error_code.val if res is not None else 'res=None'
                    self.get_logger().warn(
                        f'[ik_check] ⚠ IK 실패, error_code={err_val} '
                        f'(NO_IK_SOLUTION=-31, GOAL_IN_COLLISION=-12 등 확인용, '
                        f'avoid_collisions={avoid_collisions})')
                return None
            sol = res.solution.joint_state
            return dict(zip(sol.name, sol.position))
        except Exception as e:
            if not quiet:
                self.get_logger().warn(f'[ik_check] 조회 실패: {e}')
            return None

    def _diagnose_ik_failure(self, position, quat_xyzw) -> str:
        """[2026-08-31 추가] fail-fast IK 사전조회(avoid_collisions=True)가
        실패했을 때, 충돌을 무시하면 풀리는지 다시 확인해서 "진짜 도달
        불가"와 "충돌회피 때문에 막힘"을 구분한다(실측 지적 -- 기존엔
        둘 다 error_code=-31로 뭉뚱그려져서 구분이 안 됐음). 진단
        목적으로만 1회 추가 호출하는 것이라 매 후보마다 부르면 느려짐
        -- 실패가 확정된 후보 하나당 한 번만 호출할 것."""
        no_collision_ik = self._compute_ik_joints(position, quat_xyzw,
                                                   avoid_collisions=False, quiet=True)
        if no_collision_ik is not None:
            return ('collision_blocked — 충돌회피를 무시하면 IK가 풀림. '
                    '충돌회피 로직(등록된 keep-out 오브젝트)이 이 pose를 막고 있음.')
        return ('truly_unreachable — 충돌 무시해도 IK 자체가 없음. '
                '순수 기구학적 도달 불가(작업공간/관절한계 밖).')
    def _pick_best_yaw_candidate(self, position, angle_deg, ik_check_z=None):
        """[2026-08 리팩터] 순수 비교 로직은 grasp_kinematics.YawCandidateSelector
        로 이관됨 (mod-90 후보 구성, FK 기반 회전량 비교 등 전체 설계/이력은
        grasp_kinematics.py 모듈 docstring + YawCandidateSelector.pick_best
        docstring 참고). 이 메서드는 ROS 서비스(IK 조회, FK 조회, backend별
        quat 생성)를 콜백으로 주입하는 얇은 어댑터로만 남는다.
        """
        current_fk = self._fk_gripper_pos()
        current_quat = current_fk[1] if current_fk is not None else None
        check_z = ik_check_z if ik_check_z is not None else position.get('z', 0.0)

        def _ik_check(x, y, z, cand_quat):
            # IK는 도달 가능성 확인용으로만 사용 (해 자체는 cost에 안 씀).
            # 검사 높이는 실제 이동 목표(approach_z 등)와 맞춰야 함
            # (2026-08 코드리뷰 발견 -- ik_check_z 파라미터 도입 배경).
            return self._compute_ik_joints([x, y, z], cand_quat)

        selector = YawCandidateSelector(
            ik_check_fn=_ik_check,
            quat_for_angle_fn=self._box_aligned_quat_for,
        )
        return selector.pick_best(
            position, angle_deg, current_quat,
            log_fn=lambda msg: self.get_logger().info(msg),
            ik_check_z=check_z,
        )
    def _check_state_validity(self, joint_state=None):
        """[2026-07 추가] START_STATE_IN_COLLISION(-31) 진단용.
        joint_state 생략 시 현재 상태(self.latest_joint_state_sim) 사용.
        (is_valid, contact_pairs, joint_limit_violations) 튜플 반환,
        조회 실패 시 (None, [], []).
        """
        try:
            from moveit_msgs.srv import GetStateValidity
            if not self._state_validity_client.service_is_ready():
                self.get_logger().warn('[validity_debug] service_is_ready()=False')
                return None, [], []
            js = joint_state if joint_state is not None else self.latest_joint_state_sim
            if js is None:
                self.get_logger().warn('[validity_debug] joint_state=None')
                return None, [], []
            req = GetStateValidity.Request()
            req.group_name = 'arm'
            req.robot_state.joint_state = js
            future = self._state_validity_client.call_async(req)
            _deadline = time.time() + 1.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None:
                self.get_logger().warn('[validity_debug] 서비스 타임아웃')
                return None, [], []
            contacts = [
                f'{c.contact_body_1}<->{c.contact_body_2} (depth={c.depth:.4f})'
                for c in res.contacts
            ]
            self.get_logger().warn(
                f'[validity_debug] valid={res.valid} contacts={contacts}')
            return res.valid, contacts, []
        except Exception as e:
            self.get_logger().warn(f'[validity_debug] 조회 실패: {e}')
            return None, [], []
    def _fk_for_joints(self, joint_names, joint_positions):
        """[2026-07 추가, 07 확장] 임의의 조인트값 목록에 대해
        gripper_flange의 base_link 기준 position+orientation을 조회.
        OMPL 궤적 웨이포인트 안전성 검사(position+orientation 둘 다)용.
        반환: (position_tuple(x,y,z), quat_tuple(x,y,z,w)) 또는 실패 시 None.
        [주의] 반환 타입이 튜플 1개(position만)에서 (position, quat) 쌍으로
        바뀜 -- 이 함수를 새로 호출하는 곳을 추가할 때 언패킹 잊지 말 것.
        """
        try:
            from moveit_msgs.srv import GetPositionFK
            from sensor_msgs.msg import JointState as JointStateMsg
            if not self._fk_client.service_is_ready():
                return None
            js = JointStateMsg()
            js.name = list(joint_names)
            js.position = list(joint_positions)
            req = GetPositionFK.Request()
            req.header.frame_id = 'base_link'
            req.header.stamp = self.get_clock().now().to_msg()
            req.fk_link_names = ['gripper_flange']
            req.robot_state.joint_state = js
            future = self._fk_client.call_async(req)
            _deadline = time.time() + 1.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None or not res.pose_stamped:
                return None
            p = res.pose_stamped[0].pose.position
            q = res.pose_stamped[0].pose.orientation
            return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
        except Exception as e:
            self.get_logger().warn(f'[fk_batch] 조회 실패: {e}')
            return None
    def _is_trajectory_reasonable(self, traj, max_ratio=1.8, max_ori_ratio=3.0):
        """[2026-07 추가, 07 확장] OMPL은 랜덤 샘플링 기반이라, 물리적으로는
        유효하지만 그리퍼가 목적지까지 크게 우회하는 경로를 찾을 때가
        있음이 실측 확인됨. execute 전에 궤적의 gripper_flange 경로
        총 길이를 시작-끝 직선거리와 비교해 걸러낸다.
        [2026-07 확장 배경] position 비율만으로는 못 잡는 사고가 실측
        확인됨: position 경로는 거의 직선(비율 1.24)인데, 그 안에서
        joint5/7이 왕복 1.5rad 이상 스윙하며 그리퍼가 순간적으로
        뒤집히는 궤적이 "정상"으로 통과해 실행된 사례 (place descend,
        2026-07 로그 대조로 확인). tol_ori는 시작/끝점에만 걸리고
        중간 웨이포인트의 orientation은 원래 자유였던 게 원인.
        position과 같은 방식(경로길이/직선거리 비율)을 orientation
        각도 거리에도 그대로 적용해서 우회를 잡는다. 하드 게이트가
        아니라 기존 재계획 루프(_move_try_ompl_safe)에 태워지는
        소프트 판정이라는 점은 동일하게 유지.
        max_ori_ratio=3.0은 초기 제안값, 실측 튜닝 필요(3.6절의
        max_ratio=1.8도 실측으로 정한 값이었음 참고).
        """
        points = traj.points
        if len(points) < 3:
            return True
        n = len(points)
        sample_idx = sorted(set([0, n - 1] + [int(i * (n - 1) / 6) for i in range(1, 6)]))
        positions_3d = []
        quats = []
        for idx2 in sample_idx:
            joint_pos = list(points[idx2].positions)
            fk = self._fk_for_joints(traj.joint_names, joint_pos)
            if fk is None:
                return True
            pos, quat = fk
            positions_3d.append(pos)
            quats.append(quat)
        path_length = sum(
            math.dist(positions_3d[i], positions_3d[i + 1])
            for i in range(len(positions_3d) - 1)
        )
        straight_dist = math.dist(positions_3d[0], positions_3d[-1])
        ratio = (path_length / straight_dist) if straight_dist >= 0.01 else 0.0
        reasonable_pos = (straight_dist < 0.01) or (ratio <= max_ratio)
        ori_path = sum(
            quat_angle_diff(quats[i], quats[i + 1])
            for i in range(len(quats) - 1)
        )
        ori_straight = quat_angle_diff(quats[0], quats[-1])
        # 목표 orientation 변화 자체가 거의 없는 이동(approach/descend처럼
        # tol_ori_tight로 orientation을 고정하는 단계)일수록 분모가
        # 0에 가까워 비율이 과민해지므로 최소 5도로 바닥을 둠.
        ori_ratio = ori_path / max(ori_straight, math.radians(5.0))
        reasonable_ori = ori_ratio <= max_ori_ratio
        self.get_logger().info(
            f'  [safety] 궤적 검사: 경로길이={path_length:.3f}m '
            f'직선거리={straight_dist:.3f}m 비율={ratio:.2f} | '
            f'orientation 경로={math.degrees(ori_path):.1f}\xb0 '
            f'직선={math.degrees(ori_straight):.1f}\xb0 비율={ori_ratio:.2f}')
        if not reasonable_ori and reasonable_pos:
            self.get_logger().warn(
                '  [safety] position은 정상이나 orientation 우회(뒤집힘 의심) 감지')
        return reasonable_pos and reasonable_ori
    def _fk_gripper_pos(self):
        """[2026-07 확장] 현재 조인트 상태로 gripper_flange의 실제
        base_link 기준 position+orientation을 조회. 실패 시 None 반환.
        반환: (position_tuple(x,y,z), quat_tuple(x,y,z,w))
        [주의] 예전엔 position 튜플 1개만 반환했음 -- 이 함수를 쓰는
        곳이 새로 생기면 반드시 언패킹할 것 (지금은 호출부 없음).
        """
        try:
            from moveit_msgs.srv import GetPositionFK
            if not self._fk_client.service_is_ready():
                self.get_logger().warn('[fk_debug] service_is_ready()=False')
                return None
            js = self.latest_joint_state_sim
            if js is None:
                self.get_logger().warn('[fk_debug] latest_joint_state=None')
                return None
            req = GetPositionFK.Request()
            req.header.frame_id = 'base_link'
            req.header.stamp = self.get_clock().now().to_msg()
            req.fk_link_names = ['gripper_flange']
            req.robot_state.joint_state = js
            future = self._fk_client.call_async(req)
            _deadline = time.time() + 1.0
            while not future.done() and time.time() < _deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None or not res.pose_stamped:
                self.get_logger().warn(
                    f'[fk_debug] future.done()={future.done()} res={res} '
                    f'pose_stamped={getattr(res, "pose_stamped", None)}')
                return None
            p = res.pose_stamped[0].pose.position
            q = res.pose_stamped[0].pose.orientation
            return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
        except Exception as e:
            self.get_logger().warn(f'[fk_debug] FK 조회 실패: {e}')
            return None
    def _move_descend_with_tilt_fallback(self, px, py, descend_z, quat_candidates) -> bool:
        """
        [2026-07 추가] descend 이동을 시도하되, 실패하면 top-down 유지정도
        (tol_ori의 x/y축)만 단계적으로 완화해서 재시도한다.
        z축(그립 각도 = 박스와의 좌우 정렬)은 절대 풀지 않는다 -- 이게
        풀리면 그리퍼가 박스와 어긋난 방향으로 닫혀서 파지가 틀어진다.
        (참고: tol_ori의 축 순서가 월드축인지 목표 quat 로컬축인지는
        pymoveit2 내부까지 확인 안 했음 -- approach 단계에서 이미
        3번째=yaw축만 따로 풀던 기존 패턴을 그대로 신뢰해서 확장한 것.
        실제로 돌려서 그리퍼가 박스와 어긋나게 닫히면 축 순서 가정이
        틀린 것이니 (0,1)과 (2)를 바꿔서 재검증 필요.)

        [2026-08-17 추가, 사용자 실측 발견] tolerance 완화만으로는 못
        푸는 실패(그립각도 자체가 이 지점에서 IK 불가)가 있는데, 이전엔
        그립각도를 절대 안 건드려서 재시도할 방법이 없었다. quat_candidates
        (보통 [현재각도, 현재각도+90도] — 박스 mod-90 대칭 이용)를 순서대로
        시도하되, 각 후보마다 위 tolerance 완화 단계를 전부 거친 뒤에만
        다음 후보로 넘어간다(성급하게 각도부터 바꾸지 않도록).

        Args:
            quat_candidates: quat 리스트(우선순위 순). 하위호환을 위해
                단일 quat(list[4])도 받는다.
        """
        if quat_candidates and not isinstance(quat_candidates[0], (list, tuple)):
            quat_candidates = [quat_candidates]
        for qi, cand_quat in enumerate(quat_candidates):
            if qi > 0:
                self.get_logger().warn(
                    f'  descend 그립각도 후보 전환 {qi}/{len(quat_candidates)-1}: '
                    f'90도 회전한 각도로 재시도 (박스 mod-90 대칭)')
            for i, tol in enumerate(DESCEND_TOL_ORI_FALLBACK):
                if i > 0:
                    self.get_logger().warn(
                        f'  descend tolerance 완화 재시도 {i}/{len(DESCEND_TOL_ORI_FALLBACK)-1}: '
                        f'top-down 여유={math.degrees(tol[0]):.1f}° (그립각도는 그대로 고정)')
                ok = self._move(px, py, descend_z, cand_quat, tol_ori=tol, planner_chain=['LIN'])
                if ok:
                    return True
        return False
    def _move(self, x: float, y: float, z: float, quat: list,
              tol_pos: float = TOL_POS_LOOSE,
              tol_ori=TOL_ORI_LOOSE, planner_id: str = '',
              planner_chain: list = None) -> bool:
        """
        MoveIt2로 목표 pose 이동. 실패 시 MOVE_MAX_ATTEMPTS 회 재시도.
        성공하면 True, 모든 시도 실패 시 False 반환.
        """
        if not self.use_moveit2:
            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.header.stamp    = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            if not self._arm_status_available:
                self.pub_move.publish(pose)
                return True
            for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
                self.get_logger().info(
                    f'  → [실물] move 시도 {attempt}/{MOVE_MAX_ATTEMPTS} '
                    f'({x:.3f},{y:.3f},{z:.3f}) quat={[round(v,3) for v in quat]}')
                self.latest_arm_status = None
                publish_time = time.time()
                pose.header.stamp = self.get_clock().now().to_msg()
                self.pub_move.publish(pose)
                time.sleep(REAL_MOVE_SETTLE_SEC)
                deadline = time.time() + REAL_MOVE_TIMEOUT
                last_status = None
                saw_in_progress = False
                while time.time() < deadline:
                    status = self.latest_arm_status
                    # publish 후 settle 기간 이내 수신된 상태는 이전 상태값 — 무시
                    if status is not None and self.latest_arm_status_recv_time > publish_time + REAL_MOVE_SETTLE_SEC:
                        last_status = status
                        elapsed = time.time() - publish_time
                        if status.motion_status != 0:  # 비-SUCCESS = 이동 중
                            saw_in_progress = True
                        if status.motion_status == 0 and (saw_in_progress or elapsed > 2.0):
                            # 이동 완료: in_progress 후 SUCCESS, 또는 2s 경과 후 SUCCESS (빠른 이동 폴백)
                            suffix = '' if saw_in_progress else ' [폴백: 2s 경과]'
                            self.get_logger().info(
                                f'  ✓ [실물] move 완료 ({elapsed*1000:.0f}ms){suffix}')
                            return True
                        if saw_in_progress and status.motion_status == 1:  # FAILED
                            if status.arm_status != 0:
                                break
                    time.sleep(REAL_MOVE_POLL_SEC)
                if last_status is not None:
                    arm_s = ARM_STATUS_NAMES.get(last_status.arm_status, str(last_status.arm_status))
                    motion_s = MOTION_STATUS_NAMES.get(last_status.motion_status, str(last_status.motion_status))
                    self.get_logger().warn(
                        f'  [실물] 이동 실패/타임아웃: arm_status={arm_s} motion_status={motion_s}')
                else:
                    self.get_logger().warn('  [실물] arm_status 수신 안 됨 (타임아웃)')
                if attempt < MOVE_MAX_ATTEMPTS:
                    time.sleep(0.5)
            # [2026-08 수정, 코드리뷰 발견] MOVE_MAX_ATTEMPTS회 모두
            # 실패했을 때 return이 없어서 아래 OMPL/Pilz 코드로 그대로
            # fall-through됐음. 실물 모드에서는 self.moveit2가 None이라
            # self._move_try_pilz(...)가 AttributeError를 던지고, 그게
            # 상위 try/except에 잡혀서 'failed'로 나가긴 했지만 원래
            # 실패 원인(arm_status 등)이 뭉개지고 'NoneType has no
            # attribute...'라는 뜻 모를 에러로 대체되는 문제였음.
            # _move_joints_real_wait()는 이 return을 이미 갖고 있었음
            # (여기만 누락).
            self.get_logger().error(
                f'[실물] move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) '
                f'[{MOVE_MAX_ATTEMPTS}회 모두 실패]')
            return False
        chain = list(planner_chain) if planner_chain else ([planner_id] if planner_id else [])
        for chain_planner in chain:
            if self._move_try_pilz(x, y, z, quat, tol_pos, tol_ori, chain_planner):
                return True
            self.get_logger().warn(f'  {chain_planner} 플래너 실패, 다음 플래너로 전환...')
        # [2026-08 추가] fail-fast: OMPL 3회(+내부 재계획 3회)까지 가기
        # 전에, 순수 IK로 "애초에 이 pose에 해가 있는가"만 먼저 빠르게
        # 확인한다. 해가 아예 없으면 OMPL을 반복해도 결과가 달라지지
        # 않는데(실측 로그: lift 실패 케이스 하나에 재시도 예산을 다
        # 태우는 데 약 57초 소요), 이 경우 즉시 실패로 반환해서 상위
        # (_pick_sequence의 lift_candidates 등)가 더 빨리 대체 전략으로
        # 넘어가게 한다.
        # 주의: 이건 "명백히 불가능"만 걸러내는 용도지 OMPL 재시도
        # 로직을 대체하지 않는다 -- IK가 성공해도 OMPL이 충돌 없는
        # 경로 자체를 못 찾아 실패하는 경우는 여전히 있을 수 있으므로,
        # 그 경우는 원래대로 아래 재시도 루프가 담당한다.
        quick_ik = self._compute_ik_joints([x, y, z], quat)
        if quick_ik is None:
            # [2026-08-31 추가] 실패 원인이 "진짜 도달 불가"인지 "충돌회피
            # 때문에 막힘"인지 진단 -- 실측 지적(둘 다 error_code=-31로
            # 뭉뚱그려져 구분 불가였음). 실패가 확정된 시점에 딱 1회만
            # 추가 조회하므로 매 후보 비용에 크게 안 보탠다.
            _diag = self._diagnose_ik_failure([x, y, z], quat)
            self.get_logger().warn(
                f'  [fail-fast] IK 사전조회 실패 -- 이 pose({x:.3f},{y:.3f},{z:.3f})는 '
                f'해가 없는 것으로 보임. OMPL 재시도 생략. [원인진단: {_diag}]')
            self.get_logger().error(
                f'_move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) [IK 사전조회 단계에서 조기 종료]')
            return False
        for attempt in range(1, MOVE_MAX_ATTEMPTS + 1):
            self.get_logger().info(
                f'  → OMPL 시도 {attempt}/{MOVE_MAX_ATTEMPTS} ({x:.3f},{y:.3f},{z:.3f})')
            if self._move_try_pilz(x, y, z, quat, tol_pos, tol_ori, ''):
                return True
            if attempt < MOVE_MAX_ATTEMPTS:
                time.sleep(1.0)
        self.get_logger().error(
            f'_move 최종 실패: ({x:.3f},{y:.3f},{z:.3f}) [모든 플래너 실패]')
        return False
    def _move_try_ompl_safe(self, x, y, z, quat, tol_pos, tol_ori, max_ratio=1.8) -> bool:
        self.moveit2.pipeline_id = ''
        self.moveit2.planner_id = ''
        self.moveit2.max_velocity = 0.15
        self.moveit2.max_acceleration = 0.3
        for replan_attempt in range(1, 4):
            self.moveit2.force_reset_executing_state()
            ready = self.moveit2._MoveIt2__move_action_client.server_is_ready()
            if not ready:
                self.get_logger().warn('  action server not ready, 잠시 대기...')
                time.sleep(1.0)
                continue
            future = self.moveit2.plan_async(
                position=[x, y, z], quat_xyzw=quat,
                tolerance_position=tol_pos, tolerance_orientation=tol_ori,
            )
            if future is None:
                return False
            _deadline = time.time() + MOVE_TIMEOUT
            while not future.done() and time.time() < _deadline:
                time.sleep(0.05)
            if not future.done():
                self.get_logger().warn('  [safety] OMPL plan 타임아웃')
                continue
            traj = self.moveit2.get_trajectory(future)
            if traj is None:
                self.get_logger().info('  [sim/OMPL] plan 실패')
                continue
            reasonable = self._is_trajectory_reasonable(traj, max_ratio=max_ratio)
            if not reasonable and replan_attempt < 3:
                self.get_logger().warn(
                    f'  [safety] 비정상 우회 경로 감지, 재계획... '
                    f'({replan_attempt}/3)')
                continue
            if not reasonable:
                self.get_logger().warn(
                    '  [safety] 3회 재계획해도 우회 경로만 나옴, 마지막 결과로 진행')
            time.sleep(0.2)
            self.moveit2.execute(traj)
            time.sleep(0.1)
            deadline = time.time() + MOVE_TIMEOUT
            motion_done = False
            while time.time() < deadline:
                req = self.moveit2._MoveIt2__is_motion_requested
                exe = self.moveit2._MoveIt2__is_executing
                if not req and not exe:
                    motion_done = True
                    break
                time.sleep(0.05)
            success = motion_done and self.moveit2.motion_suceeded
            self.get_logger().info(
                f'  [sim/OMPL] move → {x:.3f} {y:.3f} {z:.3f} | '
                f'{"✅ 성공" if success else "❌ 실패"}')
            if success:
                time.sleep(0.3)
                return True
        return False
    def _move_try_pilz(self, x, y, z, quat, tol_pos, tol_ori, planner_id: str) -> bool:
        """단일 플래너로 1회 시도. 성공 시 True."""
        if not planner_id:
            return self._move_try_ompl_safe(x, y, z, quat, tol_pos, tol_ori)
        self.moveit2.force_reset_executing_state()
        ready = self.moveit2._MoveIt2__move_action_client.server_is_ready()
        if not ready:
            self.get_logger().warn('  action server not ready, 잠시 대기...')
            time.sleep(1.0)
            return False
        if planner_id:
            self.moveit2.pipeline_id = 'pilz_industrial_motion_planner'
            self.moveit2.planner_id = planner_id
        else:
            self.moveit2.pipeline_id = ''
            self.moveit2.planner_id = ''
        self.moveit2.max_velocity = 0.15
        self.moveit2.max_acceleration = 0.3
        self.moveit2.move_to_pose(
            position=[x, y, z],
            quat_xyzw=quat,
            tolerance_position=tol_pos,
            tolerance_orientation=tol_ori,
            cartesian=False,
        )
        time.sleep(0.1)
        deadline = time.time() + MOVE_TIMEOUT
        motion_done = False
        while time.time() < deadline:
            req = self.moveit2._MoveIt2__is_motion_requested
            exe = self.moveit2._MoveIt2__is_executing
            if not req and not exe:
                motion_done = True
                break
            time.sleep(0.05)
        success = motion_done and self.moveit2.motion_suceeded
        error_code = self.moveit2.get_last_execution_error_code()
        err_str = f' [error_code={error_code.val}]' if (not success and error_code) else ''
        label = planner_id if planner_id else 'OMPL'
        self.get_logger().info(
            f'  [sim/{label}] move → {x:.3f} {y:.3f} {z:.3f} | '
            f'{"✅ 성공" if success else "❌ 실패"}{err_str}')
        if not success and error_code and error_code.val == -31:
            self._check_state_validity()
        if success:
            time.sleep(0.3)
        return success
    def _gripper(self, sim_joints, real_width):
        """
        그리퍼 제어.
        시뮬(use_moveit2=True):
          /gripper_controller/follow_joint_trajectory 액션 직접 호출
          SRDF: open=[0.05,-0.05], close=[0.0,0.0]
        실물(use_moveit2=False):
          /control/joint_states 퍼블리시 (agx_arm_ros 공식)
        """
        if self.use_moveit2:
            if float(sim_joints[0]) > 0.04:
                positions = [0.05, -0.05]  # open
                label = 'open'
            else:
                positions = [0.0, 0.0]     # close
                label = 'close'
            traj = JointTrajectory()
            traj.joint_names = ['gripper_joint1', 'gripper_joint2']
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.time_from_start = Duration(sec=2, nanosec=0)
            traj.points = [pt]
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            if not self._gripper_action.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn('[gripper] action server 없음, 스킵')
                return
            future = self._gripper_action.send_goal_async(goal)
            self.get_logger().info(
                f'[sim gripper] {label} → {positions} (follow_joint_trajectory)')
        else:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name         = ['gripper']
            msg.position     = [float(real_width)]
            msg.effort       = [GRIPPER_FORCE]
            self.pub_gripper.publish(msg)
            self.get_logger().info(
                f'[gripper] width={real_width:.3f}m force={GRIPPER_FORCE}N → /control/joint_states')

    def _gripper_holding_object(self):
        """[2026-08-31 추가] pick 성공/실패 판정을 그리퍼 조인트 값으로
        내린다(사용자 지시 -- 카메라 기반 verify_pick보다 우선 신호로 씀).
        SRDF상 완전 열림=[0.05,-0.05], 완전 닫힘=[0.0,0.0] -- 그리퍼가
        물체를 물면 물체 폭에 막혀 완전히 안 닫히고 그 사이 어딘가에서
        멈춘다. 둘 다 0에 가까우면(완전히 닫힘) 허공을 문 것으로 본다.

        sim(use_moveit2=True) 전용이다 -- 실물 경로는 그리퍼가 폭(width)
        하나로 명령되는 다른 스키마([control/joint_states, 'gripper'
        단일 값)라 이 판정 로직이 안 맞는다. 실물에서는 None(판정 불가)
        반환.

        Returns: (holding: bool|None, reason: str)
        """
        if not self.use_moveit2:
            return None, '실물 로봇 경로는 그리퍼 조인트 판정 미지원'
        js = self.latest_joint_state_sim
        if js is None:
            return None, '조인트 상태 없음 -- 판정 불가'
        try:
            jd = dict(zip(js.name, js.position))
            g1 = jd.get('gripper_joint1')
            g2 = jd.get('gripper_joint2')
        except Exception as e:
            return None, f'그리퍼 조인트 조회 실패: {e}'
        if g1 is None or g2 is None:
            return None, '그리퍼 조인트 값(gripper_joint1/2) 없음'
        if abs(g1) < GRIPPER_HOLDING_TOL_M and abs(g2) < GRIPPER_HOLDING_TOL_M:
            return False, (f'그리퍼가 완전히 닫힘(g1={g1:.4f}, g2={g2:.4f}) -- '
                           f'허공을 물었을 가능성 높음')
        return True, (f'그리퍼가 중간값에서 멈춤(g1={g1:.4f}, g2={g2:.4f}) -- '
                      f'물체 폭에 막혀 완전히 안 닫힌 것으로 보임, 물고 있을 가능성 높음')

    def _publish_result(self, status, reason='', extra=None):
        payload = {'status': status, 'reason': reason}
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_result.publish(msg)
def main():
    rclpy.init()
    node = PlanningNode()
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
