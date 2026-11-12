#!/usr/bin/env python3
"""
mcp_robot_server.py  (경로 패치)

[변경점 vs 이전 버전]
1. POSES_FILE 경로를 XDG_DATA_HOME 기반으로 변경
   - 이전: ~/sj/saved_poses.json  (Jetson Thor 전용 하드코딩)
   - 이후: ~/.local/share/nero_robot/saved_poses.json
   - 환경변수 NERO_POSES_FILE 로 오버라이드 가능 (기존 Jetson 경로 유지 시 사용)
   - Jetson Thor에서 기존 파일 마이그레이션 방법:
       cp ~/sj/saved_poses.json ~/.local/share/nero_robot/saved_poses.json

그 외 로직은 이전 버전(폐루프 패치)과 동일.
"""

import os
import re
import json
import math
import time
import threading
import logging
import subprocess
from typing import Optional

_logger = logging.getLogger(__name__)

import base64
import requests as _requests

import cv2
import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — PointStamped/Vector3Stamped 변환 등록용

from mcp.server.fastmcp import FastMCP

# [2026-08 추가, 프로토타입] YOLO가 못 잡은 물체(ground_object/infer_grasp의
# VLM 폴백 경로)에도 face normal yaw를 적용하기 위해 순수 함수만 재사용.
# perception_node.py는 별도 프로세스(ROS 노드)지만 같은 sj_pickplace 패키지
# 안의 순수 함수 import라 문제 없음 (perception_node_sim.py가 이미
# _compute_box_angle_base/_dedup_3d를 같은 방식으로 재사용 중).
from .perception_node import _fit_plane_normal
from .segmentation_backend import NoOpSegmentationBackend, get_default_backend
from .point_cloud import Intrinsics as _PCIntrinsics, mask_depth_to_pointcloud
from . import geometry_3d as _geometry_3d
from .grasp_types import GeometryResult as _GeometryResult
from .grasp_types import resolve_grasp_dir as _resolve_grasp_dir_policy
# [2026-09-02 추가] place 폐루프 검증 VLM 재확인 경로에서 순수 판정 로직 재사용
# (planning_node가 /detected_objects 기반으로 쓰는 것과 같은 함수).
from .placement_verification import verify_placement as _verify_placement_core

# ── 타임아웃 설정 ──────────────────────────────────────────────────────────────
TIMEOUT_PICK  = 75.0  # [2026-07 수정] IK 후보비교, 재조회 재시도,
# calib_debug tf lookup 등이 추가되며 pick 시퀀스가 예전보다 느려짐
# (실측: 로봇은 성공했는데 MCP가 31초 타임아웃으로 먼저 실패 처리한
# 사례 확인). 넉넉하게 상향.
TIMEOUT_PLACE = 60.0  # [2026-07 수정] place가 approach->align->descend
# 4단계 구조로 바뀌고 재시도까지 겹치면 40초 이상 걸리는 것이 실측
# 확인됨(16초는 MCP가 로봇 완료 전에 먼저 포기해서 중복 명령/재시도
# 루프를 만드는 원인이었음). pick과 비슷한 수준으로 넉넉하게 상향.
TIMEOUT_MOVE  = 11.0
TIMEOUT_HOME  = 16.0

# [2026-07 추가] 박스 실측 치수 지원 전까지 임시 고정값 (추후 A안:
# depth 기반 실측으로 교체 예정). 현재 테스트 환경 박스 높이 기준.
DEFAULT_BOX_HEIGHT_M = 0.05
TIMEOUT_SCAN  = 60.0  # joint1 스윕 시간 고려 (6스텝 * 약 8초)

# [2026-09-02 추가, 사용자 지시 #1] place 폐루프 검증에서, 1차(YOLO/토픽)가
# "물체는 목표 근처에서 찾았으나 xy 오차가 이 값 초과"로 False를 냈을 때
# 그 오차가 이 값 이하이면 VLM 재검증(~20초)을 생략한다. 바구니/트레이처럼
# 넓은 리셉터클에 담는 경우 중심에서 이 정도 벗어나도 안착으로 본다.
# verify_placement의 정밀 허용치(DEFAULT_XY_TOL_M=35mm)와는 별개 -- 그건
# "정확히 그 지점에 놓였나", 이건 "넓은 통 안에 들어갔나" 판정.
WIDE_CONTAINER_XY_TOL_M = 0.12

# ── VLM 추론 서버 주소 (vlm_grasp_server.py) ──────────────────────────────────
# 오버라이드: export VLM_SERVER_URL=http://<host>:8003  (예: YOLO/VLM을 별도
# 머신에서 돌리는 분리 배포 시 — perception_node.py의 BOX_SERVER_URL과 동일 패턴)
VLM_SERVER_URL = os.environ.get('VLM_SERVER_URL', 'http://127.0.0.1:8003').rstrip('/')

# ── 포즈 저장 경로 (XDG 기반 — Jetson/PC 모두 호환) ───────────────────────────
# 오버라이드: export NERO_POSES_FILE=~/sj/saved_poses.json  (Jetson 기존 경로 유지 시)
_DEFAULT_POSES_DIR  = os.path.join(
    os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share')),
    'nero_robot'
)
POSES_FILE = os.environ.get(
    'NERO_POSES_FILE',
    os.path.join(_DEFAULT_POSES_DIR, 'saved_poses.json')
)


def _load_poses() -> dict:
    if not os.path.exists(POSES_FILE):
        return {}
    try:
        with open(POSES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_poses(poses: dict):
    os.makedirs(os.path.dirname(POSES_FILE), exist_ok=True)
    with open(POSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(poses, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 1. ROS2 Bridge Node
# ──────────────────────────────────────────────────────────────────────────────
class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('mcp_robot_bridge')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_cmd = self.create_publisher(String, '/arm_command', qos)

        self.sub_obj = self.create_subscription(
            String, '/detected_objects', self._on_objects, qos)
        self.sub_result = self.create_subscription(
            String, '/pick_result', self._on_result, qos)
        # [2026-07 수정] /feedback/joint_states는 sim 환경에서 아무도
        # 발행하지 않는 실물 전용 토픽이라(오늘 FK/IK 조회 버그 원인으로
        # 실측 확인됨), get_joint_positions/save_pose/move_joints_relative가
        # 전부 실패하는 문제가 있었음. sim에서 실제로 살아있는
        # /joint_states로 교체.
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, qos)
        # infer_grasp용 카메라 이미지 버퍼 — camera device 직접 접근 없음,
        # perception_node가 발행하는 ROS2 topic만 구독한다.
        self.sub_image = self.create_subscription(
            Image, '/camera/color/image_raw', self._on_image, qos)
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_raw', self._on_depth, qos)
        self.sub_caminfo = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._on_camera_info, qos)

        # TF 버퍼 — ground_object에서 camera_color_optical_frame → base_link 변환
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._objects_lock = threading.Lock()
        self._latest_objects: list = []
        self._last_obj_stamp: float = 0.0

        self._image_lock = threading.Lock()
        self._latest_image: Optional[np.ndarray] = None  # BGR uint8
        self._last_image_stamp: float = 0.0

        self._depth_lock = threading.Lock()
        self._latest_depth: Optional[np.ndarray] = None  # float32 meters
        self._last_depth_stamp: float = 0.0

        self._caminfo_lock = threading.Lock()
        self._cam_fx: Optional[float] = None
        self._cam_fy: Optional[float] = None
        self._cam_cx: Optional[float] = None
        self._cam_cy: Optional[float] = None
        self._cam_width: int = 640
        self._cam_height: int = 480

        self._joint_lock = threading.Lock()
        self._latest_joint_state = None
        self._last_joint_stamp: float = 0.0

        self._result_lock   = threading.Lock()
        self._result_event:  Optional[threading.Event] = None
        self._result_payload: Optional[dict]           = None

        self.get_logger().info(
            f'RosBridgeNode 준비 완료 | POSES_FILE={POSES_FILE}')

    def _on_objects(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._objects_lock:
                self._latest_objects = data.get('objects', [])
                self._last_obj_stamp = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('detected_objects JSON 파싱 실패')

    def get_objects(self) -> tuple:
        with self._objects_lock:
            return list(self._latest_objects), self._last_obj_stamp

    def _on_joint_state(self, msg: JointState):
        with self._joint_lock:
            self._latest_joint_state = dict(zip(msg.name, msg.position))
            self._last_joint_stamp = time.time()

    def get_joint_state(self):
        with self._joint_lock:
            state = dict(self._latest_joint_state) if self._latest_joint_state else None
            return state, self._last_joint_stamp

    def _on_image(self, msg: Image):
        try:
            channels = len(msg.data) // (msg.height * msg.width)
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            if msg.encoding in ('rgb8', 'RGB8'):
                arr = arr[:, :, ::-1]  # RGB→BGR
            with self._image_lock:
                self._latest_image = arr.copy()
                self._last_image_stamp = time.time()
        except Exception as e:
            self.get_logger().warn(f'_on_image 처리 실패: {e}')

    def get_image(self) -> tuple:
        with self._image_lock:
            img = self._latest_image.copy() if self._latest_image is not None else None
            return img, self._last_image_stamp

    def _on_depth(self, msg: Image):
        try:
            if msg.encoding != "32FC1":
                return
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            with self._depth_lock:
                self._latest_depth = arr.copy()
                self._last_depth_stamp = time.time()
        except Exception as e:
            self.get_logger().warn(f'_on_depth 처리 실패: {e}')

    def get_depth(self) -> tuple:
        with self._depth_lock:
            d = self._latest_depth.copy() if self._latest_depth is not None else None
            return d, self._last_depth_stamp

    def _on_camera_info(self, msg: CameraInfo):
        with self._caminfo_lock:
            k = msg.k
            self._cam_fx = k[0]
            self._cam_fy = k[4]
            self._cam_cx = k[2]
            self._cam_cy = k[5]
            self._cam_width  = msg.width
            self._cam_height = msg.height

    def get_cam_intrinsics(self) -> tuple:
        with self._caminfo_lock:
            return (self._cam_fx, self._cam_fy,
                    self._cam_cx, self._cam_cy,
                    self._cam_width, self._cam_height)

    def _on_result(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('pick_result JSON 파싱 실패')
            return
        self.get_logger().info(f'/pick_result 수신: {payload}')
        with self._result_lock:
            self._result_payload = payload
            if self._result_event is not None:
                self._result_event.set()

    def wait_for_result(self, timeout: float) -> dict:
        event = threading.Event()
        with self._result_lock:
            self._result_payload = None
            self._result_event   = event
        arrived = event.wait(timeout=timeout)
        with self._result_lock:
            self._result_event = None
            result = self._result_payload
        if not arrived or result is None:
            return {'status': 'timeout',
                    'reason': f'{timeout:.0f}초 내 응답 없음 (planning_node busy 또는 타임아웃)'}
        return result

    def publish_command(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f'/arm_command 발행: {payload}')


# ──────────────────────────────────────────────────────────────────────────────
# 2. ROS2 백그라운드 스레드
# ──────────────────────────────────────────────────────────────────────────────
_ros_node: Optional[RosBridgeNode] = None
_ros_ready = threading.Event()
_ros_lock = threading.Lock()

# [2026-07 추가] 마지막 scan_for_boxes 결과 캐시. get_scanned_boxes가
# ROS 왕복 없이 즉시 응답할 수 있게 하기 위함 (로봇이 움직이지 않는
# 물체는 재스캔 없이도 이 캐시로 충분히 정확함).
_last_scanned_boxes: list = []
_last_scan_stamp: float = 0.0

# analyze_scene 결과 캐시 — hierarchical grounding에서 parent bbox 재사용
_last_scene_objects: list = []   # [{"label":..., "bbox":[...], "source":...}, ...]
_last_scene_stamp: float = 0.0
_scene_cache_lock = threading.Lock()


# ── ground_object 헬퍼 ────────────────────────────────────────────────────────

def _px2cam(u: float, v: float, depth_m: float,
            fx: float, fy: float, cx: float, cy: float) -> Optional[dict]:
    """픽셀 좌표 + depth → camera_color_optical_frame 3D 점."""
    if not (0.05 < depth_m < 3.0):
        return None
    return {
        "x": round((u - cx) * depth_m / fx, 4),
        "y": round((v - cy) * depth_m / fy, 4),
        "z": round(float(depth_m), 4),
    }


def _sample_depth_robust(depth: np.ndarray, x1: int, y1: int,
                          x2: int, y2: int) -> Optional[float]:
    """bbox 영역에서 valid depth (0.05–3.0m)의 median. 실패 시 None."""
    h, w = depth.shape
    x1c, x2c = max(0, x1), min(w, x2)
    y1c, y2c = max(0, y1), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return None
    valid = depth[y1c:y2c, x1c:x2c]
    valid = valid[(valid > 0.05) & (valid < 3.0)]
    if len(valid) < 3:
        # center window 폴백
        cy_px = (y1c + y2c) // 2
        cx_px = (x1c + x2c) // 2
        win = 15
        patch = depth[max(0, cy_px - win):min(h, cy_px + win),
                      max(0, cx_px - win):min(w, cx_px + win)]
        valid = patch[(patch > 0.05) & (patch < 3.0)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def _cam_to_base(cam_xyz: dict) -> Optional[dict]:
    """camera_color_optical_frame 점 → base_link 좌표. 실패 시 None."""
    try:
        pt = PointStamped()
        pt.header.frame_id = "camera_color_optical_frame"
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x = cam_xyz["x"]
        pt.point.y = cam_xyz["y"]
        pt.point.z = cam_xyz["z"]
        result = _ros_node.tf_buffer.transform(
            pt, "base_link", timeout=Duration(seconds=0.3))
        return {
            "x": round(result.point.x, 3),
            "y": round(result.point.y, 3),
            "z": round(result.point.z, 3),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# [2026-09-02] ground_object 컨테이너 뷰(joint7 스윕) + 잘림 경고(B) + depth(C)
# ─────────────────────────────────────────────────────────────────────────────
# 배경: observation 자세에서 회색 바구니가 이미지 오른쪽 밖으로 반쯤 잘려서,
# VLM이 "보이는 앞쪽 슬라이스"만 bbox로 잡고 그 중심을 3D화 → 바구니 중심
# x가 팔 쪽으로 80mm+ 당겨졌고 depth median도 0.77~0.91m로 출렁였다
# (2026-09-02 같은 바구니 3회 측정).
#
# [폐기된 시도 A -- joint1 재조준] 1차 근사좌표로 joint1을 돌려 물체를
# 화면 중앙에 넣고 재측정하려 했으나, joint1을 돌리면 _cam_to_base 결과의
# y가 0.33→0.53으로 계속 틀어지는 게 실측됨(2026-09-02). joint1/카메라
# 캘리브레이션 오차가 joint1이 0(기준점)에서 멀어질수록 커지는 것으로
# 추정 -- joint1 왕복 방식은 폐기.
#
# [현재 방식 -- joint1 정렬 + joint7 스윕, 사용자 지시 2026-09-02]
# observation 자세에서:
#  1) analyze_scene으로 (조금 잘린) 컨테이너 bbox를 잡고, 그 중심 x가
#     화면 중앙(0.5)에 오도록 joint1을 비주얼 서보(Δj1 = -GAIN*err, 최대
#     2회). joint1은 방위(좌우) 정렬 담당.
#  2) 그 자세에서 joint7을 j7_start(≈1.5491)부터 STOP(1.4)까지 STEP(0.1)로
#     내리며 프레이밍(어느 변도 안 닿음)이 되는 첫 자세를 찾는다. joint7은
#     상하/미세 프레이밍 담당(체인 끝이라 TF 왜곡이 joint1보다 작음).
#  3) 찾은 (joint1, joint7)에 arm을 둔 채 grounding, 끝나면 observation 복원.
#     (joint1, joint7)는 세션 캐시(_container_view_cache).
# B: bbox가 이미지 경계에 닿으면 bbox_truncated=True + confidence 하향.
#
# [폐기 -- joint1만으로 재조준] joint1을 근사좌표 방향으로 크게 돌리면
# _cam_to_base y가 0.33→0.53으로 드리프트(실측). 여기선 joint1을 "화면
# 중앙 정렬"에만 쓰고 최종 측정은 joint7 미세조정까지 한 자세에서 한다.
CONTAINER_VIEW_J7_START_RAD  = 1.5491   # None이면 observation의 joint7 사용
CONTAINER_VIEW_J7_STOP_RAD   = 1.4      # 여기까지 (하강)
CONTAINER_VIEW_J7_STEP_RAD   = 0.1
CONTAINER_VIEW_J1_GAIN       = 2.0      # Δjoint1 = -GAIN*(bbox_center_x_norm - 0.5)
CONTAINER_VIEW_J1_TOL        = 0.06     # |중심-0.5| 이 이내면 정렬 완료
CONTAINER_VIEW_J1_MAX_ITERS  = 2
CONTAINER_VIEW_J1_LIMIT_RAD  = 2.6      # joint1 물리 한계 ±2.705 안쪽
_container_view_cache: dict = {}        # {label_lower: {'joint1':.., 'joint7':..}}
# bbox 변이 이미지 경계에서 이미지 크기의 이 비율 이내면 "잘림"으로 본다.
# 절대 픽셀(3px)로 잡았더니 VLM이 프레임 끝에서 20~30px 짧게 bbox를
# 그리는 경우(회색 바구니 실측)를 못 잡았다 -- 해상도 무관하게 2%.
# 재조준은 실패해도 손해가 15초뿐이라 관대하게 잡는다.
GROUND_BBOX_EDGE_MARGIN_FRAC = 0.02


def _bbox_edge_sides(bbox_px, img_w: int, img_h: int) -> list:
    """bbox_px=[x1,y1,x2,y2](픽셀)가 이미지 경계에 닿은 변 목록.
    닿은 게 없으면 빈 리스트."""
    if not bbox_px or len(bbox_px) != 4:
        return []
    x1, y1, x2, y2 = bbox_px
    mx = max(3, int(img_w * GROUND_BBOX_EDGE_MARGIN_FRAC))
    my = max(3, int(img_h * GROUND_BBOX_EDGE_MARGIN_FRAC))
    sides = []
    if x1 <= mx:
        sides.append('left')
    if x2 >= img_w - 1 - mx:
        sides.append('right')
    if y1 <= my:
        sides.append('top')
    if y2 >= img_h - 1 - my:
        sides.append('bottom')
    return sides


def _goto_observation() -> bool:
    """observation 저장 자세로 이동(joint space). 성공 시 True."""
    obs = _load_poses().get('observation', {}).get('joints')
    if not obs:
        return False
    jm = {f'j{i}': float(obs[f'joint{i}']) for i in range(1, 8)
          if f'joint{i}' in obs}
    if not jm:
        return False
    _ros_node.publish_command({'action': 'move_joints', 'joints': jm})
    r = _ros_node.wait_for_result(timeout=TIMEOUT_HOME)
    return r.get('status') == 'success'


def _set_single_joint(key: str, val: float) -> bool:
    _ros_node.publish_command({'action': 'move_joints',
                               'joints': {key: round(float(val), 4)}})
    r = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    ok = r.get('status') == 'success'
    if ok:
        time.sleep(0.4)  # 카메라/depth 프레임 정착
    return ok


def _set_joint1(j1: float) -> bool:
    return _set_single_joint('j1', j1)


def _set_joint7(j7: float) -> bool:
    return _set_single_joint('j7', j7)


def _container_bbox_from_scene(target_label: str):
    """현재 시야에서 컨테이너 bbox(정규화)를 얻는다. analyze_scene(VLM)의
    objects 중 container_type != none 이거나 라벨이 컨테이너 키워드인 것.
    Returns (bbox_norm|None, label|None)."""
    scene = _analyze_scene_core()
    if scene.get('status') != 'success':
        return None, None
    cands = []
    for o in scene.get('objects', []):
        bb = o.get('bbox')
        if not bb or len(bb) != 4:
            continue
        lbl = str(o.get('label', ''))
        is_cont = (str(o.get('container_type', 'none')) != 'none'
                   or _looks_like_container(lbl))
        if is_cont:
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            cands.append((area, bb, lbl))
    if not cands:
        return None, None
    # 가장 큰 컨테이너 (여러 개면 타깃 라벨 매칭 우선)
    for _a, bb, lbl in sorted(cands, reverse=True):
        if target_label.lower() in lbl.lower() or lbl.lower() in target_label.lower():
            return bb, lbl
    cands.sort(reverse=True)
    return cands[0][1], cands[0][2]


def _find_container_view(target_label: str) -> Optional[dict]:
    """observation 자세에서 (1) joint1로 컨테이너 bbox 중심 x를 화면 중앙에
    정렬 → (2) joint7을 START→STOP까지 STEP으로 내리며 프레이밍(어느 변도
    안 닿음) 되는 첫 자세를 찾는다. 최종 (joint1, joint7)에 arm을 둔 채
    dict 반환. None이면 컨테이너를 한 번도 못 봄.
    Returns {'joint1','joint7','partial','detail', ...}."""
    if not _goto_observation():
        _logger.warning('[container_view] observation 이동 실패')
        return None
    obs = _load_poses().get('observation', {}).get('joints', {})
    j7_start = (CONTAINER_VIEW_J7_START_RAD
                if CONTAINER_VIEW_J7_START_RAD is not None
                else float(obs.get('joint7', 1.5491)))
    img0, _ = _ros_node.get_image()
    if img0 is None:
        return None
    H, W = img0.shape[:2]
    detail = {'j1': [], 'j7': []}

    # ── 1) joint1: bbox 중심 x → 화면 중앙 ──
    j1 = float(obs.get('joint1', 0.0))
    seen = False
    for it in range(CONTAINER_VIEW_J1_MAX_ITERS):
        bb, lbl = _container_bbox_from_scene(target_label)
        if bb is None:
            detail['j1'].append({'it': it, 'bbox': None})
            break
        seen = True
        cx = (bb[0] + bb[2]) / 2.0
        err = cx - 0.5
        detail['j1'].append({'it': it, 'j1': round(j1, 4),
                             'cx_norm': round(cx, 3), 'err': round(err, 3)})
        if abs(err) <= CONTAINER_VIEW_J1_TOL:
            break
        j1 = max(-CONTAINER_VIEW_J1_LIMIT_RAD,
                 min(CONTAINER_VIEW_J1_LIMIT_RAD,
                     j1 - CONTAINER_VIEW_J1_GAIN * err))
        if not _set_joint1(j1):
            _logger.warning(f'[container_view] joint1={j1:.3f} 이동 실패')
            break
    if not seen:
        _logger.warning(f'[container_view] {target_label!r} 컨테이너를 한 번도 못 봄')
        return None

    # ── 2) joint7 스윕: j7_start → STOP, STEP 하강 ──
    best = None  # (n_sides_touched, j7)
    j7 = j7_start
    while True:
        bb, lbl = _container_bbox_from_scene(target_label)
        if bb is not None:
            bpx = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
            sides = _bbox_edge_sides(bpx, W, H)
            detail['j7'].append({'j7': round(j7, 4), 'touched': sides or 'none'})
            _logger.info(f'[container_view] j7={j7:.3f} label={lbl!r} '
                         f'touched={sides or "none"}')
            if not sides:
                return {'joint1': round(j1, 4), 'joint7': round(j7, 4),
                        'partial': False, 'label': lbl, 'detail': detail}
            if best is None or len(sides) < best[0]:
                best = (len(sides), j7)
        else:
            detail['j7'].append({'j7': round(j7, 4), 'touched': 'not_detected'})
        j7 -= CONTAINER_VIEW_J7_STEP_RAD
        if j7 < CONTAINER_VIEW_J7_STOP_RAD - 1e-6:
            break
        if not _set_joint7(j7):
            _logger.warning(f'[container_view] joint7={j7:.3f} 이동 실패 -- 스윕 중단')
            break

    # 완전 프레이밍 못 찾음 -> best로 이동해서 거기서 측정
    if best is not None:
        _set_joint7(best[1])
        return {'joint1': round(j1, 4), 'joint7': round(best[1], 4),
                'partial': True, 'detail': detail}
    # joint7 스윕 내내 미검출 -> j1 정렬 자세, j7_start로 되돌려 측정
    _set_joint7(j7_start)
    return {'joint1': round(j1, 4), 'joint7': round(j7_start, 4),
            'partial': True, 'detail': detail, 'note': 'j7 스윕 중 미검출'}


# [2026-09-02 C] 컨테이너(바구니/트레이 등) 판별 + depth 샘플링.
_CONTAINER_KEYWORDS = (
    'basket', 'tray', 'bin', 'drawer', 'shelf', 'bowl', 'crate', 'container',
    'hamper', 'caddy', 'receptacle', 'box',  # box는 애매하지만 "빈 상자에 담아"류 포함
    '바구니', '트레이', '선반', '서랍', '바스켓', '통',
)
# 컨테이너 depth: bbox 전체에서 이 백분위(near 쪽). 열린 바구니는 bbox
# median이 "내부 바닥"(가장 깊음)으로 쏠려서 drop 지점이 카메라에서 멀리
# 밀린다(실측 2026-09-02: median 0.92 vs 실제 rim ~0.80). 낮은 백분위는
# "바구니에서 카메라에 가장 가까운 면 = 근처 rim/윗내용물" 쪽이라 opening
# 평면 높이에 더 가깝다.
_CONTAINER_DEPTH_PCT = 20.0


def _looks_like_container(*texts) -> bool:
    """라벨/설명 문자열들에 컨테이너 키워드가 있으면 True."""
    blob = ' '.join(str(t) for t in texts if t).lower()
    return any(k in blob for k in _CONTAINER_KEYWORDS)


def _sample_depth_container(depth: np.ndarray, x1: int, y1: int,
                            x2: int, y2: int) -> Optional[float]:
    """컨테이너용 depth: bbox 전체 valid depth 중 낮은 백분위(near surface).
    _sample_depth_robust(median)가 열린 통의 내부 바닥으로 쏠리는 걸 피한다.
    포인트가 너무 적으면 None (호출부가 robust median으로 폴백)."""
    h, w = depth.shape
    x1c, x2c = max(0, int(x1)), min(w, int(x2))
    y1c, y2c = max(0, int(y1)), min(h, int(y2))
    if x2c - x1c < 4 or y2c - y1c < 4:
        return None
    patch = depth[y1c:y2c, x1c:x2c]
    valid = patch[(patch > 0.05) & (patch < 3.0)]
    if valid.size < 30:
        return None
    return float(np.percentile(valid, _CONTAINER_DEPTH_PCT))


# [2026-08 추가, 프로토타입 -- 실기 미검증] YOLO가 못 잡은 물체(VLM
# grounding으로만 bbox를 얻은 경우)용 평면적합 face normal yaw.
# perception_node._compute_face_normal_yaw와 같은 수학(_fit_plane_normal
# 재사용)이지만, 이 파일 자체의 depth/intrinsics/tf 접근자(_px2cam,
# get_depth, get_cam_intrinsics, tf_buffer)로 다시 감싼 것 -- 프로세스가
# 달라서(mcp_robot_server는 perception_node의 ROS 노드 인스턴스에 접근
# 불가) 로직만 재사용하고 배선은 이 파일 자체 것을 쓴다.
FACE_NORMAL_GRID_STEP     = int(os.environ.get("FACE_NORMAL_GRID_STEP", "8"))
FACE_NORMAL_MIN_POINTS    = int(os.environ.get("FACE_NORMAL_MIN_POINTS", "25"))
FACE_NORMAL_MAX_POINTS    = int(os.environ.get("FACE_NORMAL_MAX_POINTS", "400"))
FACE_NORMAL_PLANARITY_MIN   = float(os.environ.get("FACE_NORMAL_PLANARITY_MIN", "0.7"))
FACE_NORMAL_VERTICALITY_MIN = float(os.environ.get("FACE_NORMAL_VERTICALITY_MIN", "0.5"))


def _compute_face_normal_yaw_from_bbox(bbox_px: list, depth: np.ndarray,
                                        fx: float, fy: float, cx: float, cy: float) -> tuple:
    """bbox_px([x1,y1,x2,y2]) 영역의 depth를 점군으로 역투영해 평면을 맞추고,
    그 평면 normal의 mod-180 방위각을 base_link 기준으로 반환한다.

    perception_node._compute_face_normal_yaw와 동일한 설계(평면성/수직성
    2단 게이트, verticality 낮으면 -- 즉 카메라가 물체 윗면처럼 수평에
    가까운 면을 보고 있으면 -- None 반환)이지만, 이 노드가 이미 들고 있는
    depth/intrinsics(_ros_node.get_depth/get_cam_intrinsics)와 _px2cam으로
    다시 구현했다. 값이 없으면(신뢰 불가) 무조건 None -- angle_base_deg 같은
    필드가 YOLO 못 잡은 물체엔 원래 없었으니, 실패 시에도 기존 응답 스키마를
    깨지 않는다(단순히 필드가 null로 채워짐).

    Returns:
        (yaw_deg, confidence, n_points) -- perception_node 버전과 동일 계약.
    """
    if depth is None or fx is None:
        return None, 0.0, 0

    h, w = depth.shape[:2]
    x1 = max(0, int(bbox_px[0])); y1 = max(0, int(bbox_px[1]))
    x2 = min(w, int(bbox_px[2])); y2 = min(h, int(bbox_px[3]))
    if x2 - x1 < FACE_NORMAL_GRID_STEP or y2 - y1 < FACE_NORMAL_GRID_STEP:
        return None, 0.0, 0

    points_cam = []
    for py in range(y1, y2, FACE_NORMAL_GRID_STEP):
        for px in range(x1, x2, FACE_NORMAL_GRID_STEP):
            depth_m = float(depth[py, px])
            pt = _px2cam(float(px), float(py), depth_m, fx, fy, cx, cy)
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

    try:
        v = Vector3Stamped()
        v.header.frame_id = "camera_color_optical_frame"
        v.header.stamp = rclpy.time.Time().to_msg()
        v.vector.x, v.vector.y, v.vector.z = [float(c) for c in normal_cam]
        v_base = _ros_node.tf_buffer.transform(v, "base_link", timeout=Duration(seconds=0.3))
        nx, ny = v_base.vector.x, v_base.vector.y
    except Exception:
        return None, round(planarity, 3), len(points_cam)

    verticality = math.hypot(nx, ny)
    confidence = round(planarity * verticality, 3)
    if verticality < FACE_NORMAL_VERTICALITY_MIN:
        return None, confidence, len(points_cam)

    yaw_deg = math.degrees(math.atan2(ny, nx)) % 180.0
    return round(yaw_deg, 1), confidence, len(points_cam)


def _rotate_vec_to_base(vec) -> Optional[tuple]:
    """카메라 좌표계 방향벡터(위치 아님) 하나를 base_link로 "회전만" 변환.
    _compute_face_normal_yaw_from_bbox와 동일 Vector3Stamped 패턴 -- 여러
    지점에서 반복되던 걸 함수로 뽑음."""
    if vec is None:
        return None
    try:
        v = Vector3Stamped()
        v.header.frame_id = "camera_color_optical_frame"
        v.header.stamp = rclpy.time.Time().to_msg()
        v.vector.x, v.vector.y, v.vector.z = [float(c) for c in vec]
        v_base = _ros_node.tf_buffer.transform(v, "base_link", timeout=Duration(seconds=0.3))
        return (v_base.vector.x, v_base.vector.y, v_base.vector.z)
    except Exception:
        return None


def _geometry_to_base_link(geometry, depth_quality: float = 1.0):
    """geometry_3d.compute_geometry()가 camera_color_optical_frame 점군에서
    낸 GeometryResult(축/normal이 전부 카메라 좌표계)를 base_link 기준으로
    다시 감싼다. 점 수만 개를 전부 재변환하는 대신, 이미 계산된 축/normal
    (벡터 3~4개)과 centroid(점 1개)만 변환한다 -- 점군 자체를 base_link로
    옮긴 뒤 geometry_3d를 다시 돌리는 것과 수학적으로 동일하지만 훨씬 싸다
    (회전 변환은 선형이라 축소환 순서를 바꿔도 결과가 같음).

    실패(TF 불가 등)하면 valid=False인 빈 GeometryResult를 반환한다 --
    호출부가 무조건 뭔가를 받되, 유효성은 반드시 .valid로 확인해야 한다."""
    from dataclasses import replace as _dc_replace

    if geometry is None or not geometry.valid:
        return _GeometryResult(valid=False)

    centroid_base = _cam_to_base({'x': geometry.centroid[0], 'y': geometry.centroid[1],
                                  'z': geometry.centroid[2]})
    major_base = _rotate_vec_to_base(geometry.major_axis)
    minor_base = _rotate_vec_to_base(geometry.minor_axis)
    third_base = _rotate_vec_to_base(geometry.third_axis)
    normal_base = _rotate_vec_to_base(geometry.plane_normal) if geometry.plane_normal else None

    if centroid_base is None or major_base is None:
        return _GeometryResult(valid=False, point_count=geometry.point_count,
                               filtered_point_count=geometry.filtered_point_count)

    def _yaw(vec):
        if vec is None:
            return None
        h = math.hypot(vec[0], vec[1])
        if h < 0.5:
            return None
        return round(math.degrees(math.atan2(vec[1], vec[0])) % 180.0, 1)

    return _dc_replace(
        geometry,
        centroid=(centroid_base['x'], centroid_base['y'], centroid_base['z']),
        major_axis=major_base, minor_axis=minor_base, third_axis=third_base,
        plane_normal=normal_base,
        major_axis_yaw_deg=_yaw(major_base),
        normal_yaw_deg=_yaw(normal_base),
    )


def _crop_b64(img_bgr: np.ndarray, bbox_norm: list, padding: float = 0.07) -> tuple:
    """parent bbox 영역을 crop하고 base64 JPEG + crop pixel bbox 반환.
    padding: bbox 주변에 추가할 여유 비율 (기본 7%)"""
    h, w = img_bgr.shape[:2]
    x1n, y1n, x2n, y2n = bbox_norm
    pw = (x2n - x1n) * padding
    ph = (y2n - y1n) * padding
    x1n = max(0.0, x1n - pw)
    y1n = max(0.0, y1n - ph)
    x2n = min(1.0, x2n + pw)
    y2n = min(1.0, y2n + ph)
    x1, y1 = max(0, int(x1n * w)), max(0, int(y1n * h))
    x2, y2 = min(w, int(x2n * w)), min(h, int(y2n * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError(f'crop 너무 작음: {x2-x1}x{y2-y1}px')
    ok, buf = cv2.imencode('.jpg', img_bgr[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError('crop JPEG 인코딩 실패')
    return base64.b64encode(buf.tobytes()).decode(), [x1, y1, x2, y2]


def _restore_bbox(child_norm: list, crop_norm: list) -> list:
    """crop 기준 child bbox → 원본 이미지 normalized 좌표 복원.
    crop_norm: 원본 이미지에서의 실제 crop 영역 [x1,y1,x2,y2] (padding 포함)"""
    cx1, cy1, cx2, cy2 = child_norm
    px1, py1, px2, py2 = crop_norm
    pw, ph = px2 - px1, py2 - py1
    return [
        round(px1 + cx1 * pw, 4),
        round(py1 + cy1 * ph, 4),
        round(px1 + cx2 * pw, 4),
        round(py1 + cy2 * ph, 4),
    ]


def _label_matches_parent(candidate: str, query: str) -> bool:
    """candidate label이 query parent_label에 의미적으로 매칭되는지 확인.
    'yellow drawer' ↔ 'drawer' 같은 core word overlap 허용.
    과도한 fuzzy matching은 배제."""
    c = candidate.strip().lower()
    q = query.strip().lower()
    if c == q:
        return True
    q_words = set(q.split())
    c_words = set(c.split())
    common = q_words & c_words
    # query 단어의 절반 이상이 candidate에 포함될 때만 허용
    return len(common) >= max(1, len(q_words) // 2)


def _is_bbox_oversized(bbox_norm: list, threshold: float = 0.8) -> bool:
    """bbox 면적이 이미지의 threshold 이상이면 True (too coarse 경고용)."""
    x1, y1, x2, y2 = bbox_norm
    return (x2 - x1) * (y2 - y1) > threshold


def _make_child_target(target_label: str, parent_label: str) -> str:
    """child grounding용 target_label 생성.
    handle/knob/pull/grip 류는 내부 부품과 구분되도록 상세 설명 포함."""
    target_lower = target_label.strip().lower()
    if any(kw in target_lower for kw in ('handle', 'knob', 'pull', 'grip')):
        return (
            f"Front-facing {target_label} of the {parent_label}. "
            f"The physical handle that a person grabs to open/pull — "
            f"a horizontal bar or protruding grip on the EXTERIOR FRONT FACE of the {parent_label}. "
            f"NOT: internal rails, drawer slides, brackets, hinges, screws, "
            f"or any mechanical component inside the drawer cavity."
        )
    return f"{target_label} (part of {parent_label})"


def _ground_hierarchical(target_label: str, parent_label: str,
                          ground_url: str, vlm_timeout: float,
                          min_conf: float) -> str:
    """parent object crop → child part grounding → depth → TF → 3D (2D fallback).

    Parent bbox 우선순위:
      1. analyze_scene 캐시 (가장 최근 scene 결과)
      2. YOLO detection
      3. VLM /ground_object(parent_label) — 최후 수단
    """

    # ── A. 이미지 획득 ─────────────────────────────────────────────────────────
    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'success': False, 'reason': 'image_unavailable'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'success': False, 'reason': f'image_stale — {img_age:.1f}s'})

    h, w = img_bgr.shape[:2]

    def _encode(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    # ── B. Parent bbox 확보 ────────────────────────────────────────────────────
    # Priority 1: analyze_scene 캐시 (가장 최근 perception 결과)
    # Priority 2: YOLO detection
    # Priority 3: VLM /ground_object(parent_label) — 최후 수단

    _logger.info(f'[ground_object] target={target_label!r}  parent={parent_label!r}')

    parent_source = None
    parent_bbox_norm = None

    # Priority 1: analyze_scene 캐시 — VLM parent grounding 호출 없이 재사용
    with _scene_cache_lock:
        scene_objs = list(_last_scene_objects)
        scene_age  = time.time() - _last_scene_stamp

    if scene_objs and scene_age < 60.0:
        scene_matched = [o for o in scene_objs
                         if _label_matches_parent(str(o.get('label', '')), parent_label)]
        if scene_matched:
            # 면적 큰 순 우선 — parent는 child를 포함하는 더 넓은 영역이어야 함
            scene_matched.sort(
                key=lambda o: (
                    (o.get('bbox', [0, 0, 1, 1])[2] - o.get('bbox', [0, 0, 1, 1])[0]) *
                    (o.get('bbox', [0, 0, 1, 1])[3] - o.get('bbox', [0, 0, 1, 1])[1])
                ),
                reverse=True,
            )
            for sc in scene_matched:
                b = sc.get('bbox', [])
                if isinstance(b, list) and len(b) == 4:
                    cand = [float(v) for v in b]
                    area = (cand[2] - cand[0]) * (cand[3] - cand[1])
                    if area < 0.01:
                        _logger.warning(f'[parent] scene cache bbox too small {cand} (area={area:.4f}) — trying next candidate')
                        continue
                    if _is_bbox_oversized(cand):
                        _logger.warning(f'[parent] scene cache bbox oversized {cand} — trying next candidate')
                        continue
                    parent_bbox_norm = cand
                    parent_source = 'scene'
                    _logger.info(
                        f'[parent] reused existing analyze_scene bbox '
                        f'({sc.get("label")}): {cand}'
                    )
                    break
            # 모두 oversized여도 YOLO/VLM보다 scene 결과가 더 신뢰할 수 있으므로 최후 수단으로 사용
            if parent_bbox_norm is None and scene_matched:
                b = scene_matched[0].get('bbox', [])
                if isinstance(b, list) and len(b) == 4:
                    parent_bbox_norm = [float(v) for v in b]
                    parent_source = 'scene'
                    _logger.warning(
                        f'[parent] all scene candidates oversized — '
                        f'using smallest ({scene_matched[0].get("label")}): {parent_bbox_norm}'
                    )

    # Priority 2: YOLO
    if parent_bbox_norm is None:
        objects, _ = _ros_node.get_objects()
        yolo_matched = [o for o in objects
                        if _label_matches_parent(str(o.get('label', '')), parent_label)]
        if yolo_matched:
            b = yolo_matched[0].get('bbox', [])
            if isinstance(b, list) and len(b) == 4:
                cand = [float(v) for v in b]
                if _is_bbox_oversized(cand):
                    _logger.warning(f'[parent] YOLO bbox oversized {cand}')
                parent_bbox_norm = cand
                parent_source = 'yolo'
                _logger.info(f'[parent] using YOLO bbox ({yolo_matched[0].get("label")}): {cand}')
    else:
        # scene 경로에서는 objects 미조회 — VLM fallback용으로 lazy 조회
        objects = None

    # Priority 3: VLM parent grounding — 최후 수단
    if parent_bbox_norm is None:
        _logger.info(f'[parent] no scene/YOLO match — calling VLM ground_object({parent_label!r})')
        if objects is None:
            objects, _ = _ros_node.get_objects()
        try:
            full_b64 = _encode(img_bgr)
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'encode_failed — {e}'})

        yolo_detections = [
            {'label': o.get('label', '?'), 'bbox': o.get('bbox', []),
             'confidence': round(float(o.get('confidence', 0.0)), 3)}
            for o in objects
        ]
        try:
            resp = _requests.post(ground_url,
                                  json={'full_image_b64': full_b64,
                                        'target_label': parent_label,
                                        'detections': yolo_detections,
                                        'timestamp': time.time()},
                                  timeout=vlm_timeout)
        except _requests.exceptions.ConnectionError:
            return json.dumps({'success': False, 'reason': 'vlm_server_unavailable'})
        except _requests.exceptions.Timeout:
            return json.dumps({'success': False, 'reason': f'vlm_timeout — {vlm_timeout}s 초과'})
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'vlm_request_failed — {e}'})

        if resp.status_code != 200:
            return json.dumps({'success': False,
                               'reason': f'vlm_http_error — {resp.status_code}'})
        try:
            pdata = resp.json()
        except Exception:
            return json.dumps({'success': False, 'reason': 'vlm_invalid_json'})

        if not pdata.get('found', False):
            return json.dumps({'success': False, 'reason': 'parent_not_found'})

        p_conf = float(pdata.get('confidence', 0.0))
        if p_conf < min_conf:
            return json.dumps({'success': False,
                               'reason': f'parent_low_confidence — {p_conf:.2f} < {min_conf}'})

        raw_bbox = pdata.get('bbox_norm', [])
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            return json.dumps({'success': False, 'reason': 'parent_not_found'})
        parent_bbox_norm = [float(v) for v in raw_bbox]
        parent_source = 'vlm'
        if _is_bbox_oversized(parent_bbox_norm):
            _logger.warning(f'[parent] VLM returned oversized bbox {parent_bbox_norm}')
        _logger.info(f'[parent] VLM ground_object result: {parent_bbox_norm}')

    _logger.info(f'[parent source] {parent_source}')
    _logger.info(f'[parent bbox]   {parent_bbox_norm}')

    # ── C. Parent bbox 검증 ────────────────────────────────────────────────────
    parent_bbox_norm = [max(0.0, min(1.0, v)) for v in parent_bbox_norm]
    px1, py1, px2, py2 = parent_bbox_norm
    if px1 >= px2 or py1 >= py2 or (px2 - px1) * (py2 - py1) < 0.01:
        return json.dumps({'success': False, 'reason': 'invalid_parent_bbox'})

    # ── D. Parent crop (7% padding) ────────────────────────────────────────────
    try:
        crop_b64, crop_px = _crop_b64(img_bgr, parent_bbox_norm)
    except Exception as e:
        return json.dumps({'success': False, 'reason': f'crop_failed — {e}'})

    # ── E. Child grounding (crop 이미지에서) ──────────────────────────────────
    # _make_child_target: handle/knob/grip류는 상세 설명(내부 부품 제외) 포함
    child_label_ctx = _make_child_target(target_label, parent_label)
    _logger.info(f'[crop bbox]      px={crop_px}')
    _logger.info(f'[child target]   {child_label_ctx!r}')

    def _call_ground(image_b64: str, label: str) -> Optional[dict]:
        """VLM /ground_object 호출 → response dict 또는 None."""
        try:
            r = _requests.post(ground_url,
                               json={'full_image_b64': image_b64,
                                     'target_label': label,
                                     'detections': [],
                                     'timestamp': time.time()},
                               timeout=vlm_timeout)
        except _requests.exceptions.ConnectionError:
            return {'_error': 'vlm_server_unavailable'}
        except _requests.exceptions.Timeout:
            return {'_error': f'vlm_timeout — {vlm_timeout}s 초과'}
        except Exception as e:
            return {'_error': f'vlm_request_failed — {e}'}
        if r.status_code != 200:
            return {'_error': f'vlm_http_error — {r.status_code}'}
        try:
            return r.json()
        except Exception:
            return {'_error': 'vlm_invalid_json'}

    # 1차 시도: parent crop
    cdata = _call_ground(crop_b64, child_label_ctx)
    _error = cdata.get('_error') if isinstance(cdata, dict) else None
    if _error:
        return json.dumps({'success': False, 'reason': _error})

    child_found = (
        cdata.get('found', False)
        and isinstance(cdata.get('bbox_norm'), list)
        and len(cdata['bbox_norm']) == 4
        and float(cdata.get('confidence', 0.0)) >= min_conf
    )

    if not child_found:
        _logger.warning(
            f'[child] crop grounding failed '
            f'(found={cdata.get("found")}, conf={cdata.get("confidence")}, '
            f'raw={cdata}) — fallback to full image'
        )
        # fallback: 전체 원본 이미지로 한 번만 재시도
        try:
            full_b64_fb = _encode(img_bgr)
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'encode_failed — {e}'})

        cdata = _call_ground(full_b64_fb, child_label_ctx)
        _error = cdata.get('_error') if isinstance(cdata, dict) else None
        if _error:
            return json.dumps({'success': False, 'reason': _error})

        child_found = (
            cdata.get('found', False)
            and isinstance(cdata.get('bbox_norm'), list)
            and len(cdata['bbox_norm']) == 4
            and float(cdata.get('confidence', 0.0)) >= min_conf
        )
        if not child_found:
            _logger.error(
                f'[child] full-image fallback also failed '
                f'(found={cdata.get("found")}, conf={cdata.get("confidence")}, '
                f'raw={cdata})'
            )
            return json.dumps({'success': False, 'reason': 'child_not_found_in_parent'})

        # fallback 성공: child bbox는 이미 원본 좌표이므로 _restore_bbox 불필요
        child_bbox_in_crop = [float(v) for v in cdata['bbox_norm']]
        child_conf = float(cdata.get('confidence', 0.0))
        _logger.info(f'[child raw bbox (full-image fallback)] {child_bbox_in_crop}')

        ox1, oy1, ox2, oy2 = child_bbox_in_crop
        u_c = ((ox1 + ox2) / 2) * w
        v_c = ((oy1 + oy2) / 2) * h
        center_px = [int(round(u_c)), int(round(v_c))]
        child_bbox_px = [int(ox1 * w), int(oy1 * h), int(ox2 * w), int(oy2 * h)]
        _logger.info(f'[restored bbox (full-image fallback)] {child_bbox_in_crop}')
        _logger.info(f'[localization source] full-image VLM fallback')

        # depth → TF pipeline 공통 처리로 이동
        depth_arr_fb, _ = _ros_node.get_depth()
        fx_fb, fy_fb, cx_fb, cy_fb, _, _ = _ros_node.get_cam_intrinsics()

        def _result_2d_fb():
            return json.dumps({
                'success':             True,
                'label':               cdata.get('label', target_label),
                'source':              'vlm',
                'grounding':           'hierarchical',
                'center_px':           center_px,
                'bbox_approx':         child_bbox_px,
                'camera_point':        None,
                'base_link_point':     None,
                'confidence':          round(child_conf, 3),
                'position_confidence': '2d_only',
                'description':         cdata.get('description', ''),
                'inference_ms':        cdata.get('inference_ms', 0.0),
                'parent_label':        parent_label,
                'parent_source':       parent_source,
            }, ensure_ascii=False)

        if depth_arr_fb is None or fx_fb is None:
            _logger.info('[depth] unavailable — 2d_only')
            return _result_2d_fb()
        depth_m_fb = _sample_depth_robust(depth_arr_fb,
                                          child_bbox_px[0], child_bbox_px[1],
                                          child_bbox_px[2], child_bbox_px[3])
        if depth_m_fb is None:
            _logger.info('[depth] invalid — 2d_only')
            return _result_2d_fb()
        cam_xyz_fb = _px2cam(u_c, v_c, depth_m_fb, fx_fb, fy_fb, cx_fb, cy_fb)
        if cam_xyz_fb is None:
            return _result_2d_fb()
        base_xyz_fb = _cam_to_base(cam_xyz_fb)
        if base_xyz_fb is None:
            return _result_2d_fb()
        _logger.info(f'[3d] {base_xyz_fb}  [localization] 3d (full-image fallback)')
        return json.dumps({
            'success':             True,
            'label':               cdata.get('label', target_label),
            'source':              'vlm',
            'grounding':           'hierarchical',
            'center_px':           center_px,
            'bbox_approx':         child_bbox_px,
            'camera_point':        cam_xyz_fb,
            'base_link_point':     base_xyz_fb,
            'confidence':          round(child_conf, 3),
            'position_confidence': 'approximate',
            'description':         cdata.get('description', ''),
            'inference_ms':        cdata.get('inference_ms', 0.0),
            'depth_m':             round(depth_m_fb, 3),
            'parent_label':        parent_label,
            'parent_source':       parent_source,
        }, ensure_ascii=False)

    child_conf = float(cdata.get('confidence', 0.0))
    child_bbox_in_crop = [float(v) for v in cdata['bbox_norm']]
    _logger.info(f'[child raw bbox (crop)]  {child_bbox_in_crop}')

    # ── F. Child bbox 원본 이미지 좌표 복원 ──────────────────────────────────
    # crop_px는 padding 포함된 실제 crop 영역 [bx1,by1,bx2,by2] (pixel)
    bx1, by1, bx2, by2 = crop_px
    crop_norm = [bx1 / w, by1 / h, bx2 / w, by2 / h]
    child_bbox_orig = _restore_bbox(child_bbox_in_crop, crop_norm)
    _logger.info(f'[restored bbox]  {child_bbox_orig}')

    ox1, oy1, ox2, oy2 = child_bbox_orig
    u = ((ox1 + ox2) / 2) * w
    v = ((oy1 + oy2) / 2) * h
    center_px = [int(round(u)), int(round(v))]
    child_bbox_px = [int(ox1 * w), int(oy1 * h), int(ox2 * w), int(oy2 * h)]

    # ── 2D fallback 응답 구성 ──────────────────────────────────────────────────
    def _result_2d():
        _logger.info('[localization] 2d_only')
        return json.dumps({
            'success':             True,
            'label':               cdata.get('label', target_label),
            'source':              'vlm',
            'grounding':           'hierarchical',
            'center_px':           center_px,
            'bbox_approx':         child_bbox_px,
            'camera_point':        None,
            'base_link_point':     None,
            'confidence':          round(child_conf, 3),
            'position_confidence': '2d_only',
            'description':         cdata.get('description', ''),
            'inference_ms':        cdata.get('inference_ms', 0.0),
            'parent_label':        parent_label,
            'parent_source':       parent_source,
        }, ensure_ascii=False)

    # ── G. Depth 샘플링 ────────────────────────────────────────────────────────
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, cam_w, cam_h = _ros_node.get_cam_intrinsics()

    if depth_arr is None or fx is None:
        _logger.info('[depth] unavailable — 2d_only')
        return _result_2d()

    depth_m = _sample_depth_robust(depth_arr,
                                    child_bbox_px[0], child_bbox_px[1],
                                    child_bbox_px[2], child_bbox_px[3])
    if depth_m is None:
        _logger.info('[depth] invalid — 2d_only')
        return _result_2d()

    _logger.info(f'[depth]          {depth_m:.3f}m')

    # ── H. Camera XYZ → base_link TF ─────────────────────────────────────────
    cam_xyz = _px2cam(u, v, depth_m, fx, fy, cx, cy)
    if cam_xyz is None:
        return _result_2d()

    base_xyz = _cam_to_base(cam_xyz)
    if base_xyz is None:
        return _result_2d()

    _logger.info(f'[3d]             {base_xyz}')
    _logger.info('[localization]   3d')
    return json.dumps({
        'success':             True,
        'label':               cdata.get('label', target_label),
        'source':              'vlm',
        'grounding':           'hierarchical',
        'center_px':           center_px,
        'bbox_approx':         child_bbox_px,
        'camera_point':        cam_xyz,
        'base_link_point':     base_xyz,
        'confidence':          round(child_conf, 3),
        'position_confidence': 'approximate',
        'description':         cdata.get('description', ''),
        'inference_ms':        cdata.get('inference_ms', 0.0),
        'depth_m':             round(depth_m, 3),
        'parent_label':        parent_label,
        'parent_source':       parent_source,
    }, ensure_ascii=False)


def _ros_spin_thread():
    global _ros_node
    rclpy.init()
    _ros_node = RosBridgeNode()
    _ros_ready.set()
    try:
        rclpy.spin(_ros_node)
    finally:
        _ros_node.destroy_node()
        rclpy.shutdown()


def _is_ros_alive() -> bool:
    """브리지 노드가 살아있는지 확인."""
    try:
        return _ros_node is not None and _ros_node.context.ok()
    except Exception:
        return False


def _ensure_ros():
    global _ros_ready
    with _ros_lock:
        if _is_ros_alive():
            return
        # 노드가 죽었으면 재초기화 (ROS 재시작 후 자동 복구)
        _ros_ready.clear()
        t = threading.Thread(target=_ros_spin_thread, daemon=True)
        t.start()
        _ros_ready.wait(timeout=10.0)
        if not _is_ros_alive():
            raise RuntimeError('ROS2 bridge 노드 초기화 실패')
        time.sleep(3.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. MCP Tool 정의
# ──────────────────────────────────────────────────────────────────────────────
mcp = FastMCP('agilex-nero-pnp')


@mcp.tool()
def list_detected_objects() -> str:
    """현재 카메라 비전(perception_node)이 인식 중인 물체 목록을 조회한다.

    로봇에게 무언가를 시키기 전에 반드시 먼저 이 도구를 호출해서
    실제로 어떤 물체가 장면에 있는지 확인하라.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ① 알려진 물체 위치 확인         → 이 도구 (YOLO, 빠름)
    ② YOLO 목록에 없는 특정 물체    → ground_object(label)
    ③ 화면 전체 장면 파악 + 배치 공간 → analyze_scene()
    ④ 파지 방법 판단                → infer_grasp(label)
    ⑤ 배치 가능 공간 탐색           → find_placement()

    YOLO 결과에 target object가 있으면 ground_object()를 호출하지 마라.
    YOLO 결과만으로 해결 가능한 작업에 VLM을 호출하지 마라.

    ⚠ 중요 — 좌표/치수 해석 시 반드시 지켜야 할 규칙:
    1. center_3d는 물체의 "기하학적 중심점" 좌표다 (바닥면이 아니다).
       즉 center_3d.z는 물체 바닥이 아니라 물체 중앙 높이다.
    2. box_height_m은 "중심에서 바닥까지의 거리"가 아니라
       "직육면체 한 변(높이 방향)의 전체 길이"다. 따라서:
       - 이 물체의 바닥 높이 = center_3d.z - box_height_m / 2
       - 이 물체의 윗면 높이 = center_3d.z + box_height_m / 2
       - 이 물체 "위에" 다른 박스를 쌓으려면, 쌓을 박스의 중심 z를
         (이 물체의 윗면 높이) + (쌓을 박스의 box_height_m / 2) 로 계산해야 한다.
       - box_height_m은 현재 모든 물체에 고정값 0.05m가 채워진다.
         이 값은 신뢰할 수 있는 고정 상수다. 스캔 결과에 물체가
         안 보이거나 애매하더라도, 높이를 재확인하기 위해 재스캔하거나
         go_home으로 복귀하지 마라. 물체 자체가 안 보이는 것과 높이
         불확실성은 별개 문제이며, 후자를 이유로 전자의 조치를 반복
         호출하지 마라.

    Returns:
        {"objects": [{"label": "cup", "center_3d": {"x":0.31,"y":-0.04,"z":0.1},
                       "angle_base_deg": 45.0, "box_height_m": 0.05}, ...],
         "age_sec": 0.18}
        angle_base_deg는 물체(주로 박스류)의 base_link 기준 회전각(0~90도)이다.
        해당 없으면 null.
        objects 가 빈 배열이면 현재 인식된 물체가 없다.
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    slim = [{'label': o.get('label', '?'),
             'center_3d': o.get('center_3d', {}),
             'angle_base_deg': o.get('angle_base_deg', None),
             'box_height_m': DEFAULT_BOX_HEIGHT_M}
            for o in objects]
    return json.dumps({'objects': slim, 'age_sec': age}, ensure_ascii=False)


@mcp.tool()
def get_joint_positions() -> str:
    """현재 로봇 팔의 각 관절(joint) 각도와 그리퍼 위치를 조회한다.

    move_joints로 상대적인 움직임을 만들려면, 먼저 이 도구로
    현재 각도를 확인한 뒤 원하는 만큼 더하거나 뺀 값을 절대각도로 넘겨라.

    Returns:
        {"joints": {"joint1": 0.0, ..., "joint7": 0.0, "gripper": 0.08},
         "age_sec": 0.05}
        joints가 null이면 아직 피드백을 받지 못한 상태다.
    """
    _ensure_ros()
    joints, stamp = _ros_node.get_joint_state()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({'joints': joints, 'age_sec': age}, ensure_ascii=False)


@mcp.tool()
def save_pose(name: str) -> str:
    """현재 로봇 팔의 관절 자세를 이름을 붙여 저장한다.

    티칭 모드(웹 UI)에서 손으로 자세를 잡은 뒤 호출하면,
    그 순간의 모든 관절 각도와 그리퍼 위치를 기억해둔다.

    Args:
        name: 저장할 자세 이름 (예: "grasp_cup_1")

    Returns:
        성공: {"status": "success", "name": "...", "joints": {...}}
        실패: {"status": "failed", "reason": "..."}
    """
    _ensure_ros()
    joints, stamp = _ros_node.get_joint_state()
    if not joints:
        return json.dumps({'status': 'failed',
                           'reason': '아직 관절 피드백을 받지 못했습니다.'}, ensure_ascii=False)
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    if age > 2.0:
        return json.dumps({
            'status': 'failed',
            'reason': f'관절 피드백이 {age}초 전 값이라 오래됐습니다. 로봇 연결을 확인하세요.',
        }, ensure_ascii=False)

    poses = _load_poses()
    poses[name] = {'joints': joints, 'saved_at': time.time()}
    _save_poses(poses)
    return json.dumps({'status': 'success', 'name': name, 'joints': joints},
                      ensure_ascii=False)


@mcp.tool()
def list_saved_poses() -> str:
    """저장된 모든 자세 이름과 관절 값을 조회한다.

    Returns:
        {"poses": {"grasp_cup_1": {"joint1": 0.1, ...}, ...}}
    """
    poses = _load_poses()
    slim = {name: data.get('joints', {}) for name, data in poses.items()}
    return json.dumps({'poses': slim}, ensure_ascii=False)


@mcp.tool()
def move_to_saved_pose(name: str) -> str:
    """저장된 자세 이름으로 로봇 팔(관절 1~7)을 이동시킨다. 완료까지 블로킹.

    Args:
        name: save_pose 로 저장했던 자세 이름

    Returns:
        성공: {"status": "success", "reason": "joint_move_complete", "joints": {...}}
        실패: {"status": "failed"|"rejected"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    poses = _load_poses()
    if name not in poses:
        return json.dumps({
            'status': 'rejected',
            'reason': f"'{name}' 이라는 자세가 없습니다. 저장된 이름: {sorted(poses.keys())}",
        }, ensure_ascii=False)

    joints = poses[name].get('joints', {})
    key_map = {'joint1': 'j1', 'joint2': 'j2', 'joint3': 'j3', 'joint4': 'j4',
               'joint5': 'j5', 'joint6': 'j6', 'joint7': 'j7'}
    move_joints = {key_map[k]: v for k, v in joints.items() if k in key_map}

    if not move_joints:
        return json.dumps({'status': 'rejected',
                           'reason': '저장된 자세에 팔 관절 값이 없습니다.'}, ensure_ascii=False)

    _ros_node.publish_command({'action': 'move_joints', 'joints': move_joints})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result['joints'] = move_joints
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def scan_for_boxes(target_label: str = 'box') -> str:
    """로봇 팔을 좌우로 훑으며(joint1 스윕) 현재 시야 밖에 있는
    물체까지 찾아서 좌표를 기억해둔다. list_detected_objects로 물체를
    못 찾았을 때 이 도구를 호출한 뒤, pick_object를 from_scan=True로
    호출하면 스캔 중 발견한 위치로 집으러 갈 수 있다. 완료까지 블로킹
    (수십 초 소요).

    ⚠ 중요: 결과의 boxes 배열에 각 물체의 정확한 좌표(x,y,z)가 이미
    포함되어 있다. 이 물체들은 로봇이 스스로 움직이지 않는 한 위치가
    바뀌지 않으므로, 이후 place_object 등에 좌표가 필요할 때
    list_detected_objects로 다시 확인하지 말고 여기서 받은 boxes 값을
    그대로 사용하라. 특히 pick_object 실행 중에는 로봇 팔(카메라)이
    계속 움직이므로, pick 전후로 list_detected_objects를 다시 부르면
    카메라 각도가 달라져 다른(부정확한) 좌표를 받게 될 수 있다.

    [2026-08-26 확장] 이전엔 target_label과 정확히 일치하는 물체만 기억하고
    스윕 중 스쳐지나간 나머지는 버렸다. 이제 라벨 무관하게 스윕 중 보인
    물체를 전부(YOLO dual 백엔드가 인식하는 box + COCO 클래스 전체)
    기억한다 -- 한 번 스캔으로 box든 cup이든 bottle이든 나중에
    pick_object(from_scan=True, target_label=<원하는 라벨>)로 바로 꺼내
    쓸 수 있다. target_label 인자는 더 이상 결과를 필터링하지 않는다
    (하위호환용으로 남아있을 뿐).

    Args:
        target_label: [하위호환용, 현재는 결과에 영향 없음] 과거엔 필터링에
            썼으나 이제 스윕 중 보인 모든 라벨을 기억한다.

    Returns:
        성공: {"status": "success", "reason": "scan_complete:N",
               "boxes": [{"x":0.22,"y":0.19,"z":0.022,
                          "confidence":0.78,"angle_base_deg":45.0,
                          "label":"box"}, {"x":..,"label":"cup",...}, ...]}
               (N=발견한 개수, 라벨 다양할 수 있음. 각 항목의 x/y/z는
               물체 중심좌표)
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    global _last_scanned_boxes, _last_scan_stamp
    _ensure_ros()
    payload = {'action': 'scan_box', 'target_label': target_label.strip().lower()}
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_SCAN)
    if result.get('status') == 'success' and 'boxes' in result:
        _last_scanned_boxes = result['boxes']
        _last_scan_stamp = time.time()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_scanned_boxes() -> str:
    """가장 최근 scan_for_boxes 실행 결과(발견된 물체 좌표 목록)를
    ROS 재조회 없이 즉시 반환한다. scan_for_boxes를 이미 호출해서 물체
    위치를 확인해둔 뒤, place_object 등에 그 좌표가 다시 필요할 때
    list_detected_objects 대신 이 도구를 사용하라 (로봇이 스스로 움직이지
    않는 한 물체 위치는 바뀌지 않으므로 재스캔/재조회가 불필요하다).

    scan_for_boxes를 아직 한 번도 호출하지 않았다면 빈 목록을 반환한다.

    Returns:
        {"boxes": [{"x":0.22,"y":0.19,"z":0.022,"confidence":0.78,
                    "angle_base_deg":45.0,"label":"box"}, ...],
         "age_sec": 12.3}
        age_sec는 마지막 scan_for_boxes 호출 이후 경과 시간(초).
        boxes가 빈 배열이면 아직 스캔한 적이 없다.
    """
    age = round(time.time() - _last_scan_stamp, 2) if _last_scan_stamp > 0 else -1.0
    return json.dumps({'boxes': _last_scanned_boxes, 'age_sec': age}, ensure_ascii=False)


# [2026-08-31 추가] pick 폐루프 검증 2차 확인(VLM) -- bbox 가로세로비가
# 이 배수 이상 벌어지면 "형태가 뒤집혔다"(세운 물체가 누움/누운 물체가
# 섬 등)로 본다. 1.2는 설계 시점 추정값 -- 실측 튜닝 대상.
_BBOX_FLIP_RATIO = 1.2


def _bbox_hw_ratio(bbox):
    """bbox=[x1,y1,x2,y2] (정규화 좌표) -> 세로/가로 비율. 형식이 이상하면 None."""
    try:
        x1, y1, x2, y2 = bbox
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        return h / w
    except Exception:
        return None


def _analyze_scene_cached(max_age_sec: float) -> dict:
    """[2026-09-02 추가, 사용자 지시 #2] _analyze_scene_core() 캐시 재사용 래퍼.

    마지막으로 성공한 analyze_scene 결과(_last_scene_objects/_last_scene_stamp)가
    max_age_sec 이내면 8B VLM 재추론(~15~20초) 없이 그걸 재사용한다. pick
    검증(_check_pick_fallen_over)·place 검증(_verify_placement_vlm)·parent
    grounding 이 몇 초 안에 연달아 같은 장면을 볼 때 VLM 호출이 중복되는
    걸 막는 게 목적. 캐시가 비었거나 오래됐으면 _analyze_scene_core()로
    새로 뜬다(그 호출이 캐시도 갱신함).

    ⚠ 반환 dict은 캐시 히트 시 {'status','objects','from_cache','cache_age_sec'}
    만 담긴다 -- placement_targets/placement_regions/yolo_detected 등 나머지
    필드는 없다. 이 필드들이 필요한 호출부(analyze_scene MCP 툴 등)는 이
    래퍼를 쓰지 말고 _analyze_scene_core()를 직접 불러야 한다.
    """
    with _scene_cache_lock:
        objs = list(_last_scene_objects)
        stamp = _last_scene_stamp
    if objs and stamp > 0 and (time.time() - stamp) <= max_age_sec:
        return {'status': 'success', 'objects': [o.copy() for o in objs],
                'from_cache': True,
                'cache_age_sec': round(time.time() - stamp, 2)}
    return _analyze_scene_core()


def _check_pick_fallen_over(target_label: str, before_bbox) -> tuple:
    """[2026-08-31 추가] planning_node의 verify_pick(1차, 카메라/토픽 기반)이
    '원래 자리에 물체가 여전히 있음'(pick_verified=False)으로 보고했을 때만
    호출되는 2차 확인. 매 pick마다 부르면 VLM 추론(수 초) 때문에 느려지므로
    -- 1차 신호가 애매/실패로 나온 드문 경우에만 이 함수가 실행된다
    (속도 요구사항: "너무 느려지면 안 됨"에 대한 설계 대응).

    _analyze_scene_core()(VLM analyze_scene 로직 재사용, 사용자 지시)로
    현재 장면을 다시 보고, pick 시도 전 bbox와 세로/가로 비율을 비교해서
    "쓰러졌다"(형태가 뒤집혔다)와 "그냥 못 집었다"(형태 그대로)를 구분한다.

    Returns: (fell_over: bool, reason: str)
    """
    if not before_bbox or len(before_bbox) != 4:
        return False, 'before_bbox 없음(YOLO 미검출 상태에서 시도했을 수 있음) -- 쓰러짐 판정 불가'
    # [2026-09-02 #2] 직전 4초 이내 analyze_scene 결과가 있으면 재사용
    # (pick 직후라 장면이 거의 안 바뀌었을 시간대).
    scene = _analyze_scene_cached(4.0)
    if scene.get('status') != 'success':
        return False, f'VLM 재확인 실패 — {scene.get("reason")}'
    after_bbox = None
    for o in scene.get('objects', []):
        if str(o.get('label', '')).lower() == target_label:
            after_bbox = o.get('bbox')
            break
    if not after_bbox:
        return False, f"VLM 재확인에서도 '{target_label}' 못 찾음 -- 쓰러짐 판정 불가"
    r0, r1 = _bbox_hw_ratio(before_bbox), _bbox_hw_ratio(after_bbox)
    if r0 is None or r1 is None:
        return False, 'bbox 형식 이상 -- 쓰러짐 판정 불가'
    was_tall, now_wide = r0 > _BBOX_FLIP_RATIO, r1 < 1.0 / _BBOX_FLIP_RATIO
    was_wide, now_tall = r0 < 1.0 / _BBOX_FLIP_RATIO, r1 > _BBOX_FLIP_RATIO
    if (was_tall and now_wide) or (was_wide and now_tall):
        return True, (f'bbox 가로세로비 반전 감지(전 {r0:.2f} -> 후 {r1:.2f}) '
                      f'-- 쓰러졌을 가능성 높음')
    return False, (f'bbox 형태 변화 없음(전 {r0:.2f} -> 후 {r1:.2f}) '
                   f'-- 쓰러진 게 아니라 그냥 못 집었을 가능성')


# ─────────────────────────────────────────────────────────────────────────────
# [2026-09-02 추가, 사용자 지시] place 폐루프 검증 -- VLM analyze_scene 기반
# ─────────────────────────────────────────────────────────────────────────────
# planning_node의 1차 placement_verified는 /detected_objects(YOLO 중심,
# perception_node가 depth+TF로 center_3d까지 채워 발행하는 토픽)만 본다.
# YOLO가 놓은 물체를 잠깐 놓치거나(바구니 벽/그리퍼에 가려짐 등) 라벨이
# 흔들리면, 실제로는 잘 놓였는데 verified=False로 나오는 오탐이 실측
# 확인됐다(2026-09-02, cup/bottle을 회색 바구니에 넣은 직후 YOLO 목록에서
# 사라짐 -> "105mm 벗어남" 오탐).
#
# 그래서 1차 신호가 True가 아닐 때(False/None)만 -- pick의 _check_pick_
# fallen_over와 동일하게, 정상(verified=True) 케이스 지연시간은 건드리지
# 않는 설계 -- VLM analyze_scene으로 장면을 다시 보고, 각 물체 bbox를
# depth+TF로 3D화해서 verify_placement(planning_node와 같은 순수 로직)로
# 재판정한 뒤 그 결과로 placement_verified를 덮어쓴다.
#
# ⚠ 한계: 이 경로는 place 시작 전 VLM 스냅샷(before_objects)이 없어서
# 쌓기(stacking) 시 "베이스 박스를 새로 놓인 물체로 착각"하는 오탐을
# 걸러내지 못한다 (placement_verification.py "쌓기 시 오탐 방지" 참고).
# 어차피 1차(YOLO+before/after 델타)가 False/None으로 이미 의심 신호를
# 준 뒤에만 도는 2차 확인이고, 결과 문자열에 "[VLM 재검증]"을 붙여
# 구분하므로 수용 가능한 트레이드오프로 판단.


def _vlm_scene_objects_3d() -> tuple:
    """_analyze_scene_core()로 현재 장면을 VLM으로 다시 보고, 각 물체 bbox
    중심을 depth+TF로 3D화한 목록을 만든다. /detected_objects(YOLO)와 같은
    스키마({'label', 'center_3d': {x,y,z}})로 맞춰서 verify_placement에
    그대로 넣을 수 있게 한다.

    Returns: (objects_3d: list, err: str|None)
      err가 not None이면 VLM/depth 파이프라인 자체가 불가능한 상태 -- 판정 불가.
    """
    scene = _analyze_scene_core()
    if scene.get('status') != 'success':
        return [], f'VLM analyze_scene 실패 — {scene.get("reason")}'
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, _cam_w, _cam_h = _ros_node.get_cam_intrinsics()
    if depth_arr is None or fx is None:
        return [], 'depth/카메라 intrinsics 미수신 — VLM 물체 3D화 불가'
    h, w = depth_arr.shape[:2]
    out = []
    for o in scene.get('objects', []):
        bbox = o.get('bbox') or []
        if len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        # analyze_scene bbox는 정규화 좌표(0~1) -> 픽셀
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        u = (px1 + px2) / 2.0
        v = (py1 + py2) / 2.0
        depth_m = _sample_depth_robust(depth_arr, px1, py1, px2, py2)
        if depth_m is None:
            continue
        cam_xyz = _px2cam(u, v, depth_m, fx, fy, cx, cy)
        if cam_xyz is None:
            continue
        base_xyz = _cam_to_base(cam_xyz)
        if base_xyz is None:
            continue
        out.append({
            'label': str(o.get('label', '?')).lower(),
            'center_3d': base_xyz,
            'source': o.get('source', 'vlm'),
        })
    return out, None


def _verify_placement_vlm(expected_pos: dict) -> dict:
    """VLM analyze_scene 기반 place 재검증. 반환 스키마는 planning_node의
    placement_verified/verification_reason과 동일(verified: True|False|None)."""
    objs, err = _vlm_scene_objects_3d()
    if err is not None:
        return {'verified': None,
                'reason': f'[VLM 재검증] {err} -- 판정 불가.'}
    res = _verify_placement_core(expected_pos, objs)
    return {'verified': res['verified'],
            'reason': '[VLM 재검증] ' + res['reason']}


def _pick_object_impl(target_label: str, grasp_dir: str = 'auto',
                       from_scan: bool = False, box_index: int = None,
                       x: float = None, y: float = None, z: float = None,
                       angle_deg: float = None,
                       side_approach_deg: float = None,
                       retry_count: int = 0) -> dict:
    """[2026-08-31 추가] pick_object() MCP 툴의 실제 로직 + 폐루프 검증
    2차확인/재시도. retry_count는 내부 재귀 전용 파라미터라 MCP 툴
    시그니처(pick_object)에는 노출하지 않는다 -- Claude가 이 값을 직접
    지정할 이유가 없고, 노출하면 툴 스키마만 지저분해진다."""
    _ensure_ros()
    target_label = target_label.strip().lower()

    _before_bbox = None
    _object_z = None
    _has_override = x is not None and y is not None and z is not None
    # [2026-08-31 수정] override 좌표(x,y,z)를 준 경우는 애초에 label로
    # 위치를 찾을 게 아니라 그 좌표로 바로 접근하는 것이므로, label이
    # /detected_objects에 없다는 이유로 거부하면 안 된다 -- ground_object
    # 등 YOLO 라벨셋에 없는(VLM 전용) 물체를 override로 pick할 때마다
    # 매번 라벨을 YOLO가 아는 걸로 바꿔치기해야 하는 불편이 실측 확인됨.
    if not from_scan and not _has_override:
        objects, _ = _ros_node.get_objects()
        available = {o.get('label', '').lower() for o in objects}
        if available and target_label not in available:
            return {
                'status': 'rejected',
                'reason': f"'{target_label}' 은(는) 현재 장면에 없습니다. "
                          f"인식된 물체: {sorted(available)}. "
                          f"카메라 시야 밖에 있을 수 있으니 scan_for_boxes를 먼저 시도해보라.",
            }
        for o in objects:
            if o.get('label', '').lower() == target_label:
                _before_bbox = o.get('bbox')
                _object_z = (o.get('center_3d') or {}).get('z')
                break

    # [2026-09-02 추가] grasp_dir='auto'(기본값)를 CLAUDE.md 변환표
    # (grasp_types.resolve_grasp_dir)에 연결 -- infer_grasp를 서버가 대신
    # 호출해서 top/side/pinch를 판정한다. from_scan(박스 스캔 큐)은 기존
    # top-down 전용 흐름을 그대로 둔다(스캔 큐 물체는 대부분 box이고,
    # 카메라가 이미 그 박스를 보고 있다는 보장이 없어 infer_grasp 대상이
    # 부정확할 수 있음). override 좌표(x,y,z)를 쓴 경우엔 그 z를 바로 사용.
    _auto_grasp_reason = None
    if grasp_dir == 'auto' and not from_scan:
        if _has_override:
            _object_z = z
        _resolved_dir, _resolved_deg, _resolved_reason = _resolve_auto_grasp_dir(
            target_label, _object_z)
        if _resolved_dir is not None:
            grasp_dir = _resolved_dir
            if side_approach_deg is None:
                side_approach_deg = _resolved_deg
        _auto_grasp_reason = _resolved_reason

    payload = {'action': 'pick', 'target_label': target_label}
    if grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
    if side_approach_deg is not None:
        payload['side_approach_deg'] = side_approach_deg
    if x is not None and y is not None and z is not None:
        payload['override_pos'] = {'x': x, 'y': y, 'z': z}
        if angle_deg is not None:
            payload['override_angle_deg'] = angle_deg
    elif from_scan:
        payload['from_scan'] = True
        if box_index is not None:
            payload['box_index'] = box_index
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PICK)
    result['target_label'] = target_label
    if _auto_grasp_reason is not None:
        result['auto_grasp_resolution'] = _auto_grasp_reason

    # [2026-08-31 추가] pick 폐루프 검증 -- planning_node의 1차 신호
    # (pick_verified)가 False(원래 자리에 물체가 여전히 감지됨)일 때만
    # VLM 2차 확인을 하고, 쓰러진 걸로 판정되면 1회 재시도한다. 1차
    # 신호가 True/None이거나 이미 재시도한 상태면 여기서 끝 -- 무한
    # 재시도 방지 + 평소(1차 신호가 True인 정상 케이스) 지연시간 보존.
    if (result.get('status') == 'success'
            and result.get('pick_verified') is False
            and retry_count == 0):
        fell_over, vlm_reason = _check_pick_fallen_over(target_label, _before_bbox)
        result['pick_verification_vlm'] = vlm_reason
        if fell_over:
            result['pick_retry'] = 'fell_over_detected_retrying_once'
            return _pick_object_impl(
                target_label, grasp_dir='auto', from_scan=from_scan, box_index=box_index,
                x=x, y=y, z=z, angle_deg=angle_deg, side_approach_deg=side_approach_deg,
                retry_count=1)
        result['pick_retry'] = None
    return result


@mcp.tool()
def pick_object(target_label: str, grasp_dir: str = 'auto',
                 from_scan: bool = False, box_index: int = None,
                 x: float = None, y: float = None, z: float = None,
                 angle_deg: float = None,
                 side_approach_deg: float = None) -> str:
    """지정한 물체를 로봇 팔로 집어 올린다. pick 완료까지 블로킹.

    Args:
        target_label: 집을 물체 라벨 (예: "cup", "bottle"). 영어 소문자.
        grasp_dir: 파지 방향.
            "auto"        : [2026-09-02 변경] 서버가 infer_grasp(VLM)를 내부
                호출해서 CLAUDE.md 그립형태 변환표(grasp_types.resolve_grasp_dir)
                로 top/side/pinch를 판정한다 (기본값). from_scan=True일 때는
                기존 동작(박스 top-down 전용, 서버 내부 휴리스틱) 그대로 --
                infer_grasp 대상이 카메라 시야와 무관한 스캔 큐 항목이라
                신뢰 못 함. infer_grasp 자체가 실패하면(이미지 없음/인식 안됨)
                응답의 auto_grasp_resolution 필드에 사유가 남고 서버 내부
                휴리스틱으로 폴백한다.
            "top"         : 위에서 아래로 (auto 판정을 건너뛰고 직접 지정)
            "side"        : 앞에서 수평으로
            "side_left"   : 왼쪽에서 수평으로
            "side_right"  : 오른쪽에서 수평으로
        from_scan: True면 scan_for_boxes로 미리 찾아둔 좌표를 사용한다
            (현재 카메라 시야 밖에 있는 물체도 집을 수 있음). scan_for_boxes를
            먼저 호출해서 물체를 찾아둔 경우에만 True로 설정하라.
        side_approach_deg: [2026-08-26 기준 변경] side 그립 시 접근 방향 --
            물체 방위각(position_yaw = atan2(y,x)) 기준 상대각(도), top-down의
            angle_rel과 동일 개념. 생략(또는 0)이면 물체 정면(가장 자연스럽고
            도달 가능성이 높은 방향)에서 접근. +/-로 그 방위각 대비 회전한
            방향에서 접근한다 (world 절대각이 아님 -- 이전엔 world 절대각을
            받았으나, 물체 위치가 바뀔 때마다 "정면"에 해당하는 값을 매번
            다시 계산해야 해서 상대각 기준으로 변경).
        box_index: [신규] from_scan=True일 때, get_scanned_boxes()/이전
            pick_object 응답의 remaining_scanned_boxes 배열에서 몇 번째
            항목(0부터 시작)을 집을지 명시적으로 지정한다. 생략하면
            큐 맨 앞 항목을 집는다(하위호환, 아래 경고 참고).

    ⚠ 여러 박스를 다루는 작업(특히 쌓기)에서는 box_index를 반드시
    명시하라. box_index를 생략하면 "큐 맨 앞 항목"을 집는데, 이미 다른
    스캔된 박스의 (x,y) 위에 무언가를 place한 뒤 그 자리를 다시
    "다음 큐 항목"으로 착각해서 엉뚱한(방금 쌓은) 박스를 다시 집어버리는
    사고가 실측 확인됐다(2026-07-22). box_index로 명시하면 이 문제가
    구조적으로 발생하지 않는다.

    ⚠ from_scan=True로 pick하면, 결과의 remaining_scanned_boxes에
    "이번에 집은 것을 제외한 나머지 스캔된 물체들"의 좌표가 이미 담겨
    돌아온다. 이어서 다른 스캔된 물체를 다루려면(예: 방금 집은 물체를
    남은 물체 위에 쌓기), list_detected_objects나 scan_for_boxes를
    다시 호출하지 말고 이 필드를 그대로 사용하라(인덱스는 이 배열
    기준으로 다시 매겨짐 -- 원래 스캔 인덱스가 아님). pick 실행 중에는
    로봇 팔(카메라)이 계속 움직이므로, 재조회하면 다른(부정확한)
    좌표를 받게 될 위험이 있다. 특히 물체를 쥔 채로 scan_for_boxes를
    다시 호출하는 것은 팔이 크게 움직이며 충돌 위험이 있으니 절대
    하지 말 것.

    Returns:
        성공: {"status": "success", "reason": "pick_complete",
               "target_label": "...",
               "auto_grasp_resolution": "..." (grasp_dir='auto'였을 때만 --
                   VLM TOP/SIDE/PINCH→grasp_dir 변환 사유, 또는 infer_grasp
                   실패시 폴백 사유),
               "remaining_scanned_boxes": [{"x":..,"y":..,"z":..,
                   "confidence":..,"angle_base_deg":..,"label":".."}, ...],
               "align_angle_deg": 37.2,
               "pick_verified": true|false|null,
               "pick_verification_reason": "...",
               "pick_verification_vlm": "..." (pick_verified=false였을 때만),
               "pick_retry": "fell_over_detected_retrying_once"|null (있었을 때만)}
               (remaining_scanned_boxes는 from_scan=True였을 때만 채워짐.
               align_angle_deg는 top 그립일 때 approach 후 align 단계에서
               재측정한 각도[도, 0~90]. side 그립이면 null.)

        ⚠ [2026-08-31 추가, 폐루프 pick 검증] status="success"라고 해서
        물체가 실제로 집혔다는 보장은 아니다 -- 그리퍼가 허공에 닫혀서
        원래 자리에 물체가 그대로 남아있는 경우가 실측 반복 확인됐다.
        서버가 lift 직후 원래 위치를 재확인한 결과가 pick_verified 필드에
        담긴다:
          - true  : 원래 자리에서 물체가 사라짐. 집혔을 가능성 높음 --
                    안심하고 다음 작업(place 등) 진행 가능.
          - false : 원래 자리에 물체가 여전히 감지됨. 이 시점에 서버가
                    VLM으로 자동 2차 확인을 했고(pick_verification_vlm에
                    사유), 물체가 "쓰러진" 걸로 판단되면 자동으로 1회
                    재시도까지 마친 뒤의 최종 결과다(pick_retry 필드로
                    확인 가능). 그래도 pick_verified=false로 끝났다면
                    실제로 못 집었을 가능성이 높으니, 이 물체를 기준으로
                    후속 작업을 계속하지 말고 list_detected_objects로
                    재확인하거나 사용자에게 알려라.
          - null(필드 없거나 None) : perception 데이터가 오래됐거나 인식
                    사각지대라 판정 불가. false와 다르게 취급할 것 --
                    확실하지 않으니 필요하면 list_detected_objects로
                    직접 재확인하라.
        1차 신호(pick_verified)가 true/null이면 VLM을 안 부르므로 평소
        지연시간에는 영향이 없다 -- VLM 2차 확인+재시도는 1차 신호가
        false인 드문 경우에만 발생한다.
        실패: {"status": "failed"|"rejected"|"timeout", "reason": "..."}
    """
    return json.dumps(
        _pick_object_impl(target_label, grasp_dir=grasp_dir, from_scan=from_scan,
                          box_index=box_index, x=x, y=y, z=z, angle_deg=angle_deg,
                          side_approach_deg=side_approach_deg),
        ensure_ascii=False)


@mcp.tool()
def slide_object(target_label: str, grasp_dir: str = 'auto', x: float = None, y: float = None,
                  z: float = None, side_approach_deg: float = None,
                  slide_distance_m: float = None, from_scan: bool = False,
                  box_index: int = None) -> str:
    """문고리/서랍 손잡이 등을 잡아서 "당겨 여는" 동작. pick_object와 달리
    들어올리지 않는다 -- 접근각 반대 방향으로 slide_distance_m만큼 직선
    이동한 뒤 그리퍼를 놓는다. [2026-08-26 추가]

    ⚠ grasp_dir는 "auto"|"side"|"pinch"만 가능하다. top-down은 애초에
    수평 당김 동작과 기하학적으로 안 맞아서 미지원(요청하면 rejected).

    side/pinch의 접근 시퀀스(TCP오프셋 보정, standoff에서 직선 진입,
    접근각 후보 자동 탐색, 사전회전 없음)를 pick_object와 완전히
    동일하게 재사용한다 -- 차이는 마지막 단계뿐이다(들어올리기 대신
    당겨서 놓기).

    Args:
        target_label: 손잡이/물체 라벨 (예: "handle", "drawer").
        grasp_dir: "auto"(기본값) | "side" | "pinch".
            [2026-09-02 변경] "auto"면 서버가 infer_grasp(VLM)를 내부 호출해서
            grasp_types.resolve_grasp_dir로 side/pinch를 판정한다(top으로
            판정되면 slide 자체가 불가능하므로 rejected). from_scan=True거나
            infer_grasp가 실패하면(이미지 없음/인식 안됨 등) 안전하게 판정
            불가로 rejected -- slide는 pick과 달리 planning_node가 grasp_dir
            누락/애매값을 절대 받아주지 않아 서버 내부 휴리스틱 폴백이 없다.
            애매하면 직접 "side"/"pinch"를 지정하라.
        x, y, z: override 좌표 (base_link 기준, 미터). 지정하면 인지
            재조회 없이 이 좌표로 바로 접근한다.
        side_approach_deg: 접근각 -- 물체 방위각(atan2(y,x)) 기준
            상대각(도). 생략(또는 0)이면 정면 접근. grasp_dir="auto"면
            infer_grasp의 suggested_side_approach_deg가 자동으로 채워짐
            (직접 지정한 값이 있으면 그게 우선).
        slide_distance_m: 당길 거리(미터). 생략 시 서버 기본값(0.08m).
        from_scan: True면 scan_for_boxes로 미리 찾아둔 좌표 사용.
        box_index: from_scan=True일 때 remaining_scanned_boxes에서
            몇 번째 항목을 쓸지 (pick_object와 동일 규칙).

    Returns:
        성공: {"status": "success", "reason": "slide_complete"}
        거부: {"status": "rejected", "reason": "..."} (grasp_dir가
              side/pinch가 아니거나 접근 불가 영역, 또는 auto 판정 실패/top판정)
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()

    _auto_grasp_reason = None
    if grasp_dir == 'auto':
        if from_scan:
            return json.dumps({
                'status': 'rejected',
                'reason': "grasp_dir='auto'는 from_scan=True와 함께 쓸 수 없음 "
                          "-- 스캔 큐 항목은 카메라가 지금 보고 있다는 보장이 없어 "
                          "infer_grasp 판정을 신뢰 못 함. 'side' 또는 'pinch'를 직접 지정하라.",
            }, ensure_ascii=False)
        _object_z = z if (x is not None and y is not None and z is not None) else None
        if _object_z is None:
            objects, _ = _ros_node.get_objects()
            for o in (objects or []):
                if o.get('label', '').lower() == target_label:
                    _object_z = (o.get('center_3d') or {}).get('z')
                    break
        _resolved_dir, _resolved_deg, _resolved_reason = _resolve_auto_grasp_dir(
            target_label, _object_z)
        if _resolved_dir not in ('side', 'pinch'):
            return json.dumps({
                'status': 'rejected',
                'reason': f"grasp_dir='auto' 판정 결과 slide에 쓸 수 없음 "
                          f"(판정: {_resolved_dir!r}, 사유: {_resolved_reason}). "
                          f"grasp_dir='side' 또는 'pinch'를 직접 지정하라.",
            }, ensure_ascii=False)
        grasp_dir = _resolved_dir
        if side_approach_deg is None:
            side_approach_deg = _resolved_deg
        _auto_grasp_reason = _resolved_reason

    payload = {'action': 'slide', 'target_label': target_label, 'grasp_dir': grasp_dir}
    if side_approach_deg is not None:
        payload['side_approach_deg'] = side_approach_deg
    if slide_distance_m is not None:
        payload['slide_distance_m'] = slide_distance_m
    if x is not None and y is not None and z is not None:
        payload['override_pos'] = {'x': x, 'y': y, 'z': z}
    elif from_scan:
        payload['from_scan'] = True
        if box_index is not None:
            payload['box_index'] = box_index
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PICK)
    result['target_label'] = target_label
    if _auto_grasp_reason is not None:
        result['auto_grasp_resolution'] = _auto_grasp_reason
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def place_object(x: float, y: float, z: float, grasp_dir: str = 'auto',
                  side_approach_deg: float = None, retreat_mode: str = 'lift') -> str:
    """집은 물체를 지정한 좌표(base_link 기준, 미터)에 내려놓는다. place 완료까지 블로킹.

    Args:
        x, y, z: 내려놓을 위치 (base_link 기준, 미터)
        grasp_dir: 내려놓을 때 자세. pick_object 와 동일한 값 사용 권장.
        side_approach_deg: [2026-08-26 추가] grasp_dir이 side/pinch일 때
            접근각 -- 물체 방위각(atan2(y,x)) 기준 상대각(도). 생략(또는 0)
            이면 정면 접근. pick_object와 동일한 접근각 후보 자동 탐색이
            place에도 적용된다.
        retreat_mode: [2026-08-31 추가] grasp_dir이 side/pinch일 때, 그리퍼를
            연 뒤 물러나는 방식. top-down은 이 값과 무관하게 항상 수직으로
            물러난다.
              'lift'  (기본값) — xy 고정, z만 올림 (top-down 방식과 동일 원리).
                        놓은 지점 위쪽이 트여있을 때 적합.
              'slide' — z 고정, 접근했던 방향의 반대로 수평 후퇴(들어왔던
                        길 그대로 되돌아감). 위쪽에 다른 물체/장애물이 있어
                        수직으로 못 빠질 때 적합.

    ⚠ 요청한 좌표가 로봇의 국소 도달불가 지점(singularity)이거나 비정상적
    으로 높은 z일 경우, 서버가 자동으로 근처 좌표(±0.05m y시프트 또는
    z 하향)로 재시도한다. 성공 시 결과의 place_pos가 실제로 놓인 좌표로
    바뀌어 있을 수 있다 — 이 경우 requested_place_pos 필드에 원래 요청
    좌표가 별도로 담긴다. 다음 작업(예: 이 박스 위에 쌓기)의 기준 좌표는
    반드시 place_pos(실제 좌표)를 써야 한다.

    ⚠ [폐루프 검증] status="success"라고 해서 물체가 실제로 목표 위치에
    안착했다는 보장은 아니다. 그리퍼가 열리는 순간 물체가 미끄러지거나
    다른 물체에 부딪혀 튕겨나갈 수 있다. 서버가 lift 직후 카메라로 재확인한
    결과가 placement_verified 필드에 담긴다.

    [2026-09-02 변경] 1차 검증은 planning_node가 /detected_objects(YOLO
    중심 토픽)로 한다. 이게 True가 아니면(False/None) 서버가 VLM
    analyze_scene으로 장면을 다시 보고 재판정해서 placement_verified를
    덮어쓴다(verification_reason이 "[VLM 재검증]"으로 시작). 1차 토픽
    결과는 placement_verified_topic / verification_reason_topic에 따로
    남는다. 정상(1차 True) 케이스는 VLM을 안 부르므로 평소와 지연시간이
    같다.

    [2026-09-02 #1 가드] 1차가 False인데 "물체를 목표에서 찾았고 xy 오차가
    120mm 이하"이면 VLM 재검증(~20초)을 생략한다 -- 바구니/트레이처럼
    넓은 리셉터클이면 중심에서 그 정도 벗어나도 안착이고, 실측상 VLM
    재판정도 1차와 1~5mm로 똑같은 값을 냈다. 이 경우 placement_verified는
    1차 False가 그대로 유지되고 verification_reason 끝에 "(VLM 재검증 생략
    #1 ...)"이 붙는다 -- 좁은 목표에 정밀하게 놓아야 했다면
    list_detected_objects로 직접 재확인하라.

    placement_verified 값의 의미:
      - true  : 목표 위치 근처에서 물체가 실제로 확인됨. 안심하고 다음
                작업(그 위에 쌓기 등) 진행 가능.
      - false : 목표 위치에서 물체를 못 찾았거나 크게 벗어남
                (verification_reason에 구체적 사유). 이 물체를 기준으로
                후속 작업(쌓기 등)을 계속하지 말고, list_detected_objects로
                실제 위치를 재확인하거나 사용자에게 알려라.
      - null(필드 자체가 없거나 값이 None) : perception 데이터가 오래돼
                판정 불가. 확실하지 않으니 필요하면 list_detected_objects로
                직접 재확인하라.

    Returns:
        성공: {"status": "success", "reason": "place_complete",
               "place_pos": {...},
               "requested_place_pos": {...}  (좌표가 자동 보정된 경우만 포함),
               "placement_verified": true|false|null,
               "verification_reason": "...",
               "placement_verified_topic": true|false|null,   (VLM 재검증이
                   돌거나 #1 가드로 생략된 경우만 -- planning_node 1차
                   토픽 결과 보존),
               "verification_reason_topic": "...",
               "xy_error_m": 0.041,  (1차 검증이 물체를 찾은 경우 -- 목표
                   좌표와의 xy 거리. #1 가드 판단에 쓰임),
               "z_error_m": 0.02}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    place_pos = {'x': x, 'y': y, 'z': z}
    payload = {'action': 'place', 'place_pos': place_pos}
    if grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
        if side_approach_deg is not None:
            payload['side_approach_deg'] = side_approach_deg
        if retreat_mode and retreat_mode != 'lift':
            payload['retreat_mode'] = retreat_mode
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_PLACE)
    # [신규] planning_node가 도달 불가 지점을 감지해 좌표를 자동으로
    # 시프트했을 수 있다 (approach/descend 실패 시 ±0.05m y시프트 또는
    # z 하향 재시도, nero_robot_place_reachability memory 기반). 실제로
    # 어디 놓였는지를 place_pos에 반영해서, Claude가 요청 좌표와 실제
    # 좌표가 다르다는 걸 놓치지 않게 한다.
    if 'actual_place_pos' in result and result['actual_place_pos'] != place_pos:
        result['requested_place_pos'] = place_pos
        result['place_pos'] = result['actual_place_pos']
    else:
        result['place_pos'] = place_pos

    # [2026-09-02 추가, 사용자 지시] 1차 placement_verified(planning_node,
    # /detected_objects=YOLO 기반)가 True가 아니면 VLM analyze_scene으로
    # 재검증해서 결과를 덮어쓴다. 1차 토픽 결과는 *_topic 필드에 보존.
    # 정상(verified=True) 케이스에는 VLM을 안 부르므로 평소 지연시간 영향 없음
    # (pick의 _check_pick_fallen_over와 동일 설계). 이 place 시퀀스 직후
    # planning_node가 팔을 observation/verify-vantage 자세에 두고 끝내므로
    # 카메라가 장면을 볼 수 있는 상태다.
    #
    # [2026-09-02 #1 가드] 1차가 False인데 "물체는 목표 근처에서 찾았고
    # xy 오차가 넓은 컨테이너 허용 범위(<=WIDE_CONTAINER_XY_TOL_M)" 라면
    # VLM 재검증을 생략한다. 실측(2026-09-02, 회색 바구니에 box/cup/bottle):
    # 이 경우 VLM 재판정도 1차와 1~5mm 내로 똑같은 수치를 냈다 -- 20초
    # 더 써서 얻는 정보가 없다. 바구니/트레이처럼 넓은 리셉터클에 담을
    # 땐 중심에서 몇 cm 벗어나도 "안착"이므로, 1차 False를 그대로 두되
    # 사유에 그 취지를 덧붙인다. VLM은 1차가 정말 데이터가 없을 때(None)
    # 나 "물체 자체를 못 찾음"일 때만 값어치가 있다.
    if (result.get('status') == 'success'
            and result.get('placement_verified') is not True):
        _topic_verified = result.get('placement_verified')
        _topic_reason = result.get('verification_reason')
        _topic_xy_err = result.get('xy_error_m')
        _skip_vlm = (
            _topic_verified is False
            and isinstance(_topic_xy_err, (int, float))
            and _topic_xy_err <= WIDE_CONTAINER_XY_TOL_M
        )
        if _skip_vlm:
            result['verification_reason'] = (
                f'{_topic_reason} (VLM 재검증 생략 #1: 물체를 목표에서 '
                f'{_topic_xy_err*1000:.0f}mm 지점에서 찾음 -- 바구니/트레이 같은 '
                f'넓은 리셉터클이면 안착 범위. 좁은 목표였다면 '
                f'list_detected_objects로 재확인하라.)')
        else:
            _vlm = _verify_placement_vlm(result['place_pos'])
            result['placement_verified'] = _vlm['verified']
            result['verification_reason'] = _vlm['reason']
        result['placement_verified_topic'] = _topic_verified
        result['verification_reason_topic'] = _topic_reason

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def move_to_position(x: float, y: float, z: float, grasp_dir: str = 'auto',
                      side_approach_deg: float = None,
                      quat_override: list = None,
                      apply_side_tcp_offset: bool = False) -> str:
    """로봇 팔 끝(end-effector)을 지정 좌표로 이동한다. 완료까지 블로킹.
    pick_object와 달리 그리퍼를 닫지 않는다 -- 자세(orientation) 후보를
    실제로 뭔가 집지 않고 미리 확인해볼 때 씀. side 그립 후보 검증용으로
    [2026-08-26] 추가됨 -- 자세한 배경은 grasp-kinematics-design 스킬 참고.

    ⚠ pick_object의 side 그립과 달리 이 도구는 기본적으로
    side_reachability_check(최소거리 0.32m)나 SIDE_TCP_OFFSET 보정을
    거치지 않는다 -- 순수하게 지정한 (x,y,z)로 지정한 자세로 이동만
    해본다. apply_side_tcp_offset=True로 켜지 않는 한 실제 pick 최종
    도달 지점과 다를 수 있다.

    Args:
        x, y, z: 목표 좌표 (base_link 기준, 미터)
        grasp_dir: 자세. 생략(기본 'auto')하면 기존 동작과 동일(top-down,
            마지막 스캔 각도 기반). "top"/"side"/"side_left"/"side_right" 등
            pick_object와 동일한 값 사용 가능.
        side_approach_deg: [2026-08-26 기준 변경] side 그립일 때 접근 방향 --
            물체 방위각(atan2(y,x)) 기준 상대각(도, world 절대각 아님).
            생략(또는 0)이면 정면 접근.
        quat_override: [x,y,z,w] 쿼터니언을 직접 지정해서 grasp_dir 계산을
            완전히 건너뛴다 (여러 후보 자세를 직접 실험할 때 사용).
            지정하면 grasp_dir/side_approach_deg는 무시된다.
        apply_side_tcp_offset: [2026-08-26 추가] True + side 계열
            grasp_dir이면, pick_object와 동일하게 SIDE_TCP_OFFSET을
            (position_yaw+side_approach_deg 방향으로) 적용한 뒤 이동한다
            -- 실제 pick 시 그리퍼 손끝이 도달할 지점을 그리퍼를 닫지
            않고 미리 확인할 때 켜라. quat_override 지정 시에는 무시됨
            (원본 좌표 그대로 이동).

    Returns:
        성공: {"status": "success", "reason": "move_complete", "target_pos": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    target_pos = {'x': x, 'y': y, 'z': z}
    payload = {'action': 'move', 'target_pos': target_pos}
    if quat_override is not None:
        payload['quat_override'] = list(quat_override)
    elif grasp_dir and grasp_dir != 'auto':
        payload['grasp_dir'] = grasp_dir
        if side_approach_deg is not None:
            payload['side_approach_deg'] = side_approach_deg
        if apply_side_tcp_offset:
            payload['apply_side_tcp_offset'] = True
    _ros_node.publish_command(payload)
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result['target_pos'] = target_pos
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def move_joints(
    j1: float = None, j2: float = None, j3: float = None,
    j4: float = None, j5: float = None, j6: float = None, j7: float = None
) -> str:
    """로봇 팔의 각 관절(joint)을 직접 제어한다. 완료까지 블로킹.

    지정하지 않은 joint는 현재 각도 유지.

    Args:
        j1~j7: 각 관절 절대 각도 (라디안)

    Returns:
        성공: {"status": "success", "reason": "joint_move_complete", "joints": {...}}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    joints = {}
    if j1 is not None: joints['j1'] = j1
    if j2 is not None: joints['j2'] = j2
    if j3 is not None: joints['j3'] = j3
    if j4 is not None: joints['j4'] = j4
    if j5 is not None: joints['j5'] = j5
    if j6 is not None: joints['j6'] = j6
    if j7 is not None: joints['j7'] = j7

    if not joints:
        return json.dumps({'status': 'rejected',
                           'reason': 'joint 값을 하나 이상 지정해야 합니다.'})

    _ros_node.publish_command({'action': 'move_joints', 'joints': joints})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    result['joints'] = joints
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def move_joints_relative(j1: float = 0.0, j2: float = 0.0, j3: float = 0.0,
                          j4: float = 0.0, j5: float = 0.0, j6: float = 0.0,
                          j7: float = 0.0) -> str:
    """현재 관절 위치에서 상대적으로 이동한다 (라디안 단위).

    Args:
        j1~j7: 각 관절의 상대 이동량 (라디안, 기본값 0.0)
    """
    _ensure_ros()
    joints, stamp = _ros_node.get_joint_state()
    if not joints:
        return json.dumps({'status': 'failed', 'reason': '관절 피드백 없음'}, ensure_ascii=False)

    key_map = {'j1': 'joint1', 'j2': 'joint2', 'j3': 'joint3', 'j4': 'joint4',
               'j5': 'joint5', 'j6': 'joint6', 'j7': 'joint7'}
    deltas = {'j1': j1, 'j2': j2, 'j3': j3, 'j4': j4, 'j5': j5, 'j6': j6, 'j7': j7}

    move_joints_cmd = {}
    for k, delta in deltas.items():
        if delta != 0.0:
            jname = key_map[k]
            current = joints.get(jname, 0.0)
            move_joints_cmd[k] = round(current + delta, 6)

    if not move_joints_cmd:
        return json.dumps({'status': 'rejected', 'reason': '모든 delta가 0'}, ensure_ascii=False)

    _ros_node.publish_command({'action': 'move_joints', 'joints': move_joints_cmd})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_MOVE)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def go_home() -> str:
    """로봇 팔을 기본 복귀 자세로 되돌리고 그리퍼를 연다. 완료까지 블로킹.

    [2026-09-02 변경, 사용자 지시] 복귀 목표가 all-zero 홈이 아니라 저장된
    "observation"(지정) 자세다. all-zero 홈은 카메라가 장면을 안 보는
    자세라, go_home 직후 list_detected_objects/analyze_scene가 빈 결과를
    주고 매번 observation으로 다시 가야 했다. 이제 go_home 자체가
    observation으로 복귀하므로 그 왕복이 없다. (observation 저장 포즈가
    없으면 안전하게 all-zero로 폴백.)

    Returns:
        성공: {"status": "success", "reason": "home_complete"}
        실패: {"status": "failed"|"timeout", "reason": "..."}
    """
    _ensure_ros()
    _ros_node.publish_command({'action': 'home'})
    result = _ros_node.wait_for_result(timeout=TIMEOUT_HOME)
    return json.dumps(result, ensure_ascii=False)


def _openvocab_bbox(target_label: str, img_bgr: np.ndarray,
                     conf: float = 0.25):
    """[2026-09-02 추가] YOLO-World open-vocab 검출 티어.

    VLM_SERVER_URL/detect_open_vocab (YOLO-World, ~20~60ms) 로 target_label
    의 정밀 bbox 를 top-1 로 얻는다. 8B VLM /ground_object(2~4초, chat형
    좌표 부정확)보다 먼저 시도하는 fast-path.

    Returns:
        (bbox_norm: list[4] 또는 None, confidence: float, err: str 또는 None)
    err='openvocab_unavailable' 이면 서버에 YOLO-World 가 없다는 뜻 --
    호출부는 조용히 VLM /ground_object 로 폴백하면 된다.
    """
    OV_URL = f"{VLM_SERVER_URL}/detect_open_vocab"
    try:
        ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None, 0.0, 'encode_failed'
        b64 = base64.b64encode(buf.tobytes()).decode()
    except Exception as e:
        return None, 0.0, f'encode_failed — {e}'
    try:
        r = _requests.post(OV_URL, json={'full_image_b64': b64,
                                         'labels': [target_label],
                                         'conf': conf,
                                         'timestamp': time.time()},
                           timeout=10.0)
    except _requests.exceptions.ConnectionError:
        return None, 0.0, 'vlm_server_unavailable'
    except Exception as e:
        return None, 0.0, f'openvocab_request_failed — {e}'
    if r.status_code == 503:
        return None, 0.0, 'openvocab_unavailable'
    if r.status_code != 200:
        return None, 0.0, f'openvocab_http_error — {r.status_code}'
    try:
        dets = r.json().get('detections', [])
    except Exception:
        return None, 0.0, 'openvocab_invalid_json'
    if not dets:
        return None, 0.0, 'target_not_found'
    best = max(dets, key=lambda d: d.get('confidence', 0.0))
    bbox = best.get('bbox')
    if not bbox or len(bbox) != 4:
        return None, 0.0, 'no_valid_bbox'
    return [float(v) for v in bbox], float(best.get('confidence', 0.0)), None


def _vlm_ground_bbox_for_grasp(target_label: str, img_bgr: np.ndarray, objects: list):
    """[2026-08-26 추가] infer_grasp 전용 -- YOLO에 없는 물체의 bbox_norm 확보.

    [2026-09-02] 순서: (1) YOLO-World open-vocab(~수십 ms, 정밀 bbox) →
    (2) 8B VLM /ground_object(느림, 근사 bbox). (1) 이 서버에 없거나
    (openvocab_unavailable) 못 찾으면 (2) 로 폴백.

    Returns:
        (bbox_norm: list[4] 또는 None, confidence: float, error_reason: str 또는 None)
    """
    # ── (1) YOLO-World open-vocab fast-path ──────────────────────────────────
    ov_bbox, ov_conf, ov_err = _openvocab_bbox(target_label, img_bgr)
    if ov_bbox is not None:
        _logger.info(f'[infer_grasp/ground] open-vocab hit {target_label!r} '
                     f'conf={ov_conf:.2f} bbox={[round(v,3) for v in ov_bbox]}')
        return ov_bbox, ov_conf, None
    _logger.info(f'[infer_grasp/ground] open-vocab miss ({ov_err}) — VLM ground_object 폴백')

    GROUND_URL  = f"{VLM_SERVER_URL}/ground_object"
    VLM_TIMEOUT = 45.0

    def _encode(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode(img_bgr)
    except Exception as e:
        return None, 0.0, f'encode_failed — {e}'

    yolo_detections = [
        {'label': o.get('label', '?'), 'bbox': o.get('bbox', []),
         'confidence': round(float(o.get('confidence', 0.0)), 3)}
        for o in objects
    ]
    payload = {
        'full_image_b64': full_b64,
        'target_label':   target_label,
        'detections':     yolo_detections,
        'timestamp':      time.time(),
    }
    try:
        resp = _requests.post(GROUND_URL, json=payload, timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return None, 0.0, 'vlm_server_unavailable'
    except _requests.exceptions.Timeout:
        return None, 0.0, f'vlm_timeout — {VLM_TIMEOUT}s 초과'
    except Exception as e:
        return None, 0.0, f'vlm_request_failed — {e}'

    if resp.status_code != 200:
        return None, 0.0, f'vlm_http_error — {resp.status_code}'
    try:
        gdata = resp.json()
    except Exception:
        return None, 0.0, 'vlm_invalid_json'

    if not gdata.get('found', False):
        return None, 0.0, 'target_not_found'

    conf = float(gdata.get('confidence', 0.0))
    bbox_norm = gdata.get('bbox_norm')
    if not bbox_norm or len(bbox_norm) != 4:
        return None, conf, 'no_valid_bbox_from_vlm'
    return bbox_norm, conf, None


def _infer_grasp_core(target_label: str) -> dict:
    """[2026-09-02 추가] infer_grasp() MCP 툴 본체를 dict 반환 형태로 뽑아낸
    내부 헬퍼 -- _analyze_scene_core/_pick_object_impl과 동일한 패턴
    (로직은 여기, MCP 툴 함수는 json.dumps만 하는 얇은 wrapper).
    pick_object/slide_object의 grasp_dir='auto' 자동 판정(_resolve_auto_grasp_dir)이
    이 헬퍼를 그대로 재사용한다 -- Claude가 매번 손으로 infer_grasp를 부르고
    CLAUDE.md 변환표를 적용하던 것을 서버 내부로 옮긴 것.

    VLM은 "어떻게 잡을 것인가?"만 판단한다. "어디 있는가?"는 1차로 YOLO가
    담당한다. /detected_objects 에서 target_label과 일치하는 물체를 찾고,
    있으면 그 bbox로 crop해서 VLM_SERVER_URL/infer_grasp 에 POST한다.

    [2026-08-26 추가] YOLO 목록에 없는 라벨(예: dual-yolo box+coco 25종
    밖의 물체)이면 더 이상 바로 에러 내지 않고, ground_object()와 동일한
    VLM /ground_object 엔드포인트로 bbox를 자동으로 확보해서 계속
    진행한다 — 응답의 grounding_source 필드로 어느 쪽이었는지 확인 가능
    ("yolo"=정밀 bbox, "vlm"=근사 bbox라 crop이 다소 부정확할 수 있음).
    이제 ground_object로 미리 위치를 확보할 필요 없이 바로 호출해도 된다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    "컵 어떻게 잡아?"  → infer_grasp("cup")
    "컵 어디 있어?"   → list_detected_objects()  ← VLM 불필요

    Args:
        target_label: 추론할 물체의 YOLO 탐지 라벨 (예: "box", "cup")

    Returns:
        성공: {"status": "success", "object": "...", "grasp_type": "TOP|SIDE|PINCH",
               "orientation": "HORIZONTAL|VERTICAL", "confidence": 0.95,
               "reason": "...", "inference_ms": 1200.0,
               "bbox_px": [x1,y1,x2,y2], "image_age_sec": 0.1,
               "grounding_source": "yolo"|"vlm", "grounding_confidence": 0.85|null,
               "approach_direction": "FRONT|LEFT|RIGHT|BACK",
               "suggested_side_approach_deg": 0.0,
               "face_normal_yaw_deg": 37.2|null, "face_normal_confidence": 0.81}
               (grasp_type이 SIDE/PINCH면 suggested_side_approach_deg를
               pick_object/slide_object의 side_approach_deg에 그대로 넣어라
               -- 대략적인 시작 각도일 뿐, 틀려도 서버의 접근각 후보 탐색이
               안전망 역할을 한다. 구버전 VLM 서버는 이 필드를 안 주므로
               항상 FRONT/0.0으로 채워진다. face_normal_yaw_deg는 [2026-08
               추가, 프로토타입] depth 평면적합 기반 -- 카메라가 마침 물체
               옆면을 보고 있을 때만 값이 나오고(윗면 위주로 보고 있으면
               null) suggested_side_approach_deg보다 더 정밀할 수 있지만
               아직 모션 결정에는 연결 안 됨, 참고용으로만 볼 것.)
        실패: {"status": "error", "reason": "..."}
    """
    _ensure_ros()

    VLM_URL     = f"{VLM_SERVER_URL}/infer_grasp"
    VLM_TIMEOUT = 30.0   # Qwen2.5-VL inference 시간 여유 확보

    # ── 1. 최신 카메라 이미지 가져오기 (YOLO 폴백 시에도 필요해서 먼저 확보) ──
    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return {'status': 'error',
                'reason': 'image_unavailable — /camera/color/image_raw 수신 없음'}

    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return {'status': 'error',
                'reason': f'image_stale — 이미지가 {img_age:.1f}초 경과 (>3s)'}

    h, w = img_bgr.shape[:2]

    # ── 2. 감지된 물체 목록에서 target_label 찾기 (YOLO 우선) ──────────────
    objects, obj_stamp = _ros_node.get_objects()
    label_lower = target_label.strip().lower()
    matched = [o for o in objects if str(o.get('label', '')).lower() == label_lower] if objects else []

    grounding_source = 'yolo'
    grounding_conf = None
    if matched:
        obj = matched[0]
        bbox_norm = obj.get('bbox')   # [x_min, y_min, x_max, y_max] 0~1 정규화
    else:
        # [2026-08-26 추가] YOLO 목록에 없으면(라벨을 아예 모르거나 confidence
        # 미달) 바로 에러 내지 않고 ground_object()와 동일한 VLM
        # /ground_object 엔드포인트로 bbox를 확보해서 계속 진행한다.
        # 이전엔 여기서 무조건 'object_not_found'였는데, "book"처럼
        # dual-yolo(box+coco 25종) 밖의 라벨은 infer_grasp 자체를 못 쓰는
        # 문제가 있었다 — VLM(ground_object)은 보는데 infer_grasp만 못 보는
        # 불일치가 실측 확인됨.
        bbox_norm, grounding_conf, ground_err = _vlm_ground_bbox_for_grasp(
            target_label, img_bgr, objects or [])
        if bbox_norm is None:
            available = [o.get('label', '') for o in objects] if objects else []
            return {'status': 'error',
                    'reason': f'not_found_by_yolo_or_vlm — yolo_labels={available}, '
                              f'vlm_ground_error={ground_err}'}
        grounding_source = 'vlm'
        obj = {'label': target_label, 'confidence': grounding_conf}

    # ── 3. bbox crop 생성 ────────────────────────────────────────────────────
    if bbox_norm and len(bbox_norm) == 4:
        x1 = int(bbox_norm[0] * w)
        y1 = int(bbox_norm[1] * h)
        x2 = int(bbox_norm[2] * w)
        y2 = int(bbox_norm[3] * h)
        # 클램프 + 최소 크기 보장
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return {'status': 'error',
                    'reason': f'bbox_too_small — crop 크기 {x2-x1}x{y2-y1}px'}
        crop_bgr = img_bgr[y1:y2, x1:x2]
        bbox_px  = [x1, y1, x2, y2]
    else:
        # bbox 정보가 없으면 전체 이미지를 crop으로 사용
        crop_bgr = img_bgr.copy()
        bbox_px  = [0, 0, w, h]

    # ── 4. base64 JPEG 인코딩 ───────────────────────────────────────────────
    def _encode_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode_b64(img_bgr)
        crop_b64 = _encode_b64(crop_bgr)
    except Exception as e:
        return {'status': 'error', 'reason': f'encode_failed — {e}'}

    # ── 5. VLM 서버 POST ────────────────────────────────────────────────────
    payload = {
        'full_image_b64': full_b64,
        'crop_image_b64': crop_b64,
        'object_label':   obj.get('label', target_label),
        'bbox':           bbox_norm if bbox_norm else [0, 0, 1, 1],
        'timestamp':      time.time(),
    }

    try:
        resp = _requests.post(VLM_URL, json=payload, timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return {'status': 'error',
                'reason': f'vlm_server_unavailable — {VLM_SERVER_URL} 연결 거부'}
    except _requests.exceptions.Timeout:
        return {'status': 'error',
                'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'}
    except Exception as e:
        return {'status': 'error', 'reason': f'vlm_request_failed — {e}'}

    if resp.status_code != 200:
        return {'status': 'error',
                'reason': f'vlm_http_error — status {resp.status_code}: {resp.text[:200]}'}

    # ── 6. 응답 파싱 및 검증 ────────────────────────────────────────────────
    try:
        data = resp.json()
    except Exception:
        return {'status': 'error',
                'reason': f'vlm_invalid_json — {resp.text[:200]}'}

    grasp_type  = str(data.get('grasp_type', '')).upper()
    orientation = str(data.get('orientation', '')).upper()
    confidence  = data.get('confidence', 0.0)

    if grasp_type not in ('TOP', 'SIDE', 'PINCH'):
        return {'status': 'error',
                'reason': f'vlm_invalid_grasp_type — {grasp_type!r}'}
    if orientation not in ('HORIZONTAL', 'VERTICAL'):
        orientation = 'VERTICAL' if grasp_type == 'PINCH' else 'HORIZONTAL'
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    # [2026-08-26 추가] VLM이 준 정성적 접근방향(FRONT/LEFT/RIGHT/BACK,
    # 카메라 시점 기준)을 pick_object/slide_object의 side_approach_deg
    # (position_yaw 기준 상대각, 도)로 쓸 대략적인 시작값으로 변환한다.
    # 정밀한 각도가 아니라 "후보 탐색의 1순위 추측값" 용도 -- 틀려도
    # planning_node의 접근각 후보 스윕(±15~90°)이 안전망 역할을 한다.
    # VLM 서버가 이 필드를 아직 안 보내는 구버전이면 기본 FRONT(0°)로
    # 처리해서 하위호환된다.
    _APPROACH_DEG_MAP = {'FRONT': 0.0, 'LEFT': 90.0, 'RIGHT': -90.0, 'BACK': 180.0}
    approach_direction = str(data.get('approach_direction', 'FRONT')).upper()
    if approach_direction not in _APPROACH_DEG_MAP:
        approach_direction = 'FRONT'
    suggested_side_approach_deg = _APPROACH_DEG_MAP[approach_direction]

    # [2026-08 추가, 프로토타입 -- 실기 미검증] SIDE/PINCH일 때 접근각
    # 추정에 참고할 만한 depth 평면적합 방위각. suggested_side_approach_deg
    # (VLM의 정성적 FRONT/LEFT/RIGHT/BACK 추측)보다 더 정밀할 수 있지만,
    # 카메라가 마침 물체 옆면을 보고 있어야만 값이 나온다(윗면 위주로 보고
    # 있으면 null) -- 항상 나오는 값이 아니므로 suggested_side_approach_deg
    # 를 대체하지 않고 나란히 노출만 한다.
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, _cw, _ch = _ros_node.get_cam_intrinsics()
    face_yaw_deg, face_conf, _face_n = _compute_face_normal_yaw_from_bbox(
        bbox_px, depth_arr, fx, fy, cx, cy)

    return {
        'status':           'success',
        'object':           data.get('object', target_label),
        'grasp_type':       grasp_type,
        'orientation':      orientation,
        'confidence':       round(confidence, 3),
        'reason':           data.get('reason', ''),
        'inference_ms':     data.get('inference_ms', 0.0),
        'bbox_px':          bbox_px,
        'image_age_sec':    img_age,
        # [2026-08-26 추가] bbox 출처 -- 'yolo'면 정밀, 'vlm'이면 ground_object
        # 경유 근사 bbox라 crop이 다소 부정확할 수 있음(참고용으로 노출).
        'grounding_source': grounding_source,
        'grounding_confidence': round(grounding_conf, 3) if grounding_conf is not None else None,
        # [2026-08-26 추가] side/pinch일 때 pick_object(side_approach_deg=)에
        # 그대로 넣을 수 있는 대략적 시작 각도. approach_direction은 VLM
        # 원본 응답(FRONT/LEFT/RIGHT/BACK), 구버전 VLM 서버면 항상 FRONT/0.0.
        'approach_direction': approach_direction,
        'suggested_side_approach_deg': suggested_side_approach_deg,
        # [2026-08 추가, 프로토타입] depth 평면적합 기반 mod-180 방위각 --
        # 신뢰 불가(카메라가 물체 윗면 위주로 보고 있음/평면성 낮음)면 null.
        # pick_object 등 모션 결정에는 아직 연결 안 됨, 참고용.
        'face_normal_yaw_deg': face_yaw_deg,
        'face_normal_confidence': face_conf,
    }


@mcp.tool()
def infer_grasp(target_label: str) -> str:
    """지정한 물체에 대해 VLM grasp 타입을 추론한다 (명시적 온디맨드 호출). 상세
    동작/응답 필드는 _infer_grasp_core 참고. MCP 툴 함수는 json.dumps만 하는
    얇은 wrapper(다른 리팩터링된 함수들과 동일 패턴)."""
    return json.dumps(_infer_grasp_core(target_label), ensure_ascii=False)


def _resolve_auto_grasp_dir(target_label: str, object_z: float = None):
    """[2026-09-02 추가] pick_object/slide_object의 grasp_dir='auto' 경로를
    grasp_types.resolve_grasp_dir(CLAUDE.md 그립형태 변환표)에 연결한다.
    지금까지는 Claude가 매 pick마다 infer_grasp를 손으로 부르고 표를
    수동 적용해야 했다 -- 이제 서버가 auto일 때 이 과정을 대신한다.

    Returns:
        (grasp_dir, side_approach_deg, reason) -- infer_grasp가 실패하면
        grasp_dir=None (호출부는 이 경우 기존 폴백 동작으로 넘어가야 함).
    """
    vlm = _infer_grasp_core(target_label)
    if vlm.get('status') != 'success':
        return None, None, f"infer_grasp 실패({vlm.get('reason')}) — auto 판정 불가"
    resolved = _resolve_grasp_dir_policy(vlm, object_z)
    return resolved.grasp_dir, resolved.side_approach_deg, resolved.reason


def _analyze_scene_core() -> dict:
    """[2026-08-31 추가] analyze_scene() 본체를 dict 반환 형태로 뽑아낸
    내부 헬퍼 -- analyze_scene() MCP 툴과, pick 폐루프 검증(2차 확인)이
    둘 다 이 로직을 그대로 재사용한다(중복 유지 대신 공유). MCP 툴
    함수를 다른 코드에서 직접 호출하는 대신, 로직을 여기로 뽑고 툴
    함수는 얇은 wrapper(json.dumps)만 남기는 패턴 -- infer_grasp/
    ground_object의 VLM 폴백 경로가 이미 "서로 다른 툴이지만 내부
    로직만 공유"하는 것과 동일한 원칙."""
    _ensure_ros()

    ANALYZE_URL = f"{VLM_SERVER_URL}/analyze_scene"
    VLM_TIMEOUT = 45.0

    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return {'status': 'error',
                'reason': 'image_unavailable — /camera/color/image_raw 수신 없음'}
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return {'status': 'error', 'reason': f'image_stale — {img_age:.1f}s 경과 (>3s)'}

    yolo_objects, _ = _ros_node.get_objects()
    detections = [{'label': o.get('label', '?'), 'bbox': o.get('bbox', [])} for o in yolo_objects]

    def _encode_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode_b64(img_bgr)
    except Exception as e:
        return {'status': 'error', 'reason': f'encode_failed — {e}'}

    try:
        resp = _requests.post(ANALYZE_URL,
                              json={'full_image_b64': full_b64, 'detections': detections,
                                    'timestamp': time.time()},
                              timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return {'status': 'error',
                'reason': f'vlm_server_unavailable — {VLM_SERVER_URL} 연결 거부'}
    except _requests.exceptions.Timeout:
        return {'status': 'error', 'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'}
    except Exception as e:
        return {'status': 'error', 'reason': f'vlm_request_failed — {e}'}

    if resp.status_code != 200:
        return {'status': 'error',
                'reason': f'vlm_http_error — status {resp.status_code}: {resp.text[:200]}'}

    try:
        data = resp.json()
    except Exception:
        return {'status': 'error', 'reason': f'vlm_invalid_json — {resp.text[:200]}'}

    yolo_label_set = {str(o.get('label', '')).lower() for o in yolo_objects}
    vlm_objects    = data.get('objects', [])
    vlm_label_set  = {str(o.get('label', '')).lower() for o in vlm_objects}

    objects_merged = [
        {**o, 'source': 'both' if str(o.get('label', '')).lower() in yolo_label_set else 'vlm'}
        for o in vlm_objects
    ]
    for yo in yolo_objects:
        if str(yo.get('label', '')).lower() not in vlm_label_set:
            objects_merged.append({
                'label':  yo.get('label', '?'),
                'bbox':   yo.get('bbox', []),
                'source': 'yolo',
            })

    global _last_scene_objects, _last_scene_stamp
    with _scene_cache_lock:
        _last_scene_objects = [o.copy() for o in objects_merged]
        _last_scene_stamp   = time.time()

    # [2026-09-02] 놓을 곳 후보: VLM 이 리셉터클(container_type != "none")로
    # 태그한 물체. is_open=True 를 먼저(개구부가 카메라 쪽을 향해 드롭 가능).
    # 좌표는 여기서 계산하지 않는다 -- 호출부가 ground_object(label) 로 3D 를
    # 확보하는 게 CLAUDE.md "놓을 곳은 pick 전에 미리 좌표 확보" 규칙과 맞음.
    placement_targets = [
        {'label': o.get('label', '?'), 'bbox': o.get('bbox', []),
         'container_type': o.get('container_type', 'none'),
         'is_open': bool(o.get('is_open', False)),
         'source': o.get('source', 'vlm')}
        for o in objects_merged
        if str(o.get('container_type', 'none')) != 'none'
    ]
    placement_targets.sort(key=lambda c: (not c['is_open'], c['label']))

    return {
        'status':            'success',
        'yolo_detected':     [o.get('label', '?') for o in yolo_objects],
        'objects':           objects_merged,
        'placement_regions': data.get('placement_regions', []),
        'placement_targets': placement_targets,
        'inference_ms':      data.get('inference_ms', 0.0),
        'image_age_sec':     img_age,
    }


@mcp.tool()
def analyze_scene() -> str:
    """현재 카메라 화면에 보이는 모든 물체를 label + bbox로 반환한다.

    VLM이 카메라 이미지를 독립적으로 보고 물체를 탐지한다.
    YOLO 결과와 병합해서 전체 scene을 반환한다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ○ 사용: "화면에 뭐가 있어?" / "YOLO가 못 찾은 것도 알려줘" / 전체 장면 파악
           "어디 놓을 수 있어?" / "빈 공간 찾아줘" / placement 영역 탐색
    ✗ 금지: 단일 물체 위치 확인   → list_detected_objects() 또는 ground_object()
    ✗ 금지: YOLO에 이미 있는 물체를 다시 찾기 위한 호출
    ✗ 금지: VLM 실패 시 robot movement 명령

    source 태깅:
    - "both" = YOLO와 VLM 모두 탐지
    - "vlm"  = VLM만 탐지 (예: 선반·바구니·펜처럼 YOLO가 없는 물체)
    - "yolo" = YOLO만 탐지 (VLM이 못 찾은 경우)

    ─── VLM 실패 시 절대 금지 ──────────────────────────────────────────────────
    status가 "error"이면:
    - YOLO 좌표로 "빈 공간" / placement 위치 추론 금지
    - robot movement 명령 금지
    반드시: "시각적 scene reasoning을 수행할 수 없습니다. (이유: {reason})"

    Returns:
        성공: {"status":"success",
               "yolo_detected":["cup","box"],
               "objects":[{"label":"cup","bbox":[x1,y1,x2,y2],"source":"both",
                           "container_type":"none","is_open":false},
                          {"label":"basket","bbox":[...],"source":"vlm",
                           "container_type":"basket","is_open":true},...],
               "placement_regions":[{"bbox":[x1,y1,x2,y2],"confidence":0.8},...],
               "placement_targets":[{"label":"basket","container_type":"basket",
                           "is_open":true,"bbox":[...]}, ...],  # 리셉터클만, is_open 우선
               "inference_ms":8000.0,"image_age_sec":0.1}
        실패: {"status":"error","reason":"..."}

    "바구니/선반/서랍에 놓아" 류 요청: placement_targets 에서 원하는
    container_type 을 고른 뒤 ground_object(그 label) 로 3D 좌표를 확보하고
    (pick 전에!) place_object 에 넘긴다.
    """
    return json.dumps(_analyze_scene_core(), ensure_ascii=False)


@mcp.tool()
def ground_object(target_label: str, parent_label: str = None,
                  center_view: bool = False) -> str:
    """YOLO에 없는 물체를 VLM으로 찾아 depth + TF로 3D 좌표를 추정한다.

    동작 순서 (일반):
      1. YOLO /detected_objects에 target_label이 있으면 그 좌표를 바로 반환
         (source=yolo, grounding=detected, position_confidence=precise)
      2. YOLO에 없으면 VLM /ground_object 호출 → approximate bbox/center_norm 획득
      3. bbox 영역의 depth를 robust median으로 샘플링
      4. pixel_to_camera_xyz → camera_color_optical_frame 3D 점
      5. TF → base_link 3D 좌표 반환
         (source=vlm, grounding=approximate, position_confidence=approximate)

    ─── Hierarchical Grounding (parent_label 지정 시) ────────────────────────
    parent object crop → child part grounding → depth → TF → 3D (2D fallback).

    Use normal grounding when:
    - target이 독립적인 물체인 경우
    - 전체 scene에서 충분히 찾을 수 있는 크기

    Use hierarchical grounding when:
    - target이 큰 물체의 일부 part인 경우
    - handle, knob, button, rim, switch, opening 등 작은 조작 대상
    - 전체 이미지에서 바로 찾으면 놓칠 가능성이 높은 경우

    Examples:
      ground_object("pen")
      ground_object("drawer handle", parent_label="yellow drawer")
      ground_object("cup rim", parent_label="cup")

    라우팅 규칙:
      "은색 선반이 어디 있어?"       → ground_object("silver shelf")
      "서랍 손잡이 위치 알려줘."     → ground_object("drawer handle", parent_label="yellow drawer")
      "컵을 어떻게 잡아?"            → infer_grasp("cup")  (위치 아님)
      "화면에 뭐가 있어?"            → analyze_scene()

    Args:
        target_label: 찾을 물체 설명 (예: "silver shelf", "drawer handle")
        parent_label: [선택] hierarchical grounding 시 parent 물체 라벨
                     (예: "yellow drawer"). None이면 기존 동작 유지.
        center_view: [2026-09-02] True면, grounding 전에 observation 자세로
                     가서 joint7만 스윕하며 물체가 프레임에 다 들어오는
                     자세를 찾아 거기서 grounding한다(끝나면 observation으로
                     복원). **바구니/트레이/선반 등 큰 리셉터클의 중심
                     좌표를 place용으로 확보할 땐 True로 부르라** --
                     observation 자세에서 화면 가장자리에 걸린 컨테이너는
                     VLM이 보이는 앞쪽만 잡아 중심이 80mm+ 어긋난다(실측
                     2026-09-02). target_label이 컨테이너 키워드(basket/tray
                     /선반/...)면 center_view 없이도 자동으로 이 스윕을 한다.
                     찾은 joint7은 세션 캐시라 같은 컨테이너 재요청 시
                     스윕을 다시 안 한다. 작은 독립 물체엔 불필요(느려짐).

    Returns:
        YOLO에 있을 때:
          {"label":"cup", "source":"yolo", "grounding":"detected",
           "center_px":[u,v], "camera_point":{x,y,z}, "base_link_point":{x,y,z},
           "confidence":0.95, "position_confidence":"precise"}
        VLM grounding 성공 시:
          {"label":"silver shelf", "source":"vlm", "grounding":"approximate",
           "center_px":[u,v], "bbox_approx":[x1,y1,x2,y2],
           "camera_point":{x,y,z}, "base_link_point":{x,y,z},
           "confidence":0.78, "position_confidence":"approximate",
           "face_normal_yaw_deg":37.2|null, "face_normal_confidence":0.81,
           "depth_method":"container_p20"|"robust_median",  # [C] 컨테이너면
               near 백분위로 depth 샘플 (내부 바닥 쏠림 방지)
           "is_container":true|false,
           "bbox_truncated":true|false,  # [B] bbox가 이미지 경계에 닿음 =
               물체가 프레임 밖으로 벗어남. true면 position_confidence가
               "approximate_truncated"로 내려가고 좌표는 보이는 부분 기준.
           "bbox_truncated_sides":["right"],  # 닿은 변 (truncated=true일 때만)
           "container_view":{"joint1":-0.6,"joint7":1.45,"partial":false,
               "source":"sweep"|"cache"|"sweep_failed"}}  # [2026-09-02]
               컨테이너라서 grounding 전에 observation→joint1 정렬(중심 x를
               화면 중앙)→joint7 스윕으로 프레이밍을 잡고 측정했다는 표시.
               측정 후 observation으로 복원됨. source="sweep_failed"면
               자세를 못 잡아 현재 시야 그대로 측정한 것(좌표 불신).
          (face_normal_yaw_deg는 [2026-08 추가, 프로토타입] depth 평면적합
          기반 mod-180 방위각 -- YOLO는 원래 이 필드가 없었으니 VLM경로
          에서만 시도하는 신규 정보. 카메라가 물체 옆면을 볼 때만 값이
          나오고, 윗면을 보고 있거나 평면성이 낮으면 null. 아직
          pick_object 등 모션 결정에는 연결 안 됨 -- 참고용으로만 볼 것.)
        Hierarchical grounding 3D 성공 시:
          {"success":true, "label":"drawer handle", "source":"vlm",
           "grounding":"hierarchical", "center_px":[u,v], "bbox_approx":[x1,y1,x2,y2],
           "camera_point":{x,y,z}, "base_link_point":{x,y,z},
           "confidence":0.78, "position_confidence":"approximate",
           "parent_label":"yellow drawer", "parent_source":"yolo"|"vlm"}
        Hierarchical grounding 2D fallback (depth/TF 실패):
          {"success":true, ..., "camera_point":null, "base_link_point":null,
           "position_confidence":"2d_only"}
        실패 시:
          {"success":false, "reason":"target_not_found"|"parent_not_found"|
                            "invalid_parent_bbox"|"child_not_found_in_parent"|
                            "invalid_depth"|"tf_unavailable"|...}
    """
    return _ground_object_impl(target_label, parent_label=parent_label,
                               center_view=center_view)


def _acquire_container_view(label_lower: str, target_label: str) -> Optional[dict]:
    """컨테이너 grounding 전에 프레이밍 좋은 joint7 자세를 확보한다.
    캐시된 (joint1,joint7)이 있으면 거기로, 없으면 _find_container_view로.
    Returns dict(무엇을 했는지) 또는 None(스윕도 이동도 아무것도 안 함).
    'no_move': True면 자세를 못 잡아서 현재 시야 그대로 측정하라는 뜻."""
    cached = _container_view_cache.get(label_lower)
    if cached is not None:
        if (_goto_observation()
                and _set_joint1(cached['joint1'])
                and _set_joint7(cached['joint7'])):
            _logger.info(f'[container_view] {target_label!r} 캐시 {cached} 사용')
            return {**cached, 'source': 'cache'}
        _container_view_cache.pop(label_lower, None)  # 이동 실패 -> 캐시 폐기
    view = _find_container_view(target_label)
    if view is None:
        _logger.warning(
            f'[container_view] {target_label!r} 자세 못 찾음 -- 현재 시야에서 측정')
        return {'joint1': None, 'joint7': None,
                'source': 'sweep_failed', 'no_move': True}
    _container_view_cache[label_lower] = {'joint1': view['joint1'],
                                         'joint7': view['joint7']}
    view['source'] = 'sweep'
    return view


def _ground_object_impl(target_label: str, parent_label: str = None,
                        center_view: bool = False) -> str:
    """ground_object() MCP 툴의 실제 로직.
    [2026-09-02] center_view=True 이거나 target_label이 컨테이너 키워드면,
    grounding 전에 observation→joint1 정렬(bbox 중심 x를 화면 중앙)→joint7
    스윕(_find_container_view)으로 컨테이너가 프레임에 다 들어오는 자세를
    확보하고 거기서 측정한 뒤 observation으로 복원한다. 찾은 (joint1,
    joint7)은 _container_view_cache에 세션 캐시.
    B = bbox가 이미지 경계에 닿으면 bbox_truncated + confidence 하향.
    C = 컨테이너 depth는 near 백분위(_sample_depth_container)."""
    _ensure_ros()

    GROUND_URL  = f"{VLM_SERVER_URL}/ground_object"
    VLM_TIMEOUT = 45.0
    MIN_CONF    = 0.3

    # ── Hierarchical grounding 위임 ───────────────────────────────────────────
    if parent_label is not None:
        return _ground_hierarchical(target_label, parent_label,
                                    GROUND_URL, VLM_TIMEOUT, MIN_CONF)

    # ── 1. YOLO에서 먼저 검색 ────────────────────────────────────────────────
    objects, _ = _ros_node.get_objects()
    label_lower = target_label.strip().lower()
    matched = [o for o in objects if str(o.get('label', '')).lower() == label_lower]

    if matched:
        obj = matched[0]
        c3d = obj.get('center_3d', {})
        # YOLO는 이미 base_link 좌표를 가지고 있음
        bbox_norm = obj.get('bbox', [])
        img_bgr, _ = _ros_node.get_image()
        center_px = None
        if img_bgr is not None and len(bbox_norm) == 4:
            h, w = img_bgr.shape[:2]
            cx = int((bbox_norm[0] + bbox_norm[2]) / 2 * w)
            cy = int((bbox_norm[1] + bbox_norm[3]) / 2 * h)
            center_px = [cx, cy]
        return json.dumps({
            'label':              obj.get('label', target_label),
            'source':             'yolo',
            'grounding':          'detected',
            'center_px':          center_px,
            'camera_point':       None,
            'base_link_point':    c3d,
            'confidence':         round(float(obj.get('confidence', 1.0)), 3),
            'position_confidence': 'precise',
        }, ensure_ascii=False)

    # ── [2026-09-02] 컨테이너면 joint1 정렬 + joint7 스윕으로 프레이밍
    #    확보 후 측정, 끝나면 observation 복원 ──
    _view = None
    if center_view or _looks_like_container(target_label):
        _view = _acquire_container_view(label_lower, target_label)
    try:
        _res_json = _ground_object_vlm(target_label, GROUND_URL, VLM_TIMEOUT, MIN_CONF)
    finally:
        if _view is not None:
            _goto_observation()  # 자세를 뭐라도 건드렸으면 되돌린다
    if _view is not None:
        try:
            _d = json.loads(_res_json)
            _d['container_view'] = _view
            _res_json = json.dumps(_d, ensure_ascii=False)
        except Exception:
            pass
    return _res_json


def _ground_object_vlm(target_label: str, GROUND_URL: str,
                       VLM_TIMEOUT: float, MIN_CONF: float) -> str:
    """현재 카메라 시야에서 target_label을 VLM(open-vocab→8B 폴백)으로
    grounding하고 depth+TF로 3D를 붙여 JSON 문자열을 반환한다.
    _ground_object_impl이 (컨테이너면 joint7 스윕으로 자세를 잡은 뒤)
    호출한다."""
    objects, _ = _ros_node.get_objects()

    # ── 2. YOLO 미감지 → VLM grounding ──────────────────────────────────────
    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'success': False,
                           'reason': 'image_unavailable'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'success': False,
                           'reason': f'image_stale — {img_age:.1f}s'})

    # 이미지 인코딩
    def _encode(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    # ── 2a. YOLO-World open-vocab fast-path ─────────────────────────────────
    # [2026-09-02] 8B VLM /ground_object(느림, 근사 bbox) 전에 open-vocab
    # 검출(~수십 ms, 정밀 bbox)을 먼저 시도. 서버에 YOLO-World 가 없거나
    # (openvocab_unavailable) 못 찾으면 아래 VLM 경로로 폴백.
    gdata = None
    ov_bbox, ov_conf, ov_err = _openvocab_bbox(target_label, img_bgr, conf=MIN_CONF)
    if ov_bbox is not None:
        _logger.info(f'[ground_object] open-vocab hit {target_label!r} conf={ov_conf:.2f}')
        gdata = {
            'found': True,
            'label': target_label,
            'bbox_norm': ov_bbox,
            'center_norm': [round((ov_bbox[0] + ov_bbox[2]) / 2, 4),
                            round((ov_bbox[1] + ov_bbox[3]) / 2, 4)],
            'confidence': ov_conf,
            'description': 'open-vocab detection (YOLO-World)',
            'inference_ms': 0.0,
            '_grounding': 'open_vocab',
        }
    else:
        _logger.info(f'[ground_object] open-vocab miss ({ov_err}) — VLM ground_object 폴백')

    if gdata is None:
        try:
            full_b64 = _encode(img_bgr)
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'encode_failed — {e}'})

        yolo_detections = [
            {'label': o.get('label', '?'),
             'bbox':  o.get('bbox', []),
             'confidence': round(float(o.get('confidence', 0.0)), 3)}
            for o in objects
        ]

        payload = {
            'full_image_b64': full_b64,
            'target_label':   target_label,
            'detections':     yolo_detections,
            'timestamp':      time.time(),
        }
        try:
            resp = _requests.post(GROUND_URL, json=payload, timeout=VLM_TIMEOUT)
        except _requests.exceptions.ConnectionError:
            return json.dumps({'success': False,
                               'reason': 'vlm_server_unavailable'})
        except _requests.exceptions.Timeout:
            return json.dumps({'success': False,
                               'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'})
        except Exception as e:
            return json.dumps({'success': False, 'reason': f'vlm_request_failed — {e}'})

        if resp.status_code != 200:
            return json.dumps({'success': False,
                               'reason': f'vlm_http_error — {resp.status_code}'})

        try:
            gdata = resp.json()
        except Exception:
            return json.dumps({'success': False, 'reason': 'vlm_invalid_json'})

    if not gdata.get('found', False):
        return json.dumps({'success': False, 'reason': 'target_not_found'})

    conf = float(gdata.get('confidence', 0.0))
    if conf < MIN_CONF:
        return json.dumps({'success': False,
                           'reason': f'low_grounding_confidence — {conf:.2f} < {MIN_CONF}'})

    # ── 3. bbox_norm → pixel 좌표 변환 ──────────────────────────────────────
    h, w = img_bgr.shape[:2]
    bbox_norm = gdata.get('bbox_norm', [])
    center_norm = gdata.get('center_norm', [])

    if len(bbox_norm) == 4:
        bx1 = int(bbox_norm[0] * w)
        by1 = int(bbox_norm[1] * h)
        bx2 = int(bbox_norm[2] * w)
        by2 = int(bbox_norm[3] * h)
        bbox_px = [bx1, by1, bx2, by2]
    else:
        bbox_px = None

    # bbox_norm에서 직접 계산 (VLM center_norm은 부정확할 수 있어 사용 안 함)
    if bbox_px is not None:
        u = (bbox_px[0] + bbox_px[2]) / 2.0
        v = (bbox_px[1] + bbox_px[3]) / 2.0
    elif len(center_norm) == 2:
        u = center_norm[0] * w
        v = center_norm[1] * h
    else:
        return json.dumps({'success': False, 'reason': 'no_valid_bbox_from_vlm'})

    center_px = [int(round(u)), int(round(v))]

    # ── 4. depth 샘플링 (없으면 시각적 결과만 반환) ─────────────────────────
    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, cam_w, cam_h = _ros_node.get_cam_intrinsics()

    if depth_arr is None or fx is None:
        # depth 없음 → 3D 좌표 없이 시각적 결과만 반환
        return json.dumps({
            'label':              gdata.get('label', target_label),
            'source':             gdata.get('_grounding', 'vlm'),
            'grounding':          'visual_only',
            'center_px':          center_px,
            'bbox_approx':        bbox_px,
            'camera_point':       None,
            'base_link_point':    None,
            'confidence':         round(conf, 3),
            'position_confidence': 'visual_only — depth 없음, 3D 좌표 불가',
            'description':        gdata.get('description', ''),
            'inference_ms':       gdata.get('inference_ms', 0.0),
        }, ensure_ascii=False)

    # [2026-09-02 C] 컨테이너면 bbox median(내부 바닥으로 쏠림) 대신 낮은
    # 백분위(near rim/opening 평면 높이)로 depth를 잡는다.
    _is_container = _looks_like_container(target_label, gdata.get('label'),
                                         gdata.get('description'))
    _depth_method = 'robust_median'
    depth_m = None
    if bbox_px is not None:
        if _is_container:
            depth_m = _sample_depth_container(depth_arr, bbox_px[0], bbox_px[1],
                                              bbox_px[2], bbox_px[3])
            if depth_m is not None:
                _depth_method = f'container_p{int(_CONTAINER_DEPTH_PCT)}'
        if depth_m is None:
            depth_m = _sample_depth_robust(depth_arr, bbox_px[0], bbox_px[1],
                                            bbox_px[2], bbox_px[3])
    else:
        win = 20
        depth_m = _sample_depth_robust(depth_arr,
                                        center_px[0] - win, center_px[1] - win,
                                        center_px[0] + win, center_px[1] + win)

    if depth_m is None:
        return json.dumps({'success': False, 'reason': 'invalid_depth'})

    # ── 5. camera XYZ ────────────────────────────────────────────────────────
    cam_xyz = _px2cam(u, v, depth_m, fx, fy, cx, cy)
    if cam_xyz is None:
        return json.dumps({'success': False, 'reason': 'invalid_depth'})

    # ── 6. TF → base_link ────────────────────────────────────────────────────
    base_xyz = _cam_to_base(cam_xyz)
    if base_xyz is None:
        return json.dumps({'success': False, 'reason': 'tf_unavailable'})

    # [2026-08 추가, 프로토타입 -- 실기 미검증] YOLO 못 잡은 물체는 원래
    # angle_base_deg 자체가 없었다 -- 여기서라도 평면적합으로 방위각을
    # 시도한다. bbox_px가 있어야(면적 있는 영역이어야) 의미 있으므로
    # center_norm만 온 경우(bbox_px is None)는 스킵.
    face_yaw_deg, face_conf, _face_n = (
        _compute_face_normal_yaw_from_bbox(bbox_px, depth_arr, fx, fy, cx, cy)
        if bbox_px is not None else (None, 0.0, 0)
    )

    _ov = gdata.get('_grounding') == 'open_vocab'
    result = {
        'label':              gdata.get('label', target_label),
        'source':             'open_vocab' if _ov else 'vlm',
        'grounding':          'precise_bbox' if _ov else 'approximate',
        'center_px':          center_px,
        'bbox_approx':        bbox_px,
        'camera_point':       cam_xyz,
        'base_link_point':    base_xyz,
        'confidence':         round(conf, 3),
        'position_confidence': 'precise_bbox' if _ov else 'approximate',
        'description':        gdata.get('description', ''),
        'inference_ms':       gdata.get('inference_ms', 0.0),
        'depth_m':            round(depth_m, 3),
        'depth_method':       _depth_method,  # [2026-09-02 C] robust_median |
                                              # container_p20 (near rim 쪽)
        'is_container':       _is_container,
        # [2026-08 추가, 프로토타입] mod-180 각도, 신뢰 불가 시 null.
        # side_approach_deg 계산의 참고용 -- planning_node.py엔 아직 미연결.
        'face_normal_yaw_deg': face_yaw_deg,
        'face_normal_confidence': face_conf,
    }

    # ── [2026-09-02 B] bbox 잘림 감지 ──
    # 컨테이너면 호출부(_ground_object_impl)가 joint7 스윕으로 프레이밍을
    # 이미 맞춘 뒤라 보통 안 잘리지만, 스윕이 완전 프레이밍을 못 찾았거나
    # (partial) 작은 물체를 그냥 부른 경우엔 여전히 잘릴 수 있다.
    _trunc_sides = _bbox_edge_sides(bbox_px, w, h)
    result['bbox_truncated'] = bool(_trunc_sides)
    if _trunc_sides:
        result['bbox_truncated_sides'] = _trunc_sides
        result['position_confidence'] = 'approximate_truncated'

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def estimate_object_geometry(target_label: str, parent_label: str = None) -> str:
    """[2026-08 추가, 프로토타입 -- 실기 미검증] 물체의 3D 형상(RANSAC 평면 +
    PCA 주축)을 실측 depth로 계산한다. pick_object의 자세 결정에는 아직
    자동 연결되지 않았다 -- 이 값을 보고 판단에 참고하거나, pick_object가
    받는 grasp_candidate 필드를 직접 구성할 때 쓰기 위한 읽기 전용 도구다.

    파이프라인: bbox → (SEG_BACKEND: NoOp 사각마스크 / depth_plane / SAM)
    → depth 역투영 → point cloud → RANSAC(우세 평면) + PCA(주축)
    → base_link 좌표계로 변환.

    [2026-09-02] parent_label 을 주면 YOLO 검출 없이도 동작한다:
    ground_object(target_label, parent_label) 로 파트 bbox 를 먼저 확보한 뒤
    (예: parent_label="drawer" 로 손잡이 축 방향 계산) 그 영역에 geometry 를
    돌린다. SEG_BACKEND=sam 이면 파트 bbox 중심을 SAM point prompt 로 같이
    넘겨 마스크 정밀도를 높인다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ○ 사용: side/pinch 그립 전 물체의 실제 긴 축 방향을 확인하고 싶을 때,
           또는 infer_grasp의 grasp_relation("perpendicular_to_long_axis" 등)이
           실제로 어느 각도를 가리키는지 검증하고 싶을 때.
    ✗ 대체 아님: infer_grasp(어떻게 잡을지)나 ground_object(어디 있는지)를
           대체하지 않는다 -- 이 도구는 "물체가 실제로 어떤 모양으로 놓여
           있는지"만 답한다.

    Args:
        target_label: /detected_objects에 있는 라벨 (YOLO 검출 필요 --
            아직 VLM-only 물체는 지원 안 함, ground_object의 bbox_approx를
            나중에 여기 연결할 수 있음).

    Returns:
        성공: {"status":"success", "label":"...",
               "geometry_confidence":0.87,
               "major_axis_yaw_deg":37.2, "normal_yaw_deg":12.0|null,
               "plane_inlier_ratio":0.71, "point_count":812,
               "major_axis":[x,y,z], "plane_normal":[x,y,z]|null,
               "extents":[L,W,H]}
               (major_axis_yaw_deg/normal_yaw_deg는 mod-180 -- 부호 모호성은
               해소 안 됨, side grasp 후보 생성 시 양쪽 다 고려해야 함.
               None이면 그 축/면이 충분히 수직이 아니라 방위각이 무의미하다는
               뜻 -- geometry_3d.py의 verticality 게이트 참고.)
        실패: {"status":"error"|"low_confidence", "reason":"..."}
    """
    _ensure_ros()
    target_label = target_label.strip().lower()

    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'status': 'error', 'reason': 'image_unavailable'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'status': 'error', 'reason': f'image_stale — {img_age:.1f}s'})

    depth_arr, _ = _ros_node.get_depth()
    fx, fy, cx, cy, _cw, _ch = _ros_node.get_cam_intrinsics()
    if depth_arr is None or fx is None:
        return json.dumps({'status': 'error', 'reason': 'depth_or_intrinsics_unavailable'})

    h, w = img_bgr.shape[:2]
    point_hint_px = None
    seg_part_hint = None

    if parent_label is not None:
        # [2026-09-02] 파트 경로: ground_object(part, parent) 로 bbox 확보.
        try:
            gj = json.loads(ground_object(target_label, parent_label))
        except Exception as e:
            return json.dumps({'status': 'error', 'reason': f'part_grounding_failed — {e}'})
        bbox_px = gj.get('bbox_approx')
        if not bbox_px or len(bbox_px) != 4:
            return json.dumps({'status': 'error',
                               'reason': f"part '{target_label}' bbox 확보 실패 "
                                         f"(ground_object: {gj.get('reason', gj.get('grounding'))})"})
        cp = gj.get('center_px')
        point_hint_px = cp if (isinstance(cp, list) and len(cp) == 2) else None
        seg_part_hint = target_label
    else:
        objects, _ = _ros_node.get_objects()
        matched = [o for o in objects if str(o.get('label', '')).lower() == target_label]
        if not matched:
            return json.dumps({'status': 'error',
                               'reason': f"'{target_label}' 이(가) /detected_objects에 없습니다 "
                                         f"(YOLO 검출 필요, 또는 parent_label 지정)."})
        bbox_norm = matched[0].get('bbox')
        if not bbox_norm or len(bbox_norm) != 4:
            return json.dumps({'status': 'error', 'reason': 'invalid_bbox'})
        bbox_px = [int(bbox_norm[0] * w), int(bbox_norm[1] * h),
                   int(bbox_norm[2] * w), int(bbox_norm[3] * h)]

    # [2026-09-02] NoOp(bbox 사각 마스크) 대신 SEG_BACKEND 환경변수로 고르는
    # 기본 backend 사용(depth_plane / sam). depth 를 넘겨서 배경 테이블면을
    # 마스크에서 빼면 CLAUDE.md "major_axis_yaw_deg 신뢰 불가"의 근본 원인
    # (bbox-as-mask 배경 오염)이 완화된다. depth_plane 은 모델 0개.
    seg = get_default_backend().segment(img_bgr, bbox_px, target_part=seg_part_hint,
                                        depth_image=depth_arr, point_hint_px=point_hint_px)
    if seg is None:
        return json.dumps({'status': 'error', 'reason': 'segmentation_failed'})

    points_cam = mask_depth_to_pointcloud(
        depth_arr, seg.mask, _PCIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
        max_points=2000)
    if len(points_cam) < 20:
        return json.dumps({'status': 'error',
                           'reason': f'insufficient_points — {len(points_cam)}개 (최소 20)'})

    geo_cam = _geometry_3d.compute_geometry(points_cam, min_inliers=20)
    if not geo_cam.valid:
        return json.dumps({'status': 'error', 'reason': 'geometry_fit_failed'})

    geo = _geometry_to_base_link(geo_cam)
    if not geo.valid:
        return json.dumps({'status': 'error', 'reason': 'tf_unavailable'})

    return json.dumps({
        'status': 'success',
        'label': target_label,
        'parent_label': parent_label,
        'seg_source': seg.source,       # bbox | depth_plane | sam | ...
        'geometry_confidence': geo.geometry_confidence,
        'major_axis_yaw_deg': geo.major_axis_yaw_deg,
        'normal_yaw_deg': geo.normal_yaw_deg,
        'plane_inlier_ratio': round(geo.plane_inlier_ratio, 3),
        'point_count': geo.point_count,
        'major_axis': list(geo.major_axis) if geo.major_axis else None,
        'plane_normal': list(geo.plane_normal) if geo.plane_normal else None,
        'extents': list(geo.extents) if geo.extents else None,
    }, ensure_ascii=False)


@mcp.tool()
def find_placement() -> str:
    """카메라 화면에서 물체를 놓을 수 있는 빈 공간을 추론한다.

    ─── 라우팅 규칙 ──────────────────────────────────────────────────────────
    ○ 사용: "빈 공간에 놓아" / "정리해" / "어디 두면 좋을지 봐줘" / "선반에 놓아"
    ✗ 금지: "컵 옆에 놓아" → YOLO 3D 좌표 상대 계산으로 해결 (VLM 불필요)
    ✗ 금지: target placement가 명확한 좌표로 이미 알려진 경우

    VLM은 normalized bbox로 배치 가능 영역만 반환한다.
    실제 robot 좌표는 반환된 bbox → depth → camera coords → TF → base_link 순으로 변환한다.
    VLM placement는 approximate임을 반드시 명시한다.

    ─── VLM 실패 시 절대 금지 ──────────────────────────────────────────────────
    status가 "error"이면:
    - YOLO 좌표로 placement 위치 추측 금지
    - robot movement 명령 금지
    반드시: "시각적 scene reasoning을 수행할 수 없습니다. (이유: {reason})"

    Returns:
        성공: {"status":"success",
               "placement_regions":[{"bbox":[x1,y1,x2,y2],"confidence":0.8},...],
               "inference_ms":8000.0,"image_age_sec":0.1}
        실패: {"status":"error","reason":"..."}
    """
    _ensure_ros()

    PLACEMENT_URL = f"{VLM_SERVER_URL}/find_placement"
    VLM_TIMEOUT = 45.0

    img_bgr, img_stamp = _ros_node.get_image()
    if img_bgr is None:
        return json.dumps({'status': 'error',
                           'reason': 'image_unavailable — /camera/color/image_raw 수신 없음'})
    img_age = round(time.time() - img_stamp, 3)
    if img_age > 3.0:
        return json.dumps({'status': 'error',
                           'reason': f'image_stale — {img_age:.1f}s 경과 (>3s)'})

    yolo_objects, _ = _ros_node.get_objects()
    detections = [{'label': o.get('label', '?'), 'bbox': o.get('bbox', [])} for o in yolo_objects]

    def _encode_b64(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('JPEG 인코딩 실패')
        return base64.b64encode(buf.tobytes()).decode()

    try:
        full_b64 = _encode_b64(img_bgr)
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'encode_failed — {e}'})

    try:
        resp = _requests.post(PLACEMENT_URL,
                              json={'full_image_b64': full_b64, 'detections': detections,
                                    'timestamp': time.time()},
                              timeout=VLM_TIMEOUT)
    except _requests.exceptions.ConnectionError:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_server_unavailable — {VLM_SERVER_URL} 연결 거부'})
    except _requests.exceptions.Timeout:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_timeout — {VLM_TIMEOUT}s 초과'})
    except Exception as e:
        return json.dumps({'status': 'error', 'reason': f'vlm_request_failed — {e}'})

    if resp.status_code != 200:
        return json.dumps({'status': 'error',
                           'reason': f'vlm_http_error — status {resp.status_code}: {resp.text[:200]}'})

    try:
        data = resp.json()
    except Exception:
        return json.dumps({'status': 'error', 'reason': f'vlm_invalid_json — {resp.text[:200]}'})

    return json.dumps({
        'status':            'success',
        'placement_regions': data.get('placement_regions', []),
        'inference_ms':      data.get('inference_ms', 0.0),
        'image_age_sec':     img_age,
    }, ensure_ascii=False)


@mcp.tool()
def get_system_status() -> str:
    """로봇/비전 브리지의 현재 연결 상태를 점검한다 (헬스체크용).

    Returns:
        {"ros_bridge": "up", "vision_objects": 3, "vision_age_sec": 0.2,
         "poses_file": "~/.local/share/nero_robot/saved_poses.json"}
    """
    _ensure_ros()
    objects, stamp = _ros_node.get_objects()
    age = round(time.time() - stamp, 2) if stamp > 0 else -1.0
    return json.dumps({
        'ros_bridge': 'up',
        'vision_objects': len(objects),
        'vision_age_sec': age,
        'poses_file': POSES_FILE,
    }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 3.5 CGN+VLM+STOMP pick 파이프라인 트라이얼 (nero_pick.sh 래퍼, 2026-09-11)
# ──────────────────────────────────────────────────────────────────────────────
PHASE_C_DIR = os.path.expanduser('~/ros2_ws/src/nero_sj_pickplace/tools/phase_c')
_PIPE_GREP_RE = (r'seed\]|sem\]|regions\]|오탐|unified rank|reachable CGN|plan [0-9]|'
                  r'no reachable|중단|approach \[|Cartesian fraction|캐시 HIT|'
                  r'approach_pref|shape|ablation|kind=|합성|스킵|경쟁|seg_px|obj_world')
_EXEC_GREP_RE = r'\[p[0-9]|grasp_cfg|grip-close|verdict|NO_REACH'


def _grep_lines(text: str, pattern: str) -> list:
    return [ln for ln in text.splitlines() if re.search(pattern, ln)]


@mcp.tool()
def run_pick_trial(label: str = 'cup', obj: str = '', task: str = '',
                    sem_off: bool = False, no_cache: bool = False,
                    dry: bool = False, go_home_first: bool = True,
                    extra_env: dict = None) -> str:
    """CGN+VLM+STOMP pick 파이프라인 한 트라이얼을 통째로 실행한다
    (go_home -> capture -> Thor 파이프라인 -> exec). 실험 세션에서 손으로
    반복하던 capture/rsync/ssh run_pipeline.sh/exec_pick.py 체인을 그대로
    감싼 것 -- nero_pick.sh 를 서브프로세스로 실행하고, 완료 후 exec_pick.py
    결과(JSON)와 Thor 쪽 랭킹 로그(runs/<ts>.json)를 pull 해서 구조화된
    JSON 으로 합쳐 돌려준다.

    label: 물체 라벨 (cup/bottle/box/pan 등, YOLO 라벨과 일치해야 함)
    obj: npz target id 강제 지정 (생략하면 nero_pick.sh 가 label 로 자동 라우팅)
    task: 자연어 지시 ("손잡이 피해서 잡아줘" 등). 있으면 시맨틱 ON, 없으면
        안정 pick 기본(시맨틱 캐시 있으면 그대로 재사용)
    sem_off: True 면 시맨틱 완전히 끄고 기하 전용 파이프라인 (A/B 대조군)
    no_cache: True 면 시맨틱 의사결정 캐시 무시하고 VLM 재호출
    dry: True 면 계획(step6_pick.json)까지만 하고 exec 안 함
    go_home_first: True(기본) 면 캡처 전에 go_home 먼저 호출 -- 매 트라이얼
        전에 관측 자세로 복귀해야 카메라가 물체를 제대로 봄
    extra_env: 추가 환경변수 dict, 예 {"NERO_TIP_W": "0", "NERO_THETA4": "0.4",
        "NERO_SIDE_GATE_K": "5"} -- 세션 중 계속 새로 생기는 ablation 토글을
        이 서버 코드를 안 고치고 바로 전달하기 위한 passthrough.

    Returns (JSON):
        {"pipeline_log": [...], "exec_log": [...],
         "kind": "side"|"cgn"|"top"|None, "approach_theta": float|None,
         "run_ts": str|None, "plan_used": int|None,
         "verdict": "HELD"|"SLIPPED"|"MISSED"|"NO_REACH"|None,
         "grip_after_close": [a,b]|None, "grip_final": [a,b]|None,
         "n_cgn_raw": int|None, "n_reach": int|None,
         "chosen": {...}|None, "timing": {"sem":.., "A":.., "B":.., "C":..}|None,
         "ok": bool, "error": str|None}

    ⚠ 코드의 verdict/kind 는 참고용이다 -- Isaac 마찰기반 파지가 그리퍼
    관절값이 애매해도 실제로는 물체를 들고 있는 경우, 반대로 관절값은
    맞는데 실제로는 손잡이에 걸리거나 놓친 경우가 이번 세션에서 여러 번
    확인됐다. 최종 성공/실패는 반드시 육안(카메라/실물)으로 확인할 것.
    """
    result = {
        "pipeline_log": [], "exec_log": [], "lap_checkpoints": [], "t_total_sec": None,
        "kind": None, "approach_theta": None,
        "run_ts": None, "plan_used": None, "verdict": None,
        "grip_after_close": None, "grip_final": None, "n_cgn_raw": None,
        "n_reach": None, "chosen": None, "timing": None, "ok": False, "error": None,
    }
    try:
        if go_home_first:
            gh = json.loads(go_home())
            if gh.get('status') != 'success':
                result['error'] = f"go_home 실패: {gh}"
                return json.dumps(result, ensure_ascii=False)

        cmd = [os.path.join(PHASE_C_DIR, 'nero_pick.sh'), '--label', label]
        if obj:
            cmd += ['--obj', obj]
        if task:
            cmd += ['--task', task]
        if sem_off:
            cmd += ['--sem-off']
        if no_cache:
            cmd += ['--no-cache']
        if dry:
            cmd += ['--dry']

        env = os.environ.copy()
        for k, v in (extra_env or {}).items():
            env[str(k)] = str(v)

        _t0 = time.time()
        proc = subprocess.run(cmd, cwd=PHASE_C_DIR, env=env,
                               capture_output=True, text=True, timeout=120)
        result['t_total_sec'] = round(time.time() - _t0, 2)
        full_log = proc.stdout + proc.stderr
        result['pipeline_log'] = _grep_lines(full_log, _PIPE_GREP_RE)
        result['exec_log'] = _grep_lines(full_log, _EXEC_GREP_RE)
        # nero_pick.sh 의 _lap 체크포인트("  [ 12.3s] capture 완료" 등) 뽑기 --
        # capture(PC)/Thor 왕복/exec 각 단계가 전체 중 어디서 시간을 먹는지 보려면 필요.
        result['lap_checkpoints'] = [
            {"t": float(m.group(1)), "label": m.group(2).strip()}
            for m in re.finditer(r'\[\s*([\d.]+)s\]\s*(.+)', full_log)
        ]
        if proc.returncode != 0 and not dry:
            # exit 1(capture 실패)/2(step6_pick.json 안 생김) 도 여기로 옴 --
            # 그래도 아래서 exec_pick_res.json 파싱은 계속 시도(있으면 유효한 정보).
            result['error'] = f"nero_pick.sh 종료코드 {proc.returncode}"

        res_path = '/tmp/exec_pick_res.json'
        if not dry and os.path.exists(res_path):
            try:
                ex = json.load(open(res_path))
                if ex.get('run_ts'):        # 이번 실행이 남긴 결과인지 최소 확인
                    result['run_ts'] = ex.get('run_ts')
                    result['plan_used'] = ex.get('plan_used')
                    result['verdict'] = ex.get('verdict')
                    result['grip_after_close'] = ex.get('grip_after_close')
                    result['grip_final'] = ex.get('grip_final')
            except Exception as e:
                result['error'] = f"exec_pick_res.json 파싱 실패: {e}"

        if result['run_ts']:
            local_run_json = f"/tmp/{result['run_ts']}.rank.json"
            pull = subprocess.run(
                ['rsync', '-q', f"thor:grasp/runs/{result['run_ts']}.json", local_run_json],
                capture_output=True, text=True, timeout=15)
            if pull.returncode == 0 and os.path.exists(local_run_json):
                try:
                    rk = json.load(open(local_run_json))
                    result['n_cgn_raw'] = rk.get('n_cgn_raw')
                    result['n_reach'] = rk.get('n_reach')
                    result['chosen'] = rk.get('chosen')
                    result['timing'] = rk.get('timing')
                    if rk.get('chosen'):
                        result['kind'] = rk['chosen'].get('kind')
                        result['approach_theta'] = rk['chosen'].get('approach_theta')
                except Exception as e:
                    result['error'] = (result['error'] + ' / ' if result['error'] else '') + \
                        f"rank json 파싱 실패: {e}"

        result['ok'] = dry or (result['verdict'] is not None)
        return json.dumps(result, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        result['error'] = 'nero_pick.sh 타임아웃(120s)'
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        result['error'] = f'run_pick_trial 예외: {e}'
        return json.dumps(result, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────
def main():
    t = threading.Thread(target=_ros_spin_thread, daemon=True)
    t.start()

    transport = os.environ.get('MCP_TRANSPORT', 'stdio')
    if transport == 'stdio':
        mcp.run(transport='stdio')
    else:
        mcp.settings.host = os.environ.get('MCP_HOST', '0.0.0.0')
        mcp.settings.port = int(os.environ.get('MCP_PORT', '8000'))
        mcp.run(transport=transport)


if __name__ == '__main__':
    main()
