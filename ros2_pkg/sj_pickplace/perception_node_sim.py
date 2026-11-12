#!/usr/bin/env python3
"""
perception_node_sim.py  (Isaac Sim 카메라 연동 버전)
=================================================================================
perception_node.py의 시뮬레이션 버전. RealSense SDK(pyrealsense2) 대신
Isaac Sim이 ActionGraph로 발행하는 ROS2 토픽(/camera/color/image_raw,
/camera/depth/image_raw, /camera/camera_info)을 구독한다.

박스 인식 서버 호출, tf2 좌표변환, /detected_objects 발행 로직은
perception_node.py와 100% 동일하게 유지했다 (planning_node 변경 불필요).

[2026-07 수정 — 오탐/중복 필터링 추가]
- 그리퍼가 화면 하단에 항상 걸려서 "box"로 오탐되는 문제 발견
  → depth 기준(카메라에 너무 가까운 detection)으로 제외 (GRIPPER_MIN_DEPTH_M)
- 같은 박스가 2D bbox 두 개로 중복 검출되는 문제 발견
  → 기존 2D bbox 중심점 기준 dedup은 "쌓여있는 박스"(x,y는 겹치지만 z=depth가
    다른 별개 물체)까지 병합해버리는 문제가 있어 폐기.
  → depth 조회 완료 후, 3D 위치(x,y,z) 기준으로 dedup 하도록 변경.
    x,y가 가깝더라도 z(depth)가 다르면 쌓인 별개 박스로 보고 유지한다.

[환경변수]
  BOX_SERVER_URL  : 박스 서버 주소 (기본: http://127.0.0.1:8002/detect)
  BOX_HEALTH_URL  : 헬스체크 주소 (기본: http://127.0.0.1:8002/health)
  TARGET_LABEL    : 탐지 대상 클래스 (기본: box + vlm_boxyolo.py의 ALLOWED_COCO 24종
                    전부). 쉼표로 복수 지정 가능: "box,bottle" (지정 시 기본값 전체를
                    덮어씀 — COCO 라벨은 서버(vlm_boxyolo.py)의 ALLOWED_COCO 집합과
                    반드시 일치해야 함, 안 맞으면 서버가 조용히 필터링해서 버림)
  BASE_FRAME      : tf 변환 목표 프레임 (기본: base_link)
  CAMERA_OPTICAL_FRAME : 카메라 광학 프레임 (기본: camera_color_optical_frame)
  IMAGE_TOPIC     : RGB 이미지 토픽 (기본: /camera/color/image_raw)
  DEPTH_TOPIC     : Depth 이미지 토픽 (기본: /camera/depth/image_raw)
  CAMERA_INFO_TOPIC : CameraInfo 토픽 (기본: /camera/camera_info)
  GRIPPER_MIN_DEPTH_M : 이보다 가까운 depth는 그리퍼 오탐으로 간주해 제외 (기본: 0.15)
  DEDUP_XY_THRESH_M   : 3D dedup 시 x,y 임계값(미터) (기본: 0.03)
  DEDUP_Z_THRESH_M    : 3D dedup 시 z(depth) 임계값(미터) — 이보다 z가 가까우면 같은
                        박스로 보고 병합, 이보다 멀면 쌓인 별개 박스로 보고 유지 (기본: 0.03)
  FACE_NORMAL_GRID_STEP/MIN_POINTS/MAX_POINTS/PLANARITY_MIN/VERTICALITY_MIN :
                        [2026-08 추가, 프로토타입] face_normal_yaw_deg 계산 파라미터.
                        perception_node.py 모듈 상단 주석 참고.
"""

import os
import json
import base64
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import rclpy
import message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo

import tf2_ros
import tf2_geometry_msgs  # noqa: F401

from sj_pickplace.camera_calibration import (
    CameraIntrinsics, set_intrinsics, pixel_to_camera_xyz,
)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
# 기본 라벨 = "box" + vlm_boxyolo.py의 ALLOWED_COCO 24종 전부 (2026-08-25).
# 서버가 다른 파일이라 상수 공유 불가 — 서버 쪽 ALLOWED_COCO를 바꾸면 여기도
# 같이 맞춰야 함 (yolo/vlm_boxyolo.py 참고).
_DEFAULT_TARGET_LABELS = (
    "box,"
    "person,umbrella,tie,"
    "bottle,wine glass,cup,fork,knife,spoon,bowl,"
    "banana,apple,"
    "book,clock,vase,scissors,toothbrush,"
    "laptop,remote,keyboard,cell phone,"
    "sports ball,baseball bat,baseball glove"
)
_raw_labels      = os.environ.get("TARGET_LABEL", _DEFAULT_TARGET_LABELS)
TARGET_LABELS    = [l.strip() for l in _raw_labels.split(",") if l.strip()]
TARGET_LABEL     = TARGET_LABELS[0]  # 단일 라벨 기대 코드와의 하위 호환용
BOX_SERVER_URL   = os.environ.get("BOX_SERVER_URL", "http://127.0.0.1:8002/detect")
BOX_HEALTH_URL   = os.environ.get("BOX_HEALTH_URL", "http://127.0.0.1:8002/health")
REQUEST_TIMEOUT  = float(os.environ.get("REQUEST_TIMEOUT", "3.0"))
DISPATCH_RATE_HZ = float(os.environ.get("DISPATCH_RATE_HZ", "10.0"))

MIN_BBOX_SIZE = 0.02

# 그리퍼가 화면 하단에 고정으로 걸려서 "box"로 오탐되는 문제 대응.
# 그리퍼는 카메라에 항상 훨씬 가깝게(depth ~0.1m) 잡히므로, 이보다 가까운
# detection은 실제 작업 대상 박스가 아니라 그리퍼 자체로 간주하고 제외한다.
GRIPPER_MIN_DEPTH_M = float(os.environ.get("GRIPPER_MIN_DEPTH_M", "0.15"))

# 3D 위치 기준 dedup 임계값.
# x,y는 가깝지만 z(depth)가 이 임계값보다 더 차이나면 "쌓여있는 별개 박스"로
# 보고 병합하지 않는다 (pick & stack 작업에서 반드시 구분해야 하는 케이스).
DEDUP_XY_THRESH_M = float(os.environ.get("DEDUP_XY_THRESH_M", "0.03"))
DEDUP_Z_THRESH_M  = float(os.environ.get("DEDUP_Z_THRESH_M", "0.025"))

BASE_FRAME = os.environ.get("BASE_FRAME", "base_link")
CAMERA_OPTICAL_FRAME = os.environ.get("CAMERA_OPTICAL_FRAME", "camera_color_optical_frame")
TF_TIMEOUT_SEC = float(os.environ.get("TF_TIMEOUT_SEC", "0.2"))

# color/depth 프레임을 "같은 순간"으로 볼 허용 오차(초). 카메라가 렉이 있을
# 때(2~3Hz, 변동폭 큼) 짝이 맞는 프레임 자체가 드물게 오므로 너무 타이트하게
# 잡으면 dispatch가 거의 안 됨. 실측 기준 여유있게 잡음.
SYNC_SLOP_SEC = float(os.environ.get("SYNC_SLOP_SEC", "0.15"))

IMAGE_TOPIC       = os.environ.get("IMAGE_TOPIC", "/camera/color/image_raw")
DEPTH_TOPIC        = os.environ.get("DEPTH_TOPIC", "/camera/depth/image_raw")
CAMERA_INFO_TOPIC  = os.environ.get("CAMERA_INFO_TOPIC", "/camera/camera_info")

# perception_node.py에 정의된 함수 재사용 (동일 패키지 내 import).
# [2026-08] _dedup_3d는 perception_node.py로 포팅/통합됨 -- 이 파일에서
# 로직을 중복 유지하면 나중에 두 구현이 갈라질 위험이 있어 공용 구현을
# 그대로 가져다 쓴다. 이 노드 고유의 DEDUP_XY_THRESH_M/DEDUP_Z_THRESH_M
# 값은 그대로 이 파일에서 관리하고, 호출 시 인자로 넘긴다.
from sj_pickplace.perception_node import (   # noqa: E402
    _compute_box_angle_base, _transform_with_fallback, _dedup_3d,
    _compute_face_normal_yaw,
)


class PerceptionNodeSim(Node):
    def __init__(self):
        super().__init__('perception_node_sim')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(String, "/detected_objects", qos)

        # ── Isaac Sim 카메라 토픽 구독 ──
        # [수정] 기존엔 color/depth를 각자 독립 콜백으로 구독해서 latest_color/
        # latest_depth에 최신값만 저장 -> dispatch 시점에 둘의 timestamp 차이를
        # 사후 체크하는 방식이었는데, 카메라가 렉(2~3Hz, 변동폭 큼)이 걸리면
        # 이 방식으로는 사실상 항상 짝이 안 맞아 매 프레임이 스킵되는 문제가
        # 실측 확인됨 (색상/깊이 시간차 100~1000ms대). message_filters로 애초에
        # "짝이 맞는 프레임만" 콜백을 받도록 변경.
        self.sub_color = message_filters.Subscriber(self, Image, IMAGE_TOPIC, qos_profile=qos)
        self.sub_depth = message_filters.Subscriber(self, Image, DEPTH_TOPIC, qos_profile=qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth], queue_size=5, slop=SYNC_SLOP_SEC)
        self._sync.registerCallback(self._on_synced_frames)
        self.sub_info = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._on_camera_info, qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.frame_lock = threading.Lock()
        self.latest_color: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_frame_id_stamp = None
        self.latest_tf_stamp = None
        self._last_dispatched_stamp = None
        self._intrinsics_set = False
        # [2026-08 추가] 박스서버 요청이 밀릴 때(GPU 경합 등) 이전 요청이
        # 안 끝났는데 새 요청을 또 스레드로 띄우던 게 /detected_objects
        # 0.5~0.6Hz까지 떨어뜨린 원인으로 실측 확인됨. 이전 요청 처리 중이면
        # dispatch를 건너뛴다 (요청 큐잉 방지, 최신 프레임 우선).
        self._inflight = False
        self._inflight_lock = threading.Lock()

        if not self._wait_for_box_server():
            self.get_logger().error(
                f'박스 서버 응답 없음 ({BOX_SERVER_URL}). vlm_boxyolo.py 켜져있는지 확인.')
            raise RuntimeError("box detection server not available")

        self.timer = self.create_timer(1.0 / DISPATCH_RATE_HZ, self._dispatch_inference)

        self.get_logger().info(
            f'PerceptionNodeSim 시작 (Isaac Sim 카메라) | target={TARGET_LABELS} | '
            f'image={IMAGE_TOPIC} depth={DEPTH_TOPIC} info={CAMERA_INFO_TOPIC} | '
            f'tf: {CAMERA_OPTICAL_FRAME} -> {BASE_FRAME} | '
            f'gripper_min_depth={GRIPPER_MIN_DEPTH_M}m | '
            f'dedup_xy={DEDUP_XY_THRESH_M}m dedup_z={DEDUP_Z_THRESH_M}m')

    # ── 카메라 콜백들 ──────────────────────────────────────────────────────
    def _on_camera_info(self, msg: CameraInfo):
        if self._intrinsics_set:
            return
        k = msg.k
        set_intrinsics(CameraIntrinsics(
            fx=k[0], fy=k[4], cx=k[2], cy=k[5],
            width=msg.width, height=msg.height,
        ))
        self._intrinsics_set = True
        self.get_logger().info(f'CameraInfo 수신, intrinsics 등록 완료 (fx={k[0]:.1f}, fy={k[4]:.1f})')

    def _on_synced_frames(self, color_msg: Image, depth_msg: Image):
        """message_filters가 color/depth를 SYNC_SLOP_SEC 이내로 짝지어
        호출해주는 콜백. 이 시점부터는 두 프레임이 같은 순간을 가리킨다고
        신뢰할 수 있다 (기존 독립 콜백 + 사후 dt 체크 방식은 렉이 있을 때
        거의 항상 스킵되는 문제가 실측 확인되어 폐기)."""
        try:
            arr = np.frombuffer(color_msg.data, dtype=np.uint8)
            if color_msg.encoding in ('rgb8', 'bgr8'):
                img = arr.reshape(color_msg.height, color_msg.width, 3)
                if color_msg.encoding == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                self.get_logger().warn(f'미지원 encoding: {color_msg.encoding}', throttle_duration_sec=5.0)
                return
        except Exception as e:
            self.get_logger().warn(f'컬러 이미지 파싱 실패: {e}', throttle_duration_sec=5.0)
            return

        try:
            darr = np.frombuffer(depth_msg.data, dtype=np.uint8)
            if depth_msg.encoding == '32FC1':
                depth = darr.view(np.float32).reshape(depth_msg.height, depth_msg.width)
            elif depth_msg.encoding == '16UC1':
                depth = darr.view(np.uint16).reshape(depth_msg.height, depth_msg.width).astype(np.float32) / 1000.0
            else:
                self.get_logger().warn(f'미지원 depth encoding: {depth_msg.encoding}', throttle_duration_sec=5.0)
                return
        except Exception as e:
            self.get_logger().warn(f'뎁스 이미지 파싱 실패: {e}', throttle_duration_sec=5.0)
            return

        with self.frame_lock:
            self.latest_color = img
            self.latest_depth = depth
            # frame_id_stamp: Isaac Sim이 이미지에 박아넣은 sim-time stamp.
            # tf 트리(wall-clock)와 시간 도메인이 달라 tf 조회엔 못 쓴다
            # (/clock 미발행 확인됨) -- "같은 프레임인지" 식별용으로만 사용.
            self.latest_frame_id_stamp = color_msg.header.stamp
            # tf_stamp: 프레임이 도착한 이 순간의 노드 자체 벽시계. tf 트리와
            # 같은 도메인이라 tf 조회에는 반드시 이걸 써야 한다.
            self.latest_tf_stamp = self.get_clock().now().to_msg()

    # ── 이하 perception_node.py와 동일 ────────────────────────────────────
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
            frame_id_stamp = self.latest_frame_id_stamp
            tf_stamp = self.latest_tf_stamp

        if frame_id_stamp is None:
            return

        # Isaac Sim 렌더링이 dispatch(DISPATCH_RATE_HZ)보다 느리면 같은
        # (동기화된) 프레임 쌍이 그대로 남아있음 -> 중복 처리/전송 방지.
        # 식별은 sim-time 도메인인 frame_id_stamp로 한다 (tf_stamp는 매
        # dispatch 호출 시 어차피 새 값이라 dedup 판별에 못 씀).
        if self._last_dispatched_stamp is not None \
                and frame_id_stamp.sec == self._last_dispatched_stamp.sec \
                and frame_id_stamp.nanosec == self._last_dispatched_stamp.nanosec:
            return
        with self._inflight_lock:
            if self._inflight:
                # 이전 박스서버 요청이 아직 처리 중 -> 이번 프레임은 건너뛴다.
                # (요청을 무제한으로 쌓지 않고, 다음 dispatch에서 그때의
                # 최신 프레임을 다시 시도한다.)
                return
            self._inflight = True

        self._last_dispatched_stamp = frame_id_stamp

        threading.Thread(
            target=self._send_and_publish, args=(color, depth, tf_stamp), daemon=True
        ).start()

    def _send_and_publish(self, color: np.ndarray, depth: np.ndarray, stamp=None):
        _stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()
        try:
            ok, buf = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return
            img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            payload = {"image_b64": img_b64, "labels": TARGET_LABELS}
            r = requests.post(BOX_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return
            data = r.json()
            # ── 2026-07 수정 ────────────────────────────────────────────
            # 예전엔 여기서 2D bbox 중심점 기준 _dedup()을 먼저 적용했는데,
            # 이 방식은 depth를 모르는 상태라 "쌓여있는 박스"(x,y는 겹치지만
            # z가 다른 별개 물체)까지 병합해버리는 문제가 있었음.
            # → dedup을 depth 조회 이후로 미루고, 최종적으로 3D 위치 기준
            #   (_dedup_3d)으로 처리한다. 여기서는 bbox 크기 필터만 먼저 적용.
            dets = [
                d for d in data.get("detections", [])
                if (d["x_max"] - d["x_min"]) >= MIN_BBOX_SIZE
                and (d["y_max"] - d["y_min"]) >= MIN_BBOX_SIZE
            ]

            h, w = color.shape[:2]
            objs = []
            for d in dets:
                cx_norm = (d["x_min"] + d["x_max"]) / 2
                cy_norm = (d["y_min"] + d["y_max"]) / 2
                cx_px, cy_px = int(cx_norm * w), int(cy_norm * h)
                # [2026-07 수정] 회전된 박스 중심 보정 (perception_node.py와
                # 동일 패턴 - minAreaRect 중심을 depth 샘플링 전에 반영).
                angle_deg, refined_center_px = _compute_box_angle_base(
                    color, d, self.tf_buffer,
                    CAMERA_OPTICAL_FRAME, BASE_FRAME, TF_TIMEOUT_SEC, depth=depth,
                    stamp=_stamp, logger=self.get_logger())
                # [2026-08 추가, 프로토타입 -- 실기 미검증] perception_node.py와
                # 동일 패턴 재사용 (별개 필드로만 추가, 기존 소비 경로 무영향).
                face_yaw_deg, face_conf, _face_n = _compute_face_normal_yaw(
                    d, depth, self.tf_buffer,
                    CAMERA_OPTICAL_FRAME, BASE_FRAME, TF_TIMEOUT_SEC,
                    stamp=_stamp, logger=self.get_logger())
                if refined_center_px is not None:
                    _rx, _ry = refined_center_px
                    if 0 <= _rx < w and 0 <= _ry < h:
                        cx_px, cy_px = _rx, _ry
                        cx_norm, cy_norm = _rx / w, _ry / h
                # [2026-07 수정] 단일 중심픽셀+depth 1개 방식은 카메라가 비스듬한
                # 각도로 볼 때 원근왜곡으로 2D 중심이 실제 3D 중심과 어긋나는(Y축
                # -25~-35mm 계통오차 실측 확인) 문제가 있어, bbox 영역 안 여러
                # 점의 depth를 읽어 3D로 역투영한 뒤 평균(centroid)을 쓴다.
                _bx1, _by1 = int(d['x_min'] * w), int(d['y_min'] * h)
                _bx2, _by2 = int(d['x_max'] * w), int(d['y_max'] * h)
                depth_m, cam_xyz = self._multi_point_3d_centroid(depth, _bx1, _by1, _bx2, _by2)
                if depth_m is None or cam_xyz is None:
                    continue

                # ── 그리퍼 오탐 제외 ──────────────────────────────────
                # 그리퍼는 카메라에 고정으로 매우 가깝게(depth ~0.1m) 잡혀서
                # "box"로 오탐되는 것이 반복 테스트로 확인됨. 실제 작업 대상
                # 박스는 항상 이보다 먼 거리에 있으므로 depth로 걸러낸다.
                if depth_m < GRIPPER_MIN_DEPTH_M:
                    continue


                try:
                    xyz = self._transform_to_base(cam_xyz, _stamp)
                except Exception as e:
                    self.get_logger().warn(
                        f'tf 변환 실패 ({CAMERA_OPTICAL_FRAME} -> {BASE_FRAME}): {e}',
                        throttle_duration_sec=5.0)
                    continue


                # [2026-08-31 추가] bbox 실측 크기(미터, 카메라 광학축 기준
                # 투영 폭/높이) -- planning_node._sync_scene_collision_objects가
                # 충돌회피 keep-out zone을 고정 정육면체 대신 물체별 실제
                # 크기로 만들 수 있게 한다(실측 지적: 고정 10cm 박스가
                # 실제 물체+그리퍼 폭보다 작아서 스치는 사고가 확인됨).
                # bbox 대각선 두 점을 같은 depth_m에서 역투영해 거리로 계산 --
                # pixel_to_camera_xyz가 선형 변환이라 depth 고정이면 정확하다.
                # 카메라가 물체를 완전히 위/정면에서 보지 않으면(원근) 실제
                # 물체 치수보다 다소 왜곡될 수 있으나, keep-out 용도로는
                # 고정값보다 훨씬 낫다.
                _p1 = pixel_to_camera_xyz(_bx1, _by1, depth_m)
                _p2 = pixel_to_camera_xyz(_bx2, _by2, depth_m)
                if _p1 is not None and _p2 is not None:
                    bbox_size_m = [round(abs(_p2["x"] - _p1["x"]), 3),
                                   round(abs(_p2["y"] - _p1["y"]), 3)]
                else:
                    bbox_size_m = None

                objs.append({
                    "label": d["label"],
                    "bbox": [d["x_min"], d["y_min"], d["x_max"], d["y_max"]],
                    "bbox_size_m": bbox_size_m,
                    "center_2d": {"x": cx_norm, "y": cy_norm},
                    "center_3d": xyz,
                    "depth_m": round(float(depth_m), 3) if depth_m else None,
                    "confidence": round(float(d.get("confidence", 0.0)), 3),
                    "angle_base_deg": angle_deg,
                    "face_normal_yaw_deg": face_yaw_deg,
                    "face_normal_confidence": face_conf,
                })

            # ── 3D 위치 기준 최종 dedup ──────────────────────────────────
            # x,y,z 전부 가까운 것만 같은 박스의 중복 검출로 보고 병합한다.
            # 쌓여있는 박스처럼 z(depth)가 다르면 그대로 유지된다.
            objs = _dedup_3d(objs, xy_thresh=DEDUP_XY_THRESH_M, z_thresh=DEDUP_Z_THRESH_M)

            msg = String()
            msg.data = json.dumps({"objects": objs}, ensure_ascii=False)
            self.pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'추론/발행 실패: {e}', throttle_duration_sec=5.0)
        finally:
            with self._inflight_lock:
                self._inflight = False

    def _transform_to_base(self, cam_xyz: dict, stamp=None) -> dict:
        from geometry_msgs.msg import PointStamped
        pt = PointStamped()
        pt.header.frame_id = CAMERA_OPTICAL_FRAME
        pt.header.stamp = stamp if stamp is not None else rclpy.time.Time().to_msg()
        pt.point.x = cam_xyz['x']
        pt.point.y = cam_xyz['y']
        pt.point.z = cam_xyz['z']
        out = _transform_with_fallback(
            self.tf_buffer, pt, BASE_FRAME, TF_TIMEOUT_SEC, logger=self.get_logger())
        return {'x': out.point.x, 'y': out.point.y, 'z': out.point.z}

    @staticmethod
    @staticmethod
    def _multi_point_3d_centroid(depth, x1, y1, x2, y2):
        """[2026-07 추가] perception_node.py와 동일 -- 단일 중심픽셀
        방식의 원근왜곡 문제(Y축 -25~-35mm 계통오차 실측 확인)를 bbox
        영역 다중 포인트 3D centroid로 해결.
        """
        GRID = 5
        h, w = depth.shape
        x1c, x2c = max(0, x1), min(w, x2)
        y1c, y2c = max(0, y1), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return None, None

        pts_cam = []
        depths = []
        for gi in range(GRID):
            for gj in range(GRID):
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

        mean_x = sum(p['x'] for p in pts_cam) / len(pts_cam)
        mean_y = sum(p['y'] for p in pts_cam) / len(pts_cam)
        mean_z = sum(p['z'] for p in pts_cam) / len(pts_cam)
        depth_repr = float(np.median(depths))
        return depth_repr, {'x': mean_x, 'y': mean_y, 'z': mean_z}

    def _sample_depth(depth: np.ndarray, cx: int, cy: int) -> Optional[float]:
        h, w = depth.shape
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 0.05) & (patch < 2.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNodeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
