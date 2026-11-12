#!/usr/bin/env python3
"""rt_rviz_bridge.py -- 실시간 pick/place/삽입 루프의 상태를 RViz2 로 내보내는 시각화 노드(v4, 2026-10-07).
입력: ~/grasp/rt_state.json(루프가 쓰는 상태 파일), 카메라 RGB/깊이/camera_info, TF, joint_states(그리퍼 폭만). 이 노드는 발행만 하며 파이프라인(인식/grasp/계획/실행)에는 아무 영향이 없다.
발행:
  /nero_rt/markers   MarkerArray  네임스페이스별 층(RViz Displays > NERO RT > Namespaces 에서 개별 토글):
      floor/grid/workspace          바닥, 은은한 격자, 작업 반경 링(상시)
      obj_points/obj_hull/obj_label 물체별 분할 점(상태색), 반투명 볼록 껍질, 라벨 `ID·클래스`   (obj_box = 박스 윤곽, 기본 OFF)
      cam_frustum                   카메라 시야(기본 OFF)
      grasp_cand/grasp_chosen/grasp_ghost   CGN 후보(후보 계산 단계만) / 채택 후보(pick 실행 중만) / pre-grasp 고스트
      traj_pick/traj_place          계획 경로(얇은 점선, 해당 모션 실행 중에만)
      traj_exec                     실행 경로(굵은 실선, 끝나면 fade_s 동안 페이드아웃 후 삭제)
      place_slot/place_ghost        놓을 곳 하이라이트와 놓일 물체 예상 위치 고스트(place 계획~실행 중에만)
      result_flash                  place 판정 직후 강조(성공 초록/실패 빨강) + 문구
      insert_*                      삽입(홀더, 입구 링, 축, 막대 PCA 축, 팁 경로)
      물체를 쥐고 있는 동안 그 물체의 마커는 gripper_flange 프레임에 붙어 그리퍼와 함께 움직인다
  /nero_rt/camera_annotated  Image  카메라 영상 + 물체 마스크(알파 0.35, 얇은 외곽선) + 라벨 + grasp 2D 투영(camera_info 로 투영). 3D 마커는 일절 겹치지 않는다
  /nero_rt/camera_info       CameraInfo  위 영상과 같은 해상도(스케일 반영)
  /nero_rt/panel             Image  오른쪽 패널(카메라 영상 + HUD)을 한 장으로: panel_w x (cam_h + hud_h) 픽셀. RViz 도크 폭을 panel_w 로 두면 축소 없이 1:1
  /nero_rt/hud               Image  HUD 만(재생·다른 용도)
  /nero_rt/cloud             PointCloud2  RGB-D 컬러 점군(1 Hz, 1/8 간격)
  /nero_rt/robot_description String  URDF 사본(메쉬를 sRGB 보정 DAE 로 교체)
  TF nero_rt_focus           카메라가 따라갈 초점
상태색(2D 범례와 동일): 집을 대상 초록 / 장애물 주황 / 도착지 하늘색 / 놓은 물체 회색 / 실패·보류 빨강. 계획 경로 = 점선(얇게), 실행 경로 = 실선(굵게).
사용: python3 rt_rviz_bridge.py [--state ~/grasp/rt_state.json] [--no-cloud]   (RViz 설정: ~/nero/rviz/nero_rt.rviz, 실행: ~/nero/start_rviz_rt.sh)
환경: ROS_LOCALHOST_ONLY=1"""
import os, sys, json, math, time, argparse, re, threading
from collections import Counter as collections_Counter
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.serialization import deserialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, JointState
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

# 표시 설정 한 곳(색/수명/갱신 주기/글자 크기). 필요하면 여기만 바꾼다.
CFG = dict(
    marker_hz=5.0, marker_lifetime=1.0,           # 마커 발행 주기 / 수명(갱신이 끊기면 자동 소멸)
    cam_hz=10.0, cam_scale=0.5, hud_hz=5.0, cloud_hz=1.0,      # 카메라 주석 영상 / HUD / 점군 갱신 주기(Hz), 카메라 영상 축소 비율
    seg_hz_changed=0.8, seg_hz_idle=0.2,          # 물체 분할 점 갱신(상태가 바뀌면 빠르게)
    panel_w=640, cam_h=160, hud_h=820,            # 오른쪽 패널(카메라 + HUD) 실제 픽셀 크기 -- RViz 도크 폭을 같게 두면 1:1 로 보인다
    font_body=17, font_title=20, font_key=28,     # 글자 크기(px)
    fade_s=1.5, flash_s=3.0,                      # 실행 경로 페이드아웃 / 결과 판정 강조 시간(초)
    held_closed_max=0.040, held_open_min=0.046,   # 그리퍼 폭(gripper_joint1)이 이 값보다 작으면 물체를 쥔 것으로 본다
    label_size=0.022, point_size=0.004,
    cand_top=3,                                   # 후보 계산 단계에서 보여 줄 후보 수(점수 상위)
)
GT_CODE = r'''
import json, omni.usd
from pxr import Usd, UsdGeom
st = omni.usd.get_context().get_stage(); bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
SKIP = ("agx_arm", "Materials", "Looks", "GroundPlane", "defaultLight", "Camera", "camera", "Light", "light", "Table", "table", "Plane", "ground", "Ground", "physics")
out = []
for p in st.GetPrimAtPath("/World").GetChildren():
    nm = p.GetName()
    if any(k in nm for k in SKIP) or not p.IsA(UsdGeom.Imageable): continue
    r = bc.ComputeWorldBound(p).ComputeAlignedRange()
    if r.IsEmpty(): continue
    lo, hi = r.GetMin(), r.GetMax(); sz = [hi[i] - lo[i] for i in range(3)]
    if max(sz) > 0.7 or max(sz) < 0.01 or hi[2] < 0.0: continue
    out.append(dict(name=nm, c=[(lo[i] + hi[i]) / 2 for i in range(3)], s=sz))
json.dump(out, open("/tmp/nero_rt_gt.json", "w"))
'''
FRAME = "base_link"; CAM_FRAME = "camera_color_optical_frame"; TOOL_FRAME = "gripper_flange"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
SORT_PATH = os.environ.get("NERO_SORT_STATUS", "/tmp/multi_exec_status.json")      # 2026-10-30: 오프라인 렌더용 환경변수(기본 그대로)
MULTI_LOG = os.environ.get("NERO_MULTI_LOG", "/tmp/multi_exec_logs")               # chain.log·chain_retry.log·plan/·rep.log 가 있는 폴더(반복 실행 때 러너가 지정)
LLM_LOG = os.environ.get("NERO_LLM_LOG", "/tmp/rt_llm_log.jsonl")
DEST_COL = [(110, 190, 255), (255, 160, 70), (180, 130, 255), (110, 230, 140), (255, 120, 170), (230, 220, 100)]      # 용기별 색(HUD 칩·3D 연결선 공통, RGB 0-255)
LABELS_EN = dict(      # 영어 라벨(2026-10-30). 물체 이름(label_top1)은 스캔이 낸 영어 그대로
    phases=("SCAN", "PLAN", "EXEC", "RECOVER"),
    tag={"OK": "OK", "두기": "LEAVE", "진행": "RUN", "대기": "WAIT", "planfail": "PLANFAIL", "NOT_INSERTED": "NOT INSERTED", "MISSED": "MISSED", "SLIPPED": "SLIPPED", "TOPPLED": "TOPPLED", "NO_PLAN": "NO PLAN", "failed": "FAILED", "not_found": "NOT FOUND", "stopped": "STOPPED"},
    split={"스캔·인식": "SCAN/RECOGNIZE", "CGN+B": "CGN+B", "pick 실행": "PICK EXEC", "C 계획": "cuROBO PLAN", "place 계획": "PLACE PLAN", "place 실행": "PLACE EXEC", "전환·판정": "SWITCH/VERDICT", "기타": "OTHER"})
SCOL = dict(target=(0.22, 0.92, 0.38), obstacle=(1.0, 0.60, 0.12), dest=(0.35, 0.78, 1.0), placed=(0.66, 0.68, 0.73), failed=(1.0, 0.28, 0.28), deferred=(1.0, 0.28, 0.28))
ENT_ORANGE = (1.0, 0.60, 0.12); ENT_PINK = (1.0, 0.45, 0.72); ENT_BLUE = (0.25, 0.55, 1.0)      # 도착지 보기(NERO_RVIZ_ENT): 나머지 물체 주황 / 도착지 후보 분홍 / 최종 선택 파랑
ENT_PAL = [(0.95, 0.35, 0.35), (0.35, 0.80, 0.45), (0.40, 0.60, 1.0), (1.0, 0.80, 0.25), (0.80, 0.45, 0.95), (0.25, 0.85, 0.85), (1.0, 0.55, 0.25), (0.60, 0.75, 0.30), (0.95, 0.50, 0.75), (0.55, 0.55, 0.95), (0.80, 0.65, 0.40), (0.45, 0.80, 0.70)]      # 엔티티 보기(NERO_RVIZ_ENT): id 마다 다른 색
C_PICK = (1.0, 0.62, 0.15); C_PLACE = (0.30, 0.85, 1.0); C_EXEC = (1.0, 1.0, 1.0); C_HOLE = (1.0, 0.82, 0.2); C_PEG = (0.45, 0.6, 1.0); C_TIP = (1.0, 0.4, 0.8); C_GRASP = (0.3, 1.0, 0.5)
STAGE_IDX = {"사전 훑기": 0, "탐색": 0, "발견": 0, "재조준": 0, "정렬": 0, "CGN+B": 1, "C": 1, "pick": 2, "place 계획": 3, "place": 3, "손 안 위치 측정(오라클)": 3, "종료": 4}
CHIPS = ["탐색", "후보 계산", "집기", "놓기"]


import array


def _u8(a):
    """numpy 배열을 array('B') 로: rclpy 가 원소별 검증(메시지당 수백 ms)을 건너뛰게 한다."""
    return array.array('B', np.ascontiguousarray(a).tobytes())


def C_hex(c): return tuple(int(255 * x) for x in c[:3])


def _num(oid):
    m = re.fullmatch(r"obj_(\d+)", str(oid)); return int(m.group(1)) if m else (abs(hash(str(oid))) % 90 + 100)       # obj_N 이 아닌 ID(예: placed_25099)는 100~189 의 짧은 번호


def cr(c): return ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(c[3] if len(c) > 3 else 1.0))
def pt(p): return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
def quat_from_yaw(yaw): return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _frame(a, b):
    a = np.asarray(a, float); a = a / (np.linalg.norm(a) + 1e-12); b = np.asarray(b, float); b = b - (b @ a) * a; b = b / (np.linalg.norm(b) + 1e-12)
    return np.stack([np.cross(b, a), b, a], 1)


def quat_from_R(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0: s = math.sqrt(t + 1.0) * 2; return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3; s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2; q = [0.0] * 4
    q[i] = 0.25 * s; q[j] = (R[j, i] + R[i, j]) / s; q[k] = (R[k, i] + R[i, k]) / s; q[3] = (R[k, j] - R[j, k]) / s; return tuple(q)


def tf_to_Rt(tf):
    q = tf.rotation; x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R, np.array([tf.translation.x, tf.translation.y, tf.translation.z])


def gripper_geom(contact, approach, jaw, opening):
    """평행 조 그리퍼 형상(viz_export 규칙). 반환 (U 자 선 점 목록, 손가락 직육면체 [(중심, R, 크기)])"""
    R = _frame(approach, jaw); c = np.asarray(contact, float); a = R[:, 2]; b = R[:, 1]; hw = float(np.clip(opening, 0.02, 0.10)) / 2.0; FL = 0.06; t = 0.012; d = 0.022
    tL, tR = c + hw * b, c - hw * b; kL, kR = tL - a * FL, tR - a * FL; mid = (kL + kR) / 2
    lines = [tL, kL, kL, kR, kR, tR, mid, mid - a * 0.06]
    pads = [(c + sg * (hw + t / 2) * b - a * FL / 2, R, (d, t, FL)) for sg in (1.0, -1.0)]
    return lines, pads


def dashed(pts, seg=0.012, gap=0.010):
    """경로 점열을 점선(LINE_LIST 쌍)으로: 일정 간격으로 다시 샘플링해 선분/빈칸을 번갈아 만든다."""
    P = [np.asarray(p, float) for p in pts]; out = []; on = True; left = seg
    for a, b in zip(P[:-1], P[1:]):
        d = b - a; L = float(np.linalg.norm(d))
        if L < 1e-9: continue
        u = d / L; pos = 0.0
        while pos < L - 1e-9:
            step = min(left, L - pos)
            if on: out += [a + u * pos, a + u * (pos + step)]
            pos += step; left -= step
            if left <= 1e-9: on = not on; left = seg if on else gap
    return out


class Bridge(Node):
    def __init__(self, a):
        super().__init__("nero_rt_rviz_bridge")
        self.path = os.path.expanduser(a.state); self.ttl = a.ttl; self.cloud_on = not a.no_cloud
        qm = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(MarkerArray, "/nero_rt/markers", qm)
        self.hud_pub = self.create_publisher(Image, "/nero_rt/hud", qm); self.cam_pub = self.create_publisher(Image, "/nero_rt/camera_annotated", qm); self.ci_pub = self.create_publisher(CameraInfo, "/nero_rt/camera_info", qm)
        self.panel_pub = self.create_publisher(Image, "/nero_rt/panel", qm)
        self.pc_pub = self.create_publisher(PointCloud2, "/nero_rt/cloud", qos_profile_sensor_data)
        ql = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.rd_pub = self.create_publisher(String, "/nero_rt/robot_description", ql); self.create_subscription(String, "/robot_description", self._on_desc, ql)
        self.buf = Buffer(cache_time=Duration(seconds=30)); TransformListener(self.buf, self); self.tfb = TransformBroadcaster(self)
        self.cam = {}; cg = ReentrantCallbackGroup()
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, qos_profile_sensor_data, callback_group=cg)
        self.create_subscription(Image, "/camera/color/image_raw", self._on_rgb, qos_profile_sensor_data, callback_group=cg, raw=True)
        self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth, qos_profile_sensor_data, callback_group=cg, raw=True)
        self.create_subscription(JointState, "/joint_states", self._on_js, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=cg)
        self.grip = 0.05; self.gt = []; self.gt_t = 0.0; self.declare_parameter('gt_compare', False)
        threading.Thread(target=self._gt_loop, daemon=True).start()
        self.state = None; self.mt = 0.0; self.prev_keys = set(); self.objpts = {}; self.hull_cache = {}; self.focus = np.array([-0.1, 0.0, 0.1])
        self.trail = []; self.fading = []; self.held = None; self.seen_picked = None; self.flash = None; self.last_stage_i = -1
        self.t_hud = self.t_cam = self.t_seg = self.t_cloud = 0.0; self.seg_mt = -1.0; self.panel_img = None; self.perf = dict(t0=time.time(), cam=0, panel=0, mk=0, ft=0.0, nft=0, tk=0.0, ntk=0)
        self.clear_all(); self.create_timer(1.0 / CFG["marker_hz"], self.tick, callback_group=MutuallyExclusiveCallbackGroup()); self.create_timer(0.05, self.fast_tick, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"상태 파일 {self.path} -> /nero_rt/{{markers,camera_annotated,camera_info,panel,hud,cloud}}")

    def _gt_loop(self):
        """Isaac 정답(GT) 조회: gt_compare 파라미터가 켜져 있는 동안만 1Hz 로 Isaac 소켓에 물어본다(꺼져 있으면 시뮬에 아무 부하도 주지 않는다)."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase_c_L1")); err_t = 0.0
        while True:
            time.sleep(1.0)
            try:
                if not self.get_parameter('gt_compare').value: self.gt = []; continue
                from isaac_exec import run_code
                r = run_code(GT_CODE, timeout=5.0)
                if r.get("status") == "error": raise RuntimeError(r.get("message"))
                self.gt = json.load(open("/tmp/nero_rt_gt.json")); self.gt_t = time.time()
            except Exception as e:
                if time.time() - err_t > 30: self.get_logger().warn(f"GT 조회 실패: {e!r}"); err_t = time.time()

    def layer_gt(self, st):
        """인식 결과 vs Isaac 정답: 정답 박스(흰 반투명) + 인식 중심과 정답 중심을 잇는 선 + 오차(cm). gt_compare 파라미터를 켠 동안만 나온다. 정답은 정렬 박스(AABB)라 회전한 물체는 박스가 실제보다 크다."""
        if not self.gt or not self.get_parameter('gt_compare').value: return []
        out = []; used = set(); objs = [(k, v) for k, v in (st.get("objs") or {}).items() if "xyz" in v]
        for i, g in enumerate(self.gt):
            out.append(self.mk("gt_compare", i * 3, Marker.CUBE, (1, 1, 1, 0.13), scale=g["s"], pos=g["c"]))
            out.append(self.mk("gt_compare", i * 3 + 1, Marker.LINE_LIST, (1, 1, 1, 0.6), scale=(0.002, 0, 0), pts=self.box_edges(g["c"], g["s"], 0.0)))
        for k, o in objs:
            best, bd = None, 0.10
            for i, g in enumerate(self.gt):
                d = math.hypot(g["c"][0] - o["xyz"][0], g["c"][1] - o["xyz"][1])
                if d < bd and i not in used: best, bd = i, d
            if best is None: continue
            used.add(best); g = self.gt[best]; n = _num(k)
            out.append(self.mk("gt_compare", 1000 + n * 2, Marker.LINE_LIST, (1.0, 0.35, 0.35, 1.0), scale=(0.004, 0, 0), pts=[[o["xyz"][0], o["xyz"][1], g["c"][2]], g["c"]]))
            out.append(self.mk("gt_compare", 1001 + n * 2, Marker.TEXT_VIEW_FACING, (1.0, 0.6, 0.6, 1), scale=(0, 0, 0.022), pos=[g["c"][0], g["c"][1], g["c"][2] + g["s"][2] / 2 + 0.03], text=f"d_{100 * bd:.1f}cm"))
        return out

    def clear_all(self):
        """시작 시 이전 실행의 잔여 마커를 모두 지운다."""
        d = Marker(); d.header.frame_id = FRAME; d.action = Marker.DELETEALL
        for _ in range(2): self.pub.publish(MarkerArray(markers=[d]))

    # ── 입력 ──
    def _on_info(self, m): self.cam["info"] = m; self.cam["K"] = np.array(m.k).reshape(3, 3)
    def _on_rgb(self, raw): self.cam["rgb_raw"] = raw
    def _on_depth(self, raw): self.cam["depth_raw"] = raw

    def _on_js(self, m):
        try: self.grip = float(m.position[m.name.index("gripper_joint1")])
        except Exception: pass

    def get_rgb(self):
        raw = self.cam.get("rgb_raw")
        if raw is None: return None
        if self.cam.get("rgb_dec_src") is not raw: self.cam["rgb_dec"] = deserialize_message(raw, Image); self.cam["rgb_dec_src"] = raw
        m = self.cam["rgb_dec"]; arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]; return arr[..., ::-1] if m.encoding.startswith("bgr") else arr

    def get_depth(self):
        raw = self.cam.get("depth_raw")
        if raw is None: return None
        if self.cam.get("depth_dec_src") is not raw: self.cam["depth_dec"] = deserialize_message(raw, Image); self.cam["depth_dec_src"] = raw
        m = self.cam["depth_dec"]; d = np.frombuffer(m.data, dtype=np.float32 if m.encoding == "32FC1" else np.uint16).reshape(m.height, m.width)
        return d.astype(np.float32) / (1000.0 if m.encoding == "16UC1" else 1.0)

    def _on_desc(self, m):
        """로봇 모델: 원본 DAE 는 링크별 실제 색을 담고 있으나 선형 값이라 RViz 에서 새까맣다. sRGB 보정 복사본(~/nero/rviz/meshes)으로 메쉬 경로만 교체."""
        try:
            out = String(); mesh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rviz", "meshes")
            out.data = re.sub(r'package://agx_arm_description/agx_arm_urdf/nero/meshes/dae/([\w\-]+\.dae)', lambda mm: "file://" + os.path.join(mesh_dir, mm.group(1)), m.data)
            self.rd_pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f"로봇 모델 교체 실패: {e!r}")

    def lookup(self, tgt, src):
        try: return tf_to_Rt(self.buf.lookup_transform(tgt, src, Time()).transform)
        except Exception: return None

    # ── 물체 분할 점(박스 안쪽 깊이 점 -> 잡음 제거 -> 가장 큰 덩어리) ──
    @staticmethod
    def clean(Q):
        try:
            from scipy.spatial import cKDTree
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            tr = cKDTree(Q); cnt = tr.query_ball_point(Q, r=0.012, return_length=True); Q = Q[cnt >= 4]
            if len(Q) < 8: return Q
            tr = cKDTree(Q); pr = tr.query_pairs(0.02, output_type="ndarray")
            if len(pr) == 0: return Q
            g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(Q), len(Q))); _, lab = connected_components(g, directed=False)
            return Q[lab == np.bincount(lab).argmax()]
        except Exception: return Q

    def segment(self, st):
        objs = {k: v for k, v in (st.get("objs") or {}).items() if "xyz" in v and "raw" in v}
        for k in list(self.objpts):
            _h = self.held
            if k not in objs and not (_h and _h["oid"] == k): self.objpts.pop(k, None); self.hull_cache.pop(k, None)
        d = self.get_depth(); K = self.cam.get("K"); RT = self.lookup(FRAME, CAM_FRAME)
        if d is None or K is None or RT is None or not objs: return
        R, t = RT; s = 6; H, W = d.shape; vv, uu = np.mgrid[0:H:s, 0:W:s]; z = d[::s, ::s]; ok = np.isfinite(z) & (z > 0.1) & (z < 2.0)
        X = (uu[ok] - K[0, 2]) / K[0, 0] * z[ok]; Y = (vv[ok] - K[1, 2]) / K[1, 1] * z[ok]; P = (np.stack([X, Y, z[ok]], 1) @ R.T) + t
        P = P[(P[:, 2] > 0.012) & (np.hypot(P[:, 0], P[:, 1]) < 0.75)]
        for oid, o in objs.items():
            _h = self.held
            if _h and _h["oid"] == oid: continue
            c = np.array(o["xyz"], float); raw = o["raw"]; yaw = float(o.get("yaw", 0.0)); cs, sn = math.cos(-yaw), math.sin(-yaw); dp = P - c
            lx = cs * dp[:, 0] - sn * dp[:, 1]; ly = sn * dp[:, 0] + cs * dp[:, 1]
            m = (np.abs(lx) < raw[0] / 2 + 0.012) & (np.abs(ly) < raw[1] / 2 + 0.012) & (np.abs(dp[:, 2]) < raw[2] / 2 + 0.012)
            Q = P[m]
            if len(Q) > 700: Q = Q[np.random.default_rng(0).choice(len(Q), 700, replace=False)]
            if len(Q) >= 8: Q = self.clean(Q)
            if len(Q) >= 8:
                if len(Q) > 500: Q = Q[np.random.default_rng(0).choice(len(Q), 500, replace=False)]
                self.objpts[oid] = Q; self.hull_cache.pop(oid, None)

    def hull_of(self, Q):
        if Q is None or len(Q) < 6: return None
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(Q); tri = []; ctr = Q.mean(0)
            for s_ in h.simplices:
                p = Q[s_]; n = np.cross(p[1] - p[0], p[2] - p[0])
                if n @ (p.mean(0) - ctr) < 0: p = p[[0, 2, 1]]
                tri += [p[0], p[1], p[2]]
            return tri
        except Exception: return None

    def hull_mesh(self, oid):
        if oid not in self.hull_cache: self.hull_cache[oid] = self.hull_of(self.objpts.get(oid))
        return self.hull_cache[oid]

    # ── 마커 도우미 ──
    def mk(self, ns, i, typ, color, scale=(0.01, 0.01, 0.01), pos=None, quat=(0, 0, 0, 1), pts=None, text=None, colors=None, frame=FRAME):
        m = Marker(); m.header.frame_id = frame; m.ns = ns; m.id = int(i); m.type = typ; m.action = Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = [float(s) for s in scale]; m.color = cr(color); lt = CFG["marker_lifetime"]; m.lifetime.sec = int(lt); m.lifetime.nanosec = int((lt % 1) * 1e9)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = [float(q) for q in quat]
        if pos is not None: m.pose.position = pt(pos)
        if pts is not None: m.points = [pt(p) for p in pts]
        if colors is not None: m.colors = [cr(c) for c in colors]
        if text is not None: m.text = text
        return m

    @staticmethod
    def box_edges(center, size, yaw):
        c = np.asarray(center, float); hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2; cs, sn = math.cos(yaw), math.sin(yaw)
        pts = [c + np.array([cs * sx * hx - sn * sy * hy, sn * sx * hx + cs * sy * hy, sz * hz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        return [p for i, j in idx for p in (pts[i], pts[j])]

    def status_of(self, oid, o, st):
        stt = o.get("status")
        if oid == st.get("target") and stt not in ("dest", "placed"): return "target"
        return stt if stt in SCOL else "obstacle"

    # ── 층별 마커 ──
    def layer_floor(self):
        out = [self.mk("floor", 1, Marker.CUBE, (0.20, 0.21, 0.25, 1.0), scale=(2.6, 2.6, 0.004), pos=[0, 0, -0.0025])]
        lines = []
        for g in np.arange(-0.8, 0.801, 0.1): lines += [[g, -0.8, 0.0005], [g, 0.8, 0.0005], [-0.8, g, 0.0005], [0.8, g, 0.0005]]
        out.append(self.mk("grid", 1, Marker.LINE_LIST, (0.34, 0.36, 0.42, 0.55), scale=(0.0015, 0, 0), pts=lines))
        ring = [[0.55 * math.cos(t), 0.55 * math.sin(t), 0.001] for t in np.linspace(0, 2 * math.pi, 97)]
        out.append(self.mk("workspace", 1, Marker.LINE_STRIP, (0.62, 0.68, 0.82, 0.45), scale=(0.003, 0, 0), pts=ring))
        return out

    def layer_objects(self, st, stage_i):
        out = []; held = self.held; heldid = held["oid"] if held else None
        for oid, o in (st.get("objs") or {}).items():
            if "xyz" not in o or "raw" not in o: continue
            k = _num(oid); sk = self.status_of(oid, o, st); col = SCOL[sk]; xyz = np.array(o["xyz"], float); raw = o["raw"]; yaw = float(o.get("yaw", 0.0))
            dim = 0.55 if (stage_i in (1, 2) and sk not in ("target", "dest")) else 1.0
            lab = str(o.get("label") or "object").replace(" ", "_")[:18]; txt = f"{k}·{lab}"
            ent = bool(st.get("ent_view")) and stage_i == 0; tent = False; _imp = False      # 탐색 단계에서만 도착지 보기: 나머지 주황 / 도착지 후보 분홍 / 최종 선택 파랑, 라벨은 실시간(마지막 갱신) 값
            if ent:
                _dp = st.get("dest_pick"); _dc = set(st.get("dest_cands") or [])
                col = ENT_BLUE if oid == _dp else (ENT_PINK if oid in _dc else (SCOL["target"] if (oid == st.get("target") or o.get("match")) else ENT_ORANGE)); dim = 1.0      # 집을 대상은 초록
                _imp = (oid == _dp or oid in _dc or oid == st.get("target") or o.get("match"))      # 대상/후보/선택만 이름을 붙이고 나머지는 번호만(라벨 난립 방지)
                txt = (("*" if oid == _dp else "") + f"{k} · {str(o.get('label') or '라벨_없음')[:24].replace(' ', '_')}" + (f" · {float(o['dadj']):+.1f}" if o.get("dadj") is not None and (oid in _dc or oid == _dp) else "")) if _imp else f"{k}"      # RViz 3D 글자는 공백에서 단어가 흩어져 그려진다 -> 공백은 밑줄로
            fr = FRAME; Q = self.objpts.get(oid); tri = self.hull_mesh(oid)
            if oid == heldid:                                    # 쥔 물체: 그리퍼 프레임에 붙여 함께 움직이게(점·껍질·라벨 모두 그리퍼 기준 좌표)
                fr = TOOL_FRAME; Q = held["pts"]; tri = held["tri"]
            if Q is not None:
                out.append(self.mk("obj_points", k, Marker.POINTS, (col[0], col[1], col[2], 0.95 * dim), scale=(CFG["point_size"], CFG["point_size"], 0), pts=Q, frame=fr))
                if tri: out.append(self.mk("obj_hull", k, Marker.TRIANGLE_LIST, (col[0], col[1], col[2], (0.40 if sk in ("target", "dest") else 0.28) * dim), scale=(1, 1, 1), pts=tri, frame=fr))
                top = float(Q[:, 2].max()); ctr = Q.mean(0)
                lp = [ctr[0], ctr[1], (0.24 if ent else top + 0.075 + 0.02 * (k % 3))] if fr == FRAME else [ctr[0], ctr[1], ctr[2] + 0.09]
                if ent and fr == FRAME and _imp and not self._sort_on(): out.append(self.mk("obj_leader", k, Marker.LINE_LIST, (col[0], col[1], col[2], 0.55), scale=(0.001, 0, 0), pts=[[ctr[0], ctr[1], top], [ctr[0], ctr[1], 0.24]]))
            else:
                out.append(self.mk("obj_hull", k, Marker.CUBE, (col[0], col[1], col[2], 0.25 * dim), scale=(raw[0], raw[1], max(raw[2], 0.005)), pos=xyz, quat=quat_from_yaw(yaw)))
                lp = [xyz[0], xyz[1], (0.24 if ent else xyz[2] + raw[2] / 2 + 0.075 + 0.02 * (k % 3))]
            out.append(self.mk("obj_box", k, Marker.LINE_LIST, (col[0], col[1], col[2], 0.95 if ent else 0.7), scale=(0.004 if ent else 0.002, 0, 0), pts=self.box_edges(xyz, raw, yaw)))
            if not self._sort_on(): out.append(self.mk("obj_label", k, Marker.TEXT_VIEW_FACING, (min(1.0, col[0] + 0.15), min(1.0, col[1] + 0.15), min(1.0, col[2] + 0.15), 1.0), scale=(0, 0, (CFG["label_size"] * 1.6) if ent else CFG["label_size"]), pos=lp, text=txt, frame=fr)) if (_imp or not ent) else None
        return out

    def layer_frustum(self):
        K = self.cam.get("K"); info = self.cam.get("info"); RT = self.lookup(FRAME, CAM_FRAME)
        if K is None or RT is None: return []
        R, t = RT; W, H = info.width, info.height; L = 0.45
        P = [R @ (np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * L) + t for u, v in ((0, 0), (W, 0), (W, H), (0, H))]; edges = []; tri = []
        for i in range(4): edges += [t, P[i], P[i], P[(i + 1) % 4]]; tri += [t, P[i], P[(i + 1) % 4]]
        tri += [P[0], P[1], P[2], P[0], P[2], P[3]]
        return [self.mk("cam_frustum", 1, Marker.LINE_LIST, (0.85, 0.9, 1.0, 0.55), scale=(0.002, 0, 0), pts=edges), self.mk("cam_frustum", 2, Marker.TRIANGLE_LIST, (0.7, 0.8, 1.0, 0.07), scale=(1, 1, 1), pts=tri)]

    def gripper_markers(self, ns, idx, contact, approach, jaw, opening, color, width, pads_alpha=None):
        lines, pads = gripper_geom(contact, approach, jaw, opening); out = [self.mk(ns, idx * 10, Marker.LINE_LIST, color, scale=(width, 0, 0), pts=lines)]
        if pads_alpha:
            for k, (c, R, sz) in enumerate(pads): out.append(self.mk(ns + "_pad", idx * 10 + k, Marker.CUBE, (color[0], color[1], color[2], pads_alpha), scale=sz, pos=c, quat=quat_from_R(R)))
        return out

    @staticmethod
    def cand_jaw(c):
        if c.get("jaw") is not None: return c["jaw"]
        if c.get("quat"):
            x, y, z, w = c["quat"]; return [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)]
        return [0, 1, 0]

    def layer_grasp(self, st, stage_i):
        """후보는 후보 계산 단계에서만(점수 상위 몇 개, 점수 높을수록 진하게), 채택 후보는 pick 실행 중에만(HELD 면 stage 가 place 로 넘어가 사라진다)."""
        out = []; cs = st.get("cands") or []; chosen = (st.get("plan") or {}).get("cand")
        if not cs or stage_i not in (1, 2): return out
        top = cs[:CFG["cand_top"]]; sc = [c.get("score") for c in top if c.get("score") is not None]; lo, hi = (min(sc), max(sc)) if sc else (0.0, 1.0); rank = hi <= lo
        for i, c in enumerate(cs):
            if not c.get("contact") or not c.get("approach"): continue
            is_ch = chosen is not None and c.get("idx", i) == chosen
            if (stage_i == 2 and not is_ch) or (stage_i == 1 and not is_ch and i >= CFG["cand_top"]): continue
            jaw = self.cand_jaw(c); op = c.get("opening", 0.06)
            f = (1.0 - i / 4.0) if rank else ((c["score"] - lo) / (hi - lo) if c.get("score") is not None else 0.5)
            if is_ch:
                out += self.gripper_markers("grasp_chosen", i, c["contact"], c["approach"], jaw, op, C_GRASP + (1.0,), 0.012, pads_alpha=0.35)
                out.append(self.mk("grasp_chosen", i * 10 + 5, Marker.SPHERE, (1, 1, 1, 0.95), scale=(0.014, 0.014, 0.014), pos=c["contact"]))
                ap = np.array(c["approach"], float); ap = ap / (np.linalg.norm(ap) + 1e-9)
                out += self.gripper_markers("grasp_ghost", i, np.array(c["contact"]) - ap * 0.07, c["approach"], jaw, op, C_GRASP + (0.25,), 0.006)
            else:
                col = (0.55 + 0.45 * f, 0.62 + 0.18 * f, 0.95 - 0.70 * f, 0.35 + 0.35 * f)
                out += self.gripper_markers("grasp_cand", i, c["contact"], c["approach"], jaw, op, col, 0.006)
        return out

    def plan_line(self, ns, pts, color):
        """계획 경로 = 얇은 점선 + 끝점 표시(한 스타일). 실행 경로(굵은 실선)와 구분된다."""
        if not pts or len(pts) < 2: return []
        return [self.mk(ns, 1, Marker.LINE_LIST, color + (0.9,), scale=(0.004, 0, 0), pts=dashed(pts)), self.mk(ns, 2, Marker.SPHERE, color + (1.0,), scale=(0.014, 0.014, 0.014), pos=pts[-1])]

    def exec_line(self, ns, pts, alpha):
        """실행 경로 = 굵은 실선(앞은 연하게 -> 현재 위치는 선명하게) + 끝점. alpha 로 페이드아웃."""
        if len(pts) < 3: return []
        n = len(pts); cols = [(1, 1, 1, alpha * (0.2 + 0.8 * i / (n - 1))) for i in range(n)]
        return [self.mk(ns, 1, Marker.LINE_STRIP, C_EXEC + (alpha,), scale=(0.011, 0, 0), pts=pts, colors=cols), self.mk(ns, 2, Marker.SPHERE, (1, 1, 1, alpha), scale=(0.02, 0.02, 0.02), pos=pts[-1])]

    def layer_traj(self, st, stage_i):
        """계획 경로는 해당 모션을 실행하는 단계에서만, 실행 경로는 실행 중 + 끝난 뒤 fade_s 동안 페이드아웃."""
        out = []; pl = st.get("plan") or {}; ppl = st.get("place_plan") or {}; now = time.time()
        if stage_i == 2 and pl.get("ee_path"): out += self.plan_line("traj_pick", pl["ee_path"], C_PICK)
        if stage_i == 3 and ppl.get("ee_path"): out += self.plan_line("traj_place", ppl["ee_path"], C_PLACE)
        if stage_i in (2, 3): out += self.exec_line("traj_exec", self.trail, 1.0)
        self.fading = [(p, t0) for p, t0 in self.fading if now - t0 < CFG["fade_s"]]
        for i, (p, t0) in enumerate(self.fading): out += self.exec_line(f"traj_exec_fade{i}", p, max(0.0, 1.0 - (now - t0) / CFG["fade_s"]))
        return out

    def layer_place(self, st, stage_i):
        """놓을 곳 하이라이트와 놓일 물체 고스트는 place 계획~실행 중(stage place)에만, 판정이 나면(stage 종료) 사라진다."""
        out = []; ppl = st.get("place_plan") or {}
        if stage_i != 3 or not ppl: return out
        slot = ppl.get("slot")
        if slot:
            z = (ppl.get("slot_z") or 0.0) + 0.002; ring = [[slot[0] + 0.05 * math.cos(t), slot[1] + 0.05 * math.sin(t), z] for t in np.linspace(0, 2 * math.pi, 49)]
            out.append(self.mk("place_slot", 1, Marker.LINE_STRIP, SCOL["dest"] + (1.0,), scale=(0.006, 0, 0), pts=ring))
        oid = ppl.get("oid"); _h = self.held; Q = _h["pts0"] if (_h and _h["oid"] == oid) else self.objpts.get(oid)
        if slot and Q is not None and len(Q) >= 6:
            tri = self.hull_of(Q)
            if tri:
                c0 = Q.mean(0); dz = (ppl.get("slot_z") or 0.0) - Q[:, 2].min(); shift = np.array([slot[0] - c0[0], slot[1] - c0[1], dz])
                out.append(self.mk("place_ghost", 1, Marker.TRIANGLE_LIST, SCOL["target"] + (0.30,), scale=(1, 1, 1), pts=[np.array(p) + shift for p in tri]))
        return out

    def layer_flash(self):
        f = self.flash
        if not f or time.time() > f["until"]: self.flash = None; return []
        col = SCOL["target"] if f["ok"] else SCOL["failed"]; out = []
        if f.get("tri"): out.append(self.mk("result_flash", 1, Marker.TRIANGLE_LIST, col + (0.55,), scale=(1, 1, 1), pts=f["tri"]))
        out.append(self.mk("result_flash", 2, Marker.TEXT_VIEW_FACING, col + (1.0,), scale=(0, 0, 0.045), pos=f["pos"], text=f["text"]))
        return out

    def layer_insert(self, st):
        ins = st.get("insert")
        if not ins: return []
        if time.time() - float(st.get("t", 0) or 0) > 25.0: return []      # 2026-10-15: 상태 갱신이 멈춘 뒤(실행 종료)에도 마지막 삽입 디버그 문구(PEG_PCA, HOLE_..)가 허공에 남아 라벨이 흩어져 보였다
        out = []; E = np.array(ins["E"], float); d = np.array(ins["d"], float); d = d / (np.linalg.norm(d) + 1e-9)
        inner = float(ins.get("inner", 0.06)); wall = float(ins.get("wall", 0.01)); L = float(ins.get("len", 0.14)); h = inner / 2 + wall
        u = np.cross(d, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0]); v = np.cross(d, u)
        corners = lambda c, hh: [c + u * hh * sx + v * hh * sy for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        top = corners(E, h); bot = corners(E + d * L, h); inn = corners(E, inner / 2); edges = []
        for i in range(4):
            j = (i + 1) % 4; edges += [top[i], top[j], bot[i], bot[j], top[i], bot[i]]
        out.append(self.mk("insert_holder", 1, Marker.LINE_LIST, C_HOLE + (1.0,), scale=(0.004, 0, 0), pts=edges))
        out.append(self.mk("insert_mouth", 1, Marker.LINE_STRIP, (1.0, 0.5, 0.15, 1.0), scale=(0.007, 0, 0), pts=inn + [inn[0]]))
        out.append(self.mk("insert_axis", 1, Marker.ARROW, C_HOLE + (1.0,), scale=(0.01, 0.022, 0.028), pts=[E - d * 0.10, E + d * (L + 0.02)]))
        pg = ins.get("peg") or {}
        if pg.get("ok") and pg.get("axis") is not None:
            c = np.array(pg["center"], float); ax = np.array(pg["axis"], float); Lp = float(pg["length"])
            out.append(self.mk("insert_peg", 1, Marker.ARROW, C_PEG + (1.0,), scale=(0.01, 0.022, 0.028), pts=[c - ax * Lp / 2, c + ax * Lp / 2]))
        hp = ins.get("hole_pts")
        if hp:      # 2026-10-08: 홀더 마스크 점(주황), 홀더 점군 PCA 축(입구 축 추정, 자홍), 레이블
            out.append(self.mk("insert_holemask", 1, Marker.POINTS, (1.0, 0.6, 0.1, 0.9), scale=(CFG["point_size"], CFG["point_size"], 0), pts=hp))
        hc = ins.get("hole_pca") or {}
        if hc.get("ok"):
            c_ = np.array(hc["center"], float); a_ = np.array(hc["axis"], float); a_ = a_ / (np.linalg.norm(a_) + 1e-9); Lh = max(0.10, float(hc.get("length", 0.12)))
            out.append(self.mk("insert_holepca", 1, Marker.ARROW, (0.95, 0.2, 0.9, 1.0), scale=(0.008, 0.02, 0.026), pts=[c_ - a_ * Lh / 2, c_ + a_ * Lh / 2]))
            out.append(self.mk("insert_holepca_txt", 1, Marker.TEXT_VIEW_FACING, (0.95, 0.5, 0.95, 1), scale=(0, 0, 0.026), pos=c_ + a_ * (Lh / 2 + 0.025), text="HOLE_PCA"))
        out.append(self.mk("insert_dir_txt", 1, Marker.TEXT_VIEW_FACING, (0.6, 0.9, 1.0, 1), scale=(0, 0, 0.026), pos=E + d * (L + 0.05), text="INSERT_DIR"))
        if pg.get("ok") and pg.get("axis") is not None:
            out.append(self.mk("insert_peg_txt", 1, Marker.TEXT_VIEW_FACING, (0.6, 1.0, 0.6, 1), scale=(0, 0, 0.026), pos=np.array(pg["center"], float) + np.array([0, 0, 0.06]), text="PEG_PCA"))
        tp = ins.get("tip_path")
        if tp and len(tp) > 1: out += self.plan_line("insert_tip", tp, C_TIP)
        out.append(self.mk("insert_text", 1, Marker.TEXT_VIEW_FACING, (1.0, 0.92, 0.5, 1), scale=(0, 0, 0.032), pos=E + np.array([0, 0, 0.12]), text=f"HOLE_{ins.get('alpha', 0):.0f}deg" + (f"_{ins['verdict']}" if ins.get("verdict") else "")))
        return out

    # ── 다물체 정리(2026-10-14): 입력 목적지 박스 + 계획(물체 -> 목적지) -- 글씨는 물체당 한 줄, 목적지당 한 줄만 쓴다 ──
    def _sort_on(self):
        """정리 모드(새 HUD)를 켠다. 예전엔 상태 파일이 10분 안에 갱신됐을 때만 켜서, 조사/계획/LLM 대기로 10분 넘게 멈추면 옛 단일 물체 HUD 로 되돌아갔다 -> 상태 파일 또는 LLM 소통 기록이 3시간 안이면 유지."""
        try:
            t = max(os.path.getmtime(SORT_PATH), os.path.getmtime(LLM_LOG) if os.path.exists(LLM_LOG) else 0.0)
            return time.time() - t < 3 * 3600
        except Exception: return False

    def _sort_status(self):
        global MULTI_LOG
        if "NERO_MULTI_LOG" not in os.environ:      # 2026-10-30: 러너가 상태 파일 링크(/tmp/multi_exec_status.json -> <반복 폴더>/status.json)를 걸면 그 반복 폴더를 로그 폴더로 쓴다(chain.log·plan/·planner_prompt.txt·rep.log)
            try:
                _d = os.path.dirname(os.path.realpath(SORT_PATH))
                if os.path.exists(os.path.join(_d, "chain.log")) or os.path.exists(os.path.join(_d, "planner_prompt.txt")): MULTI_LOG = _d
            except Exception: pass
        """정리 상태 파일을 읽는다. 쓰는 쪽이 파일을 비우고 다시 쓰는 도중에 읽으면 파싱이 실패한다 -> 그때 빈 값을 돌려주면 라벨/카드가 통째로 지워졌다가 다시 나타나 깜박였다. 실패하면 마지막 정상 값을 쓴다."""
        try: S = json.load(open(SORT_PATH)); self._sort_cache = S; return S
        except Exception: return getattr(self, "_sort_cache", None)

    def _sort_cur_id(self, S):
        m = re.search(r"현재 (obj_\d+)", str(S.get("stage", ""))); return m.group(1) if m else None

    def _sort_track(self, S, st):
        """계획 항목마다 xy + 윗면 높이를 따라간다. 이웃이 4~7cm 간격이라 넓게 찾으면 이웃으로 옮겨 붙는다 -> 반경 6cm 안에서 하나만 맞을 때만 갱신, 모호하면 마지막 위치 유지."""
        sig = json.dumps([(x["id"], x["xy"]) for x in S.get("plan", [])])
        if not hasattr(self, "_found"): self._found = {}
        if getattr(self, "_trk_sig", None) != sig: self._trk_sig = sig; self._found = {}; self._trk = {x["id"]: [x["xy"][0], x["xy"][1], 0.12] for x in S.get("plan", [])}
        skip = ("storage bin", "basket", "saucer", "black cup")
        objs = [(k, v) for k, v in (st.get("objs") or {}).items() if "xyz" in v and v.get("status") not in ("dest", "placed", "obstacle_dest") and not str(k).startswith("placed_") and not any(t in str(v.get("label")) for t in skip)]
        for x in S.get("plan", []):
            cur = self._trk[x["id"]]
            c = [(math.hypot(v["xyz"][0] - cur[0], v["xyz"][1] - cur[1]), k, v) for k, v in objs]; c = sorted([t for t in c if t[0] < 0.06], key=lambda t: t[0])
            self._found[x["id"]] = len(c) >= 1
            if len(c) >= 1 and (len(c) == 1 or c[1][0] - c[0][0] > 0.02):
                v = c[0][2]; rz = (v.get("raw") or [0, 0, 0.1])[2]; self._trk[x["id"]] = [v["xyz"][0], v["xyz"][1], float(v["xyz"][2]) + float(rz) / 2]
        return self._trk

    def layer_sort(self):
        if not self._sort_on(): return []
        S = self._sort_status()
        if not S: return []
        try: D = {d["id"]: d for d in json.load(open(S["dests"]))}
        except Exception: return []
        st = self.state or {}; trk = self._sort_track(S, st)
        out = []; cur_id = self._sort_cur_id(S); ok_ids = {r["id"] for r in S.get("results", []) if r.get("ok")}; last = {}
        for r in S.get("results", []): last[r["id"]] = r
        cnt = {}
        for j, (k, d) in enumerate(D.items()):
            z = d["size"][2]
            out.append(self.mk("sort_dest", 10 + j, Marker.LINE_LIST, (0.35, 0.78, 1.0, 0.9), scale=(0.004, 0, 0), pts=self.box_edges((d["xy"][0], d["xy"][1], z / 2), d["size"], d.get("yaw", 0.0))))
        for i, st_ in enumerate(S.get("plan", [])):      # 마커 id 는 항목 번호로 고정(상태가 바뀌어도 같은 id 를 덮어써서 유령 라벨이 안 남는다)
            oid = st_["id"]; r = last.get(oid); dst = D.get(st_.get("dest")); x, y, zt = trk.get(oid, [st_["xy"][0], st_["xy"][1], 0.12])
            _lb = str(st_.get("label_top1", oid)); _lb = _lb.split(" ", 1)[1] if _lb.split(" ", 1)[0] in ("a", "an") and " " in _lb else _lb; _lb = _lb[:12]
            active = (oid == cur_id) and oid not in ok_ids; base = 100 * (i + 1)
            if not active and not self._found.get(oid, False): continue      # 지금 인식에 안 보이는 물체(날아갔거나 아직 못 본 것)는 허공에 라벨을 두지 않는다 -- 상태는 HUD ITEMS 에 표시
            if time.time() - float(st.get("t", 0) or 0) > 25.0 and not active: continue      # 상태 갱신이 멈췄으면(실행 종료 후) 물체 라벨은 숨긴다
            if oid in ok_ids: cnt[st_["dest"]] = cnt.get(st_["dest"], []) + [f"{i + 1}"]; continue      # 끝난 항목은 도착지 위에 모아서 표시
            if st_.get("dest") == "leave": continue
            if r is not None and not r.get("ok") and r.get("outcome") is not None: col = (1.0, 0.3, 0.3, 0.95); tag = f"X {i + 1} {_lb} ({str(r.get('place') or r.get('outcome'))[:12]})"
            elif active: col = (1.0, 0.88, 0.25, 1.0); tag = f"> {i + 1} {_lb} -> {st_['dest']}"
            else: col = (0.72, 0.76, 0.86, 0.6); tag = f"{i + 1}"      # 대기 중인 물체는 번호만(물체가 서로 4~7cm 라 글자가 겹쳐 흩어져 보였다)
            out.append(self.mk("sort_plan", base, Marker.TEXT_VIEW_FACING, col, scale=(0, 0, 0.03 if active else 0.022), pos=(x, y, zt + 0.04), text=tag))      # 물체 윗면 바로 위
            if active and dst is not None: out.append(self.mk("sort_plan", base + 1, Marker.ARROW, col, scale=(0.008, 0.016, 0.02), pts=[(x, y, zt + 0.02), (dst["xy"][0], dst["xy"][1], dst["size"][2] + 0.09)]))
            if dst is not None:      # 계획 용기 연결선(용기별 색; 현재 항목 굵게, 나머지 가늘고 흐리게) -- 읽기 전용 표시
                _dc = DEST_COL[list(D).index(st_["dest"]) % len(DEST_COL)]; out.append(self.mk("sort_link", base + 2, Marker.LINE_LIST, (_dc[0] / 255, _dc[1] / 255, _dc[2] / 255, 0.95 if active else 0.45), scale=(0.007 if active else 0.002, 0, 0), pts=[(x, y, zt + 0.01), (dst["xy"][0], dst["xy"][1], dst["size"][2] + 0.09)]))
        for j, (k, d) in enumerate(D.items()):
            z = d["size"][2]; done = cnt.get(k, [])
            out.append(self.mk("sort_dest", 30 + j, Marker.TEXT_VIEW_FACING, (0.7, 0.9, 1.0, 1.0), scale=(0, 0, 0.04), pos=(d["xy"][0], d["xy"][1], z + 0.07), text=k + (f"  [{len(done)}]" if done else "")))
            if done: out.append(self.mk("sort_dest", 50 + j, Marker.TEXT_VIEW_FACING, (0.4, 0.95, 0.5, 1.0), scale=(0, 0, 0.022), pos=(d["xy"][0], d["xy"][1], z + 0.12), text="done " + ",".join(done)))
        return out

    def build(self, st):
        stage_i = STAGE_IDX.get(st.get("stage"), 0); out = []
        out += self.layer_floor(); out += self.layer_objects(st, stage_i); out += self.layer_frustum(); out += self.layer_grasp(st, stage_i)
        out += self.layer_traj(st, stage_i); out += self.layer_place(st, stage_i); out += self.layer_flash(); out += self.layer_gt(st); out += self.layer_sort()
        try: out += self.layer_insert(st)
        except Exception as e: self.get_logger().warn(f"삽입 마커 실패 {e!r}", throttle_duration_sec=10.0)
        return out

    # ── 카메라 주석 영상: 원본 RGB + 마스크(알파 0.35) + 외곽선 + 라벨 + grasp 2D 투영. 3D 마커는 포함하지 않는다 ──
    @staticmethod
    def project(P, RT, K):
        R, t = RT; C = np.asarray(P, float) @ R.T + t
        if np.any(C[:, 2] < 0.05): return None
        return np.stack([K[0, 0] * C[:, 0] / C[:, 2] + K[0, 2], K[1, 1] * C[:, 1] / C[:, 2] + K[1, 2]], 1)

    def camera_image(self, st, stage_i):
        rgb = self.get_rgb(); K = self.cam.get("K"); RT = self.lookup(CAM_FRAME, FRAME)
        if rgb is None or K is None: return None
        sc = CFG["cam_scale"]; img = cv2.resize(np.ascontiguousarray(rgb), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA); K = K.copy(); K[:2] *= sc; ov = img.copy(); lab = []
        if RT is None: return img
        R, t = RT
        for oid, o in (st.get("objs") or {}).items():
            _h = self.held; Q = self.objpts.get(oid) if not (_h and _h["oid"] == oid) else None
            if Q is None: continue
            C = Q @ R.T + t; C = C[C[:, 2] > 0.05]
            if len(C) < 6: continue
            uv = np.stack([K[0, 0] * C[:, 0] / C[:, 2] + K[0, 2], K[1, 1] * C[:, 1] / C[:, 2] + K[1, 2]], 1)
            hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
            if cv2.contourArea(hull) < 30: continue
            col = tuple(int(255 * x) for x in SCOL[self.status_of(oid, o, st)]); txt_ = f"{_num(oid)} · {str(o.get('label') or 'object')[:16]}"
            if st.get("ent_view") and stage_i == 0:      # 도착지 보기: 나머지 주황 / 후보 분홍 / 최종 선택 파랑 / 대상 초록. 라벨은 대상, 후보, 선택만(나머지는 번호만)
                _dp = st.get("dest_pick"); _dc = set(st.get("dest_cands") or []); _isT = (oid == st.get("target") or o.get("match"))
                _c = ENT_BLUE if oid == _dp else (ENT_PINK if oid in _dc else (SCOL["target"] if _isT else ENT_ORANGE)); col = tuple(int(255 * x) for x in _c)
                txt_ = (("★ " if oid == _dp else "") + txt_) if (oid == _dp or oid in _dc or _isT) else f"{_num(oid)}"
            cv2.fillConvexPoly(ov, hull, col); lab.append((hull, col, txt_))
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        for hull, col, _ in lab: cv2.polylines(img, [hull], True, col, 1, cv2.LINE_AA)
        # grasp 2D 투영: 후보 계산 단계에서는 점수 상위 몇 개를 얇게, pick 중에는 채택 후보를 굵고 밝게
        cs = st.get("cands") or []; chosen = (st.get("plan") or {}).get("cand")
        if cs and stage_i in (1, 2):
            for i, c in enumerate(cs):
                if not c.get("contact") or not c.get("approach"): continue
                is_ch = chosen is not None and c.get("idx", i) == chosen
                if not is_ch and (stage_i == 2 or i >= CFG["cand_top"]): continue
                lines, _ = gripper_geom(c["contact"], c["approach"], self.cand_jaw(c), c.get("opening", 0.06)); uvl = self.project(lines, RT, K)
                if uvl is None: continue
                colg = (110, 255, 140) if is_ch else (255, 190, 90); w = 3 if is_ch else 1
                for j in range(0, len(uvl), 2): cv2.line(img, tuple(int(v) for v in uvl[j]), tuple(int(v) for v in uvl[j + 1]), colg, w, cv2.LINE_AA)
        from PIL import Image as PI, ImageDraw, ImageFont
        pim = PI.fromarray(img); dr = ImageDraw.Draw(pim)
        if not hasattr(self, "_font14"):
            try: self._font14 = ImageFont.truetype(FONT, 14)
            except Exception: self._font14 = ImageFont.load_default()
        f = self._font14
        for hull, col, txt in lab:
            x0, y0 = int(hull[:, 0, 0].min()), int(hull[:, 0, 1].min()); tw = dr.textlength(txt, font=f); y0 = max(18, y0 - 6)
            x0 = int(min(max(x0, 4), max(4, pim.width - tw - 8))); y0 = int(min(y0, pim.height - 6))      # 라벨이 영상 가장자리에서 잘리지 않게 안으로 고정
            dr.rectangle([x0 - 3, y0 - 17, x0 + tw + 4, y0 + 2], fill=(20, 22, 28)); dr.text((x0, y0 - 16), txt, font=f, fill=col)
        return np.asarray(pim)

    # ── HUD + 패널(카메라 영상과 HUD 를 한 장으로; 패널 크기 그대로 그린다) ──
    def _time_split(self):
        """chain.log 의 [rt] +t 타임스탬프로 구간별 시간을 3초마다 다시 계산(time_breakdown.py 와 같은 분류)."""
        now = time.time()
        if now - getattr(self, "_ts_t", 0) < 3.0: return getattr(self, "_ts", {})
        self._ts_t = now; cat = {}
        RU = [("스캔·인식", r"embed-id|local-scan|정지 재채점|사전 훑기 완료|발견:|필요한 부분만|국소 탐색|조기 종료|잘린 검출|프레임 게이트"), ("CGN+B", r"CGN \d+ms|도달 가능한 파지 후보 없음|시점을 바꿔"), ("pick 실행", r"pick 시간표|exec pick"), ("C 계획", r"C 완료|C 실패|C단계"), ("place 계획", r"place 계획"), ("place 실행", r"place verdict"), ("전환·판정", r"pick 완료|\[chain\]|보류|치우")]
        try:
            prev = 0.0
            for ln in open(os.path.join(MULTI_LOG, "chain.log"), errors="ignore"):
                m = re.match(r"^\[rt\] \+\s*([\d.]+)s (.*)", ln)
                if not m: continue
                t = float(m.group(1)); body = m.group(2); c = next((nm for nm, rg in RU if re.search(rg, body)), "기타")
                if t > prev: cat[c] = cat.get(c, 0.0) + t - prev
                prev = t
        except Exception: pass
        self._ts = cat; return cat

    def _plan_info(self, S):
        """PLAN SPLIT 용 읽기 전용 정보: 용기 수용량(프롬프트 '최대 약 N개'), 문구 조건(프롬프트 파일에서 판별: A/B/current), note(plan_llm.json, 없으면 LLM 로그), 계획 대 실제 도착 용기(chain.log '다음 항목 -> dest')."""
        now = time.time(); c = getattr(self, "_pi", None)
        if c and now - c[0] < 2.0: return c[1]
        txt = ""; caps = {}; note = ""; actual = {}
        try: txt = open(os.environ.get("NERO_SORT_PROMPT") or os.path.join(MULTI_LOG, "planner_prompt.txt"), errors="ignore").read()      # 2026-10-30: NERO_SORT_PROMPT 로 프롬프트 파일 직접 지정 가능
        except Exception: pass
        for m in re.finditer(r"^- (\S+) \((\w+)\)[^\n]*?최대 약 (\d+)개", txt, re.M): caps[m.group(1)] = int(m.group(3))
        cond = "n/a" if not txt else ("A" if "세 가지를 함께 저울질" in txt else ("B0" if "2-1. 순서를 다 정한 뒤" in txt else ("B" if "이름·용도가 비슷한 묶음으로 나눈다" in txt else "current")))
        if txt and "## 지정 목적지(1단계 의미 묶음" in txt: cond = "C" if cond == "current" else "C+" + cond      # 2026-10-30: 변형 C(NERO_PLAN_GROUP2) 표기
        try: note = str(json.load(open(os.path.join(MULTI_LOG, "plan", "plan_llm.json"))).get("note", ""))
        except Exception:
            try:
                for l in reversed(open(LLM_LOG).read().splitlines()[-40:]):
                    e = json.loads(l)
                    if e.get("dir") == "llm>robot":
                        m = re.search(r'"note"\s*:\s*"([^"]*)"', str(e.get("text", "")))
                        if m: note = m.group(1)
                        break
            except Exception: pass
        for fn in ("chain.log", "chain_retry.log"):
            try:
                for ln in open(os.path.join(MULTI_LOG, fn), errors="ignore"):
                    m = re.search(r"다음 항목 (obj_\d+) -> (\S+?)\(", ln)
                    if m: actual[m.group(1)] = m.group(2)
            except Exception: pass
        diff = [(x["id"], x.get("dest"), actual[x["id"]]) for x in S.get("plan", []) if actual.get(x["id"]) and x.get("dest") not in (None, "leave") and actual[x["id"]] != x.get("dest")]
        lab = {}
        for m in re.finditer(r"^- (obj_\d+): 라벨 VLM 추정 '([^']*)' / 일반 어휘 CLIP 후보 ([^|]*)", txt, re.M): lab[m.group(1)] = (m.group(2).strip(), m.group(3).strip().split(",")[0].strip())      # (VLM 라벨, CLIP top1)
        r = dict(caps=caps, cond=cond, note=note, diff=diff, lab=lab); self._pi = (now, r); return r

    def _last_t(self, path):
        """[rt] +t 로그 파일의 마지막 t(초). 없거나 못 읽으면 0."""
        v = 0.0
        try:
            for ln in open(path, errors="ignore"):
                m = re.match(r"^\[rt\] \+\s*([\d.]+)s", ln)
                if m: v = float(m.group(1))
        except Exception: pass
        return v

    def _phase_state(self, S, now, running, el):
        """SCAN/PLAN/EXEC/RECOVER 소요(초, 상태)를 기존 산출물에서만 읽어 만든다(읽기 전용). 반환 [(이름, 초|None, 'done'|'run'|'wait'|'est'|'na')].
        SCAN: MULTI_LOG/rep.log 의 '스캔 끝 N s' 가 있으면 정확, 없으면 None(n/a) -- MCP 경로는 스캔 시작이 파일로 안 남음. PLAN: MULTI_LOG/plan/plan_meta.json secs, 없으면 LLM 로그 마지막 robot>llm -> llm>robot 쌍(추정).
        EXEC/RECOVER: 상태 stage '놓친 물체 N개 재시도' 가 나온 시점부터 RECOVER(라이브는 브리지가 처음 본 시각, 종료 후엔 chain_retry.log 마지막 t). 프로세스 안 보류 재시도는 EXEC 에 포함된다."""
        stage = str(S.get("stage", "")); scan = plan = None; ss = ps = "na"
        try:
            for ln in open(os.path.join(MULTI_LOG, "rep.log"), errors="ignore"):
                m = re.search(r"스캔 끝 ([\d.]+)s", ln)
                if m: scan = float(m.group(1)); ss = "done"
        except Exception: pass
        try: plan = float(json.load(open(os.path.join(MULTI_LOG, "plan", "plan_meta.json"))).get("secs")); ps = "done"
        except Exception:
            try:
                ev = [json.loads(l) for l in open(LLM_LOG).read().splitlines()[-40:]]
                for i in range(len(ev) - 1, 0, -1):
                    if ev[i]["dir"] == "llm>robot" and ev[i - 1]["dir"] == "robot>llm": plan = ev[i]["t"] - ev[i - 1]["t"]; ps = "est"; break
            except Exception: pass
        rec = stage.startswith("놓친 물체")
        if rec and getattr(self, "_rec_t0", None) is None: self._rec_t0 = (now, el)      # 라이브: 처음 본 시점
        if not rec and running and getattr(self, "_rec_t0", None) is not None and el < self._rec_t0[1]: self._rec_t0 = None      # 새 실행으로 시계가 되돌아가면 초기화
        retry = self._last_t(os.path.join(MULTI_LOG, "chain_retry.log"))
        if rec and getattr(self, "_rec_t0", None): rec_s = max(0.0, el - self._rec_t0[1]); ex_s = self._rec_t0[1]; es, rs = "done", "run"
        elif not running and retry > 0: rec_s = retry; ex_s = max(0.0, el - retry); es, rs = "done", "done"
        else: rec_s = None; ex_s = el; es = "run" if running else "done"; rs = "wait" if running else "na"
        return [("SCAN", scan, ss), ("PLAN", plan, ps), ("EXEC", ex_s, es), ("RECOVER", rec_s, rs)]

    def _lanes(self, S, el, scan=None, plan=None):
        """겹침 줄용 구간(초). 축 원점 = 스캔 시작(스캔·계획 시간을 알 때), 아니면 구동 시작. ROBOT = SCAN(rep.log) + results[].secs 누적으로 복원한 항목 구간(근사), LLM = PLAN(plan_meta.json secs 우선) + rt_llm_log robot>llm -> llm>robot 쌍.
        반환 (robot, llm, 축 길이)."""
        sc0 = scan or 0.0; pl0 = plan or 0.0; pre = sc0 + pl0; robot = []; llm = []
        if scan: robot.append((0.0, sc0))
        if plan: llm.append((sc0, pre))
        a = pre
        for r in S.get("results", []):
            if r.get("id") == "_summary": continue
            sc = float(r.get("secs") or 0.0)
            if sc > 0: robot.append((a, a + sc)); a += sc
        t0 = float(S.get("updated", time.time())) - float(S.get("t", 0))
        try:
            ev = [json.loads(l) for l in open(LLM_LOG).read().splitlines()[-200:]]
            for i in range(1, len(ev)):
                if ev[i]["dir"] == "llm>robot" and ev[i - 1]["dir"] == "robot>llm":
                    x0 = ev[i - 1]["t"] - t0; x1 = ev[i]["t"] - t0
                    if x1 > 0 and x0 < el and not plan: llm.append((pre + max(0.0, x0), pre + min(el, x1)))      # plan_meta 가 있으면 그 PLAN 구간만 쓴다(같은 호출 이중 표시 방지)
                    elif x1 > 0 and x0 < el and x0 > 1.0: llm.append((pre + x0, pre + min(el, x1)))      # 구동 중 재계획 대기(시작 직후 쌍은 PLAN 과 같은 호출로 보고 제외)
        except Exception: pass
        return robot, llm, pre + max(el, 1.0)

    def sort_hud_image(self, st, stale, W, H):
        from PIL import Image as PI, ImageDraw, ImageFont
        S = self._sort_status() or {}
        BG = (20, 22, 29); CARD = (30, 33, 43); INK = (236, 240, 248); SUB = (150, 160, 180); im = PI.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
        if not hasattr(self, "_fonts"):
            try: self._fonts = (ImageFont.truetype(FONT, CFG["font_body"]), ImageFont.truetype(FONT_B, CFG["font_title"]), ImageFont.truetype(FONT_B, CFG["font_key"]), ImageFont.truetype(FONT, 14))
            except Exception: self._fonts = (ImageFont.load_default(),) * 4
        f_b, f_t, f_k, f_s = self._fonts; fb = CFG["font_body"]; ft = CFG["font_title"]; y = 6

        if not hasattr(self, "_tw"): self._tw = {}
        def tlen(t_, f):      # 글자 폭 측정은 비싸다(문장마다 글자 수만큼 호출) -> (글자, 폰트) 단위 캐시
            k_ = (t_, id(f)); v_ = self._tw.get(k_)
            if v_ is None:
                v_ = dr.textlength(t_, font=f)
                if len(self._tw) < 20000: self._tw[k_] = v_
            return v_
        def fit(t_, f, maxw):
            t_ = str(t_); k_ = ("fit", t_, id(f), maxw); v_ = self._tw.get(k_)
            if v_ is not None: return v_
            r_ = t_
            if tlen(r_, f) > maxw:
                lo_, hi_ = 4, len(r_)      # 이분 탐색: 폭 안에 들어가는 가장 긴 접두
                while lo_ < hi_:
                    mid_ = (lo_ + hi_ + 1) // 2
                    if tlen(r_[:mid_], f) <= maxw: lo_ = mid_
                    else: hi_ = mid_ - 1
                r_ = r_[:lo_]
            if len(self._tw) < 20000: self._tw[k_] = r_
            return r_
        def wrap(t_, f, maxw, maxl):
            k_ = ("wrap", str(t_), id(f), maxw, maxl); v_ = self._tw.get(k_)
            if v_ is not None: return v_
            out_, cur_ = [], ""
            for ch in str(t_):
                if tlen(cur_ + ch, f) > maxw: out_.append(cur_); cur_ = ch
                else: cur_ += ch
            if cur_: out_.append(cur_)
            if len(out_) > maxl: out_ = out_[:maxl]; out_[-1] = fit(out_[-1] + "…", f, maxw)
            if len(self._tw) < 20000: self._tw[k_] = out_
            return out_
        def card(h, title):
            nonlocal y
            h = max(h, ft + 14)
            dr.rounded_rectangle([6, y, W - 6, y + h], radius=10, fill=CARD); dr.text((16, y + 5), title, font=f_t, fill=SUB); y0 = y + 5 + ft + 6; y += h + 6; return y0

        plan = S.get("plan", []); res = S.get("results", []); stage = str(S.get("stage", "")); now = time.time()
        running = bool(S) and not (stage.startswith("종료") or stage.startswith("판단 대기")); el = float(S.get("t", 0)) + ((now - float(S.get("updated", now))) if running else 0.0)
        ok_ids = {r["id"] for r in res if r.get("ok")}; secs = {}
        for r in res:
            if r["id"] != "_summary": secs[r["id"]] = secs.get(r["id"], 0.0) + float(r.get("secs") or 0)
        last = {r["id"]: r for r in res}; cur_id = self._sort_cur_id(S); n_ok = len(ok_ids); N = len([x for x in plan if x.get("dest") != "leave"])
        # 1) 헤더: 경과 시간 + 진행 + NOW 줄
        yy = card(80, "SORT RUN  (clock starts at EXEC)" + ("  running" if running else "  " + ("done" if stage.startswith("종료") else "waiting")))
        dr.text((16, yy - 2), f"{int(el) // 60:02d}:{int(el) % 60:02d}", font=f_k, fill=(255, 225, 130)); n_fail = sum(1 for x in plan if x.get("dest") != "leave" and x["id"] not in ok_ids and x["id"] in last and not (running and x["id"] == cur_id)); dr.text((130, yy + 2), f"OK {n_ok}/{N}", font=f_k, fill=(120, 240, 150)); dr.text((300, yy + 2), f"FAIL {n_fail}", font=f_k, fill=(255, 110, 110) if n_fail else SUB)
        cur_it = next((x for x in plan if x["id"] == cur_id), None); done_t = sorted(v for k_, v in secs.items() if k_ != cur_id)
        med = done_t[len(done_t) // 2] if done_t else None; spent = sum(secs.values()); cur_s = max(0.0, el - spent) if (running and cur_it) else None
        ph = self._phase_state(S, now, running, el)
        # 1a) STAGE 띠(크게): SCAN | PLAN | EXEC | RECOVER -- 지금 단계만 밝고 굵게, 지난 단계는 흐리게+v, 다음 단계는 점선. 아래 줄에 단계 세부와 그 단계 경과 시계(~ = 추정)
        _pi0 = self._plan_info(S); _order = ["SCAN", "PLAN", "EXEC", "RECOVER"]; _llm_wait = None
        try:
            _ev = [json.loads(l_) for l_ in open(LLM_LOG).read().splitlines()[-6:]]
            if _ev and _ev[-1].get("dir") == "robot>llm" and now - _ev[-1]["t"] < 600 and (not running or _ev[-1]["t"] > float(S.get("updated", 0))): _llm_wait = now - _ev[-1]["t"]
        except Exception: pass
        _stg = str(S.get("stage", ""))
        cur_ph = "RECOVER" if (running and _stg.startswith("놓친 물체")) else ("EXEC" if running else ("PLAN" if _llm_wait is not None else None))
        ci = _order.index(cur_ph) if cur_ph else (len(_order) if S else -1)      # 상태 없음: 모든 단계를 점선(대기)으로
        hb = 78; dr.rounded_rectangle([6, y, W - 6, y + hb], radius=10, fill=CARD); xs = 14; wseg = (W - 28 - 3 * 6) // 4; yb = y + 8
        for i_, nm_ in enumerate(_order):
            x0 = xs + i_ * (wseg + 6); colp = {"SCAN": (110, 170, 255), "PLAN": (190, 150, 255), "EXEC": (120, 230, 150), "RECOVER": (255, 190, 90)}[nm_]; vv = next((v_ for n_, v_, _ in ph if n_ == nm_), None)
            if i_ == ci: dr.rectangle([x0, yb, x0 + wseg, yb + 30], fill=colp, outline=(255, 255, 255), width=2); dr.text((x0 + 8, yb + 2), nm_, font=f_t, fill=(20, 22, 29))
            elif i_ < ci and not (nm_ == "RECOVER" and vv is None) and not (vv is None and nm_ in ("SCAN",)):
                dr.rectangle([x0, yb, x0 + wseg, yb + 30], fill=tuple(int(c_ * 0.35) for c_ in colp)); dr.text((x0 + 8, yb + 5), "v " + nm_, font=f_b, fill=(190, 198, 214))
            else:
                for xx_ in range(x0, x0 + wseg, 8): dr.line([xx_, yb, xx_ + 4, yb], fill=SUB); dr.line([xx_, yb + 30, xx_ + 4, yb + 30], fill=SUB)
                dr.line([x0, yb, x0, yb + 30], fill=SUB); dr.line([x0 + wseg, yb, x0 + wseg, yb + 30], fill=SUB); dr.text((x0 + 8, yb + 5), nm_, font=f_b, fill=SUB)
        det = "-"; clk = None
        if cur_ph == "EXEC":
            det = (f"{cur_id} -> {cur_it.get('dest')}  (item {1 + sum(1 for x in plan if x['id'] in last and x['id'] != cur_id)}/{len(plan)})" if cur_it else "executing chain"); clk = next((v_ for n_, v_, _ in ph if n_ == "EXEC"), None)
        elif cur_ph == "RECOVER":
            _tx = ""
            for fn_ in ("chain_retry.log", "chain.log"):
                try: _tx = "".join(open(os.path.join(MULTI_LOG, fn_), errors="ignore").readlines()[-25:]); break
                except Exception: pass
            tag_ = "next target search" if re.search(r"국소 탐색|발견:|\[chain\] 다음 항목", _tx) else "-"
            if re.search(r"시점을 바꿔", _tx): tag_ = "viewpoint retry"
            if re.search(r"홈 복귀|시작: 관측 자세", _tx) and tag_ == "-": tag_ = "home"
            det = f"RE-OBSERVE: {tag_}" + (f"   target {cur_id}" if cur_id else ""); clk = next((v_ for n_, v_, _ in ph if n_ == "RECOVER"), None)
        elif cur_ph == "PLAN": det = "waiting for LLM plan"; clk = _llm_wait
        elif not running: det = "finished" if _stg.startswith("종료") else (_stg[:30] or "idle")
        dr.text((14, yb + 38), fit(det, f_b, W - 28 - 90), font=f_b, fill=INK)
        if clk is not None: dr.text((W - 96, yb + 36), f"{'~' if cur_ph == 'PLAN' else ''}{int(clk) // 60}:{int(clk) % 60:02d}", font=f_k, fill=(255, 225, 130))
        y += hb + 6
        # 1b) PHASE STRIP: SCAN | PLAN | EXEC | RECOVER (폭은 소요 비례, 최소 폭 보장; ~ = 추정, n/a = 산출물 없음)
        ph = self._phase_state(S, now, running, el); _pre = sum(v for nm, v, _ in ph if nm in ("SCAN", "PLAN") and v); _sum = sum(v for _, v, _ in ph if v); _gap = _sum - (_pre + el)
        yy = card(70, f"PHASE  pre-run {int(_pre) // 60}:{int(_pre) % 60:02d} + run {int(el) // 60}:{int(el) % 60:02d}" + (f"   GAP {_gap:+.0f}s" if abs(_gap) > 3 else "")); tot_p = sum(v for _, v, _ in ph if v) or 1.0; x_ = 12; wavail = W - 24 - 3 * 4
        cols = {"SCAN": (110, 170, 255), "PLAN": (190, 150, 255), "EXEC": (120, 230, 150), "RECOVER": (255, 190, 90)}
        for nm, v, stt in ph:
            w_ = 104 + int((wavail - 4 * 104) * (v or 0) / tot_p); w_ = min(w_, W - 12 - x_)
            if stt in ("done", "run", "est"): dr.rectangle([x_, yy, x_ + w_, yy + 30], fill=tuple(int(c * (1.0 if stt == "run" else 0.62)) for c in cols[nm]), outline=(255, 255, 255) if stt == "run" else None)
            else: dr.rectangle([x_, yy, x_ + w_, yy + 30], outline=SUB)
            txt = "n/a" if v is None else f"{'~' if stt == 'est' else ''}{int(v) // 60}:{int(v) % 60:02d}"
            dr.text((x_ + 5, yy + 4), fit(f"{nm} {txt}", f_s, w_ - 6), font=f_s, fill=(20, 22, 29) if stt in ("done", "run", "est") else SUB); x_ += w_ + 4
        # 2) ITEM BARS: 항목별 상태 + 걸린 시간(막대 길이 = 초, 공통 축)
        pi = self._plan_info(S)
        def names(x_):      # 표시 이름: VLM 라벨 먼저(빈 값이면 obj_N), CLIP top1 은 괄호로 작게. 프롬프트가 없으면 계획의 label_top1
            vlm_, clip_ = pi["lab"].get(x_["id"], (None, None))
            if vlm_ is None: lb_ = str(x_.get("label_top1", "")); vlm_ = lb_.split(" ", 1)[1] if lb_.split(" ", 1)[0] in ("a", "an") and " " in lb_ else lb_; clip_ = ""
            return (vlm_ if vlm_ not in ("", "?") else x_["id"]), (f"(clip: {clip_})" if clip_ else "")
        rh = 19; yy = card(ft + 14 + rh * max(1, len(plan)), "ITEMS  (bar = seconds)"); tmax = max(list(secs.values()) + [1.0]); ntry = collections_Counter(r["id"] for r in res if r.get("id") != "_summary")
        bx0 = 300; bx1 = W - 170
        for i, x_ in enumerate(plan):
            oid = x_["id"]; r = last.get(oid); t_ = secs.get(oid)
            if oid in ok_ids: tag, col = "OK", (120, 240, 150)
            elif x_.get("dest") == "leave": tag, col = "두기", (150, 160, 180)
            elif oid == cur_id and running: tag, col = "진행", (255, 225, 90)
            elif r is not None: _tg = str(r.get("place") or r.get("outcome")); tag, col = _tg, (255, 110, 110)
            else: tag, col = "대기", (150, 160, 180)
            tag = LABELS_EN["tag"].get(tag, tag)[:12]
            _m, _c = names(x_); _t = fit(f"{i + 1:>2} {_m}", f_b, 150); dr.text((16, yy + i * rh), _t, font=f_b, fill=col); _xs = 16 + tlen(_t, f_b) + 5
            if _c: dr.text((_xs, yy + i * rh + 3), fit(_c, f_s, 252 - _xs), font=f_s, fill=SUB)
            dr.text((256, yy + i * rh + 2), str(x_.get("dest")), font=f_s, fill=col)
            if t_ is not None: dr.rectangle([bx0, yy + i * rh + 4, bx0 + max(2, int((bx1 - bx0) * t_ / tmax)), yy + i * rh + fb - 2], fill=col)
            elif oid == cur_id and running and cur_s is not None: dr.rectangle([bx0, yy + i * rh + 4, bx0 + max(2, int((bx1 - bx0) * min(cur_s, tmax) / tmax)), yy + i * rh + fb - 2], outline=col)
            dr.text((W - 160, yy + i * rh), tag, font=f_b, fill=col)
            if t_ is not None: dr.text((W - 52, yy + i * rh), f"{t_:.0f}s" + (f"x{ntry[oid]}" if ntry[oid] > 1 else ""), font=f_s, fill=INK)
        # 2b) OVERLAP LANES: ROBOT(항목 구간 복원, 근사) / LLM(대기 구간). 겹친 시간은 빗금.
        yy = card(76, "LANES  (from scan start; ROBOT reconstructed)"); _scan = next((v for nm, v, _ in ph if nm == "SCAN"), None); _plan = next((v for nm, v, _ in ph if nm == "PLAN"), None); rb, lm, span = self._lanes(S, el, _scan, _plan); ax1 = W - 20; ax0 = 70
        for ln_i, (nm, iv, col) in enumerate((("ROBOT", rb, (120, 230, 150)), ("LLM", lm, (190, 150, 255)))):
            y_ = yy + ln_i * 26; dr.text((16, y_ + 2), nm, font=f_s, fill=SUB); dr.rectangle([ax0, y_ + 4, ax1, y_ + 20], outline=(60, 66, 84))
            for a_, b_ in iv: dr.rectangle([ax0 + int((ax1 - ax0) * a_ / span), y_ + 5, max(ax0 + int((ax1 - ax0) * a_ / span) + 1, ax0 + int((ax1 - ax0) * b_ / span)), y_ + 19], fill=col)
        for ra, rb_ in rb:
            for la, lb_ in lm:
                a_, b_ = max(ra, la), min(rb_, lb_)
                if b_ > a_:
                    xa = ax0 + int((ax1 - ax0) * a_ / span); xb = ax0 + int((ax1 - ax0) * b_ / span)
                    for xx_ in range(xa, xb + 1, 4): dr.line([xx_, yy + 4, xx_ + 4, yy + 46], fill=(255, 90, 90), width=1)
        # 3) 시간 분해
        ts = self._time_split(); tot = sum(ts.values()) or 1.0; items = sorted(ts.items(), key=lambda kv: -kv[1])[:2]
        yy = card(ft + 14 + (fb + 4) * max(1, len(items)), "TIME  (where it goes)")
        for i, (nm, v) in enumerate(items):
            r_ = yy + i * (fb + 4); dr.text((16, r_), LABELS_EN["split"].get(nm, nm)[:14], font=f_b, fill=SUB); dr.rectangle([180, r_ + 4, 180 + int((W - 180 - 120) * v / items[0][1]), r_ + fb - 2], fill=(110, 170, 255)); dr.text((W - 112, r_), f"{v:.0f}s {100 * v / tot:.0f}%", font=f_b, fill=INK)
        # 4) PLAN SPLIT: LLM 이 물체를 용기별로 어떻게 나눴나(plan_llm.json/status plan 읽기 전용). 칩=용기 색, n/전체, 수용량 표(프롬프트 최대 약 N개) 대비 채움 막대(참고), 물체 토큰(v=완료, x=실패)
        pi = self._plan_info(S); dests_ = sorted({x.get("dest") for x in plan if x.get("dest") not in (None, "leave")} | set(pi["caps"]), key=str); ntot = len([x for x in plan if x.get("dest") != "leave"])
        n_leave = len([x for x in plan if x.get("dest") == "leave"]); nrows = len(dests_) + (1 if n_leave else 0)
        why_it = cur_it or next((x for x in reversed(plan) if x["id"] in last), None) or (plan[0] if plan else None)
        yy = card(ft + 14 + 22 * max(1, nrows) + 17 * (2 + (1 if pi["diff"] else 0)) + 4, f"PLAN SPLIT  [prompt: {pi['cond']}]"); r_ = yy
        def _lab(x_): return names(x_)[0][:7]
        for j, dk in enumerate(dests_):
            col = DEST_COL[j % len(DEST_COL)]; its = [x for x in plan if x.get("dest") == dk]; n = len(its); cap = pi["caps"].get(dk)
            dr.rectangle([16, r_ + 3, 30, r_ + 17], fill=col); dr.text((36, r_), f"{dk} {n}/{ntot}", font=f_b, fill=INK); bx, bw = 150, 64
            dr.rectangle([bx, r_ + 4, bx + bw, r_ + 16], outline=SUB)
            if cap: dr.rectangle([bx, r_ + 4, bx + max(1, int(bw * min(1.0, n / cap))), r_ + 16], fill=col if n <= cap else (255, 110, 110)); dr.text((bx + bw + 4, r_ + 2), f"cap {cap}", font=f_s, fill=SUB)
            x_ = bx + bw + (50 if cap else 8)
            for x2 in its:
                oid = x2["id"]; mk_, mc = ("v", (120, 240, 150)) if oid in ok_ids else (("x", (255, 110, 110)) if oid in last else ("", SUB)); tok = _lab(x2) + (f"[{mk_}]" if mk_ else ""); w_ = tlen(tok, f_s) + 7
                if x_ + w_ > W - 14: dr.text((x_, r_ + 2), "…", font=f_s, fill=SUB); break
                dr.text((x_, r_ + 2), tok, font=f_s, fill=mc if mk_ else (205, 212, 226)); x_ += w_
            r_ += 22
        if n_leave: dr.text((36, r_), f"leave {n_leave}", font=f_b, fill=SUB); r_ += 22
        if why_it: dr.text((16, r_), fit(f"why {why_it['id']} {names(why_it)[0]} {names(why_it)[1]}: {why_it.get('why', '')}", f_s, W - 32), font=f_s, fill=(205, 212, 226)); r_ += 17
        dr.text((16, r_), fit("note: " + (pi["note"] or "-"), f_s, W - 32), font=f_s, fill=SUB); r_ += 17
        if pi["diff"]: dr.text((16, r_), fit("changed: " + ", ".join(f"{o} plan {a} -> actual {b}" for o, a, b in pi["diff"]), f_s, W - 32), font=f_s, fill=(255, 190, 90)); r_ += 17
        # 5) LOG: 남는 자리
        if H - y - 10 >= 60:
            yy = card(H - y - 12, "LOG"); lg = (st.get("log") or [])[-max(1, (H - y - 12 - ft - 14) // (fb + 4)):]
            for i, l in enumerate(lg): dr.text((16, yy + i * (fb + 4)), fit(l, f_b, W - 32), font=f_b, fill=(205, 212, 226))
        return np.asarray(im)

    def hud_image(self, st, stale, W, H):
        from PIL import Image as PI, ImageDraw, ImageFont
        # 2026-11-02: 예전엔 상태 파일/LLM 기록이 3시간 안일 때만 새 HUD(sort_hud_image)를 썼다 -> RViz 를 새로 켜면(계획 전·기록이 오래됨) 옛 단일 물체 HUD 가 보이다가 계획이 서면 새 HUD 로 바뀌었다. 이제 기본은 항상 새 HUD(상태가 없으면 waiting 표시). 옛 동작은 NERO_RVIZ_OLD_HUD=1
        if (os.environ.get("NERO_RVIZ_OLD_HUD", "0") != "1" or self._sort_on()) and CFG.get("sort_hud", True): return self.sort_hud_image(st, stale, W, H)
        BG = (20, 22, 29); CARD = (30, 33, 43); INK = (236, 240, 248); SUB = (150, 160, 180); im = PI.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
        fb, ft, fk = CFG["font_body"], CFG["font_title"], CFG["font_key"]
        if not hasattr(self, "_fonts"):
            try: self._fonts = (ImageFont.truetype(FONT, fb), ImageFont.truetype(FONT_B, ft), ImageFont.truetype(FONT_B, fk), ImageFont.truetype(FONT, 14))
            except Exception: self._fonts = (ImageFont.load_default(),) * 4
        f_b, f_t, f_k, f_s = self._fonts
        rgb = lambda c: tuple(int(255 * x) for x in c); y = 6

        def card(h, title):
            nonlocal y
            dr.rounded_rectangle([6, y, W - 6, y + h], radius=10, fill=CARD); dr.text((16, y + 5), title, font=f_t, fill=SUB); y0 = y + 5 + ft + 6; y += h + 6; return y0

        def fit(s_, f, maxw):
            s_ = str(s_)
            while dr.textlength(s_, font=f) > maxw and len(s_) > 4: s_ = s_[:-2]
            return s_
        stage = st.get("stage", "-"); idx = STAGE_IDX.get(stage, -1); age = time.time() - float(st.get("t", 0)); stats = st.get("stats") or {}; cmd = st.get("command") or {}
        yy = card(168, "STAGE"); x = 16; cw = (W - 32 - 18) // 4                                    # 1) STAGE 진행 + 명령 + 성공/시도(핵심 값은 크게)
        for i, name in enumerate(CHIPS):
            on = (i == idx) and not stale; done = (idx > i) or idx == 4; col = (46, 190, 100) if on else ((70, 92, 120) if done else (44, 48, 60))
            dr.rounded_rectangle([x, yy, x + cw, yy + 30], radius=8, fill=col); tw = dr.textlength(name, font=f_b); dr.text((x + (cw - tw) / 2, yy + 3), name, font=f_b, fill=(255, 255, 255) if on or done else (140, 148, 166)); x += cw + 6
        pl = (str(cmd.get("place") or "").split("|")[0] or "-"); tg = str(cmd.get("target") or "전부").split("|")[0]
        dr.text((16, yy + 34), fit(f"대상  {tg}", f_b, W - 32), font=f_b, fill=(255, 225, 130)); dr.text((16, yy + 56), fit(f"목적지  {pl}", f_b, W - 32), font=f_b, fill=(255, 225, 130))
        picks = list(st.get("picked") or []); pk_ok = stats.get("picks_ok", 0); pk_n = stats.get("attempts", 0); pl_ok = sum(1 for e in picks if e.get("place") == "PLACED"); pl_n = sum(1 for e in picks if e.get("place"))
        dr.text((16, yy + 76), f"pick {pk_ok}/{pk_n}", font=f_k, fill=(120, 240, 150)); dr.text((W // 2, yy + 76), f"place {pl_ok}/{pl_n}", font=f_k, fill=(120, 200, 255))
        dr.text((16, yy + 114), f"현재 {stage}  ·  갱신 {age:.0f}s 전" + ("  ⚠ 상태 갱신 없음" if stale else ""), font=f_b, fill=(255, 90, 90) if stale else SUB)
        if st.get("ent_view") and idx == 0:      # 탐색 단계 도착지 보기: DEST(큰 카드) + LOG. 비어 있는 LATENCY/PLAN/DECISION 카드는 생략
            yy = card(360, "DEST"); dr_ = st.get("dest_reason") or {}; r = yy
            if not dr_: dr.text((16, r), "도착지 후보 탐색 중", font=f_b, fill=SUB)
            else:
                dr.text((16, r), fit(f"질의  {dr_.get('query', '')}", f_b, W - 32), font=f_b, fill=(255, 225, 130)); r += fb + 4
                dr.text((16, r), f"종류  {dr_.get('kind') or '-'}", font=f_b, fill=(255, 225, 130)); r += fb + 12
                for row in (dr_.get("rows") or [])[:5]:
                    ch = bool(row.get("chosen")); col_ = rgb(ENT_BLUE) if ch else rgb(ENT_PINK)
                    dr.text((16, r), fit(("★ " if ch else "● ") + f"{str(row.get('id', '')).replace('obj_', '')}  {str(row.get('label') or '라벨 없음')[:26]}", f_b, W - 32), font=f_b, fill=col_); r += fb + 3
                    dr.text((34, r), fit(f"문구 {row.get('s_p', 0):+.1f}" + (f"   색 {row.get('color')} {row.get('bonus', 0):+.1f}" if row.get("color") else "") + f"   합계 {row.get('adj', 0):+.1f}", f_s, W - 50), font=f_s, fill=(205, 212, 226)); r += 14 + 10
                if dr_.get("note"): dr.text((16, r + 4), fit(str(dr_["note"]), f_b, W - 32), font=f_b, fill=(205, 212, 226))
            if H - y - 56 >= 70:
                yy = card(H - y - 56, "LOG"); lg = (st.get("log") or [])[-max(1, (H - y - 56 - 40) // (fb + 4)):]
                for i, l in enumerate(lg): dr.text((16, yy + i * (fb + 4)), fit(l, f_b, W - 32), font=f_b, fill=(205, 212, 226))
            yl = H - 50; xl = 14
            for name, c in (("대상", SCOL["target"]), ("후보", ENT_PINK), ("선택", ENT_BLUE), ("나머지", ENT_ORANGE)):
                dr.rounded_rectangle([xl, yl + 3, xl + 13, yl + 16], radius=3, fill=rgb(c)); dr.text((xl + 18, yl), name, font=f_s, fill=INK); xl += int(dr.textlength(name, font=f_s)) + 36
            return np.asarray(im)
        tm = st.get("timing") or {}; yy = card(160, "LATENCY")                                       # 2) 지연 + 사이클 타임라인
        rows = [("정렬", tm.get("align"), (150, 160, 190)), ("CGN", None if tm.get("cgn_ms") is None else tm["cgn_ms"] / 1000.0, (255, 170, 60)), ("B", None if tm.get("b_ms") is None else tm["b_ms"] / 1000.0, (90, 170, 255)), ("cuRobo", tm.get("c_s"), (120, 230, 140)), ("발견→준비", tm.get("ready"), (255, 230, 90))]
        mx = max([v for _, v, _ in rows if isinstance(v, (int, float))] + [0.5])
        for i, (lab_, v, c) in enumerate(rows):
            r = yy + i * (fb + 4); dr.text((16, r), lab_, font=f_b, fill=SUB)
            if isinstance(v, (int, float)): dr.rectangle([112, r + 4, 112 + int((W - 112 - 80) * v / mx), r + fb - 2], fill=c); dr.text((W - 70, r), f"{v:.2f}s", font=f_b, fill=INK)
        ty = yy + len(rows) * (fb + 4) + 2
        if tm.get("ready"):                                                    # 사이클 타임라인: 발견→pick 준비 안에서 단계별 비중
            al = tm.get("align") or 0.0; rtt = tm.get("cgn_b_rtt") or 0.0; cs_ = tm.get("c_s") or 0.0; oth = max(tm["ready"] - al - rtt - cs_, 0.0); tot = max(al + rtt + cs_ + oth, 0.01); bx = 16; bw = W - 32
            for nm, v, c in (("정렬", al, (150, 160, 190)), ("CGN+B", rtt, (190, 130, 255)), ("cuRobo", cs_, (120, 230, 140)), ("기타", oth, (255, 230, 90))):
                w = int(bw * v / tot)
                if w > 0:
                    dr.rectangle([bx, ty, bx + w, ty + 16], fill=c)
                    if w > 60: dr.text((bx + 3, ty), nm, font=f_s, fill=(20, 22, 29))
                    bx += w
        pl_, ppl, ins = st.get("plan"), st.get("place_plan"), st.get("insert"); yy = card(78, "PLAN"); r = yy   # 3) PLAN
        if pl_: dr.text((16, r), fit(f"pick  {pl_.get('oid')}  후보 #{pl_.get('cand')}  접근 {pl_.get('pre_used')}", f_b, W - 32), font=f_b, fill=C_hex(C_PICK)); r += fb + 5
        if ppl: dr.text((16, r), fit(f"place  {ppl.get('oid')}  {ppl.get('mode')}  완화 {ppl.get('relax') or '-'}  z {float(ppl.get('slot_z') or 0.0):.3f}", f_b, W - 32), font=f_b, fill=C_hex(C_PLACE)); r += fb + 5
        if ins: dr.text((16, r), fit(f"삽입  기울기 {float(ins.get('alpha', 0) or 0):.0f}°  {ins.get('verdict') or ''}", f_b, W - 32), font=f_b, fill=C_hex(C_HOLE)); r += fb + 5
        if not (pl_ or ppl or ins): dr.text((16, r), "계획 없음", font=f_b, fill=SUB)
        if st.get("ent_view") and idx == 0:      # 탐색 단계: 도착지 선정 근거 카드(DECISION 자리, NERO_RVIZ_ENT=1 일 때)
            yy = card(172, "DEST"); dr_ = st.get("dest_reason") or {}
            if not dr_: dr.text((16, yy), "도착지 후보 탐색 중", font=f_b, fill=SUB); r = yy + fb + 6
            else:
                dr.text((16, yy), fit(f"질의 '{dr_.get('query', '')}'  종류 {dr_.get('kind') or '-'}", f_b, W - 32), font=f_b, fill=(255, 225, 130)); r = yy + fb + 6
                for row in (dr_.get("rows") or [])[:4]:
                    ch = bool(row.get("chosen")); col_ = rgb(ENT_BLUE) if ch else rgb(ENT_PINK)
                    dr.text((16, r), fit(("★ " if ch else "· ") + f"{row.get('id')}  {str(row.get('label') or '라벨 없음')[:22]}  문구 {row.get('s_p', 0):+.1f}" + (f"  색 {row.get('color')} {row.get('bonus', 0):+.1f}" if row.get("color") else "") + f"  = {row.get('adj', 0):+.1f}", f_b, W - 32), font=f_b, fill=col_); r += fb + 4
                if dr_.get("note"): dr.text((16, r), fit(str(dr_["note"]), f_b, W - 32), font=f_b, fill=(205, 212, 226))
            dc = None; _skip_dc = True
        else: _skip_dc = False
        dc = st.get("decision"); yy = (card(172, "DECISION") if not _skip_dc else yy)                                   # 3.5) 놓기/꽂기 판단: 어느 분기를 탔는지 + 근거(2026-10-12)
        if _skip_dc: pass
        elif not dc: dr.text((16, yy), "판단 전", font=f_b, fill=SUB)
        else:
            md = dc.get("mode"); how = dc.get("how"); lab_ = {"insert": "INSERT 꽂기(세워서)", "place": "PLACE 놓기", None: "ABORT 안 들어감"}.get(md, str(md)); col_ = {"insert": (255, 170, 60), "place": (120, 200, 255), None: (255, 110, 110)}.get(md, INK)
            dr.text((16, yy), fit("→ " + lab_ + (f"  ({how})" if how and md == "place" else ""), f_k, W - 32), font=f_k, fill=col_); r = yy + fk + 8
            ob, ds = dc.get("obj") or {}, dc.get("dest") or {}
            lines = []
            if ob: lines.append(f"물체  {ob.get('kind', '')}  L {100 * float(ob.get('L', 0)):.1f}cm  단면 {100 * float(ob.get('c1', 0)):.1f}×{100 * float(ob.get('c2', 0)):.1f}cm")
            if ds: lines.append(f"목적지  {ds.get('kind', '')}  " + ("개구부 %.1f×%.1fcm" % (100 * ds['open'][0], 100 * ds['open'][1]) if ds.get("open") else "평면") + (f"  기울기 {float(ds.get('tilt', 0)):.0f}°" if ds.get("open") else ""))
            if dc.get("clear_lie") is not None: lines.append(f"여유(한 변)  눕힘 {1000 * float(dc['clear_lie']):.0f}mm  세움 {1000 * float(dc['clear_up']):.0f}mm  (기준 ≥ {1000 * float(dc.get('min_clear', 0.006)):.0f}mm)")
            if dc.get("hole_check"): lines.append(str(dc["hole_check"]))
            lines.append({"insert": "적용: 삽입 계획으로 전환", "place": ("적용: 기존 놓기 경로" if dc.get("applied") else "홀이 아니라 모드 전환 없이 기존 놓기 경로"), None: "적용: 서버 호출 없이 중단, 사유 반환"}.get(md, ""))
            for l_ in lines[:5]: dr.text((16, r), fit(l_, f_b, W - 32), font=f_b, fill=(205, 212, 226)); r += fb + 4
        yy = card(76, "RESULTS"); pk = picks[-2:]                                               # 4) RESULTS
        for i, e in enumerate(pk):
            ok = e.get("verdict") == "HELD" and e.get("place") in (None, "PLACED"); s_ = f"{e.get('id')}  {str(e.get('label', ''))[:12]}  {e.get('verdict')}" + (f" → {e['place']}" if e.get("place") else "")
            dr.text((16, yy + i * (fb + 4)), fit(s_, f_b, W - 32), font=f_b, fill=(120, 240, 150) if ok else (255, 110, 110))
        if not pk: dr.text((16, yy), "아직 없음", font=f_b, fill=SUB)
        try:      # 다물체 정리 진행 카드(2026-10-14)
            if self._sort_on():
                S = json.load(open(SORT_PATH)); pl = [x for x in S.get("plan", [])]; rs = {r["id"]: r for r in S.get("results", [])}
                yy = card(32 + 22 * max(1, len(pl)) + 6, f"SORT  {S.get('stage', '')}")
                for i, x_ in enumerate(pl):
                    r = rs.get(x_["id"]); tag = "대기" if r is None else ("두기" if r.get("result") else ("성공" if r.get("ok") else "실패:" + str(r.get("place"))))
                    col = (160, 168, 186) if r is None else ((120, 240, 150) if r.get("ok") else ((200, 205, 215) if r.get("result") else (255, 110, 110)))
                    dr.text((16, yy + i * 22), fit(f"{i + 1}. {x_['id']}  {str(x_.get('label_top1', ''))[:14]} → {x_.get('dest')}   {tag}", f_b, W - 32), font=f_b, fill=col)
        except Exception: pass
        if H - y - 56 >= 70:      # 판단 카드 때문에 자리가 모자라면 LOG 카드는 생략한다
            yy = card(H - y - 56, "LOG"); lg = (st.get("log") or [])[-max(1, (H - y - 56 - 40) // (fb + 4)):]            # 5) LOG(들어가는 만큼, 한 줄 유지)
            for i, l in enumerate(lg): dr.text((16, yy + i * (fb + 4)), fit(l, f_b, W - 32), font=f_b, fill=(205, 212, 226))
        y = H - 50; x = 14                                                                         # 6) 범례(물체 상태색 / 경로 스타일)
        for name, c in (("대상", SCOL["target"]), ("장애물", SCOL["obstacle"]), ("도착지", SCOL["dest"]), ("놓은", SCOL["placed"]), ("실패", SCOL["failed"])):
            dr.rounded_rectangle([x, y + 3, x + 13, y + 16], radius=3, fill=rgb(c)); dr.text((x + 18, y), name, font=f_s, fill=INK); x += int(dr.textlength(name, font=f_s)) + 36
        dr.text((14, y + 24), "- - -  계획 경로(얇은 점선)       ━━  실행 경로(굵은 실선)", font=f_s, fill=SUB)
        return np.asarray(im)

    # ── 점군(RGB-D) ──
    def scene_cloud(self):
        d = self.get_depth(); rgb = self.get_rgb(); K = self.cam.get("K"); RT = self.lookup(FRAME, CAM_FRAME)
        if d is None or rgb is None or K is None or RT is None: return None
        R, t = RT; s = 8; H, W = d.shape; vv, uu = np.mgrid[0:H:s, 0:W:s]; z = d[::s, ::s]; ok = np.isfinite(z) & (z > 0.1) & (z < 2.0)
        X = (uu[ok] - K[0, 2]) / K[0, 0] * z[ok]; Y = (vv[ok] - K[1, 2]) / K[1, 1] * z[ok]; P = (np.stack([X, Y, z[ok]], 1) @ R.T + t).astype(np.float32); c = rgb[::s, ::s][ok].astype(np.uint32)
        packed = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).astype(np.uint32).view(np.float32)
        arr = np.zeros(len(P), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("rgb", "f4")]); arr["x"], arr["y"], arr["z"], arr["rgb"] = P[:, 0], P[:, 1], P[:, 2], packed
        m = PointCloud2(); m.header.frame_id = FRAME; m.header.stamp = self.get_clock().now().to_msg(); m.height = 1; m.width = len(arr); m.is_dense = True; m.is_bigendian = False
        m.fields = [PointField(name=n_, offset=o, datatype=PointField.FLOAT32, count=1) for n_, o in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))]; m.point_step = 16; m.row_step = 16 * len(arr); m.data = _u8(arr); return m

    # ── 초점 TF ──
    def focus_target(self, st, stage_i):
        objs = {k: v for k, v in (st.get("objs") or {}).items() if "xyz" in v}; tg = objs.get(st.get("target")); dest = next((v for v in objs.values() if v.get("status") == "dest"), None)
        if st.get("insert"):
            E = np.array(st["insert"]["E"], float); p = np.array(tg["xyz"], float) if tg else E; return (E + p) / 2
        if stage_i == 3 and tg and dest: return (np.array(tg["xyz"], float) + np.array(dest["xyz"], float)) / 2
        if stage_i in (1, 2, 3) and tg: return np.array(tg["xyz"], float)
        if objs: return np.mean([np.array(v["xyz"], float) for v in objs.values()], axis=0)
        return np.array([-0.1, 0.0, 0.1])

    # ── 쥔 물체 부착 / 실행 경로 기록 / 판정 강조 ──
    def update_dynamic(self, st, stage_i, now):
        pl = st.get("plan") or {}; ppl = st.get("place_plan") or {}; oid = ppl.get("oid") or pl.get("oid") or st.get("target"); RT = self.lookup(FRAME, TOOL_FRAME)
        if stage_i != self.last_stage_i:                                      # 단계가 바뀌면 지금까지의 실행 경로는 페이드아웃 목록으로(계획 경로는 단계가 끝나는 즉시 사라진다)
            if self.last_stage_i in (2, 3) and len(self.trail) > 2: self.fading.append((self.trail, now))
            self.trail = []; self.last_stage_i = stage_i
        if stage_i in (2, 3) and RT is not None:
            p = RT[1].tolist()
            if not self.trail or np.linalg.norm(np.array(p) - np.array(self.trail[-1])) > 0.006: self.trail.append(p); self.trail = self.trail[-900:]
        if self.held is None:                                                 # 그리퍼가 닫혀(물체 폭에서 멈춤) 있으면 물체 마커를 gripper_flange 에 붙인다
            if stage_i in (2, 3) and RT is not None and 0.004 < self.grip < CFG["held_closed_max"] and oid in self.objpts and len(self.objpts[oid]) >= 6:
                R, t = RT; P = self.objpts[oid]; tri = self.hull_mesh(oid)
                self.get_logger().info(f"부착: {oid} -> {TOOL_FRAME} (grip {self.grip:.3f}, stage {stage_i})"); self.held = dict(oid=oid, pts0=P, pts=(P - t) @ R, tri=[(np.asarray(p) - t) @ R for p in tri] if tri else None)
        elif self.grip > CFG["held_open_min"] or stage_i in (0, 1):            # 열리면 놓인 자리(월드 좌표)로 되돌린다
            if RT is not None:
                R, t = RT; self.objpts[self.held["oid"]] = self.held["pts"] @ R.T + t; self.hull_cache.pop(self.held["oid"], None)
            self.get_logger().info(f"해제: {self.held['oid']} (grip {self.grip:.3f}, stage {stage_i})"); self.held = None
        if self.state is None: return                                          # 첫 상태를 읽기 전에는 판정 강조를 하지 않는다(시작 시 옛 항목으로 오발동 방지)
        pk = st.get("picked") or []                                           # 놓은 결과 판정: place 판정이 새로 붙은 항목이 생기면 놓인 위치에서 몇 초 강조
        done = {(e.get("id"), e.get("t")) for e in pk if e.get("place")}
        if self.seen_picked is None: self.seen_picked = done                 # 시작 시 이미 있던 판정은 강조하지 않는다
        elif done - self.seen_picked:
            key = sorted(done - self.seen_picked, key=lambda k: k[1] or 0)[-1]; e = next(x for x in pk if (x.get("id"), x.get("t")) == key)
            tl = [l for l in (st.get("log") or []) if "place verdict" in l]; mt = re.search(r"'tilt': ([0-9.]+)", tl[-1]) if tl else None
            Q = self.objpts.get(e.get("id")); ok = e.get("place") == "PLACED"; txt = f"{e['place']}" + (f"_tilt_{float(mt.group(1)):.1f}deg" if mt else "")
            self.get_logger().info(f"판정 강조: {txt}"); self.flash = dict(until=now + CFG["flash_s"], ok=ok, text=txt, tri=self.hull_of(Q) if Q is not None else None, pos=(Q.mean(0) + np.array([0, 0, 0.13])).tolist() if Q is not None else [0, 0, 0.3])
        self.seen_picked = (self.seen_picked or set()) & done | done if self.seen_picked is not None else None

    def fast_tick(self):
        t_in = time.time()
        try: self._fast_tick()
        except Exception as e: self.get_logger().warn(f"fast_tick 예외(무시하고 계속): {e!r}", throttle_duration_sec=5.0)
        p = self.perf; p['ft'] += time.time() - t_in; p['nft'] += 1
        if t_in - p['t0'] > 10.0:
            dt = t_in - p['t0']; self.get_logger().info(f"성능: 마커 {p['mk']/dt:.1f}Hz 카메라 {p['cam']/dt:.1f}Hz 패널 {p['panel']/dt:.1f}Hz | fast_tick 평균 {1000*p['ft']/max(p['nft'],1):.0f}ms 호출 {p['nft']/dt:.0f}Hz | tick 평균 {1000*p['tk']/max(p['ntk'],1):.0f}ms")
            self.perf = dict(t0=t_in, cam=0, panel=0, mk=0, ft=0.0, nft=0, tk=0.0, ntk=0)

    def _fast_tick(self):
        st = self.state or {}; stage_i = STAGE_IDX.get(st.get("stage"), 0); now = time.time()
        try: tgt = self.focus_target(st, stage_i)
        except Exception: tgt = self.focus
        self.focus = self.focus + 0.1 * (np.asarray(tgt, float) - self.focus)
        t = TransformStamped(); t.header.stamp = self.get_clock().now().to_msg(); t.header.frame_id = FRAME; t.child_frame_id = "nero_rt_focus"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = [float(v) for v in self.focus]; t.transform.rotation.w = 1.0; self.tfb.sendTransform(t)
        try: self.update_dynamic(st, stage_i, now)
        except Exception as e: self.get_logger().warn(f"동적 상태 갱신 실패 {e!r}", throttle_duration_sec=10.0)
        if now - self.t_cam > 1.0 / CFG["cam_hz"] and self.cam.get("rgb_raw") is not None:
            self.t_cam = now
            try:
                img = self.camera_image(st, stage_i)
                if img is not None:
                    stamp = self.get_clock().now().to_msg(); m = Image(); m.header.frame_id = CAM_FRAME; m.header.stamp = stamp; m.height, m.width = img.shape[:2]; m.encoding = "rgb8"; m.step = m.width * 3; m.data = _u8(img); self.cam_pub.publish(m); self.perf['cam'] += 1
                    ci = self.cam.get("info"); sc = CFG["cam_scale"]
                    if ci is not None:
                        ci2 = CameraInfo(); ci2.header.frame_id = CAM_FRAME; ci2.header.stamp = stamp; ci2.height, ci2.width = img.shape[:2]; ci2.distortion_model = ci.distortion_model; ci2.d = list(ci.d)
                        ci2.k = [v * sc for v in ci.k[:6]] + [0.0, 0.0, 1.0]; ci2.r = list(ci.r); ci2.p = [v * sc for v in ci.p[:8]] + list(ci.p[8:]); self.ci_pub.publish(ci2)
                    self.panel_img = img
            except Exception as e: self.get_logger().warn(f"카메라 영상 실패 {e!r}", throttle_duration_sec=10.0)
        if now - self.t_hud > 1.0 / CFG["hud_hz"]:
            self.t_hud = now; stale = (now - float(st.get("t", 0))) > self.ttl
            try:
                W, H = CFG["panel_w"], CFG["hud_h"]; key = (self.mt, int(now * 2), stale, stage_i)                    # HUD 는 내용이 바뀔 때만(상태 파일 갱신/1초 경과) 다시 그린다 -- 그리는 비용이 크다
                if key != getattr(self, "_hud_key", None) or getattr(self, "_hud", None) is None:
                    self._hud_key = key; self._hud = self.hud_image(st, stale, W, H)
                    m = Image(); m.header.frame_id = FRAME; m.height, m.width = self._hud.shape[:2]; m.encoding = "rgb8"; m.step = m.width * 3; m.data = _u8(self._hud); self.hud_pub.publish(m)
                cam = self.panel_img; ch = CFG["cam_h"]
                camp = np.zeros((ch, W, 3), np.uint8)
                if cam is not None:      # 종횡비 유지(레터박스): 패널이 가로로 길어져도 영상이 찌그러지지 않게
                    sc_ = min(W / cam.shape[1], ch / cam.shape[0]); rs_ = cv2.resize(cam, (max(1, int(cam.shape[1] * sc_)), max(1, int(cam.shape[0] * sc_))), interpolation=cv2.INTER_AREA); y0_, x0_ = (ch - rs_.shape[0]) // 2, (W - rs_.shape[1]) // 2; camp[y0_:y0_ + rs_.shape[0], x0_:x0_ + rs_.shape[1]] = rs_
                pan = np.vstack([camp, self._hud]); m = Image(); m.header.frame_id = FRAME; m.height, m.width = pan.shape[:2]; m.encoding = "rgb8"; m.step = m.width * 3; m.data = _u8(pan); self.panel_pub.publish(m); self.perf['panel'] += 1
            except Exception as e: self.get_logger().warn(f"HUD 실패 {e!r}", throttle_duration_sec=10.0)
        if self.cloud_on and now - self.t_cloud > 1.0 / CFG["cloud_hz"]:
            self.t_cloud = now
            try:
                pc = self.scene_cloud()
                if pc is not None: self.pc_pub.publish(pc)
            except Exception as e: self.get_logger().warn(f"점군 실패 {e!r}", throttle_duration_sec=10.0)

    def tick(self):
        t_in = time.time()
        try: self._tick()
        except Exception as e: self.get_logger().warn(f"tick 예외(무시하고 계속): {e!r}", throttle_duration_sec=5.0)
        self.perf['tk'] += time.time() - t_in; self.perf['ntk'] += 1

    def _tick(self):
        try:
            mt = os.path.getmtime(self.path)
            if mt != self.mt: self.state = json.load(open(self.path)); self.mt = mt
        except Exception:
            if self.state is None: return
        st = self.state or {}; now = time.time()
        if now - self.t_seg > 1.0 / (CFG["seg_hz_changed"] if self.mt != self.seg_mt else CFG["seg_hz_idle"]):
            self.t_seg = now; self.seg_mt = self.mt
            try: self.segment(st)
            except Exception as e: self.get_logger().warn(f"분할 점 실패 {e!r}", throttle_duration_sec=10.0)
        ma = MarkerArray(); cur = self.build(st); keys = {(m.ns, m.id) for m in cur}
        for ns, i in self.prev_keys - keys:                                    # 사라진 마커만 명시적으로 지운다(DELETEALL 은 깜빡임)
            d = Marker(); d.header.frame_id = FRAME; d.ns = ns; d.id = i; d.action = Marker.DELETE; ma.markers.append(d)
        ma.markers += cur; self.prev_keys = keys; self.pub.publish(ma); self.perf['mk'] += 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-cloud", action="store_true"); ap.add_argument("--cloud", action="store_true"); ap.add_argument("--hz", type=float, default=5.0); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--state", default=os.environ.get("NERO_RT_STATE", "~/grasp/rt_state.json")); ap.add_argument("--ttl", type=float, default=float(os.environ.get("NERO_RT_STATE_TTL", "6")))
    a = ap.parse_args(); rclpy.init(); n = Bridge(a); ex = MultiThreadedExecutor(num_threads=4); ex.add_node(n)
    try: ex.spin()
    except KeyboardInterrupt: pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
