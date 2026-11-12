#!/usr/bin/env python3
"""
capture_multi_npz.py — PC. flange 카메라 한 프레임으로 "여러 물체를 동시에" 캡처.
capture_flange_npz.py(--sam 모드)의 다중물체 버전 -- RGB/depth는 1번만 찍고,
YOLO+SAM 서버(/detect)에 라벨 리스트를 한 번에 보내서(이미 다중라벨 지원됨,
vlm_boxyolo.py DetectRequest.labels: List[str]) 물체마다 다른 정수 ID로 seg에 채운다.

capture_flange_npz.py 와 다른 점(그것 말고는 로직 100% 재사용):
  - seg 값이 0/1 이 아니라 물체마다 다른 id (1,2,3,...)
  - box_id/bottle_id 대신 target_labels(문자열 배열, id 순서와 매칭) 저장
  - 물체 하나가 못 잡히면(마스크 없음/포인트 부족) 그 라벨만 스킵하고 나머지는 저장

사용:
  python3 capture_multi_npz.py --labels scissors,cup,banana --out ~/grasp/multi.npz
  python3 capture_multi_npz.py --labels box:2,banana --out ~/grasp/multi.npz   # box 2개까지
  (source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash 필요)

2026-09-17 추가: "라벨:개수" 문법으로 같은 라벨의 여러 인스턴스를 한 캡처에서 동시에
잡을 수 있다(예: 박스 2개). 다중 인스턴스면 target_labels 에 box_1/box_2 처럼 저장됨
(개수 1개짜리 라벨은 기존처럼 접미어 없음, 하위호환). scan_for_boxes 같은 "스캔해서
캐시"가 아니라 이 캡처 자체의 세그멘테이션에서 바로 계산된 실측 좌표(target_worlds)라,
CGN grasp 추론과 place 목표 좌표를 항상 같은 캡처에서 동시에 얻을 수 있다.
"""
import argparse, base64, io, json, os, sys, time, urllib.request
from thor_net import thor_ip
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, JointState
from tf2_ros import Buffer, TransformListener

SAM_URL = os.environ.get("SAM_SERVER_URL", f"http://{thor_ip()}:8002/detect")

BOTTLE_LABELS = {"bottle", "cup", "vase", "cylinder", "can", "wine glass", "banana",
                 "pot", "kettle", "mug"}


def _overlap_ratio(a, b):
    """두 bbox([x1,y1,x2,y2]) 의 교집합 / 더 작은 쪽 면적. 라벨이 달라도 같은 물체를
    가리키는(포함/거의겹침) 박스쌍을 잡아내기 위함 -- 표준 IoU 는 한쪽이 다른쪽 안에
    쏙 들어간 "중첩 박스"(예: 프라이팬 큰 박스 안에 그 일부가 또 다른 라벨로 잡힌 작은
    박스) 케이스에서 분모(합집합)가 커져 과소평가되므로 min-area 기준을 쓴다."""
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    return inter / min(area_a, area_b)


_DEDUPE_EXEMPT_LABELS = {"table"}  # 2026-09-24 추가, 사용자 실측 지적: 아래 docstring 참고


def multi_detect(rgb, label_counts, H, W, overlap_thresh=0.6, contain_thresh=0.7, area_ratio=1.4, container_max_area=0.25):
    """RGB + {라벨: 최대개수} -> YOLO@Thor(/detect, want_mask) 한 번 호출 ->
    {label: [(mask HxW bool, bbox_norm, confidence), ...]} (라벨당 최대 label_counts[label]개,
    confidence 내림차순).

    2026-09-17 추가 -- (1) 라벨 간(cross-label) 중복제거: Isaac Sim 합성 이미지에서 YOLO가
    불안정해서 같은 물체 하나가 서로 다른 라벨로 겹치는 박스 여러 개를 만들어내는 경우가
    실측 확인됨(프라이팬을 가위로 오검출 + 그 안의 일부를 또 다른 라벨(칫솔/스푼)로 검출).
    (2) 같은 라벨의 서로 다른 물체(예: box 2개)를 동시에 다루기 위한 다중 인스턴스 지원 --
    예전엔 라벨당 최고-confidence 1개만 남겨서 "박스 2개 중 하나가 그냥 버려짐" 문제가
    있었음(사용자 지적). 전역 confidence 내림차순으로 훑으면서, 이미 채택된 박스와
    overlap_ratio > overlap_thresh 면 같은 물체로 보고 버리고, 그 라벨의 채택 개수가
    label_counts[label]에 닿으면 그 이상은 안 받는다.

    2026-09-24 수정(사용자 지적, 실측 확인): 이 cross-label 중복제거가 "가위=프라이팬"
    같은 실제 오검출(작은 물체끼리 시각적으로 혼동)을 잡으려던 건데, "박스가 테이블
    위/근처에 있어 2D 화면상 겹쳐 보이는" 완전히 다른 두 실물체의 정상적인 겹침까지
    같은 물체로 오판했다 -- 본실험(N=10)에서 discovery 실패 4건 중 3건이 정확히
    "table이 box와 겹침 -- 스킵"으로 테이블이 매번 지워진 게 원인이었음(실측 로그
    확인). _DEDUPE_EXEMPT_LABELS(테이블 등 "다른 물체가 정상적으로 근처/위에 있을 수
    있는" 큰 표면/리셉터클 라벨)에 속하면 겹침 여부와 무관하게 절대 이 dedup으로
    제거되지 않는다(다른 라벨을 dedup으로 지우는 것도 안 함 -- 양방향 면제)."""
    import cv2
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    labels = list(label_counts.keys())
    print(f"[capture-multi] SAM 요청: jpg {len(buf)//1024}KB, labels={labels} -> {SAM_URL}", file=sys.stderr)
    body = json.dumps({"image_b64": b64, "labels": labels, "want_mask": True}).encode()
    # 2026-09-18 (사용자 지시): 라벨에 open-vocab(YOLO-World) 대상(예: "table")이 섞이면
    # 서버(vlm_boxyolo.py)가 간헐적으로 500을 낸다(실측 재현 -- 같은 각도에서도 매번은
    # 아님, world_model 공유상태 관련 레이스로 추정, 근본원인 미확정). 이 한 스텝만 놓치고
    # 스윕 전체가 죽는 걸 막기 위해 1회 재시도 후에도 실패하면 빈 결과로 계속 진행한다.
    for _attempt in range(2):
        req = urllib.request.Request(SAM_URL, data=body, headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=90).read())
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            print(f"[capture-multi] /detect HTTP {e.code} (시도 {_attempt+1}/2): {detail}", file=sys.stderr)
            if _attempt == 1:
                return {}
            time.sleep(0.5)

    cand = [d for d in r["detections"] if d.get("mask_b64") and d["label"] in label_counts]
    cand.sort(key=lambda d: -d["confidence"])

    # 포함 억제(2026-09-29, 사용자 지시 "머스타드 병 아래쪽을 box로 오탐" 일반규칙화): 작은 bbox가
    # 더 큰 다른 검출 bbox 안에 거의 통째로 들어 있으면(포함비 >= contain_thresh, 면적비 <=
    # 1/area_ratio, 컨테이너는 물체 크기 bbox만) 큰 물체의 일부(하단/부분크롭)를 다른 라벨로 오탐한 것으로 보고 작은 쪽을
    # confidence와 무관하게 버린다. 아래 conf순 dedupe는 오탐 conf가 더 높으면 진짜 큰 물체를
    # 지워버렸다(실측: box 0.89가 mustard bottle 0.52를 스킵시킴). 면제 라벨(table)은 제외.
    def _bb_area(bb):
        return max(1e-9, (bb[2] - bb[0]) * (bb[3] - bb[1]))
    _bbs = [[d["x_min"], d["y_min"], d["x_max"], d["y_max"]] for d in cand]
    if os.environ.get("NERO_DEBUG_DET"):
        for d, bb in zip(cand, _bbs):
            print(f"[capture-multi][det] {d['label']} conf={d['confidence']:.2f} bbox={[round(float(x),3) for x in bb]} area={_bb_area(bb):.4f}", file=sys.stderr)
    _keep = []
    for i, d in enumerate(cand):
        drop_by = None
        if d["label"] not in _DEDUPE_EXEMPT_LABELS:
            for j, e in enumerate(cand):
                # 화면 25% 초과 bbox는 물체가 아니라 표면/배경(테이블 크기 box 오탐 등)이라 컨테이너 자격 없음
                if i == j or e["label"] in _DEDUPE_EXEMPT_LABELS or _bb_area(_bbs[j]) > container_max_area:
                    continue
                if (_bb_area(_bbs[j]) >= area_ratio * _bb_area(_bbs[i])
                        and _overlap_ratio(_bbs[i], _bbs[j]) >= contain_thresh):
                    drop_by = e; break
        if drop_by is not None:
            print(f"[capture-multi] {d['label']}(conf={d['confidence']:.2f})는 더 큰 "
                  f"{drop_by['label']}(conf={drop_by['confidence']:.2f}) bbox 안에 포함 -- "
                  f"부분 오탐으로 보고 스킵", file=sys.stderr)
        else:
            _keep.append(d)
    cand = _keep

    accepted = []           # [{label,bbox,confidence}] -- 라벨 무관 전체 (겹침판정용)
    by_label = {}            # {label: [(mask,bbox,conf), ...]}
    for d in cand:
        lb = d["label"]
        bbox = [d["x_min"], d["y_min"], d["x_max"], d["y_max"]]
        dup = None
        if lb not in _DEDUPE_EXEMPT_LABELS:
            dup = next((a for a in accepted
                        if a["label"] not in _DEDUPE_EXEMPT_LABELS
                        and _overlap_ratio(bbox, a["bbox"]) > overlap_thresh), None)
        if dup is not None:
            print(f"[capture-multi] {lb}(conf={d['confidence']:.2f})가 "
                  f"{dup['label']}(conf={dup['confidence']:.2f})와 겹침(같은 물체로 판단) -- 스킵",
                  file=sys.stderr)
            continue
        if len(by_label.get(lb, [])) >= label_counts[lb]:
            continue  # 이 라벨은 이미 원하는 개수만큼 다 찼음
        m = cv2.imdecode(np.frombuffer(base64.b64decode(d["mask_b64"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        accepted.append({"label": lb, "bbox": bbox, "confidence": d["confidence"]})
        by_label.setdefault(lb, []).append((m > 127, bbox, d["confidence"]))
    return by_label


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="쉼표구분 라벨 리스트, 예: scissors,cup,banana")
    ap.add_argument("--out", default="/tmp/cgn_multi.npz")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--floor-z", type=float, default=0.012)
    a = ap.parse_args()
    # "box:2,cup" 처럼 "라벨:개수" 지원 (2026-09-17 추가) -- 같은 라벨 여러 인스턴스를
    # 한 캡처에서 동시에(예: 박스 2개) 다루기 위함. 개수 생략 시 1.
    labels, label_counts = [], {}
    for tok in a.labels.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            lb, cnt = tok.split(":", 1)
            lb, cnt = lb.strip(), int(cnt.strip())
        else:
            lb, cnt = tok, 1
        labels.append(lb)
        label_counts[lb] = max(label_counts.get(lb, 0), cnt)
    if not labels:
        print("[capture-multi] --labels 비어있음", file=sys.stderr); sys.exit(1)

    rclpy.init()
    n = Node("capture_multi_npz")
    buf = Buffer(); TransformListener(buf, n)
    got = {}

    def on_info(m):
        got["K"] = np.array(m.k).reshape(3, 3)

    def on_depth(m):
        d = np.frombuffer(m.data, dtype=np.float32 if m.encoding == "32FC1" else np.uint16)
        d = d.reshape(m.height, m.width).astype(np.float32)
        if m.encoding == "16UC1":
            d = d / 1000.0
        got["depth"] = d

    def on_rgb(m):
        arr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
        got["rgb"] = arr[..., ::-1].copy() if m.encoding.startswith("bgr") else arr.copy()

    ARM_J = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    def on_js(m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM_J):
            got["arm_q"] = [float(d[j]) for j in ARM_J]

    n.create_subscription(CameraInfo, "/camera/camera_info", on_info, qos_profile_sensor_data)
    n.create_subscription(Image, "/camera/depth/image_raw", on_depth, qos_profile_sensor_data)
    n.create_subscription(Image, "/camera/color/image_raw", on_rgb, qos_profile_sensor_data)
    n.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)

    t0 = time.time()
    pre = ("K", "depth", "rgb")
    while time.time() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
        if all(k in got for k in pre):
            try:
                got["tf"] = buf.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time())
                break
            except Exception:
                pass
    for k in pre + ("tf",):
        if k not in got:
            print(f"[capture-multi] 못 받음: {k}", file=sys.stderr); rclpy.shutdown(); sys.exit(1)
    # 2026-09-17 버그이력: 15회x0.1s(1.5초)로는 가끔 arm_q 를 못 받아서(간헐적, 원인 미확인)
    # npz 에 arm_q 가 아예 빠진 채로 저장됐고, 그러면 다운스트림(B단계 NERO_ARM_Q 파싱)에서
    # 훨씬 헷갈리는 에러로 몇 단계 뒤에야 터졌다. 재시도 창을 5초로 늘리고, 그래도 안
    # 들어오면 여기서 바로(캡처 단계에서) 명확하게 실패시킨다 -- downstream 이 조용히
    # "기본 seed로 폴백"하는 것보다, 뭐가 문제인지 바로 아는 게 낫다.
    for _ in range(50):
        if "arm_q" in got:
            break
        rclpy.spin_once(n, timeout_sec=0.1)
    if "arm_q" not in got:
        print("[capture-multi] arm_q(/joint_states) 5초 넘게 못 받음 -- 중단 "
              "(NERO_ARM_Q 없이 진행하면 B단계가 잘못된 seed로 계획할 수 있음)",
              file=sys.stderr)
        rclpy.shutdown(); sys.exit(1)

    K3 = got["K"]; fx, fy, cx, cy = K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]
    depth = got["depth"]; H, W = depth.shape

    tr = got["tf"].transform
    R = quat_to_R(tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w)
    base_T_opt = np.eye(4)
    base_T_opt[:3, :3] = R
    base_T_opt[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    W_T_cam = base_T_opt @ flip

    zf = depth.copy(); zf[~np.isfinite(zf)] = 0.0
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Xc = (uu - cx) / fx * zf
    Yc = (vv - cy) / fy * zf
    pts_opt = np.stack([Xc, Yc, zf, np.ones_like(zf)], -1)
    pts_base = pts_opt @ base_T_opt.T

    # === 캡처 1번, 검출 1번(다중라벨 + 라벨당 다중인스턴스) ===
    by_label = multi_detect(got["rgb"], label_counts, H, W)

    seg = np.zeros((H, W), dtype=np.int32)
    target_labels, target_ids, target_worlds, target_is_bottle = [], [], [], []
    next_id = 1
    for lb in labels:
        insts = by_label.get(lb, [])
        if not insts:
            print(f"[capture-multi] {lb}: 마스크 없음 -- 스킵", file=sys.stderr)
            continue
        multi = label_counts[lb] > 1
        for i, (smask, sbbox, conf) in enumerate(insts):
            out_lb = f"{lb}_{i+1}" if multi else lb
            m = smask & (zf > 0.1) & (zf < 2.0) & (pts_base[..., 2] > a.floor_z)
            npx = int(m.sum())
            if npx < 50:
                print(f"[capture-multi] {out_lb}: seg 픽셀 {npx} -- 스킵", file=sys.stderr)
                continue
            seg[m] = next_id
            target_labels.append(out_lb)
            target_ids.append(next_id)
            target_worlds.append(np.median(pts_base[m][:, :3], axis=0))
            target_is_bottle.append(lb in BOTTLE_LABELS)
            print(f"[capture-multi] {out_lb}: id={next_id} seg_px={npx} conf={conf:.2f}", file=sys.stderr)
            next_id += 1

    if not target_labels:
        print("[capture-multi] 검출된 물체 0개", file=sys.stderr); rclpy.shutdown(); sys.exit(1)

    extra = {}
    if "arm_q" in got:
        extra["arm_q"] = np.array(got["arm_q"], dtype=np.float64)
    np.savez(a.out, depth_m=zf.astype(np.float32), seg=seg,
              K=np.array([fx, fy, cx, cy], dtype=np.float64),
              W_T_cam=W_T_cam.astype(np.float64),
              target_labels=np.array(target_labels),
              target_ids=np.array(target_ids, dtype=np.int32),
              target_worlds=np.array(target_worlds, dtype=np.float64),
              target_is_bottle=np.array(target_is_bottle, dtype=bool),
              **extra)

    rgb_png = a.out[:-4] + ".rgb.png" if a.out.endswith(".npz") else a.out + ".rgb.png"
    try:
        import cv2
        cv2.imwrite(rgb_png, np.ascontiguousarray(got["rgb"][..., :3][..., ::-1]))
    except Exception as e:
        print(f"[capture-multi] RGB 저장 실패: {e}", file=sys.stderr); rgb_png = None

    # 다중인스턴스면 target_labels 는 "box_1"/"box_2" 형태라, base 라벨 기준 스킵 여부는
    # 그 접두어를 가진 인스턴스가 하나라도 있는지로 판단.
    def _base_ok(lb):
        pfx = lb + "_"
        return lb in target_labels or any(t.startswith(pfx) for t in target_labels)
    print(json.dumps({"out": a.out, "rgb_png": rgb_png, "labels_ok": target_labels,
                      "labels_skipped": [l for l in labels if not _base_ok(l)],
                      "K": [round(float(v), 1) for v in (fx, fy, cx, cy)]}, indent=1))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
