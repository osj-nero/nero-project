#!/usr/bin/env python3
"""
segmentation_backend.py

[신설, 2026-08] Segmentation 단계 backend abstraction.

[2026-09-02 확장] 실제 segmentation backend 2종 추가:
  - DepthPlaneSegmentationBackend : 모델 0개. bbox frustum 안에서 "지배 평면
    (테이블) inlier가 아닌 점"만 남기는 depth 기반 세그. geometry_3d.py가
    어차피 RANSAC 평면을 구하므로 추가 비용 ~수 ms.
  - SamSegmentationBackend        : ultralytics MobileSAM / SAM2. bbox를 box
    prompt로, target_part 힌트를 point prompt로 받는다. Thor iGPU ~20-40ms.

기본 backend는 `SEG_BACKEND` 환경변수로 고른다(`noop`|`depth_plane`|`sam`).
지정 안 하면 기존과 동일하게 `noop`(하위호환) — 실기 검증 후 기본값을
`depth_plane`으로 올리는 것을 권장.

    export SEG_BACKEND=depth_plane
    export SEG_BACKEND=sam           # SAM_MODEL 로 가중치 지정(기본 mobile_sam.pt)
    export SAM_MODEL=mobile_sam.pt   # 또는 sam2.1_t.pt (약간 느리지만 정확)

향후 실제 segmentation을 붙이려면 SegmentationBackend를 구현한 새 클래스를
추가하고 `get_default_backend()`의 분기만 늘리면 된다 — 이 인터페이스를
쓰는 point_cloud.py/grasp_pose_generator.py/mcp_robot_server.py 쪽 코드는
변경 불필요(backend abstraction의 목적 자체가 이거다).
"""
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray           # bool 배열, 이미지와 동일 H,W. True=object 픽셀
    bbox_px: list               # [x1,y1,x2,y2], mask의 bounding box(디버깅/재사용용)
    source: str = "bbox"        # bbox(NoOp) | depth_plane | sam | yolo_seg | grounded_sam
    confidence: float = 1.0


class SegmentationBackend(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, bbox_px: list,
                target_part: Optional[str] = None,
                depth_image: Optional[np.ndarray] = None,
                point_hint_px: Optional[list] = None) -> Optional[SegmentationResult]:
        """image(HxWx3, BGR/RGB 무관 -- 마스크만 만들 뿐 색공간 안 씀) +
        bbox_px([x1,y1,x2,y2] 픽셀좌표)로 마스크를 만든다.

        target_part는 prompt 기반 backend(SAM 등)가 참고할 수 있는 힌트
        (예: "handle") -- NoOp/DepthPlane은 무시한다.
        depth_image는 DepthPlane backend가 필요로 한다(없으면 bbox 사각
        마스크로 폴백). point_hint_px([u,v])는 SAM point prompt용(파트
        추출 시 VLM이 찍은 점).
        실패(bbox 무효 등) 시 None."""
        raise NotImplementedError


def _clamp_bbox(bbox_px: list, w: int, h: int) -> Optional[list]:
    if not bbox_px or len(bbox_px) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class NoOpSegmentationBackend(SegmentationBackend):
    """YOLO bbox를 그대로 사각형 마스크로 반환. 실제 segmentation 없음."""

    def segment(self, image, bbox_px, target_part=None,
                depth_image=None, point_hint_px=None):
        if image is None:
            return None
        h, w = image.shape[:2]
        cb = _clamp_bbox(bbox_px, w, h)
        if cb is None:
            return None
        x1, y1, x2, y2 = cb
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return SegmentationResult(mask=mask, bbox_px=cb, source="bbox", confidence=1.0)


class DepthPlaneSegmentationBackend(SegmentationBackend):
    """모델 없이 depth만으로 물체 픽셀을 분리한다.

    아이디어(CLAUDE.md "major_axis_yaw_deg 신뢰 불가" 항목의 근본 원인 대응):
    NoOp가 bbox 사각형을 그대로 마스크로 써서 배경(테이블면) depth가 PCA에
    섞이는 게 문제였다. 여기서는 bbox 안에서
      1. 유효 depth 픽셀만 추림
      2. bbox 중앙부(안쪽 60%) depth의 median을 "물체 대표 깊이"로 잡고
      3. |depth - median| 이 band_m 이내인 연결 성분만 남긴다(테이블면은
         보통 물체보다 더 멀어서 제외됨)
    RANSAC 평면 적합까지는 안 하고(그건 geometry_3d.py 담당), depth 히스토그램
    기반의 싼 전경 분리만 한다. depth_image가 없으면 NoOp와 동일하게 폴백.

    band_m 기본 0.06 은 실측 튜닝값이 아니라 합리적 시작값 -- 실기 검증 후
    조정 대상(geometry_3d.py의 다른 threshold와 동일 성격).
    """

    def __init__(self, band_m: float = 0.06, min_valid_ratio: float = 0.15):
        self.band_m = band_m
        self.min_valid_ratio = min_valid_ratio

    def segment(self, image, bbox_px, target_part=None,
                depth_image=None, point_hint_px=None):
        if image is None:
            return None
        h, w = image.shape[:2]
        cb = _clamp_bbox(bbox_px, w, h)
        if cb is None:
            return None
        x1, y1, x2, y2 = cb

        if depth_image is None or depth_image.shape[:2] != (h, w):
            # depth 없으면 NoOp 폴백(조용히 실패하지 않게 source로 표시)
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return SegmentationResult(mask=mask, bbox_px=cb,
                                       source="bbox_nodepth", confidence=0.5)

        d = depth_image[y1:y2, x1:x2].astype(np.float64)
        valid = np.isfinite(d) & (d > 0.05) & (d < 3.0)
        if valid.sum() < max(20, self.min_valid_ratio * d.size):
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = valid
            return SegmentationResult(mask=mask, bbox_px=cb,
                                       source="depth_plane_lowvalid", confidence=0.4)

        # bbox 안쪽 60% 영역의 median 을 물체 대표 깊이로
        ih, iw = d.shape
        iy1, iy2 = int(ih * 0.2), int(ih * 0.8)
        ix1, ix2 = int(iw * 0.2), int(iw * 0.8)
        core = d[iy1:iy2, ix1:ix2]
        core_valid = core[np.isfinite(core) & (core > 0.05) & (core < 3.0)]
        if core_valid.size < 10:
            core_valid = d[valid]
        obj_depth = float(np.median(core_valid))

        near = valid & (np.abs(d - obj_depth) <= self.band_m)
        # 물체보다 명백히 가까운 픽셀(그리퍼 등)도 포함하되, 훨씬 먼 배경만 제외
        near = near | (valid & (d < obj_depth - self.band_m))

        full = np.zeros((h, w), dtype=bool)
        full[y1:y2, x1:x2] = near
        if full.sum() < 20:
            full[y1:y2, x1:x2] = valid  # 폴백
            return SegmentationResult(mask=full, bbox_px=cb,
                                       source="depth_plane_fallback", confidence=0.4)
        return SegmentationResult(mask=full, bbox_px=cb,
                                   source="depth_plane", confidence=0.8)


class SamSegmentationBackend(SegmentationBackend):
    """ultralytics MobileSAM / SAM2 로 픽셀 정밀 마스크.

    - bbox_px 를 box prompt 로 넘긴다.
    - point_hint_px([u,v]) 가 오면 point prompt(label=1)로 같이 넘긴다 --
      큰 물체 안의 특정 파트(손잡이 등)를 VLM이 점으로 찍어 보낸 경우.
    - 모델은 첫 호출 때 lazy load(무거운 import를 서버 기동 시 물지 않도록).
      로드 실패(패키지 없음/가중치 다운로드 실패)하면 None 대신 NoOp 사각
      마스크로 폴백해서 파이프라인을 죽이지 않는다.

    가중치: SAM_MODEL 환경변수(기본 mobile_sam.pt). 첫 실행 시 ultralytics가
    자동 다운로드하므로 인터넷 필요 -- 오프라인 배포는 yolo/weights/ 에 미리
    받아두고 절대경로 지정.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.environ.get("SAM_MODEL", "mobile_sam.pt")
        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False

    def _get_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from ultralytics import SAM
                self._model = SAM(self.model_path)
                print(f"[SamSegmentationBackend] loaded {self.model_path}", flush=True)
            except Exception as e:
                self._load_failed = True
                print(f"[SamSegmentationBackend] load 실패 → NoOp 폴백: "
                      f"{type(e).__name__}: {e}", flush=True)
            return self._model

    def segment(self, image, bbox_px, target_part=None,
                depth_image=None, point_hint_px=None):
        if image is None:
            return None
        h, w = image.shape[:2]
        cb = _clamp_bbox(bbox_px, w, h)
        if cb is None:
            return None

        model = self._get_model()
        if model is None:
            x1, y1, x2, y2 = cb
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return SegmentationResult(mask=mask, bbox_px=cb,
                                       source="bbox_sam_unavailable", confidence=0.5)

        kwargs = {"bboxes": [cb], "verbose": False}
        if point_hint_px is not None and len(point_hint_px) == 2:
            kwargs["points"] = [[int(point_hint_px[0]), int(point_hint_px[1])]]
            kwargs["labels"] = [1]

        try:
            results = model.predict(image, **kwargs)
        except Exception as e:
            print(f"[SamSegmentationBackend] predict 실패 → NoOp 폴백: {e}", flush=True)
            x1, y1, x2, y2 = cb
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return SegmentationResult(mask=mask, bbox_px=cb,
                                       source="bbox_sam_error", confidence=0.4)

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            x1, y1, x2, y2 = cb
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return SegmentationResult(mask=mask, bbox_px=cb,
                                       source="bbox_sam_empty", confidence=0.4)

        m = results[0].masks.data[0].cpu().numpy().astype(bool)
        if m.shape != (h, w):
            import cv2
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return SegmentationResult(mask=m, bbox_px=cb,
                                       source="sam_emptymask", confidence=0.3)
        mbbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return SegmentationResult(mask=m, bbox_px=mbbox, source="sam", confidence=0.9)


_DEFAULT_BACKEND: Optional[SegmentationBackend] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_backend() -> SegmentationBackend:
    """현재 기본 backend를 반환한다 -- 호출부는 이 함수를 통해서만 backend를
    얻어야 나중에 기본값 교체가 한 곳으로 끝난다.

    SEG_BACKEND 환경변수: noop(기본) | depth_plane | sam
    한 번 만든 인스턴스를 캐시한다(SAM 모델 재로딩 방지)."""
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is not None:
        return _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        if _DEFAULT_BACKEND is not None:
            return _DEFAULT_BACKEND
        kind = os.environ.get("SEG_BACKEND", "noop").strip().lower()
        if kind == "sam":
            _DEFAULT_BACKEND = SamSegmentationBackend()
        elif kind in ("depth_plane", "depthplane", "depth"):
            _DEFAULT_BACKEND = DepthPlaneSegmentationBackend()
        else:
            _DEFAULT_BACKEND = NoOpSegmentationBackend()
        print(f"[segmentation_backend] default = {type(_DEFAULT_BACKEND).__name__} "
              f"(SEG_BACKEND={kind!r})", flush=True)
        return _DEFAULT_BACKEND
