#!/usr/bin/env python3
"""
placement_verification.py

[신설 — 폐루프(closed-loop) place 검증]

지금까지 place_object는 완전히 열린 루프였다: 로봇이 approach->descend->
그리퍼 열기->lift를 마치면 그걸로 "성공"이라고 보고했다. 실제로 물체가
의도한 위치에 안착했는지는 아무도 확인하지 않았다 — 그리퍼가 열리는
순간 물체가 미끄러지거나, 다른 물체와 부딪혀 튕겨나가거나, 살짝 어긋난
자세로 놓여도 다음 명령(예: 그 위에 쌓기)은 "성공"했다는 잘못된 전제
위에서 그대로 진행된다.

이 모듈은 place 직후 perception이 관측한 물체 목록을 받아서, 기대한
위치 근처에 실제로 물체가 있는지 확인하는 순수 로직만 담당한다.
ROS 의존성이 없어서 pytest로 검증 가능하다 (perception_node 없이
가짜 detected_objects 리스트만 넣어서 테스트).

책임 밖: 카메라 재조회 시점 결정, settle delay, ROS 토픽 구독 —
이건 planning_node.py가 담당한다.

## 알려진 한계
같은 높이(5cm) 박스 3개의 인식 z가 26mm까지 벌어졌던 사례가 있다
(z 좌표 인식 노이즈, outlier 미필터링 평균 문제 — 아직 코드 수정 안 됨,
다음 과제로 남아있는 항목). 그래서 z_tol을 xy_tol보다 훨씬 관대하게
잡는다. z 노이즈가 median 기반 필터링으로 해결되면 z_tol을 좁혀야 한다.

## 쌓기(stacking) 시 오탐 방지 -- before/after 델타 비교 (2026-08 추가)
초기 버전은 "목표 좌표 근처에 뭔가 있나?"만 봤는데, 이러면 심각한
맹점이 있다: B를 A 위에 쌓았는데 B가 넘어지거나 미끄러져도, 애초에
그 자리에 있던 A(베이스 박스)를 "확인됨"으로 잘못 판정할 수 있다.
쌓기는 목표 z가 베이스보다 딱 박스 한 개 높이(5cm)만 위인데, z_tol
(4cm)이 이보다 작긴 해도 26mm 노이즈를 감안하면 여유가 얇다.

이를 막기 위해 place 시작 전 장면(before_objects)을 스냅샷해두고,
place 후 장면(after_objects)에서 "before에 이미 있던 것과 같은
물체"(dedup_xy_tol=2cm, dedup_z_tol=2cm 이내)는 후보에서 제외한다.
베이스 박스는 항상 필터링되므로, 남는 후보는 이번에 새로 나타난
물체뿐이다. B가 넘어져서 안 보이거나 A와 겹쳐서 하나의 블롭으로
합쳐지면, 필터링 후 후보가 아예 없어져 정확히 verified=False로
판정된다.

**여전히 남는 한계**: 정육면체(5cm×5cm×5cm에 가까운)가 "제자리에서
옆으로 넘어진" 경우 -- 위치(xy)도 거의 그대로고 중심 높이(z)도 정육면체
특성상 세워졌을 때와 크게 다르지 않을 수 있어, 이 케이스는 여전히
false positive 가능성이 있다. 이건 center_3d(중심점)만으로는 원리적으로
구분 불가능하고, 물체의 자세(orientation)까지 봐야 잡을 수 있는
문제라 이 모듈의 책임 범위를 넘어선다 (perception이 orientation을
안정적으로 주기 시작하면 다음 개선 대상).
"""
import math


# 물체 중심(x,y)이 목표 위치에서 이 거리 이내면 "같은 물체"로 간주.
# 박스 한 변(약 5cm)의 절반보다 살짝 넉넉하게.
DEFAULT_XY_TOL_M = 0.035

# z는 인식 노이즈가 커서(최대 26mm 관측) 관대하게 잡는다.
# median 필터링 적용 전까지는 이 값을 좁히지 말 것.
DEFAULT_Z_TOL_M = 0.04

# perception 데이터가 이 시간(초)보다 오래됐으면 "신뢰 불가"로 간주하고
# verified=None(판정 불가)을 반환한다 — False(실패)로 단정하지 않는다.
MAX_STALE_SEC = 2.0

# before/after 델타 비교용: 이 거리 이내면 "이미 있던 그 물체"로 간주해
# 후보에서 제외한다. 배치 판정용 xy_tol/z_tol보다 훨씬 타이트해야
# 한다 -- 여기 목적은 "같은 물체 재감지"를 잡는 것이지 "근처의 다른
# 물체"까지 걸러내면 안 되기 때문.
DEDUP_XY_TOL_M = 0.02
DEDUP_Z_TOL_M = 0.02


def _already_existed(obj: dict, before_objects: list,
                      xy_tol: float = DEDUP_XY_TOL_M,
                      z_tol: float = DEDUP_Z_TOL_M) -> bool:
    """obj가 before_objects(place 시작 전 스냅샷) 중 하나와 사실상 같은
    물체인지 판정한다 (같은 물체 재감지 vs 새로 나타난 물체 구분용)."""
    c = obj.get('center_3d', {})
    ox, oy, oz = c.get('x'), c.get('y'), c.get('z')
    if ox is None or oy is None:
        return False
    for prev in before_objects:
        pc = prev.get('center_3d', {})
        px, py, pz = pc.get('x'), pc.get('y'), pc.get('z')
        if px is None or py is None:
            continue
        if math.hypot(ox - px, oy - py) <= xy_tol:
            if oz is None or pz is None or abs(oz - pz) <= z_tol:
                return True
    return False


def verify_placement(expected_pos: dict, detected_objects: list,
                      before_objects: list = None, label: str = None,
                      xy_tol: float = DEFAULT_XY_TOL_M,
                      z_tol: float = DEFAULT_Z_TOL_M,
                      obj_age_sec: float = None) -> dict:
    """place 목표 위치 근처에 실제로 (새로) 물체가 감지됐는지 확인한다.

    Args:
        expected_pos: {'x','y','z'} place_object에 요청했던(또는 자동
            보정된 실제) 목표 좌표.
        detected_objects: place 후(after) perception이 보고한 물체 목록.
        before_objects: place 시작 전(before) 스냅샷. 주어지면, after
            목록에서 before에 이미 있던 것과 같은 물체(모듈 상단
            "쌓기 시 오탐 방지" 참고)는 후보에서 제외한다. 생략하면
            이 필터링 없이 단순 근접 매칭만 한다(하위호환, 쌓기
            시나리오에서는 오탐 위험이 있으니 가급적 항상 넘길 것).
        label: 지정하면 이 라벨과 일치하는 물체만 후보로 본다. None이면
            라벨 무관하게 가장 가까운 물체로 판단 (place 시점엔 perception
            라벨이 안정적이지 않을 수 있어 기본값은 라벨 무시 쪽을 권장).
        xy_tol, z_tol: 허용 오차(미터).
        obj_age_sec: detected_objects가 관측된 시점으로부터 경과한 시간
            (초). None이면 신선도 검사를 생략한다.

    Returns:
        {
          'verified': True|False|None,   # None = 판정 불가(데이터 오래됨/없음)
          'reason': str,                  # 사람이 읽을 설명
          'matched_object': dict|None,    # 매칭된 물체 (있으면)
          'xy_error_m': float|None,
          'z_error_m': float|None,
        }
    """
    if obj_age_sec is not None and obj_age_sec > MAX_STALE_SEC:
        return {
            'verified': None,
            'reason': f'perception 데이터가 {obj_age_sec:.1f}초 전 값이라 '
                      f'신뢰할 수 없음 (최대 {MAX_STALE_SEC}초). 재확인 필요.',
            'matched_object': None, 'xy_error_m': None, 'z_error_m': None,
        }

    # [신설, 2026-08] 근접 top-down 각도에서는 카메라가 뭘 보든 잘 못 잡는
    # "인식 사각지대"가 실측 확인됨 (박스 바로 위, 가까운 거리에서 detection
    # 자체가 아예 안 뜨는 경우). 이걸 "물체가 없다"(False)로 단정하면 억울한
    # 오탐이 쌓인다. 판별 신호: 이번 프레임에 장면 전체에서 물체가 하나도
    # 감지되지 않았다면, 이건 "이 박스만 없다"보다 "지금 카메라가 뭘 봐도
    # 잘 못 잡고 있다"는 신호로 보는 게 더 타당하다. 반대로 다른 물체는
    # 잘 보이는데 이번에 놓은 것만 안 보이면, 그건 훨씬 신뢰할 수 있는
    # 진짜 실패 신호다.
    if not detected_objects:
        return {
            'verified': None,
            'reason': '이 시점에 장면에서 물체가 하나도 감지되지 않음 -- '
                      '카메라 각도/거리 문제로 판정 자체가 불가능할 수 있음 '
                      '(예: 박스 바로 위 근접 top-down 각도는 인식 사각지대로 '
                      '알려짐). 물체가 실제로 없다고 단정하지 말고, list_detected_objects'
                      '로 다른 각도에서 재확인하는 걸 권장.',
            'matched_object': None, 'xy_error_m': None, 'z_error_m': None,
        }

    ex, ey, ez = expected_pos.get('x', 0.0), expected_pos.get('y', 0.0), expected_pos.get('z', 0.0)

    candidates = detected_objects
    if label:
        candidates = [o for o in candidates if o.get('label') == label]

    excluded_count = 0
    if before_objects:
        _filtered = []
        for o in candidates:
            if _already_existed(o, before_objects):
                excluded_count += 1
            else:
                _filtered.append(o)
        candidates = _filtered

    best = None
    best_xy_err = None
    for obj in candidates:
        c = obj.get('center_3d', {})
        ox, oy = c.get('x'), c.get('y')
        if ox is None or oy is None:
            continue
        xy_err = math.hypot(ox - ex, oy - ey)
        if best is None or xy_err < best_xy_err:
            best, best_xy_err = obj, xy_err

    if best is None:
        label_desc = f" (라벨='{label}')" if label else ""
        excl_note = (f' (기존에 있던 물체 {excluded_count}개는 "새 물체"'
                     f' 후보에서 제외함 -- 베이스 박스 재감지 가능성 배제)'
                     if excluded_count else '')
        return {
            'verified': False,
            'reason': f'목표 위치({ex:.3f},{ey:.3f},{ez:.3f}) 근처에서'
                      f'{label_desc} 새로 나타난 물체를 감지하지 못함{excl_note}.'
                      f' 놓친 채 떨어졌거나 미끄러졌을 가능성.',
            'matched_object': None, 'xy_error_m': None, 'z_error_m': None,
        }

    oz = best.get('center_3d', {}).get('z', ez)
    z_err = abs(oz - ez)

    if best_xy_err > xy_tol:
        return {
            'verified': False,
            'reason': f'가장 가까운 물체가 목표에서 {best_xy_err*1000:.0f}mm '
                      f'떨어져 있음 (허용 {xy_tol*1000:.0f}mm). 놓는 과정에서'
                      f' 밀렸거나 튕겨났을 가능성.',
            'matched_object': best, 'xy_error_m': best_xy_err, 'z_error_m': z_err,
        }

    if z_err > z_tol:
        return {
            'verified': False,
            'reason': f'물체는 목표 위치 근처(xy 오차 {best_xy_err*1000:.0f}mm)에'
                      f' 있으나 z가 {z_err*1000:.0f}mm 어긋남 (허용 '
                      f'{z_tol*1000:.0f}mm). 비스듬히 놓였거나 다른 물체'
                      f' 위에 걸쳤을 가능성.',
            'matched_object': best, 'xy_error_m': best_xy_err, 'z_error_m': z_err,
        }

    return {
        'verified': True,
        'reason': f'목표 위치 확인됨 (xy 오차 {best_xy_err*1000:.0f}mm, '
                  f'z 오차 {z_err*1000:.0f}mm, 둘 다 허용 범위 내).',
        'matched_object': best, 'xy_error_m': best_xy_err, 'z_error_m': z_err,
    }
