#!/usr/bin/env python3
"""
pick_verification.py

[신설 — 폐루프(closed-loop) pick 검증]

place_object는 이미 verify_placement(placement_verification.py)로 "그리퍼가
열린 자리에 물체가 실제로 있는지"를 확인한다. 반면 pick_object는 지금까지
완전히 열린 루프였다 — approach->descend->그리퍼 닫기->lift를 마치면 그걸로
"성공"이라고 보고했다. 실제로 그리퍼가 물체를 물었는지는 아무도 확인하지
않았다 — 실측으로 "success"인데 그리퍼가 허공에 닫혀서 물체가 원래
자리에 그대로 남아있는 사례가 여러 번 재현됐다(2026-08-31).

이 모듈은 place와 반대 방향의 검사를 한다: "목표 위치에 새로 뭔가
생겼나"가 아니라 "원래 있던 물체가 그 자리에서 사라졌나"를 본다. ROS
의존성 없는 순수 로직만 담당(placement_verification.py와 동일한 모듈
경계 원칙) — pytest로 검증 가능.

## 비대칭성 참고
place 검증은 z_tol을 관대하게 잡아야 했다(인식 z 노이즈 최대 26mm).
pick 검증은 "있다/없다"만 보면 되므로 원리적으로 place보다 판정이
단순하다 — 다만 "물체가 사라짐"이 "성공적으로 집힘"과 동의어가 아니라는
한계가 있다(예: 그리퍼가 밀쳐서 카메라 시야 밖으로 날아간 경우도
"사라짐"으로 오판할 수 있음). 이 모듈은 "1차 신호"만 제공하고, 그
신호가 애매하거나 "여전히 있음"으로 나오면 호출부(mcp_robot_server.py)가
VLM(analyze_scene)으로 2차 확인을 하는 게 설계 의도다.
"""
import math


# 물체 중심(x,y)이 원래 위치에서 이 거리 이내면 "아직 거기 있다"로 간주.
DEFAULT_XY_TOL_M = 0.035

# pick 후 z 판정은 원래 물체 위치 근처인지만 보면 되고, place처럼 "정확한
# 목표 높이"에 안착했는지를 볼 필요가 없어서 place보다 더 관대하게 잡는다.
DEFAULT_Z_TOL_M = 0.05

# perception 데이터가 이 시간(초)보다 오래됐으면 "신뢰 불가"로 간주.
MAX_STALE_SEC = 2.0


def verify_pick(original_pos: dict, detected_objects: list, label: str = None,
                 xy_tol: float = DEFAULT_XY_TOL_M, z_tol: float = DEFAULT_Z_TOL_M,
                 obj_age_sec: float = None) -> dict:
    """pick 시도 전 물체가 있던 위치(original_pos) 근처에서, pick 후에도
    같은 라벨의 물체가 여전히 감지되는지 확인한다.

    Args:
        original_pos: {'x','y','z'} pick을 시도했던 물체의 (pick 전) 위치.
        detected_objects: pick(lift 포함) 후 perception이 보고한 물체 목록.
        label: 지정하면 이 라벨과 일치하는 물체만 "같은 물체"로 본다.
            None이면 라벨 무관하게 근접한 아무 물체나 매칭(비권장 -- 가능하면
            항상 넘길 것, place와 달리 pick은 라벨을 이미 알고 시작하므로).
        xy_tol, z_tol: 허용 오차(미터) -- 이 범위 안에 뭔가 있으면 "그대로
            있다"로 판정.
        obj_age_sec: detected_objects 관측 후 경과 시간(초). None이면 신선도
            검사 생략.

    Returns:
        {
          'verified': True|False|None,
            # True  = 원래 자리에서 물체가 사라짐 (집혔을 가능성 높음)
            # False = 원래 자리에 물체가 여전히 감지됨 (집기 실패 가능성 높음
            #         -- 허공을 물었거나, 밀치기만 하고 못 집었을 수 있음)
            # None  = 판정 불가 (perception 데이터 오래됨/없음)
          'reason': str,
          'matched_object': dict|None,  # verified=False일 때 매칭된(여전히 거기 있는) 물체
        }
    """
    if obj_age_sec is not None and obj_age_sec > MAX_STALE_SEC:
        return {
            'verified': None,
            'reason': f'perception 데이터가 {obj_age_sec:.1f}초 전 값이라 '
                      f'신뢰할 수 없음 (최대 {MAX_STALE_SEC}초). 재확인 필요.',
            'matched_object': None,
        }

    # placement_verification.verify_placement와 동일한 원칙: 이번 프레임에
    # 장면 전체가 텅 비었으면 "물체가 없다"보다 "카메라가 지금 뭘 봐도
    # 못 잡고 있다"는 인식 사각지대 신호로 보는 게 타당하다.
    if not detected_objects:
        return {
            'verified': None,
            'reason': '이 시점에 장면에서 물체가 하나도 감지되지 않음 -- '
                      '인식 사각지대일 수 있음. 물체가 사라졌다고 단정하지 말 것.',
            'matched_object': None,
        }

    ox0, oy0, oz0 = (original_pos.get('x', 0.0), original_pos.get('y', 0.0),
                     original_pos.get('z', 0.0))

    candidates = detected_objects
    if label:
        candidates = [o for o in candidates if o.get('label') == label]

    best = None
    best_xy_err = None
    for obj in candidates:
        c = obj.get('center_3d', {})
        ox, oy = c.get('x'), c.get('y')
        if ox is None or oy is None:
            continue
        xy_err = math.hypot(ox - ox0, oy - oy0)
        if xy_err > xy_tol:
            continue
        oz = c.get('z')
        if oz is not None and abs(oz - oz0) > z_tol:
            continue
        if best is None or xy_err < best_xy_err:
            best, best_xy_err = obj, xy_err

    if best is None:
        return {
            'verified': True,
            'reason': f'원래 위치({ox0:.3f},{oy0:.3f},{oz0:.3f}) 근처에서 '
                      f'더 이상 감지되지 않음 -- 집혔을 가능성 높음.',
            'matched_object': None,
        }

    return {
        'verified': False,
        'reason': f'원래 위치 근처(xy 오차 {best_xy_err*1000:.0f}mm)에서 '
                  f'여전히 감지됨 -- 못 집었거나 밀치기만 했을 가능성.',
        'matched_object': best,
    }
