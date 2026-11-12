"""shelf_cell.py -- (개발 사본, 2026-11-04; NERO_SHELF_CELL=1 일 때만 discover_object 가 사용) 선반 칸(kind='shelf_cell') 지원.
- --dests 명세의 shelf_cell 칸은 목적지 엔티티로 등록하되 **칸 부피를 속 찬 상자(other_<id>)로 장애물에 넣지 않는다**.
- 칸이 속한 선반의 판(바닥·옆·뒤[+천장 변형 시 천장])은 명세의 plates 를 그대로 장애물 box({id:'shelf_<이름>', xyz, raw, yaw})로 추가한다(dest_ids 밖 → 하강·후퇴 구간 충돌에도 남음).
- 판 id 가 'shelf_' 로 시작하면 place 요청에 slot_filter=False 를 넣어(서버 _slot_blocker 가 ≥8 mm 판을 슬롯 막힘으로 보지 않게; 서버 변경 없음) 슬롯 순서 재배치를 끈다.
기본 꺼짐: 환경변수가 없으면 이 모듈은 쓰이지 않는다(출력 동일)."""
import os


def enabled():
    return os.environ.get("NERO_SHELF_CELL", "0") == "1"


def apply_obstacles(objs, obstacles):
    """등록부(objs)와 기존 장애물 목록 -> shelf_cell 칸 상자를 빼고 판 box 를 (id 중복 없이) 추가한 새 목록."""
    cell_ids = {"other_" + oid for oid, o in objs.items() if o.get("shelf_cell")}
    out = [o for o in obstacles if o.get("id") not in cell_ids]
    seen = {o.get("id") for o in out}
    for oid, o in objs.items():
        if not o.get("shelf_cell"): continue
        for p in (o.get("plates") or []):
            if p["id"] in seen: continue
            out.append(dict(id=p["id"], xyz=[float(v) for v in p["xyz"]], raw=[float(v) for v in p["raw"]], yaw=float(p.get("yaw", 0.0)))); seen.add(p["id"])
    return out


def has_plates(obstacles):
    return any(str(o.get("id", "")).startswith("shelf_") for o in obstacles)
