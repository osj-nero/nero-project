"""2026-11-10: 물체 자세를 시뮬레이터 정답 대신 인식 결과(덤프의 place_check.req.obj: centre, dims, yaw[rad])로 정한다.
면 이름의 기준 좌표(로컬 x/y)는 인식 상자에서 결정적으로 고른다:
 - 로컬 x = 인식된 수평 두 축 중 물체 형판(GRID_DIMS[asset][0]: 상자 7, 판 12, 막대 20cm)에 길이가 더 가까운 축
 - 앞뒤 부호 = 앞면(로컬 -y)이 로봇(원점) 쪽을 향하도록. 상자는 앞뒤가 대칭이라 기하로는 어느 쪽이 '원래 앞면'인지 알 수 없다 -> '로봇을 향한 면 = 앞면' 으로 정의
사진 이름표와 플래너 회전이 같은 이 좌표를 쓰므로 둘이 어긋나지 않는다. 반환 yaw 는 도(°)."""
import math, json
import numpy as np


def frame(obj, asset, grid_dims):
    c = [float(v) for v in obj["centre"]]; yw = float(obj.get("yaw", 0.0)); d = [float(v) for v in obj["dims"]]
    ax = [np.array([math.cos(yw), math.sin(yw)]), np.array([-math.sin(yw), math.cos(yw)])]      # 인식 축 0(dims[0]), 축 1(dims[1])
    t0 = float(grid_dims[asset][0])
    i = 0 if abs(d[0] - t0) <= abs(d[1] - t0) else 1
    x = ax[i]; y = np.array([-x[1], x[0]])      # 오른손 좌표(z 위)
    to_robot = -np.array(c[:2]); to_robot /= (np.linalg.norm(to_robot) + 1e-9)
    if float(-y @ to_robot) < 0: x, y = -x, -y      # 앞면(-y)이 로봇 쪽
    yaw_deg = math.degrees(math.atan2(x[1], x[0]))
    dims = [d[i], d[1 - i], float(obj.get("height") or d[2])]
    return dict(centre=c, yaw=yaw_deg, dims=dims)


def from_dump(path, asset, grid_dims):
    return frame(json.load(open(path))["place_check"]["req"]["obj"], asset, grid_dims)
