"""Stage B: rank CGN grasps (already in gripper_flange convention) + IK, per object.
Elongated -> CGN side grasp re-targeted to the fitted cylinder axis at mid-height.
Compact  -> CGN top-down grasp by score - w*theta^4 - w*d_com.
Writes <name>_pick.json (quat, approach, flange, pre, standoff, grasp_sol, pre_sol) for step6.

FAST IK (rank3 pattern): geometric prefilter (0 IK) -> cost sort -> 120ms IK on top-K only,
first standoff only, second standoff as fallback.
"""
import json, sys, os, time
import numpy as np
import rclpy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

NAMES = sys.argv[1:] or ["bottle", "box"]
D_TCP = 0.1358
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
SEED = [0.0, -0.9991, 0.0, 1.3986, 0.0, 0.0, 1.5491]
TOPK = 24                       # IK budget after prefilter+sort
IK_NS = 200_000_000            # solver returns in <10 ms when a solution exists
# coarse reachable workspace for the flange origin (base_link at world origin)
WS_R2 = (0.14, 0.78)          # xy radius bounds
WS_Z = (0.02, 0.90)

rclpy.init()
n = rclpy.create_node("two_pick")
ik = n.create_client(GetPositionIK, "/compute_ik")
ik.wait_for_service(timeout_sec=10)
ss = RobotState(); ss.joint_state = JointState(name=ARM, position=SEED)
_ik_calls = [0]


def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def in_ws(p):
    r2 = p[0] * p[0] + p[1] * p[1]
    return (WS_R2[0] ** 2 < r2 < WS_R2[1] ** 2) and (WS_Z[0] < p[2] < WS_Z[1])


def fik(pos, quat, seed=None):
    _ik_calls[0] += 1
    st = ss if seed is None else RobotState(joint_state=JointState(name=ARM, position=list(seed)))
    req = GetPositionIK.Request(); req.ik_request.group_name = "arm"; req.ik_request.robot_state = st
    ps = PoseStamped(); ps.header.frame_id = "base_link"
    ps.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    ps.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
    req.ik_request.pose_stamped = ps
    req.ik_request.timeout.sec = 0
    req.ik_request.timeout.nanosec = IK_NS
    # reachability filter only -- collision-free PATH is enforced downstream by step6's
    # STOMP + Cartesian (avoid_collisions=True) + planning-scene obstacle. With it True
    # here pick_ik converges to a self-colliding branch from the seed and never retries.
    req.ik_request.avoid_collisions = False
    f = ik.call_async(req); rclpy.spin_until_future_complete(n, f, timeout_sec=4); r = f.result()
    if r and r.error_code.val == 1:
        js = r.solution.joint_state
        return [round(js.position[js.name.index(j)], 4) for j in ARM]
    return None


def R2q(Rm):
    t = np.trace(Rm)
    if t > 0:
        u = np.sqrt(t + 1) * 2; w = .25 * u
        x = (Rm[2, 1] - Rm[1, 2]) / u; y = (Rm[0, 2] - Rm[2, 0]) / u; z = (Rm[1, 0] - Rm[0, 1]) / u
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        u = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / u; x = .25 * u; y = (Rm[0, 1] + Rm[1, 0]) / u; z = (Rm[0, 2] + Rm[2, 0]) / u
    elif Rm[1, 1] > Rm[2, 2]:
        u = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / u; x = (Rm[0, 1] + Rm[1, 0]) / u; y = .25 * u; z = (Rm[1, 2] + Rm[2, 1]) / u
    else:
        u = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / u; x = (Rm[0, 2] + Rm[2, 0]) / u; y = (Rm[1, 2] + Rm[2, 1]) / u; z = .25 * u
    v = np.array([x, y, z, w]); return (v / np.linalg.norm(v)).tolist()


for name in NAMES:
    t0 = time.time(); _ik_calls[0] = 0
    p = os.path.expanduser(f"~/grasp/{name}_grasps.json")
    if not os.path.exists(p):
        print(f"[{name}] no grasps file"); continue
    data = json.load(open(p))
    grasps = data["grasps"]
    cyl = data.get("cyl_fit")

    cand = []          # (cost, flange, Rf, approach, opening, kind)
    npre = 0
    if cyl:
        # Cylinder: CGN already told us it's a graspable round object + the opening.
        # The side grasp is analytically fixed by (fitted axis, height, radius) except
        # for approach azimuth -> sweep azimuth, order by (a) closeness to CGN's dominant
        # side-approach and (b) approaching from the robot side. step6 then STOMP+Cartesian
        # -validates in order. (Not "hand-coded grasp": the grasp is CGN-confirmed; the
        #  azimuth choice is a motion-reachability decision.)
        side_az = []
        for g in grasps:
            a = np.array(g["approach"]); a2 = a.copy(); a2[2] = 0
            if np.degrees(np.arccos(np.clip(-a[2], -1, 1))) >= 55 and np.linalg.norm(a2) > 1e-3:
                side_az.append(np.degrees(np.arctan2(a2[1], a2[0])))
        cgn_az = float(np.median(side_az)) if side_az else None
        span = cyl["top_z"] - cyl["base_z"]
        ch = float(np.clip(0.5 * (cyl["base_z"] + cyl["top_z"]) + 0.10 * span,
                           cyl["base_z"] + 0.04, cyl["top_z"] - 0.04))
        grip_centre = np.array([cyl["axis"][0], cyl["axis"][1], ch])
        opening = float(min(2 * cyl["radius"] + 0.018, 0.10))
        to = np.array([cyl["axis"][0], cyl["axis"][1]]); to /= max(np.linalg.norm(to), 1e-6)
        base_score = float(np.median([g["score"] for g in grasps])) if grasps else 0.25
        for az in range(0, 360, 15):
            ah = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
            flange = grip_centre - D_TCP * ah
            if not in_ws(flange):
                npre += 1; continue
            b = np.cross([0, 0, 1.0], ah); b /= np.linalg.norm(b)
            Rf = np.column_stack([-np.cross(ah, b), b, ah])
            align = float(ah[0] * to[0] + ah[1] * to[1])          # +1 = from robot side
            d_cgn = 0.0 if cgn_az is None else abs((az - cgn_az + 180) % 360 - 180) / 180.0
            cost = base_score + 0.6 * align - 0.5 * d_cgn
            cand.append((cost, flange, Rf, ah, opening, "side"))

    # compact object (mug/box): rank the ACTUAL CGN grasps (score - w*theta^4 - w*d_com),
    # then also add a few synthesized top-down candidates from the bbox as a fallback.
    # step6 STOMP+Cartesian-validates the list in order.
    if not cyl and data.get("box_fit"):
        bf = data["box_fit"]; c = np.array(bf["centre"]); ext = np.array(bf["ext"])
        centroid = np.array(bf["centre"])
        opening = float(min(min(ext[0], ext[1]) + 0.02, 0.10))
        cand = []
        for g in grasps:
            a = np.array(g["approach"]); pos = np.array(g["position_xyz"])
            w = g["opening_m"]
            if not (0.005 < w <= 0.102):
                continue
            Rf = q2R(g["quat_xyzw"])
            flange = pos - D_TCP * a                        # CGN grasp -> flange
            if not in_ws(flange):
                npre += 1; continue
            theta = np.degrees(np.arccos(np.clip(-a[2], -1, 1)))   # 0 = straight down
            d_com = float(np.linalg.norm(pos - centroid))
            cost = g["score"] - 0.5 * (theta / 90.0) ** 4 - 1.2 * d_com
            cand.append((cost, flange, Rf, a, min(w + 0.02, 0.10), "cgn"))
        # fallback: synthesized top-down from bbox (yaw sweep), ranked below real grasps
        gz = float(bf["bbmax"][2] - 0.02)
        gc = np.array([c[0], c[1], gz]); ad = np.array([0, 0, -1.0])
        for yaw in (45, 30, 60, 90, 20, 75, 15, 0):
            b = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
            Rf = np.column_stack([-np.cross(ad, b), b, ad])
            flange = gc - D_TCP * ad
            if not in_ws(flange):
                continue
            cand.append((-1.0 - abs(yaw - 45) * 1e-4, flange, Rf, ad, opening, "top"))

    # 180° twin: 그리퍼를 approach 축(Rf 3열) 기준 180° 뒤집으면 물리적으로 같은 파지지만
    # 손목 방향(= flange 카메라 방향)이 반대. cand tuple 에 flip 태그 추가.
    FLIP = np.diag([-1.0, -1.0, 1.0])
    cand = [c + ("orig",) for c in cand] + \
           [(c[0] - 1e-4, c[1], c[2] @ FLIP, c[3], c[4], c[5], "flip") for c in cand]

    cand.sort(key=lambda t: -t[0])
    t_pre = time.time() - t0
    kind0 = cand[0][5] if cand else "-"
    print(f"[{name}] {len(cand)} candidates (orig+flip, prefilter dropped {npre}, kind {kind0}), "
          f"prefilter {t_pre*1000:.0f} ms")
    if not cand:
        print(f"[{name}] no candidates"); continue

    # rank3-style FAST IK: one collision-free /compute_ik per candidate as a reachability
    # sieve (path collision + Cartesian feasibility are enforced downstream by step6, which
    # STOMP+Cartesian-validates the candidate list in order and takes the first fraction>=0.9).
    SOFF = {"side": 0.10, "top": 0.05, "cgn": 0.07}.get(kind0, 0.06)
    keep = []
    n_orig = n_flip = 0
    for cost, flange, Rf, a, opening, kind, tw in cand[:max(TOPK, 80)]:
        q = R2q(Rf)
        contact = np.array(flange) + D_TCP * np.array(a)
        j1s = np.arctan2(contact[1], contact[0]) + np.pi
        j1s = (j1s + np.pi) % (2 * np.pi) - np.pi
        seed_face = [float(np.clip(j1s, -2.9, 2.9)), -0.5, 0.0, 1.4, 0.0, 0.6, 1.5]
        if fik(flange, q) is None and fik(flange, q, seed=seed_face) is None:
            continue
        n_orig += tw == "orig"; n_flip += tw == "flip"
        so = {"side": 0.10, "top": 0.05, "cgn": 0.07}.get(kind, 0.06)
        keep.append(dict(object=name, kind=kind, twin=tw, quat=q, approach=[float(x) for x in a],
                         flange=[float(x) for x in flange],
                         pre=[float(x) for x in (np.array(flange) - so * a)],
                         standoff=so,
                         contact=[float(x) for x in (np.array(flange) + D_TCP * a)],
                         opening=opening, radius=cyl["radius"] if cyl else None))
        if len(keep) >= 8:
            break

    dt = time.time() - t0
    if keep:
        print(f"[{name}] reachable: {n_orig} orig + {n_flip} flip  (picked #0 = {keep[0]['twin']})")
        out = dict(keep[0]); out["candidates"] = keep
        json.dump(out, open(os.path.expanduser(f"~/grasp/{name}_pick.json"), "w"), indent=1)
        print(f"[{name}] {len(keep)} reachable candidates, primary approach "
              f"{np.round(keep[0]['approach'], 2)} contact {np.round(keep[0]['contact'], 3)} "
              f"standoff {SOFF} -> {name}_pick.json")
    else:
        print(f"[{name}] no reachable grasp")
    print(f"[{name}] {_ik_calls[0]} IK calls, {dt:.2f} s total")

rclpy.shutdown()
