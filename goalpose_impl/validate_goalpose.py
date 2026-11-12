#!/usr/bin/env python3
"""goalpose(obj_rot) 오프라인 검증 (로봇/Isaac 없음). 실서버 8020 은 읽기 전용 계획 요청만, 복사본 8021 에 obj_rot 요청.
  (a) 회귀: obj_rot 없는 같은 /plan, /place 요청을 8020/8021 에 보내 비교 (+ 8021 에 obj_rot=identity 를 보내 새 경로 = 기존 경로 확인)
  (b,c) 높은 작업대(윗면 0.30) 위 슬롯 (-0.46,0) 에 x90 / x-90 / y90 / x180 (+ 기준: 회전 없음) 으로 /plan
  (d) 풀린 계획의 place_cfg 를 /fk 로 실제 플랜지 자세로 바꾸고, 집은 자세 대비 강체 변환으로 목표 물체 자세를 역산해 기대값과 비교
사용: python3 validate_goalpose.py [--only a|b] [--out results.json]"""
import os as _nr_os; NERO_ROOT = _nr_os.environ.get("NERO_ROOT") or _nr_os.path.dirname(_nr_os.path.dirname(_nr_os.path.abspath(__file__)))  # repo root (override: NERO_ROOT)
import argparse, glob, json, math, time, urllib.request, copy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DUMP = sorted(glob.glob("" + NERO_ROOT + "/precise/plate_feas/dump/*.json"))
D_TCP = 0.1358
PLATFORM = dict(id="platform", xyz=[-0.46, 0.0, 0.15], raw=[0.24, 0.24, 0.30], yaw=0.0)
TOP = 0.30
PICK_PLATFORM = False


def post(port, path, req, timeout=1200):
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=timeout)
    out = json.loads(r.read()); out["_wall"] = round(time.time() - t0, 2); return out


def rot_of(spec):
    if spec is None: return None
    if isinstance(spec, list): return Rot.from_quat(spec)
    R = Rot.identity()
    for t in spec.split("_"): R = Rot.from_euler(t[0], float(t[1:]), degrees=True) * R
    return R


def summ_place(pp):
    if not pp: return None
    keys = ("ok", "mode", "n_cands", "n_ik_ok", "yaw_deg", "slot", "slot_z", "relax_level", "retreat_len", "half_h_rot", "cand_rank", "obj_goal", "sec", "mode_tried")
    s = {k: pp.get(k) for k in keys if k in pp}
    for k in ("tcp", "flange", "quat"):
        if pp.get(k) is not None: s[k] = [round(float(x), 4) for x in pp[k]]
    s["n_tried"] = len(pp.get("tried") or []); s["tried"] = [dict(yaw=t.get("yaw"), relax=t.get("relax"), fails=t.get("fails"), fail=t.get("fail")) for t in (pp.get("tried") or [])]
    return s


def summ_plan(o):
    s = dict(ok=o.get("ok"), sec=o.get("sec"), wall=o.get("_wall"), why=[w[:110] for w in (o.get("why") or [])][:20], error=o.get("error"))
    p = o.get("plan") or {}
    if p: s.update(cand_index=p.get("cand_index"), place=summ_place(p.get("place_plan")))
    return s


def backcompute(port, req, plan, R_rel):
    """place_cfg -> FK 플랜지 -> 물체 목표 자세 역산(집은 순간의 손-물체 강체 관계 사용)."""
    c = req["candidates"][plan["cand_index"]]; pp = plan["place_plan"]; ob = req["place_check"]["req"]["obj"]
    R0 = Rot.from_quat(c["quat"]); FL0 = np.array(c["flange"], float)
    CTR3 = np.array([ob["centre"][0], ob["centre"][1], ob["base_z"] + ob["height"] / 2.0])
    fk = post(port, "/fk", {"q": [pp["place_cfg"][:7]]})
    fl = np.array(fk["pos"][0]); w, x, y, z = fk["quat_wxyz"][0]; Rf = Rot.from_quat([x, y, z, w])
    Rd = Rf * R0.inv()                                   # 손이 집은 자세 대비 돈 양 = 물체가 돈 양
    cen = fl + Rd.apply(CTR3 - FL0); Robj = Rd * Rot.from_euler("z", float(ob.get("yaw", 0.0)))
    dims = np.array([ob["dims"][0], ob["dims"][1], ob["height"]], float)
    cn = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float) * dims / 2.0
    low = float((cen + Robj.apply(cn))[:, 2].min())
    clear = float(req["place_check"]["req"].get("clear", 0.006))
    hz = 0.5 * float(np.abs(Robj.as_matrix()[2, :]) @ dims)
    exp_z = TOP + clear + hz
    # 기대 회전: Rz(yaw_deg) * R_rel (yaw 는 서버가 고른 값)
    Rexp = Rot.from_euler("z", pp["yaw_deg"], degrees=True) * (R_rel if R_rel is not None else Rot.identity())
    ang = math.degrees((Rd * Rexp.inv()).magnitude())
    srv = pp.get("obj_goal") or {}
    loc = Robj.as_matrix()
    axes_up = {n: round(float(loc[2, i]), 3) for i, n in enumerate(("w(x,%.3f)" % dims[0], "d(y,%.3f)" % dims[1], "h(z,%.3f)" % dims[2]))}
    return dict(fk_flange_err_mm=round(1000 * float(np.linalg.norm(fl - np.array(pp["flange"]))), 2),
                centre=[round(float(v), 4) for v in cen], centre_xy_err_mm=round(1000 * float(np.hypot(cen[0] - pp["slot"][0], cen[1] - pp["slot"][1])), 2),
                centre_z=round(float(cen[2]), 4), expected_centre_z=round(exp_z, 4), z_err_mm=round(1000 * (float(cen[2]) - exp_z), 2),
                lowest_corner_z=round(low, 4), gap_above_top_mm=round(1000 * (low - TOP), 2), half_h=round(hz, 4),
                rot_err_vs_expected_deg=round(ang, 3), local_axes_world_z=axes_up,
                server_obj_goal_centre=srv.get("centre"))


def platform_req(base, rot):
    r = copy.deepcopy(base)
    # 작업대는 place_check.req 에만 넣는다(사양). pick 월드에도 넣으면 저장된 시작 자세(start_q)가 작업대와 충돌해 pick 접근이 전부 실패한다(1차 실행 확인)
    if PICK_PLATFORM: r["obstacles"] = list(r.get("obstacles") or []) + [dict(PLATFORM)]
    pr = r["place_check"]["req"]
    pr["slots"] = [[-0.46, 0.0]]; pr["top_z"] = TOP
    pr["obstacles"] = list(pr.get("obstacles") or []) + [dict(PLATFORM)]; pr["dest_ids"] = list(pr.get("dest_ids") or []) + ["platform"]
    if rot is not None: pr["obj_rot"] = [float(v) for v in rot_of(rot).as_quat()]
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="ab"); ap.add_argument("--out", default="" + NERO_ROOT + "/goalpose_impl/results.json")
    ap.add_argument("--dumps", default="all"); ap.add_argument("--rots", default=""); a = ap.parse_args()
    res = dict(a_place=[], a_plan=[], b=[])
    dumps = DUMP if a.dumps == "all" else DUMP[:int(a.dumps)]
    def save(): json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    if "a" in a.only:
        for f in dumps:
            base = json.load(open(f))
            # /place 직접(파지 = 후보 0): 기존 8020, 복사본 8021(obj_rot 없음), 복사본 8021(obj_rot=identity -> 새 경로)
            c0 = base["candidates"][0]; pr = copy.deepcopy(base["place_check"]["req"])
            pr["grasp"] = dict(quat=c0["quat"], approach=c0["approach"], flange=c0["flange"]); pr["start_q"] = base["start_q"]
            row = dict(dump=f.split("/")[-1])
            for tag, port, extra in (("8020", 8020, {}), ("8021", 8021, {}), ("8021_identity", 8021, {"obj_rot": [0, 0, 0, 1]})):
                o = post(port, "/place", dict(pr, **extra)); row[tag] = summ_place(o); row[tag]["wall"] = o["_wall"]; print("[a/place]", row["dump"], tag, row[tag].get("ok"), row[tag].get("yaw_deg"), row[tag].get("tcp"), o["_wall"], flush=True)
            res["a_place"].append(row); save()
            for port in (8020, 8021):
                o = post(port, "/plan", base); s = summ_plan(o); s.update(dump=row["dump"], port=port); res["a_plan"].append(s); save()
                print("[a/plan]", row["dump"], port, s["ok"], s.get("cand_index"), (s.get("place") or {}).get("yaw_deg"), (s.get("place") or {}).get("tcp"), s["wall"], flush=True)
    if "b" in a.only:
        base = json.load(open(DUMP[0]))
        for rot in (a.rots.split(",") if a.rots else ("x90", "x-90", "y90", "y-90", "x180", None)):
            rot = None if rot == "none" else rot
            req = platform_req(base, rot)
            o = post(8021, "/plan", req); s = summ_plan(o); s.update(dump=DUMP[0].split("/")[-1], rot=rot or "none(세움)")
            if o.get("ok"):
                _pp = o["plan"]["place_plan"]; s["raw"] = dict(cand_index=o["plan"]["cand_index"], place_cfg=_pp["place_cfg"], yaw_deg=_pp["yaw_deg"], slot=_pp["slot"], flange=_pp["flange"], tcp=_pp["tcp"], quat=_pp["quat"], obj_goal=_pp.get("obj_goal"), durations=_pp.get("durations"))
                try: s["back"] = backcompute(8021, req, o["plan"], rot_of(rot))
                except Exception as e: s["back"] = dict(error=repr(e))
            res["b"].append(s); save()
            print("[b]", s["rot"], s["ok"], "pick", s.get("cand_index"), "place", (s.get("place") or {}).get("cand_rank"), (s.get("place") or {}).get("yaw_deg"), s["wall"], s.get("back"), flush=True)
    save()


if __name__ == "__main__":
    main()
