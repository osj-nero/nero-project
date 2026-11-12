#!/usr/bin/env python3
"""scene_sort.py <이름> -- 정리/분류 시험 장면(2026-10-14): 실제 GSO 물체 여러 개 + 목적지(파란 보관함, 도자기 받침) 를 배치하고 목적지 입력 명세(/tmp/dests_<이름>.json)를 만든다.
S1: 장난감 2(사자, 라쿤) + 식품통 + 마우스. 장면 초기화는 사용자가 지시할 때만 실행한다(이 스크립트가 곧 초기화다)."""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BIN, SAU = "Curver_Storage_Bin_Blue", "Cole_Hardware_Saucer_Glazed_6"
A = dict(yoshi="Nintendo_Yoshi_Action_Figure", blackbasket="Basket_Wide_Black_v2", redbasket="Basket_Wide_Red_v2", lion="Schleich_Lion_Action_Figure", raccoon="Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure", nesquik="Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister",
         mouse="Razer_Abyssus_Ambidextrous_Gaming_Mouse", cosmetic="CoQ10_BjTLbuRVt1t", nips="Nestle_Nips_Hard_Candy_Peanut_Butter", eagle="Schleich_Bald_Eagle", wilton="Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold", holder="Pencil_Holder_Black_v3", flashlight="HeavyDuty_Flashlight",
         mug="ACE_Coffee_Mug_Kristen_16_oz_cup", bell="Cole_Hardware_School_Bell_Solid_Brass_38", coffee="Nescafe_Momento_Mocha_Specialty_Coffee_Mix_8_ct", clock="Crosley_Alarm_Clock_Vintage_Metal", tape="SNAIL_MEASURING_TAPE", jarrow="Nikon_1_AW1_w11275mm_Lens_Silver")
LAY = {"S1": {"lion": (-0.28, -0.28, 20), "raccoon": (-0.12, -0.36, 100), "nesquik": (-0.25, -0.12, 0), "mouse": (-0.30, 0.20, 70), BIN: (-0.40, 0.0, 0), SAU: (-0.12, 0.34, 0)},
       "S3": {"lion": (-0.30, -0.18, 20), "raccoon": (-0.22, -0.30, 100), "nesquik": (-0.42, -0.26, 0), "wilton": (-0.30, 0.22, 0), BIN: (-0.40, 0.0, 0), SAU: (-0.12, 0.34, 0), "holder": (-0.07, -0.39, 0)},
       "S4": {"lion": (-0.32, -0.31, 20), "raccoon": (-0.20, -0.22, 100), "eagle": (-0.44, -0.24, 60), "wilton": (-0.30, 0.22, 0), BIN: (-0.40, 0.0, 0), SAU: (-0.12, 0.34, 0), "holder": (-0.07, -0.39, 0)},
       "S5": {"flashlight": (0.20, -0.28, 0), "wilton": (-0.38, 0.21, 0), "lion": (-0.36, -0.26, 20), "raccoon": (-0.22, -0.17, 100), "eagle": (-0.27, -0.36, 60), BIN: (-0.40, 0.0, 0), SAU: (-0.12, 0.36, 0), "holder": (-0.07, -0.39, 0)},
       "S6": {"flashlight": (-0.14, -0.32, 0), "lion": (-0.30, -0.34, 20), "eagle": (-0.44, -0.28, 60), "raccoon": (-0.22, -0.17, 100), "bell": (-0.38, -0.14, 0), "coffee": (-0.18, 0.31, 0), "cosmetic": (-0.17, 0.16, 30), "clock": (-0.28, 0.0, 0), "tape": (-0.44, 0.03, 0), "jarrow": (-0.37, 0.15, 0), "wilton": (-0.30, 0.31, 0),
              BIN: (0.28, 0.24, 0), SAU: (0.26, -0.26, 0), "holder": (0.10, -0.40, 0)},
       "S2": {"lion": (-0.28, -0.28, 20), "raccoon": (-0.12, -0.36, 100), "nesquik": (-0.25, -0.12, 0), "mouse": (-0.30, 0.20, 70), "nips": (-0.22, 0.05, 30), "eagle": (-0.40, -0.30, 60), BIN: (-0.43, 0.0, 0), SAU: (-0.10, 0.34, 0)}}
def rand_layout(seed, split="y", stable=False, spread=1.0, conts=None, objs=("flashlight","nesquik","eagle","raccoon","clock","tape","bell","jarrow","wilton","coffee","cosmetic")):
    """무작위 장면. split="y": 용기 Y>0 / 물체 Y<0. split="diag": 직선 Y=X 로 나눠 용기 Y>X 쪽 / 물체 Y<X 쪽(경계에서 7cm 이상). 로봇 원점에서 R_MIN~R_MAX, 서로 중심 간격 확보."""
    import random, math
    rnd = random.Random(seed); placed = []
    def put(side, gap, rmin, rmax):      # side=+1: 용기 쪽, -1: 물체 쪽
        for _ in range(30000):
            x, y = rnd.uniform(-0.5, 0.5), rnd.uniform(-0.5, 0.5); r = math.hypot(x, y)
            if split == "diag":
                if side * (y - x) / math.sqrt(2) < 0.07: continue
            elif split == "anti":      # 직선 Y=-X: 용기 X+Y>0 쪽, 물체 X+Y<0 쪽
                if side * (x + y) / math.sqrt(2) < 0.07: continue
            elif (y < 0.14 if side > 0 else y > -0.10): continue
            if stable and (x > 0.02 if side < 0 else x > 0.40): continue      # 2026-10-15(stable): X>0 쪽은 j1 이 관절 한계에 가까워 불안정 -> 물체는 X<0, 용기는 X<=0.40
            if rmin <= r <= rmax and all(math.hypot(x - a, y - b) >= max(gap, g2) for a, b, g2 in placed): placed.append((x, y, gap)); return x, y
        raise RuntimeError("배치 실패")
    L = {}
    for k, gap in (conts or ((BIN, 0.24), ("redbasket", 0.24), (SAU, 0.20), ("holder", 0.16))):
        x, y = put(+1, gap, 0.44, 0.52); L[k] = (round(x, 3), round(y, 3), rnd.choice([0, 90]))
    for k in objs:
        for _t in range(300):      # 2026-10-15: 손전등과 홀더(cup)는 55cm 이상 떨어뜨린다(홀더 근처에서 손전등이 쓰러지는 문제)
            x, y = put(-1, spread * {"flashlight": 0.22, "coffee": 0.17, "cosmetic": 0.17, "yoshi": 0.17, "wilton": 0.15}.get(k, 0.13), (0.38 if k in ("flashlight", "nesquik", "eagle", "mouse", "raccoon", "clock", "coffee", "cosmetic", "yoshi") else 0.36), (0.43 if k == "flashlight" else (0.55 if spread > 1 else 0.52)))
            if k != "flashlight" or "holder" not in L or math.hypot(x - L["holder"][0], y - L["holder"][1]) >= 0.55: break
            placed.pop()
        else: raise RuntimeError("손전등-홀더 간격 배치 실패")
        L[k] = (round(x, 3), round(y, 3), rnd.randrange(0, 180, 10)); continue
        x, y = put(-1, spread * {"flashlight": 0.22, "coffee": 0.17, "cosmetic": 0.17, "wilton": 0.15}.get(k, 0.13), (0.32 if (stable or spread > 1) and k in ("flashlight", "nesquik", "eagle", "raccoon") else 0.25), (0.54 if spread > 1 else 0.50)); L[k] = (round(x, 3), round(y, 3), rnd.randrange(0, 180, 10))
    return L
SEAL_MIN = 4000000
def _seal_guard(seed):      # 2026-10-29(봉인 보호): 시드 >= 4,000,000 은 NERO_SEAL_HOLDER=1 없이는 거부한다(메시지에 시드 미포함). 개발 시드(3,000,000대·9xxx 등)는 영향 없음
    try: v = int(str(seed).lstrip("A-Za-z"))
    except Exception: return
    if v >= SEAL_MIN and os.environ.get("NERO_SEAL_HOLDER") != "1": raise PermissionError("봉인 범위 시드: 거부(NERO_SEAL_HOLDER=1 필요)")
_rand_layout_orig = rand_layout
def rand_layout(*a, **k):
    if a: _seal_guard(a[0])
    elif "seed" in k: _seal_guard(k["seed"])      # 2026-10-29(C10, 기본 꺼짐): NERO_LAYOUT_MINGAP(mm) 가 없으면 원 함수를 그대로 부른다
    if os.environ.get("NERO_LAYOUT_MINGAP"):
        import layout_c10; return layout_c10.rand_layout_c10(_rand_layout_orig, A, float(os.environ["NERO_LAYOUT_MINGAP"]), *a, **k)
    return _rand_layout_orig(*a, **k)
def g_pick(seed, n=10, _use=None):
    """G 배치: 일반 용품 풀(batch_assets/G_pool_curated.json, 집는 폭 7cm 이하로 스케일 조정)에서 범주를 고루 섞어 n개. -> [(이름, 스케일, 범주)]
    같은 모양이 겹치지 않게: (1) 같은 제조사(이름 첫 토큰), (2) 스케일 뒤 치수가 모두 12% 안, (3) 잉크 카트리지 1개 제한 -- 배치 안에서.
    배치 사이: 앞선 시드(1..seed-1)에서 덜 쓰인 물체를 먼저 고른다(풀이 작은 공구/전자도 최대한 번갈아 나오게)."""
    import random
    if _use is None:
        _use = {}
        for sd in range(1, seed):
            for q in g_pick(sd, n, _use): _use[q[0]] = _use.get(q[0], 0) + 1
    _ex = json.load(open(os.path.join(HERE, "batch_assets", "G_exclude.json")))["exclude"]
    pool = [q for q in json.load(open(os.path.join(HERE, "batch_assets", os.environ.get("NERO_G_POOL", "G_pool_curated.json")))) if q["name"] not in _ex]; rnd = random.Random(seed * 7919 + 13); by = {}
    for q in pool: by.setdefault(q["category"], []).append(q)
    fam = lambda q: q["name"].lower().split("_")[0]
    def similar(a, b):
        return fam(a) == fam(b) or all(abs(x - y) <= 0.12 * max(x, y) for x, y in zip(a["dims"], b["dims"])) or ("cartridge" in a["name"].lower() and "cartridge" in b["name"].lower())
    order = lambda lst: sorted(lst, key=lambda q: (_use.get(q["name"], 0), rnd.random()))
    _bm = os.environ.get("NERO_G_BOX_MAX"); boxfull = lambda q: _bm is not None and q.get("is_box", False) and sum(1 for o in out if o.get("is_box", False)) >= int(_bm)    # 장면당 박스 상한(풀의 is_box 표시 기준). 미설정이면 기존 동작
    quota = [("health", 2), ("personal", 2), ("stationery", 1), ("toys", 2), ("kitchen", 1), ("tools", 1), ("electronics", 1)]; out = []
    for c, k in quota:
        for q in order(by.get(c, [])):
            if sum(1 for o in out if o["category"] == c) >= k: break
            if not any(similar(q, o) for o in out) and not boxfull(q): out.append(q)
    for q in order(pool):
        if len(out) >= n: break
        if q not in out and not any(similar(q, o) for o in out) and not boxfull(q): out.append(q)
    return [(q["name"], q["scale"], q["category"]) for q in out]

if __name__ == "__main__":
    name = sys.argv[1]; _seal_guard(name[1:]); GP = g_pick(int(name[1:])) if name[0] == "G" else []; E = dict(os.environ, ROS_LOCALHOST_ONLY="1"); L = rand_layout(int(name[1:]), {"D": "diag", "A": "anti", "B": "anti", "C": "anti", "E": "anti", "H": "anti", "G": "anti"}.get(name[0], "y"), stable=(name[0] == "B"), spread=(1.1 if name[0] == "C" else (0.9 if name[0] == "H" else 1.0)), **({"objs": ("flashlight","lion","eagle","raccoon","clock","jarrow","wilton","coffee","cosmetic")} if name[0] == "B" else {}), **({"objs": ("flashlight","nesquik","cosmetic","raccoon","clock","tape","bell","jarrow","wilton","yoshi")} if name[0] in "EH" else {}), **({"objs": tuple(q[0] for q in GP), "conts": ((BIN, 0.24), ("redbasket", 0.24), ("blackbasket", 0.24))} if name[0] == "G" else {}))      # B: X<0 안정 범위에 간격(손전등 22cm 이상)을 지키며 들어가는 9개 if name[0] in "RDABCEH" and name[1:].isdigit() else LAY[name]
    print("layout", json.dumps(L))
    subprocess.run([sys.executable, "scene_park.py"], cwd=HERE, env=E, capture_output=True, timeout=120)
    if name[0] == "G":      # 2026-10-19: 마스크 불량/CGN 후보 없음으로 확정 제외한 물체는 스폰하지 않는다(나머지 물체 위치는 그대로 -- 배치 계획 자료가 유효하다)
        _ex = json.load(open(os.path.join(HERE, "batch_assets", "G_exclude.json")))["exclude"]; L = {k: v for k, v in L.items() if k not in _ex}
    objs = [A.get(k, k) for k in L]; pls = [dict(name=A.get(k, k), x=v[0], y=v[1], yaw=v[2]) for k, v in L.items()]
    r = subprocess.run([sys.executable, "gso_scene.py", "--placements", json.dumps(pls), "--objs", ",".join(objs), "--pose", "natural"], cwd=HERE, env=E, capture_output=True, text=True, timeout=200)
    print(r.stdout.strip()[-300:], r.stderr.strip()[-300:]); time.sleep(8)
    if GP:      # G 배치: 물체별 스케일(집는 폭 7cm 이하)을 스폰 뒤 프림 스케일로 적용한다. 범주표는 평가 전용으로 저장.
        try:
            from isaac_exec import run_code
            run_code("""
import omni.usd
from pxr import UsdGeom, Gf
st = omni.usd.get_context().get_stage()
SC = %s
for p in st.GetPrimAtPath("/World").GetChildren():
    for nm, sc in SC.items():
        if nm[:30] in p.GetName() and sc < 1.0:
            for o in UsdGeom.Xformable(p).GetOrderedXformOps():
                if o.GetOpName() == "xformOp:scale": o.Set(Gf.Vec3d(sc, sc, sc))
""" % json.dumps({q[0]: q[1] for q in GP}), timeout=20.0)
        except Exception as e_: print("G 스케일 실패", e_)
        os.makedirs(os.path.join(HERE, "batch_assets", name), exist_ok=True); json.dump([dict(name=q[0], scale=q[1], category=q[2]) for q in GP], open(os.path.join(HERE, "batch_assets", name, "objects_truth.json"), "w"), ensure_ascii=False)
    try:      # 2026-10-15: 손전등이 너무 커서(23cm) 삽입/놓기가 자주 실패 -> 스케일을 줄인다(NERO_FLASH_SCALE, 기본 0.7 => 약 16cm). 스폰 뒤 프림 스케일만 바꾼다.
        from isaac_exec import run_code
        _fs = float(os.environ.get("NERO_FLASH_SCALE", "0.7"))
        _lie = os.environ.get("NERO_FLASH_LIE", "1") == "1"
        run_code(f"""
import omni.usd
from pxr import UsdGeom, Gf
st = omni.usd.get_context().get_stage()
for p in st.Traverse():
    q = str(p.GetPath())
    if "Flashlight" in q and q.count("/") == 2:
        for o in UsdGeom.Xformable(p).GetOrderedXformOps():
            if o.GetOpName() == "xformOp:scale": o.Set(Gf.Vec3d({_fs}, {_fs}, {_fs}))
        if {_lie}:      # 쓰러뜨려 눕힌다: 원래 자세(서 있음) 위에 세계 X 축 90도 회전을 얹고 높이를 낮춘다(물리가 마저 안착시킨다)
            import math
            tr = [o for o in UsdGeom.Xformable(p).GetOrderedXformOps() if o.GetOpName() == "xformOp:translate"][0]
            oo = [o for o in UsdGeom.Xformable(p).GetOrderedXformOps() if o.GetOpName() == "xformOp:orient"][0]
            q0 = oo.Get(); c, s_ = math.cos(math.pi / 4), math.sin(math.pi / 4)
            w1, x1, y1, z1 = c, s_, 0.0, 0.0; a = q0.GetReal(); b, cc, d = q0.GetImaginary()
            QT = type(q0); VT = Gf.Vec3f if QT is Gf.Quatf else Gf.Vec3d
            nq = QT(w1 * a - x1 * b - y1 * cc - z1 * d, VT(w1 * b + x1 * a + y1 * d - z1 * cc, w1 * cc - x1 * d + y1 * a + z1 * b, w1 * d + x1 * cc - y1 * b + z1 * a))
            oo.Set(nq)
            t = tr.Get(); tr.Set(Gf.Vec3d(t[0], t[1], 0.06))
""", timeout=15.0)
    except Exception as e_: print("손전등 스케일 실패", e_)
    try:      # 2026-10-16: 목적지(용기/접시/컵 홀더) 질량 -- 시뮬 기본은 모든 물체 0.1kg 이라 물체가 떨어지면 용기가 밀렸다. NERO_DEST_MASS(기본 2.0kg)로 bin/바구니/접시/홀더 전부 같은 질량(0 이면 기본 유지)
        from isaac_exec import run_code
        _md = float(os.environ.get("NERO_DEST_MASS", "2.0"))
        run_code(f"""
import omni.usd
from pxr import UsdPhysics
st = omni.usd.get_context().get_stage()
for p in st.GetPrimAtPath("/World").GetChildren():
    nm = p.GetName()
    if {_md} > 0 and any(k in nm for k in ("Curver_Storage_Bin", "Basket_Wide", "Saucer", "Pencil_Holder")) and p.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.MassAPI(p).GetMassAttr().Set({_md})
""", timeout=15.0)
    except Exception as e_: print("목적지 질량 설정 실패", e_)
    out = f"/tmp/dests_{name}.json"; r = subprocess.run([sys.executable, "make_dest_spec.py", out], cwd=HERE, env=E, capture_output=True, text=True, timeout=120)
    print("목적지 명세", out, r.stderr.strip()[-200:])
