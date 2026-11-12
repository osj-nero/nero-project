"""VLM 승격(2026-11-10): 조건·순서·풀기 세 질문 모두에서 물체 설명 글(obj) 대신 '카메라 사진 2장(원본, 면 이름표) + 기하로 아는 모양 글'을 준다.
질문 문장은 글 입력판과 같게 두고 obj/cmd 만 바꾼다(공정 비교). enable(items) 뒤 set_item(it, scene) 으로 지금 문항의 사진을 정하면 P3.call 이 사진을 붙인다.
사진: Isaac 로봇 카메라 크롭이 있으면(vlm_images/<scene>/<texture>_{raw,lab}.jpg) 그것, 없으면 render_box_vlm 오프라인 렌더(같은 시점 근사)."""
import os, json, base64, subprocess, time
H = os.path.dirname(os.path.abspath(__file__)); IMG = H + "/vlm_images"
SHAPE = {"box": "직육면체(가로 {a}cm, 깊이 {b}cm, 높이 {c}cm). 지금 서 있음(윗면이 위)",
         "plate": "직육면체(가로 {a}cm, 두께 {b}cm, 높이 {c}cm). 지금 긴 모서리가 바닥에 닿게 세워져 있음. 넓은 두 면이 앞면과 뒷면",
         "bar": "직육면체(길이 {a}cm, 폭 {b}cm, 높이 {c}cm). 지금 바닥에 눕혀져 있음"}      # 2026-11-10: 치수는 인식(깊이)으로 잰 값
OBJ_V = "사진 속 물체(첫 사진: 로봇 카메라 원본, 둘째 사진: 같은 사진에 면 이름 표시, 앞면과 뒷면 중 지금 로봇에 더 가까운 쪽이 앞면). 깊이로 잰 모양: {shape}. 재질·내용물·표시는 사진으로 판단"
CUR = {"imgs": None}


def images_for(texture, asset, scene_yaw, scene_id):
    d = f"{IMG}/{scene_id}"; raw, lab = f"{d}/{texture}_raw.jpg", f"{d}/{texture}_lab.jpg"
    if not (os.path.exists(raw) and os.path.exists(lab)): raise FileNotFoundError(f"Isaac 로봇 카메라 크롭 없음: {raw} (capture_vlm_all.py)")      # 오프라인 렌더로 대신하지 않음
    return [raw, lab]


def enable():
    import rq1_decide3_prompt as P3
    if getattr(P3, "_vlm_on", False): return
    orig = P3.call; P3._orig_call = orig

    def call(p, seed):
        if not CUR["imgs"]: return orig(p, seed)
        g = P3._g; a = g["a"]
        content = [dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + base64.b64encode(open(f, "rb").read()).decode())) for f in CUR["imgs"]] + [dict(type="text", text=p)]
        req = dict(model=a.model, messages=[{"role": "user", "content": content}], temperature=a.temp, seed=seed, max_tokens=a.mt); t0 = time.time()
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "thor", "curl -s -m 900 http://127.0.0.1:8005/v1/chat/completions -H 'Content-Type: application/json' -d @-"], input=json.dumps(req).encode(), capture_output=True, timeout=940)
        jj = json.loads(r.stdout.decode()); m = jj["choices"][0]["message"]
        return (m.get("content") or ""), (m.get("reasoning_content") or m.get("reasoning") or ""), jj["usage"]["completion_tokens"], round(time.time() - t0, 2), jj["choices"][0]["finish_reason"]
    P3.call = call; P3._vlm_on = True


def vlm_item(it, scene):
    """글 입력 문항 -> VLM 문항(obj = 사진+모양, cmd = 안 보이는 사실을 더한 cmd_v). 사진을 지금 문항으로 정한다."""
    CUR["imgs"] = images_for(it["texture"], it["asset"], scene["yaw"], scene["id"] + "_perc")      # 인식 좌표로 그린 사진(vlm_images/<scene>_perc)
    p = scene["perc"]; a_, b_, c_ = [round(100 * v, 1) for v in p["dims"]]; fmt = lambda v: (str(int(v)) if float(v).is_integer() else str(v))
    return dict(it, obj=OBJ_V.format(shape=SHAPE[it["asset"]].format(a=fmt(a_), b=fmt(b_), c=fmt(c_))), cmd=it["cmd_v"], obj_text_orig=it["obj"])
