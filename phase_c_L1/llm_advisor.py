"""llm_advisor.py -- 실패한 항목에 대한 LLM 조언을 비동기로 받는다(2026-10-18, NERO_LLM_RECOVER=1).
로봇은 기다리지 않고 규칙 기본 동작(다음 항목으로)으로 계속 가고, 응답은 그 항목이 다시 차례가 될 때 적용한다. 안 왔으면 기본 동작(변경 없음)."""
import json, subprocess, threading, time

class Advisor:
    def __init__(self):
        self.res = {}; self.log = []
    def submit(self, key, prompt):
        rec = dict(id=key, t_sent=round(time.time(), 1), t_reply=None, action=None, dest=None); self.log.append(rec)
        def run():
            t0 = time.time()
            try:
                out = subprocess.run(["claude", "-p", "--tools", "", "--output-format", "text"], input=prompt, capture_output=True, text=True, timeout=90, cwd="/tmp").stdout
                js = json.loads(out[out.index("{"): out.rindex("}") + 1]); self.res[key] = js
                rec.update(action=js.get("action"), dest=js.get("dest"), why=str(js.get("why", ""))[:100], secs=round(time.time() - t0, 1))
            except Exception as e:
                rec.update(error=repr(e)[:100], secs=round(time.time() - t0, 1))
        threading.Thread(target=run, daemon=True).start()
    def take(self, key):
        return self.res.pop(key, None)
