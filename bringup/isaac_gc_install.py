#!/usr/bin/env python3
"""isaac_gc_install.py -- Isaac 안에 '주기적 GC' 를 설치한다(멱등). Kit 의 자동 GC 가 꺼져 있어 쌓이는 OmniGraph Controller 순환 참조를 N초마다 수거한다.
시행 사이 수거(isaac_gc.py)가 주 방어선이고, 이건 사람이 손으로 오래 쓸 때(시행 경계가 없을 때)의 안전망이다. 기본 90초."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
CODE = r'''
import builtins, gc, time, ctypes
import omni.kit.app
if getattr(builtins, "_nero_gc_sub", None) is None:
    st = {"t": time.time(), "n": 0}
    def _on_update(e):
        if time.time() - st["t"] >= %f:
            st["t"] = time.time(); st["n"] += 1
            gc.collect()
            if st["n"] %% 10 == 0:
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass
    builtins._nero_gc_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(_on_update, name="nero_periodic_gc")
    builtins._nero_gc_state = st
''' % SEC
from isaac_exec import run_code
r = run_code(CODE, timeout=30)
print("주기적 GC 설치:", r.get("status", r))
