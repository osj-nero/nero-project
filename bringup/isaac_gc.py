#!/usr/bin/env python3
"""isaac_gc.py [--quiet] -- Isaac(Kit) 파이썬 가비지를 수거한다. 시뮬레이션(시행)이 끝나고 다음으로 넘어갈 때 부른다.
원인(2026-10-02 진단): Kit 은 파이썬 자동 GC 가 꺼져 있어(gc.isenabled()==False) OmniGraph Controller 객체(메서드를 속성으로 가진 순환 참조)가 초당 약 100개씩 쌓여 시간당 1GB 이상 늘었다.
gc.collect() 한 번(약 0.7s)으로 객체 499만 -> 62만, Controller 20만 -> 0 으로 풀린다. 이어 malloc_trim 으로 풀린 힙을 OS 에 돌려준다.
반환: {before/after: {objects, controllers, rss_gb}, collected, sec}  (Isaac 이 안 떠 있으면 조용히 실패)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CODE = r'''
import gc, json, os, time, ctypes
def _n(): return sum(1 for o in gc.get_objects() if type(o).__name__ == "Controller")
def _rss():
    try:
        import psutil; return round(psutil.Process(os.getpid()).memory_info().rss / 1e9, 3)
    except Exception: return -1
b = dict(objects=len(gc.get_objects()), controllers=_n(), rss=_rss())
t = time.time(); c = gc.collect(); sec = round(time.time() - t, 2)
try: ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception: pass
a = dict(objects=len(gc.get_objects()), controllers=_n(), rss=_rss())
json.dump(dict(before=b, after=a, collected=c, sec=sec, gc_enabled=gc.isenabled()), open("/tmp/_isaac_gc_out.json", "w"))
'''
def collect(timeout=60):
    from isaac_exec import run_code
    try:
        os.remove("/tmp/_isaac_gc_out.json")
    except OSError:
        pass
    try:
        run_code(CODE, timeout=timeout)
        return json.load(open("/tmp/_isaac_gc_out.json"))
    except Exception as e:
        return dict(error=repr(e)[:120])
if __name__ == "__main__":
    r = collect()
    if "--quiet" not in sys.argv:
        print(json.dumps(r, ensure_ascii=False))
