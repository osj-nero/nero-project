#!/usr/bin/env python3
"""isaac_exec.py — Isaac Sim addon 소켓(localhost:8766)에 직접 붙어 execute_script를
실행하는 CLI. mcp__isaac-sim__execute_script와 완전히 동일한 프로토콜(단순 JSON,
{"type":"execute_script","params":{"code":...}})을 직접 구현한 것 -- Claude(대화형 MCP
호출) 없이 bash 스크립트가 반복시행(N trial) 중에 스스로 씬을 리셋할 수 있게 하기 위함.
프로토콜은 ~/Documents/isaac-sim-mcp/isaac_mcp/server.py의 IsaacConnection.send_command를
그대로 따름(길이 프리픽스 없음, 응답이 완전한 JSON으로 파싱될 때까지 반복 recv).

사용: python3 isaac_exec.py --code-file /tmp/snippet.py
      python3 isaac_exec.py --code "print(1+1)"
      echo "print(1)" | python3 isaac_exec.py
"""
import argparse, json, socket, sys


def run_code(code, host="localhost", port=8766, timeout=60.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    try:
        sock.sendall(json.dumps({"type": "execute_script", "params": {"code": code}}).encode("utf-8"))
        chunks = []
        while True:
            chunk = sock.recv(16384)
            if not chunk:
                if not chunks:
                    raise ConnectionError("Isaac 소켓이 데이터 없이 닫힘 -- Isaac Sim이 안 떠있을 수 있음")
                break
            chunks.append(chunk)
            try:
                data = b"".join(chunks)
                return json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue
        return json.loads(b"".join(chunks).decode("utf-8"))
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--code-file")
    g.add_argument("--code")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    if a.code_file:
        code = open(a.code_file).read()
    elif a.code:
        code = a.code
    else:
        code = sys.stdin.read()

    try:
        resp = run_code(code, a.host, a.port, a.timeout)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

    print(json.dumps(resp))
    if resp.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
