"""thor_net.py -- Thor 접속 IP 선택(nero/thor_net.sh 와 같은 규칙). NERO_THOR_IP 가 있으면 우선, 없으면 NERO_THOR_NET(lan|wifi, 기본 lan)."""
import os

def thor_ip():
    ip = os.environ.get("NERO_THOR_IP")
    if ip:
        return ip
    net = os.environ.get("NERO_THOR_NET", "lan")
    if net == "lan":
        return os.environ.get("NERO_THOR_IP_LAN", "")
    if net == "wifi":
        ip = os.environ.get("NERO_THOR_IP_WIFI", "")
        if not ip:
            raise RuntimeError("thor_net: NERO_THOR_NET=wifi 인데 NERO_THOR_IP_WIFI 가 비어 있음")
        return ip
    raise RuntimeError(f"thor_net: NERO_THOR_NET 은 lan|wifi (받은 값: {net})")
