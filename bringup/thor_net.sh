# thor_net.sh -- source 용. Thor 접속 IP 를 NERO_THOR_NET(lan|wifi)으로 고른다. 기본 lan = 기존 동작.
# 이미 NERO_THOR_IP 가 설정돼 있으면 그 값을 그대로 쓴다. 같은 규칙의 파이썬 판: nero/tools/phase_c/thor_net.py
NERO_THOR_NET="${NERO_THOR_NET:-lan}"
NERO_THOR_IP_LAN="${NERO_THOR_IP_LAN:-}"   # set in config/nero_env.sh
NERO_THOR_IP_WIFI="${NERO_THOR_IP_WIFI:-}"
if [ -z "${NERO_THOR_IP:-}" ]; then
    case "$NERO_THOR_NET" in
        lan) NERO_THOR_IP="$NERO_THOR_IP_LAN" ;;
        wifi)
            if [ -z "$NERO_THOR_IP_WIFI" ]; then
                echo "thor_net: NERO_THOR_NET=wifi 인데 NERO_THOR_IP_WIFI 가 비어 있음" >&2; return 1 2>/dev/null || exit 1
            fi
            NERO_THOR_IP="$NERO_THOR_IP_WIFI" ;;
        *) echo "thor_net: NERO_THOR_NET 은 lan|wifi (받은 값: $NERO_THOR_NET)" >&2; return 1 2>/dev/null || exit 1 ;;
    esac
fi
export NERO_THOR_NET NERO_THOR_IP_LAN NERO_THOR_IP_WIFI NERO_THOR_IP
