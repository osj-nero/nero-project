#!/usr/bin/env bash
# rt_run.sh <discover_object.py 인자...> -- 실시간 pick 루프 1회 실행(환경은 gso_rt.sh 와 동일). MCP(rt_mcp_server.py)가 백그라운드로 띄운다.
cd "$(dirname "$0")"
export NERO_GRIP_TRAJ_SEC="${NERO_GRIP_TRAJ_SEC:-0.15}" ROS_LOCALHOST_ONLY=1 NERO_NO_PHYSICS=0 PYTHONUNBUFFERED=1 NERO_TOL_FIX=1 NERO_GT_HOLD=1 NERO_GT_HOLD_SEC=1.5
exec python3 -u discover_object.py "$@"
