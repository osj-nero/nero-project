#!/bin/bash
# ============================================================================
# start_nero_isaac_all.sh
#
# Isaac Sim + MoveIt2/ROS2 스택 전체를 tmux 없이 한 번에 띄우는 스크립트.
# 백그라운드 job + trap 방식 (tmux 세션 관리 없이, 이 스크립트를 실행한
# 터미널 하나에서 전부 관리됨).
#
# 이 스크립트가 켜는 것 (순서대로):
#   0. Isaac Sim (USD 로드, --open 만 함 — ▶ 재생은 GUI에서 직접 눌러야 함)
#   1. robot_state_publisher (rsp)
#   2. static_virtual_joint_tfs (world -> base_link)
#   3. ros2_control_node (topic_based_ros2_control, Isaac Sim과 브릿지)
#   4. spawn_controllers (1회 실행, arm/gripper/joint_state_broadcaster)
#   5. move_group
#   6. rqt_joint_trajectory_controller (조인트 슬라이더 UI, ENABLE_RQT_JTC=true 일 때)
#   10. planning_node + nero-robot MCP(:9000) + nero-rt MCP(:9100)  (ENABLE_PC_CONTROL=true — nero_pc_control.sh up)
#   11. cuRobo 서버(:8020, C단계 계획/IK/place)  (ENABLE_CUROBO=true, 2026-10-01 추가) -- 종료 때 같이 내려간다
#   12. RViz 시각화(빠른 TF + rt_rviz_bridge + RViz2)  (ENABLE_RVIZ=true, 2026-10-07 추가) -- 종료 때 같이 내려간다
#   -1. Thor 서비스(CGN/CLIP/move_group; YOLO 는 ENABLE_YOLO=true 일 때만)  (ENABLE_THOR=true, ssh thor ~/grasp/thor_all.sh up) -- 종료 때는 THOR_DOWN_ON_EXIT=true 일 때만 내림
#
# 기본 off (2026-09-08, CGN 재설계 이후 — 구 비전 파지 경로에만 필요):
#   - 카메라 정적 TF          ENABLE_CAMERA_TF=true 로 켬
#   - 시각화 UI                ENABLE_VISUALIZE=true 로 켬 (기본: cuRobo 실시간 pick UI visualize_rt_curobo.py, VISUALIZE_UI=legacy 면 예전 visualize_3d_bpdl)
#   - perception_node_sim      ENABLE_PERCEPTION=true 로 켬 (8002 서버도 필요)
#   - RViz(moveit_rviz)        여전히 주석
#
# 이 스크립트가 켜지 않는 것 (직접 별도 터미널에서 실행):
#   - Claude Code CLI
#   - CGN pick 파이프라인은 이 스택 불필요 (Thor + Isaac execute_script)
#
# [2026-09-10] planning_node + nero-robot MCP(:9000) 도 이 스크립트가 같이 켠다
#   (ENABLE_PC_CONTROL=true, 내부에서 nero_pc_control.sh up 호출).
#   Ctrl+C 시 cleanup 이 nero_pc_control.sh down 도 같이 부른다.
#
# 종료: 이 스크립트가 실행 중인 터미널에서 Ctrl+C 한 번
#       -> trap이 모든 백그라운드 프로세스(Isaac Sim 포함)를 정리함
#
# 사용 전 꼭 확인/수정할 것 (환경마다 다를 수 있음):
#   - ISAACSIM_PATH, USD_PATH
#   - ROS_DISTRO_SETUP, WS_SETUP 경로
#   - MOVEIT_CFG_DIR 안의 xacro/yaml 경로
#   - 카메라 정적 TF 값 (hand-eye calibration 결과, ~/.nero_calib/flange_to_camera.json 참고)
#
# 사용법:
#   chmod +x start_nero_isaac_all.sh
#   ./start_nero_isaac_all.sh
# ============================================================================

set -e

# ── 환경 설정 (필요시 이 값들만 수정) ──────────────────────────────────────
ISAACSIM_PATH="$HOME/isaacsim/isaac-sim.sh"
NERO_ROOT="${NERO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; export NERO_ROOT   # repo root
USD_PATH="${NERO_USD:-$NERO_ROOT/assets/nero_simul.usd}"

ROS_DISTRO_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$HOME/ros2_ws/install/setup.bash"

# [2026-09-24 추가] Fast-DDS SHM 세그먼트 기본 크기(~512KB)가 카메라 이미지
# (1280x720x4 ≈ 3.6MB)보다 훨씬 작아서, 큰 메시지가 조용히 전송 실패하고
# "sequence size exceeds remaining buffer" 경고만 남기는 문제를 실측으로 확인
# (joint_states는 작아서 통과, camera_info/image_raw/tf는 전부 막힘 -- OmniGraph
# 노드/렌더 데이터는 전부 정상인데 ROS2 발행만 안 되는 것까지 직접 검증함).
# 이 프로필로 세그먼트를 20MB로 키운다. Isaac Sim(자식 프로세스로 기동)과 이
# 스크립트가 띄우는 모든 ROS2 노드가 export로 상속받는다 -- 이 스크립트로
# 기동하는 스택 전체에 일괄 적용.
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/.config/fastdds/nero_profile.xml"

MOVEIT_PKG="nero_gripper_moveit_config"
MOVEIT_CFG_DIR="$HOME/ros2_ws/src/agx_arm_sim/Moveit2/${MOVEIT_PKG}/config"
XACRO_FILE="${MOVEIT_CFG_DIR}/nero.urdf.xacro"
CONTROLLERS_YAML="${MOVEIT_CFG_DIR}/ros2_controllers.yaml"

# 카메라 정적 TF (hand-eye calibration 결과값 — 최신 값으로 갱신 필요시 여기 수정)
CAM_X=0.073089; CAM_Y=0.026438; CAM_Z=0.038050
CAM_ROLL=-0.350796; CAM_PITCH=-0.000000; CAM_YAW=1.569204
CAM_PARENT="gripper_base"; CAM_CHILD="camera_color_optical_frame"

# YOLO 추론 서버가 도는 Jetson Thor 주소 (perception_node_sim 이 여기로 프레임 POST)
# ~/.ssh/config 의 Host thor = <THOR_IP>.  ~/grasp/yolo_server.sh 로 Thor 에서 기동.
. "$NERO_ROOT/bringup/thor_net.sh" || exit 1
THOR_YOLO_HOST="${THOR_YOLO_HOST:-$NERO_THOR_IP}"

# ── 조인트 슬라이더 UI (rqt_joint_trajectory_controller) ──────────────────
# UI 슬라이더로 arm_controller / gripper_controller에 직접 궤적을 넣어
# Isaac Sim 로봇을 수동 제어. move_group(충돌회피)을 안 거치므로 주의.
# 패키지: sudo apt install ros-humble-rqt-joint-trajectory-controller
#
# ⚠️ [2026-09-08] 기본 false 로 바꿈. rqt_jtc 는 전원 버튼을 켜면
#    /arm_controller/joint_trajectory 로 10Hz 로 "현재 슬라이더값 유지" 궤적을
#    계속 발행하는데, 이게 move_group/nero-robot MCP 가 보내는 모든
#    FollowJointTrajectory 액션 goal 을 매 100ms 마다 선점(preempt)해서
#    로봇이 안 움직이고 STATUS_ABORTED / TIMED_OUT 난다 (실측 확인).
#    MCP/move_group 제어와 공존 불가. 수동 슬라이더가 꼭 필요하면 이 스크립트
#    밖에서 따로 띄우고, MCP 제어 전에 전원 버튼을 끄거나 창을 닫아라.
ENABLE_RQT_JTC=false

# ── 선택 노드 토글 (2026-09-08, CGN 재설계 이후 정리) ────────────────────
# [2026-09-08] CGN pick 은 flange 카메라 → YOLO(Thor) bbox → Thor CGN 이므로
# camera TF + perception 이 다시 필요. visualize 만 선택.
ENABLE_CAMERA_TF=true       # gripper_base -> camera_color_optical_frame 정적 TF (PC 카메라 뷰 3D화)
ENABLE_VISUALIZE="${ENABLE_VISUALIZE:-false}"       # 시각화 UI 창(q 로 종료). [2026-10-07] 기본 false: RViz 시각화(ENABLE_RVIZ)가 대체. 2D 창이 필요하면 ENABLE_VISUALIZE=true ./start_nero_isaac_all.sh
# [2026-10-01] 기본 UI 를 cuRobo 실시간 pick 용으로 교체: 카메라 영상 + 등록 물체 박스 + cuRobo 계획 경로 + 위에서 본 지도 + 단계/시간 패널.
#   (~/grasp/rt_state.json 은 discover_object.py 가 쓴다.) 예전 UI(visualize_3d_bpdl: YOLO bbox + cam/base XYZ, STOMP 파이프라인 상태)로 되돌리려면 VISUALIZE_UI=legacy
VISUALIZE_UI="${VISUALIZE_UI:-rt_curobo}"
ENABLE_PERCEPTION=false     # perception_node_sim (옛 YOLO 박스 검출 경로; 새 시스템은 CLIP 접지라 불필요 -- 2026-10-01 OFF. 켜려면 true 와 ENABLE_YOLO=true)
ENABLE_YOLO=false           # Thor YOLO 서버(:8002): 옛 인지 경로(perception_node_sim, nero-robot MCP 검출 도구, discover_boxes.py) 전용. 켜면 thor_all.sh 에 NERO_YOLO=1 을 넘긴다
ENABLE_PC_CONTROL=true      # planning_node + nero-robot MCP(:9000) 도 같이 (nero_pc_control.sh up/down)
PC_CONTROL_SH="$NERO_ROOT/bringup/nero_pc_control.sh"
# [2026-10-01] cuRobo 서버(PC GPU, :8020): C단계(접근/lift/place 계획) + B단계 IK + /fk. 실시간 pick 루프(discover_object.py)가 항상 필요로 한다.
ENABLE_CUROBO=true
# [2026-10-07] RViz 3D 시각화(빠른 TF + 브리지 + RViz): start_rviz_rt.sh 를 같이 띄운다. 끄려면 ENABLE_RVIZ=false ./start_nero_isaac_all.sh
ENABLE_RVIZ="${ENABLE_RVIZ:-true}"
# [2026-10-01] Thor(Jetson, ssh thor) 서비스도 같이: ~/grasp/thor_all.sh up = thor_stack(move_group) + YOLO :8002 + CGN :8010 + CLIP :8030.
#   이미 떠 있으면 건너뛴다(각 *.sh 가 멱등). 모델 로드가 길어 Isaac 부팅과 병렬로 먼저 시작한다.
ENABLE_THOR=true
THOR_HOST="thor"                    # ssh 별칭(키 인증, user@<THOR_IP>)
THOR_DOWN_ON_EXIT=false             # true 면 Ctrl+C 때 Thor 서비스도 내린다(기본 false: 모델 재로드가 오래 걸려 계속 켜 둔다)
CUROBO_SERVER_PY="$NERO_ROOT/pc_curobo/curobo_c_server.py"; export NERO_CUROBO_DIR="${NERO_CUROBO_DIR:-$NERO_ROOT/config}"
CUROBO_VENV="$HOME/venvs/curobo"
CUROBO_CUDA_HOME="$HOME/venvs/cuda121"

# ── 박스 위치 랜덤화 설정 (2026-07 추가) ──────────────────────────────────
# 실행할 때마다 TestBox/TestBox1/TestBox2를 안전 영역 내 새 랜덤 위치로
# 재배치. auto_stack_demo.py와 짝을 이뤄서 "매번 다른 배치에서도 되는지"
# 자동 반복 검증할 때 씀. 순수 usd 파일 수정이라 Isaac Sim을 (재)여는
# 시점에만 반영됨 -- 그래서 safety map 갱신과 마찬가지로 Isaac Sim 열기
# "직전"에 실행해야 함.
RANDOMIZE_BOXES=false   # 매번 랜덤 배치로 검증하고 싶으면 true로 (CGN 원통 작업 중엔 씬 흔들림 방지로 false)
RANDOMIZE_SCRIPT="$NERO_ROOT/bringup/randomize_box_positions.py"
RANDOMIZE_SEED=""       # 비워두면 매번 다른 랜덤, 재현하려면 숫자 지정 (예: 42)

# ── 안전지도(도달가능/불일치 맵) 자동 갱신 설정 ──────────────────────────
# 메인 usd에는 embed_reference_once.py로 SAFETY_MAP_USD를 참조(Reference)로
# 한 번만 심어두면 됨. 이후로는 아래 CSV/스크립트로 SAFETY_MAP_USD 파일
# 내용만 매번 새로 덮어써도 Isaac Sim이 열 때마다 자동으로 최신 내용을 읽음.
# build_safety_map_layer.py는 순수 OpenUSD(pip usd-core)만 쓰므로 Isaac Sim의
# python.sh가 아니라 일반 python3을 씀 (pip install --user usd-core 로 설치됨).
# venv를 따로 만드셨다면 그 venv의 python 경로로 바꿔도 됨.
SAFETY_MAP_PYTHON="python3"
SAFETY_MAP_BUILD_SCRIPT="$HOME/nero/build_safety_map_layer.py"
SAFETY_MAP_USD="$HOME/nero/safety_map_layer.usd"

# ★ 지도 종류를 바꾸고 싶으면 이 한 줄만 수정 ★
#   "mismatch"      : sim/실물 컨벤션 불일치 지도 (summary csv 필요)
#   "sim_reachable" : sim 기준 IK 해 존재 지도   (raw csv 필요)
SAFETY_MAP_SOURCE="mismatch"

# CSV가 들어있는 폴더 (파일명이 실행마다 바뀌므로, 폴더 안에서 SAFETY_MAP_SOURCE에
# 맞는 가장 최근 CSV를 자동으로 찾아 씀). 폴더 구조가 다르면 이 경로만 수정.
SAFETY_MAP_CSV_DIR="$HOME/nero"

# SAFETY_MAP_SOURCE에 맞춰 찾을 CSV 패턴이 자동으로 정해짐 (따로 손댈 필요 없음)
if [ "$SAFETY_MAP_SOURCE" = "sim_reachable" ]; then
    SAFETY_MAP_CSV_PATTERN="*mismatch_raw_*.csv"
else
    SAFETY_MAP_CSV_PATTERN="*mismatch_summary_*.csv"
fi

ENABLE_SAFETY_MAP=false   # 지도 자동 갱신 (CGN 원통 작업 중엔 USD 레이어 재생성 불필요로 false)

# 공통 source 커맨드
SRC_CMD="source ${ROS_DISTRO_SETUP} && source ${WS_SETUP}"

# 로그를 남길 디렉토리 (각 노드 stdout/stderr 확인용)
LOG_DIR="/tmp/nero_isaac_logs"
mkdir -p "$LOG_DIR"

PIDS=()
_cleaned=0
cleanup() {
    [ "$_cleaned" = 1 ] && return; _cleaned=1
    echo ""
    echo "🛑 전체 종료 중..."
    # planning_node + nero-robot MCP 는 별도 tmux 세션(nero_pc)이라 PIDS 로 안 잡힘 → 따로 down
    if [ "${ENABLE_PC_CONTROL:-false}" = true ] && [ -x "$PC_CONTROL_SH" ]; then
        echo "  nero_pc_control.sh down (planning_node + MCP :9000)"
        "$PC_CONTROL_SH" down 2>/dev/null || true
    fi
    if [ "${ENABLE_THOR:-false}" = true ] && [ "${THOR_DOWN_ON_EXIT:-false}" = true ]; then
        echo "  Thor 서비스 종료 (thor_all.sh down)"
        timeout 60 ssh -o ConnectTimeout=5 "$THOR_HOST" '~/grasp/thor_all.sh down' 2>/dev/null || true
    fi
    if [ "${ENABLE_CUROBO:-false}" = true ]; then
        echo "  cuRobo 서버 종료"
        pkill -f "^python .*curobo_c_server.py" 2>/dev/null || true
    fi
    if [ "${ENABLE_RVIZ:-false}" = true ]; then
        echo "  RViz 시각화 종료 (rviz2 / 브리지 / rsp_fast)"
        for _p in $(ps -eo pid,args | awk '($2=="python3" && $3=="rt_rviz_bridge.py") || $2=="rviz2" || /robot_state_publisher_fast/ && !/awk/ {print $1}'); do kill "$_p" 2>/dev/null || true; done
    fi
    echo "  백그라운드 프로세스: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "✅ 정리 완료. 로그는 $LOG_DIR 에 남아있습니다."
}
trap cleanup EXIT INT TERM

run_bg() {
    # run_bg <로그파일이름> <bash -c로 실행할 커맨드 전체>
    local logname="$1"; shift
    bash -c "$*" > "${LOG_DIR}/${logname}.log" 2>&1 &
    PIDS+=($!)
    echo "  -> PID $! ($logname), 로그: ${LOG_DIR}/${logname}.log"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " NERO Isaac Sim + MoveIt2 전체 스택 기동"
echo " 종료하려면 이 터미널에서 Ctrl+C 한 번"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$RANDOMIZE_BOXES" = true ]; then
    echo "사전 단계) 박스 위치 랜덤화 — Isaac Sim 열기 전에 usd 안 박스 좌표 새로 씀"
    SEED_ARG=""
    if [ -n "$RANDOMIZE_SEED" ]; then
        SEED_ARG="--seed $RANDOMIZE_SEED"
    fi
    python3 "$RANDOMIZE_SCRIPT" --main-usd "$USD_PATH" $SEED_ARG \
        > "${LOG_DIR}/0_randomize_boxes.log" 2>&1 \
        && echo "   -> 박스 재배치 완료 (로그: ${LOG_DIR}/0_randomize_boxes.log)" \
        || echo "   ⚠️ 박스 재배치 실패 (로그: ${LOG_DIR}/0_randomize_boxes.log) — 이전 위치 그대로 진행합니다."
fi

if [ "$ENABLE_SAFETY_MAP" = true ]; then
    echo "사전 단계) 안전지도(safety map) 갱신 — Isaac Sim 열기 전에 usd 레이어 새로 씀"
    # 폴더(하위 폴더 포함) 안에서 패턴에 맞는 가장 최근(mtime 기준) CSV 자동 탐색
    SAFETY_MAP_CSV="$(find "$SAFETY_MAP_CSV_DIR" -type f -name "$SAFETY_MAP_CSV_PATTERN" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -n1 | cut -d' ' -f2-)"
    if [ -n "$SAFETY_MAP_CSV" ] && [ -f "$SAFETY_MAP_CSV" ]; then
        echo "   -> 사용할 CSV: $SAFETY_MAP_CSV"
        "$SAFETY_MAP_PYTHON" "$SAFETY_MAP_BUILD_SCRIPT" \
            --source "$SAFETY_MAP_SOURCE" \
            --csv "$SAFETY_MAP_CSV" \
            --out "$SAFETY_MAP_USD" \
            > "${LOG_DIR}/0_safety_map_build.log" 2>&1 \
            && echo "   -> 지도 갱신 완료: $SAFETY_MAP_USD" \
            || echo "   ⚠️ 지도 갱신 실패 (로그: ${LOG_DIR}/0_safety_map_build.log) — 지도 없이 계속 진행합니다."
    else
        echo "   ⚠️ '$SAFETY_MAP_CSV_DIR' 안에서 '$SAFETY_MAP_CSV_PATTERN' 패턴의 CSV를 못 찾았습니다 (건너뜀)"
    fi
fi

if [ "$ENABLE_THOR" = true ]; then
    echo "-1) Thor 서비스 기동 (ssh $THOR_HOST ~/grasp/thor_all.sh up) — Isaac 부팅과 병렬"
    run_bg "thor_up" "ssh -o ConnectTimeout=8 $THOR_HOST 'NERO_YOLO=$([ "$ENABLE_YOLO" = true ] && echo 1 || echo 0) ~/grasp/thor_all.sh up'"
else
    echo "-1) Thor 서비스 — 건너뜀 (ENABLE_THOR=false)"
fi

echo "0) Isaac Sim 실행 + USD 로드"
run_bg "0_isaacsim" "${SRC_CMD} && '$ISAACSIM_PATH' --open '$USD_PATH' --ext-folder $HOME/Documents/isaac-sim-mcp/ --enable isaac.sim.mcp_extension"
echo "   Isaac Sim 창이 뜨면 반드시 ▶ 재생 버튼을 눌러주세요 (ActionGraph는 재생 중에만 동작)."
sleep 25   # Isaac Sim 자체 부팅이 오래 걸림 — 환경에 따라 늘려야 할 수 있음
# [2026-10-02] Isaac 메모리 누수 대책: Kit 은 파이썬 자동 GC 가 꺼져 있어 OmniGraph Controller 객체가 초당 ~100개씩 쌓여 시간당 1GB 이상 늘었다(실측 진단).
#   부팅 직후 주기적 GC(90초마다 gc.collect)를 Isaac 안에 설치한다. 시행 사이 수거(tools/phase_c/isaac_gc.py)는 실험 큐가 따로 부른다.
(
    for i in $(seq 1 40); do
        if (exec 3<>/dev/tcp/127.0.0.1/8766) 2>/dev/null; then
            python3 $NERO_ROOT/bringup/isaac_gc_install.py 90 > "${LOG_DIR}/0b_isaac_gc.log" 2>&1 && break
        fi
        sleep 3
    done
) &


echo "1) robot_state_publisher"
run_bg "1_rsp" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} rsp.launch.py"
sleep 3

echo "2) static_virtual_joint_tfs"
run_bg "2_static_tf" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} static_virtual_joint_tfs.launch.py"
sleep 2

echo "3) ros2_control_node"
run_bg "3_ros2_control" "${SRC_CMD} && ros2 run controller_manager ros2_control_node --ros-args \
    -p robot_description:=\"\$(xacro ${XACRO_FILE})\" \
    --params-file ${CONTROLLERS_YAML}"
sleep 5

echo "4) spawn_controllers"
run_bg "4_spawn_ctrl" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} spawn_controllers.launch.py"
sleep 5

echo "5) move_group"
run_bg "5_move_group" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} move_group.launch.py"
sleep 10

#echo "6) RViz"
#run_bg "6_rviz" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} moveit_rviz.launch.py"
#sleep 3


if [ "$ENABLE_CAMERA_TF" = true ]; then
    echo "6) 카메라 정적 TF"
    run_bg "7_camera_tf" "${SRC_CMD} && ros2 run tf2_ros static_transform_publisher \
        --x ${CAM_X} --y ${CAM_Y} --z ${CAM_Z} \
        --roll ${CAM_ROLL} --pitch ${CAM_PITCH} --yaw ${CAM_YAW} \
        --frame-id ${CAM_PARENT} --child-frame-id ${CAM_CHILD}"
else
    echo "6) 카메라 정적 TF — 건너뜀 (ENABLE_CAMERA_TF=false)"
fi

if [ "$ENABLE_VISUALIZE" = true ]; then
    if [ "$VISUALIZE_UI" = "legacy" ]; then
        echo "7) 시각화 (visualize_3d_bpdl, 예전 UI)"
        run_bg "8_visualize" "${SRC_CMD} && python3 ~/nero/visualize_3d_bpdl.py"
    else
        echo "7) 시각화 (visualize_rt_curobo: 카메라 + 지도 + cuRobo 계획 경로 + 단계/시간)"
        run_bg "8_visualize" "${SRC_CMD} && python3 ~/nero/visualize_rt_curobo.py"
    fi
    sleep 2
else
    echo "7) 시각화 — 건너뜀 (ENABLE_VISUALIZE=false)"
fi

if [ "$ENABLE_RQT_JTC" = true ]; then
    echo "8) 조인트 슬라이더 UI (rqt_joint_trajectory_controller)"
    run_bg "8b_rqt_jtc" "${SRC_CMD} && ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller"
    echo "   창에서 controller manager ns=/controller_manager, controller=arm_controller 선택 후"
    echo "   빨간 전원 버튼을 눌러 활성화하면 슬라이더로 로봇 제어 가능."
    sleep 2
fi

# ── 선택 사항: perception/planning까지 여기서 같이 켜고 싶으면 주석 해제 ──
# [2026-08-31] 박스 YOLO 서버(vlm_boxyolo.py)는 이제 이 데스크탑이 아니라
# 같은 연구실 Jetson Thor(<THOR_IP>:8002)에서 띄운다 -- 로컬에서 또
# 띄우면 perception_node_sim이 BOX_SERVER_URL 미지정 시 기본값
# 127.0.0.1:8002(로컬)를 보게 돼 Thor 배포와 따로 노는 상황이 생겨서,
# 로컬 기동 자체를 제거하고 아래 9단계에서 Thor 주소를 명시적으로
# export한다. VLM grasp 서버(8003)도 Thor에서 띄움
# (run_mcp_robot_server.sh의 VLM_SERVER_URL 참고, mcp_robot_server는
# 이 스크립트가 안 켬 -- Claude Code가 별도로 기동).
# [2026-08-19] vlm_boxyolo.py가 듀얼 모델 서버(box 커스텀 + COCO 25종)로
# 바뀌면서 인자명이 --conf → --conf-box/--conf-coco로 변경됨(Thor에서
# 띄울 때도 이 인자명 그대로 씀. 참고용으로 커맨드 남겨둠, 로컬 기동
# 안 함):
#   python3 ~/nero/yolo/vlm_boxyolo.py --host 0.0.0.0 --port 8002 \
#       --model ~/nero/yolo/best.pt --model-coco yolov8n.pt \
#       --conf-box 0.75 --conf-coco 0.25

if [ "$ENABLE_PERCEPTION" = true ]; then
    echo "9) perception_node_sim (YOLO 서버: Jetson Thor ${THOR_YOLO_HOST}:8002)"
    run_bg "9_perception" "export BOX_SERVER_URL=http://${THOR_YOLO_HOST}:8002/detect BOX_HEALTH_URL=http://${THOR_YOLO_HOST}:8002/health && ${SRC_CMD} && ros2 run sj_pickplace perception_node_sim"
    sleep 3
else
    echo "9) perception_node_sim — 건너뜀 (ENABLE_PERCEPTION=false)"
fi

if [ "$ENABLE_THOR" = true ]; then
    echo "   Thor 서비스 확인 (CGN :8010, CLIP :8030$([ "$ENABLE_YOLO" = true ] && echo ', YOLO :8002'))"
    THOR_SPECS="CGN:8010 CLIP:8030"; [ "$ENABLE_YOLO" = true ] && THOR_SPECS="YOLO:8002 $THOR_SPECS"
    for spec in $THOR_SPECS; do
        nm="${spec%%:*}"; pt="${spec##*:}"; okk=0
        for i in $(seq 1 60); do
            if timeout 6 ssh -o ConnectTimeout=5 "$THOR_HOST" "curl -s -m 3 http://127.0.0.1:${pt}/health" >/dev/null 2>&1; then okk=1; break; fi
            sleep 3
        done
        [ "$okk" = 1 ] && echo "   - $nm OK" || echo "   ⚠️ $nm :$pt 응답 없음 (로그: ${LOG_DIR}/thor_up.log, Thor ~/grasp/logs/)"
    done
fi

if [ "$ENABLE_CUROBO" = true ]; then
    echo "11) cuRobo 서버 (:8020) — GPU 로 모델 로드/워밍업(약 1분)"
    if curl -s -m 2 http://localhost:8020/health >/dev/null 2>&1; then
        echo "   이미 :8020 응답 중 -- 새로 띄우지 않음(종료 때도 건드리지 않으려면 수동 기동 상태 유지)"
    else
        # exec 로 파이썬이 이 job 의 PID 가 되게 한다(Ctrl+C 때 PIDS kill 로 정확히 종료)
        run_bg "11_curobo" "source ${CUROBO_VENV}/bin/activate && export CUDA_HOME=${CUROBO_CUDA_HOME} PATH=${CUROBO_CUDA_HOME}/bin:\$PATH && cd ${NERO_CUROBO_DIR} && exec python ${CUROBO_SERVER_PY}"
        printf "   cuRobo 준비 대기"
        for i in $(seq 1 60); do
            if curl -s -m 2 http://localhost:8020/health >/dev/null 2>&1; then echo " — OK (${i}x3s)"; break; fi
            printf "."; sleep 3
            [ "$i" -eq 60 ] && echo " — 타임아웃 (로그: ${LOG_DIR}/11_curobo.log) 계속 진행"
        done
    fi
else
    echo "11) cuRobo 서버 — 건너뜀 (ENABLE_CUROBO=false)"
fi

if [ "$ENABLE_PC_CONTROL" = true ]; then
    echo "10) planning_node + nero-robot MCP(:9000)  —  nero_pc_control.sh up"
    if [ -x "$PC_CONTROL_SH" ]; then
        # move_group 가 실제로 뜬 뒤 호출 (nero_pc_control 의 prereq 체크 통과하도록 최대 30s 대기)
        printf "   move_group 대기"
        for i in $(seq 1 30); do
            if bash -c "${SRC_CMD} && ros2 node list 2>/dev/null" | grep -qx /move_group; then
                echo " — OK"; break
            fi
            printf "."; sleep 1
            [ "$i" -eq 30 ] && echo " — 타임아웃 (그래도 시도)"
        done
        # set -e 상태라 실패해도 스크립트가 죽지 않게 if 로 감쌈
        if "$PC_CONTROL_SH" up; then
            echo "   -> planning_node + MCP :9000 기동 완료. Claude Code 에서 /mcp 로 확인."
        else
            echo "   ⚠️ nero_pc_control.sh up 실패 — planning/MCP 없이 계속."
            echo "      수동 재시도: $PC_CONTROL_SH up"
        fi
    else
        echo "   ⚠️ $PC_CONTROL_SH 없음/실행불가 — 건너뜀"
    fi
else
    echo "10) planning_node + MCP — 건너뜀 (ENABLE_PC_CONTROL=false)"
fi

if [ "$ENABLE_RVIZ" = true ]; then
    echo "12) RViz 시각화 (빠른 TF + 브리지 + RViz) — start_rviz_rt.sh"
    if [ -x "$NERO_ROOT/bringup/start_rviz_rt.sh" ]; then
        run_bg "12_rviz" "$NERO_ROOT/bringup/start_rviz_rt.sh"
    else
        echo "   ⚠️ bringup/start_rviz_rt.sh 없음/실행불가 — 건너뜀"
    fi
else
    echo "12) RViz 시각화 — 건너뜀 (ENABLE_RVIZ=false)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 전체 기동 완료. 확인용 명령어:"
echo "   ros2 node list"
echo "   ros2 topic list | grep -E 'joint_states|controller|camera'"
echo "   tail -f ${LOG_DIR}/5_move_group.log   # move_group 로그 실시간 확인"
echo "   $PC_CONTROL_SH status                 # planning_node + MCP :9000/:9100 상태"
echo "   curl -s localhost:8020/health         # cuRobo 서버"
echo "   ssh thor ~/grasp/thor_all.sh status   # Thor 서비스"
echo ""
echo "Claude Code 에서:  /mcp  (nero-robot 이 connected 인지 확인)"
echo "종료: 이 터미널에서 Ctrl+C  (Isaac Sim·ROS 스택 + planning_node + MCP(:9000/:9100) + cuRobo 서버 + RViz 시각화 전부 정리)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wait
