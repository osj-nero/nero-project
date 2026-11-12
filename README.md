# NERO Manipulation — LLM이 판단하고 로봇이 집어 놓는 시스템

자연어 명령을 받아 7축 로봇팔(AgileX NERO + 평행 그리퍼 + 손목 카메라)이 물체를 찾아 집고, **어디에 어떤 자세로 놓을지를 LLM이 판단**해 놓는 시스템입니다. 모든 실행은 NVIDIA Isaac Sim 시뮬레이션에서 이뤄집니다.

*A language-driven pick-and-place system for a 7-DoF arm in Isaac Sim. An LLM decides where and in what pose to place each object; grasp candidates come from Contact-GraspNet and motions from cuRobo.*

**프로젝트 페이지(소개·흐름도·데모 영상): <https://osj-nero.github.io/nero-project/>**

> 이 저장소는 연구용 코드입니다. 한 연구실의 PC 1대 + Jetson 1대 구성에 맞춰 개발됐고, 다른 환경에서의 설치는 아래 절차를 따라 직접 맞춰야 합니다.

---

## 1. 무엇을 하는가

| 과제 | 내용 | 실행 스크립트 |
|---|---|---|
| **높은 작업대 의미 배치** | 30cm 높이 작업대 위에 물체를 "명령의 뜻에 맞는 자세"로 놓는다. 장애물로 막힌 장면에서는 LLM이 조건을 풀거나 포기를 판단한다. | `rq1/stage2_vlm.py` |
| **정밀 배치** | 막대로 井(우물 정)자, 고인돌, 삼각형, 3층 구조를 만든다. LLM이 구조 명세를 쓰고 코드가 검사한 뒤 로봇이 실행한다. 기둥 4개 위에 판을 얹는 과제도 포함. | `precise/run_claude_tasks.py`, `precise/run_floor_pillars_lid.py` |
| **분류 정리** | 여러 물체가 섞인 장면을 조사한 뒤, LLM이 물체마다 목적지를 정해 순서대로 옮긴다. | MCP 도구 `rt_survey` → `rt_sort` |

## 2. 시스템 흐름

```
 자연어 명령
     │
     ▼
 ┌───────────────────────────── LLM 판단 ─────────────────────────────┐
 │  · 과제 계획: 어떤 물체를 어느 목적지로, 어떤 순서로 (MCP 도구 호출)      │
 │  · 놓기 자세: 사진 2장 + 형상 설명을 보고 놓을 면·방향 조건을 정함        │
 │  · 실패 대응: 계획이 서지 않으면 이유를 받아 조건 완화 / 포기 판단        │
 └────────────────────────────────────────────────────────────────────┘
     │ 대상·목적지·놓기 조건(place_spec)
     ▼
 ① 탐색·인식   팔을 돌려 훑기 → 이름 없는 덩어리 검출 → CLIP 으로 대상 접지
     ▼
 ② 파지 후보   Contact-GraspNet (물체 점군 → 6자유도 파지 후보)
     ▼
 ③ 후보 선별   기하·IK 필터 + 학습 모델로 순위 (B 단계)
     ▼
 ④ 동작 계획   cuRobo: 집기와 놓기를 함께 계획, 충돌·놓기 조건 검사 (C 단계)
     ▼
 ⑤ 실행       ros2_control 궤적 실행 (Isaac Sim)
     ▼
 ⑥ 검증·복구   놓은 뒤 카메라로 확인 → 필요하면 재계획 / LLM 에 되묻기
```

### 구성 요소와 위치

| 위치 | 포트 | 코드 | 역할 |
|---|---|---|---|
| PC | — | `phase_c_L1/discover_object.py` (`rt_run.sh`) | 실시간 집기·놓기 루프 본체 |
| PC | 8020 | `pc_curobo/curobo_c_server.py` | cuRobo IK |
| PC | 9100 | `phase_c_L1/rt_mcp_server.py` (`mcp/rt_plus.py`) | LLM이 로봇을 부리는 MCP 도구 서버 (nero-rt) |
| PC | — | `ros2_pkg/` (`planning_node`) | ROS 2 노드 (관측 자세 기록 등) |
| PC | — | `bringup/start_nero_isaac_all.sh` | Isaac Sim + ros2_control + MoveIt + 위 서버 일괄 기동 |
| Thor | 8012 | `thor/grasp/cgn_server_fp.py` | Contact-GraspNet 파지 후보 + 후보 선별(`two_pick_plan_fast.py`) |
| Thor | 8021 | `thor/curobo/curobo_c_server.py` | cuRobo 집기+놓기 계획 (놓기 조건·충돌 검사) |
| Thor | 8030 | `thor/grasp/clip_ground_server.py` | CLIP ViT-L/14 자연어 접지 |
| Thor | 8005 | `thor/vllm/serve_big.sh` (vLLM) | 로컬 멀티모달 LLM (Gemma 4) |

PC는 x86 + NVIDIA GPU(Isaac Sim 구동), Thor는 NVIDIA Jetson AGX Thor입니다. Thor 역할은 GPU가 있는 다른 리눅스 서버로 대신할 수 있습니다.

## 3. 폴더 구조

```
phase_c_L1/     로봇 실행 본체: 탐색, 인식, 집기, 놓기, 검증, 복구, MCP 서버
rq1/            과제 1 (높은 작업대 의미 배치): 장면 생성, LLM 판단, 채점
precise/        과제 2 (정밀 배치): 구조 명세 생성·검사·실행
contplace/      놓기 조건 판정 공통 코드
goalpose_impl/  계획 서버 호출 유틸
mcp/            MCP 서버 확장 (호출 기록, 응답 보완)
pc_curobo/      PC cuRobo IK 서버
ros2_pkg/       ROS 2 패키지 sj_pickplace
bringup/        PC 기동 스크립트, RViz 시각화
thor/           Thor 쪽 서버 코드와 기동 스크립트 (grasp/, curobo/, vllm/)
config/         로봇 설정(cuRobo yml), 환경변수 예시
assets/         Isaac Sim 장면 (nero_simul.usd)
docs/           프로젝트 페이지 (GitHub Pages)
```

폴더 이름(`phase_c_L1`, `rq1` 등)은 개발 중 이름을 그대로 둔 것입니다. 스크립트끼리 이 이름으로 서로를 찾으므로 바꾸지 마세요.

## 4. 설치

### 4.1 필요한 것

| 구분 | 버전·출처 |
|---|---|
| PC | Ubuntu 22.04, ROS 2 Humble, NVIDIA Isaac Sim, Python 3.10 |
| Thor | ROS 2 Jazzy, CUDA |
| 로봇 모델·MoveIt 설정 | [agilexrobotics/agx_arm_sim](https://github.com/agilexrobotics/agx_arm_sim) (`nero_gripper_moveit_config`) |
| 동작 계획 | [cuRobo](https://github.com/NVlabs/curobo) (PC와 Thor 각각 가상환경) |
| 파지 후보 | [contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch) + 체크포인트 |
| 접지 | [OpenAI CLIP](https://github.com/openai/CLIP) ViT-L/14 |
| LLM | [vLLM](https://github.com/vllm-project/vllm) + `google/gemma-4-26B-A4B-it` |
| 분류 과제 물체 | [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) (USD 변환본) |

### 4.2 PC

```bash
git clone https://github.com/osj-nero/nero-project.git ~/nero-manip
cd ~/nero-manip

# 1) 환경변수
cp config/nero_env.sh.example config/nero_env.sh    # Thor IP 등 값 채우기
source config/nero_env.sh

# 2) ROS 2 작업공간: 로봇 모델 + 이 저장소의 패키지
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/agilexrobotics/agx_arm_sim.git
ln -s ~/nero-manip/ros2_pkg sj_pickplace
cd ~/ros2_ws && colcon build && source install/setup.bash

# 3) 로봇 설정 yml 의 경로 채우기
~/nero-manip/config/apply_paths.sh ~/ros2_ws

# 4) cuRobo 가상환경 (~/venvs/curobo) 은 cuRobo 설치 문서를 따른다.

# 5) Thor 에 ssh 별칭으로 접속되게 한다 (~/.ssh/config 에 Host thor, 키 인증)
ssh thor hostname
```

### 4.3 Thor

```bash
git clone https://github.com/osj-nero/nero-project.git ~/nero-manip
~/nero-manip/thor/install_thor.sh        # 서버 코드를 ~/grasp, ~/nero/tools/phase_c_goalpose 등으로 복사
```

가상환경은 서버별로 따로 씁니다: `~/grasp/cgn_venv`(Contact-GraspNet, CLIP), `~/venvs/curobo_thor`(cuRobo), `~/vllm-venv-g4`(vLLM).

### 4.4 모델 파일

저장소에는 가중치가 들어 있지 않습니다. 아래 파일을 Thor에 넣어야 합니다.

| 파일 | 위치 | 출처 |
|---|---|---|
| Contact-GraspNet 체크포인트 | `~/grasp/contact_graspnet_pytorch/checkpoints/` | 원 저장소 |
| `ViT-L-14.pt` | `~/weights/clip/` | OpenAI CLIP |
| `phold_v1.joblib`, `place_reach_model.joblib` | `~/grasp/` | 이 저장소의 Releases |
| Gemma 4 | Hugging Face 캐시 | `google/gemma-4-26B-A4B-it` (이용 약관 동의 필요) |

## 5. 실행

### 5.1 서버 띄우기

```bash
# Thor
~/grasp/thor_all.sh up                                   # MoveIt 스택 + CLIP(:8030)
cd ~/grasp && CGN_FP=10 CGN_PORT=8012 ~/grasp/cgn_venv/bin/python3 cgn_server_fp.py          # 파지 후보(:8012)
cd ~/nero/tools/phase_c_goalpose && NERO_NB_FIRST=1 NERO_CUROBO_BUDGET_SCALE=3.0 \
    ~/venvs/curobo_thor/bin/python curobo_c_server.py --port 8021                            # 계획(:8021)
MODEL=google/gemma-4-26B-A4B-it SERVED=qwen3-vl GPU_MEM_UTIL=0.45 MAXLEN=16384 MAXSEQ=8 \
    EXTRA="--quantization fp8 -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY --max-num-batched-tokens 4096" \
    ~/vllm-venv/serve_big.sh                                                                 # LLM(:8005). 코드가 모델 이름 qwen3-vl 로 부른다

# PC
source ~/nero-manip/config/nero_env.sh
ssh -fN -L 8021:127.0.0.1:8021 thor                      # 계획 서버 터널
~/nero-manip/bringup/start_nero_isaac_all.sh             # Isaac Sim 이 뜨면 ▶(재생)을 눌러야 한다
```

### 5.2 과제 1 — 높은 작업대 의미 배치

```bash
cd ~/nero-manip/rq1
S2V_SAME_PERC=1 python3 stage2_vlm.py blocked_v4/exec_vlm_cases.json     # LLM 판단
python3 stage2_rules.py blocked_v4/exec_vlm_cases87.json S,F,C           # 비교용 규칙 정책
```

장면 생성 → 로봇 인식 → LLM이 놓기 조건 결정 → 계획·실행 → 채점까지 한 번에 돕니다. 결과는 `rq1/stage2_vlm/`에 쌓입니다.

### 5.3 과제 2 — 정밀 배치

```bash
cd ~/nero-manip/precise
python3 run_claude_tasks.py jeong        # dolmen | jeong | jeong3 | triangle
FPL_AB=1 FPL_TWIST=20 FPL_CX=-0.42 python3 run_floor_pillars_lid.py demo 0.07    # 기둥 4개(간격 7cm) + 판
```

`run_claude_tasks.py`는 구조 명세를 만드는 데 Claude(`claude` CLI)를 부릅니다.

### 5.4 과제 3 — 분류 정리

MCP 클라이언트(예: Claude Code)에 `http://127.0.0.1:9100/mcp`를 `nero-rt`로 등록한 뒤 자연어로 지시합니다.

```bash
cd ~/nero-manip/phase_c_L1 && python3 scene_sort.py S1       # 물체 여러 개 + 목적지 장면, /tmp/dests_S1.json 생성
```

```
"책상 위 물건을 조사해서 장난감은 파란 보관함에, 나머지는 받침 위에 정리해줘"
  → rt_survey(dests_path)   장면 조사: 물체 표(id, 라벨 후보, 색, 크기, 위치)
  → rt_sort(plan, dests_path)   LLM 이 정한 계획을 순서대로 실행
  → rt_sort_status()   진행 확인
```

단일 물체는 `rt_pick`, `rt_place`로 지시할 수 있습니다.

## 6. 알아둘 점

- **시뮬레이션 전용**입니다. 채점과 일부 판정은 Isaac Sim의 정답 자세를 읽습니다.
- 코드에는 `ssh thor ...` 호출과 `~/grasp` 작업 폴더가 그대로 들어 있습니다. PC와 Thor의 사용자 홈 아래에 같은 `~/grasp` 경로가 있어야 합니다.
- 동작 옵션은 대부분 `NERO_*` 환경변수입니다. 과제 스크립트가 필요한 값을 스스로 넣으며, 기본값은 `rq1/exec_main.py`의 `ENV`에 있습니다.
- 코드 주석과 로그는 한국어입니다.
- 이 공개본은 개발 저장소에서 실제로 쓰이는 파일만 골라 경로를 설정값으로 바꾼 것입니다. 경로를 바꾼 뒤의 전체 동작은 공개 시점에 다시 확인하지 못한 부분이 있을 수 있으니, 문제가 있으면 Issue로 알려주세요.

## 7. 라이선스와 출처

- 이 저장소 코드의 라이선스: **MIT** ([LICENSE](LICENSE)). 아래 외부 구성 요소에는 적용되지 않습니다.
- 로봇 모델·MoveIt 설정은 AgileX Robotics `agx_arm_sim`, 동작 계획은 NVIDIA cuRobo, 파지 후보는 Contact-GraspNet, 접지는 OpenAI CLIP, LLM은 Google Gemma를 사용합니다. 각각의 라이선스와 이용 약관을 따릅니다.

## 8. 인용

논문 정보는 게재 후 추가합니다.
