#!/usr/bin/env bash
# pick_one.sh — pick 전용 모드(2026-09-29): place 를 걷어내고 pick 만 최적화/평가한다.
# 홈 복귀 -> (선계산 계획 재사용 | 안전망 재계산) -> exec pick -> 파지 기록 -> HELD 면 그 자리에서 그리퍼를 열어 물체를 떨군다.
# place 관련(테이블/슬롯/place 계획/복구)은 전혀 안 쓴다. place 개발은 place_on_target_one.sh 쪽에서 병렬로 계속.
# 사용: ./pick_one.sh <DISC.json> <gid> <box_index_i>
set -eo pipefail
source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
cd "$(dirname "$0")"
DISC="$1"; name="$2"; i="$3"
[ -n "$DISC" ] && [ -n "$name" ] && [ -n "$i" ] || { echo "사용: $0 <DISC.json> <gid> <box_index_i>" >&2; exit 1; }
OTHER_OBJS=$(python3 -c "import json; print(' '.join(b for b in json.load(open('$DISC'))['boxes']))")
PRE_OK=$(python3 -c "import json; d=json.load(open('$DISC')); print(d.get('precomputed',{}).get('$name', False))")

# 2026-09-30 홈 복귀 없이 이어 붙이기(chain): 앞 물체 pick 의 lift 끝 자세에서 곧바로 이 물체로 간다.
# discover 가 앞 물체를 집는 동안 그 lift 끝 자세를 시작점으로 이 물체의 C단계(transit)를 미리 계산해 두었다(파일 태그 NERO_CHAIN_PLAN_TAG).
# 현재 자세가 그 시작 자세와 다르면(pick 실패, 다른 plan 사용 등) 조용히 기존 경로(홈 복귀 + 선계산 계획)로 되돌아간다.
CHAIN_TAG=""
if [ "${NERO_RT_LOCAL:-0}" = "1" ] && [ -n "${NERO_CHAIN_START:-}" ]; then
    # 2026-10-01 실시간 루프(discover_object.py): 계획이 PC 에 이미 있고 현재 자세도 호출자가 안다 -> arm_q 읽기/ssh 확인 생략(pick 준비 오버헤드 단축)
    CHAIN_TAG="${NERO_CHAIN_PLAN_TAG:-}"; ARM_Q="$NERO_CHAIN_START"
    echo "══ [$name] 홈 복귀 생략 -- 앞 물체 pick 자세에서 바로 이어서 pick (chain) ══"
    echo "  arm_q=$ARM_Q"
elif [ -n "${NERO_CHAIN_PLAN_TAG:-}" ] && [ -n "${NERO_CHAIN_START:-}" ]; then
    ARM_Q=$(python3 read_armq.py 2>/dev/null || true)
    CH_NEAR=$(python3 - "$ARM_Q" "$NERO_CHAIN_START" <<'PYEOF'
import sys
try:
    a = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2].split()]
    print(1 if len(a) == 7 and len(b) == 7 and max(abs(x - y) for x, y in zip(a, b)) < 0.06 else 0)
except Exception:
    print(0)
PYEOF
)
    if [ "$CH_NEAR" = "1" ] && timeout 30 ssh thor "test -f ~/grasp/${name}_step6_pick${NERO_CHAIN_PLAN_TAG}.json"; then
        CHAIN_TAG="$NERO_CHAIN_PLAN_TAG"
        echo "══ [$name] 홈 복귀 생략 -- 앞 물체 pick 자세에서 바로 이어서 pick (chain) ══"
        echo "  arm_q=$ARM_Q"
    else
        echo "  [$name] chain 불가(현재 자세가 시작 자세와 다르거나 계획 없음) -- 홈 복귀 경로로 폴백"
    fi
fi

if [ -z "$CHAIN_TAG" ]; then
echo "══ [$name] 홈 복귀 ══"
# 2026-09-30 속도: 앞 물체 처리 뒤 백그라운드로 이미 홈에 가 있다 -> 전체 홈 스크립트(약 2s, DDS 디스커버리+정착 대기) 대신
# 현재 관절만 읽어(0.5s) observation 근처면 그대로 쓰고, 아니면 기존처럼 홈 복귀한다.
ARM_Q=$(python3 read_armq.py 2>/dev/null || true)
NEAR=$(python3 - "$ARM_Q" <<'PYEOF'
import json, os, sys
try:
    q = [float(x) for x in sys.argv[1].split()]
    ref = json.load(open(os.path.expanduser("~/.local/share/nero_robot/saved_poses.json")))["observation"]["joints"]
    print(1 if len(q) == 7 and max(abs(q[i] - ref["joint%d" % (i + 1)]) for i in range(7)) < 0.05 else 0)
except Exception:
    print(0)
PYEOF
)
if [ "$NEAR" != "1" ]; then
    ARM_Q=$(python3 home_and_get_armq.py 2>/dev/null)
fi
echo "  arm_q=$ARM_Q"

if [ "$PRE_OK" = "True" ] && timeout 30 ssh thor "test -f ~/grasp/${name}_step6_pick.json"; then
    echo "  [$name] discover_boxes.py 가 미리 계산해둔 계획 재사용 (B/C단계 생략)"
else
    echo "══ [$name] B단계 IK 랭킹 (선계산 없음/실패 -- 안전망으로 재계산) ══"
    timeout 30 ssh thor "rm -f ~/grasp/${name}_pick.json ~/grasp/${name}_step6_pick.json ~/grasp/step6_pick.json"
    timeout 180 ssh thor "source /opt/ros/jazzy/setup.bash && source \$HOME/ros2_ws/install/setup.bash && cd ~/grasp && export NERO_ARM_Q='$ARM_Q' && python3 two_pick_plan_fast.py $name" 2>&1 | grep -E 'reachable|no candidate|Error|저장|->' || true
    echo "══ [$name] C단계 STOMP ══"
    timeout 300 ssh thor "source /opt/ros/jazzy/setup.bash && source \$HOME/ros2_ws/install/setup.bash && cd ~/grasp && export NERO_ARM_Q='$ARM_Q' NERO_VEL_SCALE='${NERO_VEL_SCALE:-0.8}' NERO_OTHER_OBJS='$OTHER_OBJS' && bash step6_single.sh $name" 2>&1 | grep -E 'plan [0-9]|fraction|확실|없음|step6|스윕 충돌|lift|타깃' || true
fi
if ! timeout 30 ssh thor "test -f ~/grasp/${name}_step6_pick.json"; then
    echo "  !! $name : 계획 실패 -- 스킵"
    exit 0
fi

fi

echo "══ [$name] exec pick ══"
rm -f "/tmp/${name}.exec_res.json"
[ "${SNAP:-1}" = "1" ] && python3 snap_objects.py "/tmp/snap_${name}_a.json" >/dev/null 2>&1 || true     # 이웃 영향 측정(시뮬 하네스): pick 전 스냅샷
if [ "${NERO_WARM_EXEC:-0}" = "1" ]; then
    # 2026-10-01 예열 실행기: 호출자(discover_object.py)가 이미 exec_pick 을 띄워 신호를 줬다 -- 끝날 때까지만 기다린다(결과 파일은 호출자가 복사해 둔다)
    for _w in $(seq 1 6000); do [ -f "/tmp/${name}.warm_done" ] && break; sleep 0.05; done
    [ -f "/tmp/${name}.warm_done" ] || echo "  !! $name : 예열 실행기 완료 신호 없음"
else
    EXEC_PLAN_TAG="$CHAIN_TAG" NERO_NO_PHYSICS="${NERO_NO_PHYSICS:-1}" timeout 300 ./exec_multi.sh "$name" || echo "  !! $name : exec 타임아웃/실패(rc=$?)"
fi
V=$(python3 -c "import json;print(json.load(open('/tmp/${name}.exec_res.json')).get('verdict'))" 2>/dev/null || echo "?")
# 2026-09-30 강화 정답의 유지 스냅샷을 이웃 스냅샷보다 먼저(pick 직후) 찍는다: 예전엔 이웃 스냅샷(1~2초) 뒤라 들어 올린 지 2~3초 지난 시점을 쟀다.
if [ "${NERO_GT_HOLD:-0}" = "1" ] && [ -f "/tmp/snap_${name}_a.json" ]; then
    python3 snap_hold.py "/tmp/snap_${name}_hold.json" "${NERO_GT_HOLD_SEC:-1.5}" >/dev/null 2>&1 || true
fi
if [ "${SNAP:-1}" = "1" ] && [ -f "/tmp/snap_${name}_a.json" ]; then
    python3 snap_objects.py "/tmp/snap_${name}_b.json" >/dev/null 2>&1 || true
    SC=$(python3 -c "
import json
used = json.load(open('/tmp/${name}.exec_res.json')).get('plan_used', 0) or 0
c = json.load(open('$HOME/grasp/${name}_step6_pick.json'))['plans'][int(used)]['contact']
print(c[0], c[1], c[2])" 2>/dev/null || true)
    [ -n "$SC" ] && [ -f "/tmp/snap_${name}_b.json" ] && echo "  [$name] 이웃 영향: $(python3 snap_compare.py /tmp/snap_${name}_a.json /tmp/snap_${name}_b.json $SC 2>&1)" || true
fi

# 2026-09-30 강화된 시뮬 정답(NERO_GT_HOLD=1): 들어 올린 직후 + 유지(NERO_GT_HOLD_SEC, 기본 1.5s) 후에도 타깃이 올라가 있고 그리퍼 중앙 근처(손에 쥠)이며 높이가 안정적인가.
# NERO_VISUAL_PAUSE=<초>: 들어 올린 자세에서 그만큼 더 멈춘다(육안 확인용). NERO_VISUAL_FRAME=1: 그때 손목 카메라 프레임을 저장. 모든 pick 은 visual_check.csv 에 기록한다.
GTH=""; V_RAW="$V"
if [ "${NERO_GT_HOLD:-0}" = "1" ] && [ -f "/tmp/snap_${name}_a.json" ]; then
    GTH=$(python3 gt_hold_verdict.py "$name" 2>/dev/null)
    FRAME=""
    if [ "${NERO_VISUAL_FRAME:-0}" = "1" ]; then
        mkdir -p ${NERO_RUNS_DIR:-$HOME/nero_runs}/visual_frames; FRAME="${NERO_RUNS_DIR:-$HOME/nero_runs}/visual_frames/$(date +%m%d_%H%M%S)_${name}.png"
        timeout 15 python3 grab_frame.py "$FRAME" >/dev/null 2>&1 || FRAME=""
    fi
    VPRIM=$(echo "${GTH:-NA}" | awk '{print $5}')
    [ -n "${NERO_VISUAL_PAUSE:-}" ] && { echo "  >>> [$name] 육안 확인 시작: 대상 물체 [$VPRIM] 를 들어 올린 자세에서 ${NERO_VISUAL_PAUSE}초 멈춤 -- 지금 물체를 손에 들고 있습니까?"; sleep "${NERO_VISUAL_PAUSE}"; echo "  <<< [$name] 육안 확인 종료"; }
    echo "  [$name] 강화 정답(유지 ${NERO_GT_HOLD_SEC:-1.5}s): ${GTH:-NA}  (규칙 verdict=$V)"
    mkdir -p ${NERO_RUNS_DIR:-$HOME/nero_runs}
    VCSV="${NERO_VISUAL_CSV:-${NERO_RUNS_DIR:-$HOME/nero_runs}/visual_check.csv}"
    [ -f "$VCSV" ] || echo "time,name,rule_verdict,gt_hold,rise_after_lift,rise_after_hold,dist_to_hand,prim,reason,frame" > "$VCSV"
    echo "$(date +%H:%M:%S),$name,$V,$(echo ${GTH:-NA} | awk '{print $1","$2","$3","$4","$5","$6}'),$FRAME" >> "$VCSV"
fi
# 2026-09-30 시뮬 정답 판정(교차검증): 그리퍼 폭 규칙은 물체가 손 안에서 돌아 눕거나(커피 상자 0.0272->0.0151) 경계값에서 거짓 SLIPPED 를 냈다(줄자는 기록 누락).
# pick 직후 스냅샷에서 타깃 물체(잡은 지점에 가장 가까웠던 강체)가 실제로 들렸는지(z +6cm 이상)를 본다. 규칙과 다르면 경고하고, NERO_GT_VERDICT=1(기본)이면 정답으로 판정을 바로잡는다.
# 시뮬 전용(실물에서는 정답이 없다 -> 그때는 그리퍼 신호로 학습한 판정기가 필요). NERO_GT_VERDICT=0 이면 기록만 한다.
if [ "${SNAP:-1}" = "1" ] && [ -f "/tmp/snap_${name}_a.json" ] && [ -f "/tmp/snap_${name}_b.json" ]; then
    GTR=$(python3 - "$name" <<'GTEOF' 2>/dev/null
import json, math, os, sys
name = sys.argv[1]
try:
    a = json.load(open(f"/tmp/snap_{name}_a.json")); b = json.load(open(f"/tmp/snap_{name}_b.json"))
    used = json.load(open(f"/tmp/{name}.exec_res.json")).get("plan_used", 0) or 0
    c = json.load(open(os.path.expanduser(f"~/grasp/{name}_step6_pick.json")))["plans"][int(used)]["contact"]
    tp = min(a, key=lambda k: math.dist(a[k], c))
    if tp not in b:
        print("NA"); sys.exit(0)
    dz = b[tp][2] - a[tp][2]
    print("%s %.3f %s" % ("HELD" if dz > 0.06 else "NOT", dz, tp))
except Exception:
    print("NA")
GTEOF
)
    if [ -n "$GTH" ] && [ "${GTH%% *}" != "NA" ]; then GTR="$(echo $GTH | awk '{print $1, $3, $5}')"; fi     # 강화 정답이 있으면 그것을 쓴다(verdict, 유지 후 높이 변화, prim)
    case "$GTR" in
        HELD*|NOT*)
            GT_V=$(echo "$GTR" | awk '{print $1}'); GT_DZ=$(echo "$GTR" | awk '{print $2}'); GT_P=$(echo "$GTR" | awk '{print $3}')
            echo "  [$name] 시뮬 정답: $GT_P 높이 변화 ${GT_DZ}m -> $GT_V (규칙 verdict=$V)"
            NEWV=""
            if [ "$GT_V" = "HELD" ] && { [ "$V" = "SLIPPED" ] || [ "$V" = "MISSED" ]; }; then NEWV="HELD"; fi
            if [ "$GT_V" = "NOT" ] && [ "$V" = "HELD" ]; then NEWV="SLIPPED"; fi
            if [ -n "$NEWV" ]; then
                echo "  !! [$name] 판정 불일치: 규칙 $V vs 시뮬 정답 $GT_V"
                if [ "${NERO_GT_VERDICT:-1}" = "1" ]; then
                    python3 -c "
import json
p='/tmp/${name}.exec_res.json'; d=json.load(open(p)); d['verdict_rule']=d.get('verdict'); d['verdict']='$NEWV'; d['gt_dz']=$GT_DZ; json.dump(d,open(p,'w'))" 2>/dev/null || true
                    V="$NEWV"; echo "  [$name] 판정을 정답 기준으로 정정: $V"
                fi
            fi ;;
    esac
fi

# 파지 기록 (place_on_target_one.sh 와 동일 형식 -> /tmp/grasp_log.csv)
python3 - "$name" "$V" <<'PYEOF' &
import csv, json, os, subprocess, sys, time
name, v = sys.argv[1], sys.argv[2]
try:
    used = json.load(open(f"/tmp/{name}.exec_res.json")).get("plan_used", 0) or 0
    pl = json.load(open(os.path.expanduser("~/grasp/%s_step6_pick.json" % name)))["plans"][int(used)]
    r = subprocess.run(["ssh", "thor", f"python3 -c \"import json;b=json.load(open(__import__('os').path.expanduser('~/grasp/{name}_grasps.json')))['box_fit'];print(b['bbmin'][2],b['bbmax'][2])\""],
                       capture_output=True, text=True, timeout=20).stdout.split()
    lo, hi = float(r[0]), float(r[1])
    ratio = (pl["contact"][2] - max(0.0, min(0.03, lo))) / max(hi - min(lo, 0.03), 1e-6)
    row = [time.strftime("%H:%M:%S"), name, v, "%.3f" % pl["contact"][2], "%.3f" % (hi - lo), "%.2f" % ratio,
           "%.0f" % (pl.get("approach_theta") or -1), "%.3f" % (pl.get("opening") or 0), pl.get("kind"), int(used)]
    new = not os.path.exists("/tmp/grasp_log.csv")
    with open("/tmp/grasp_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "gid", "verdict", "contact_z", "obj_h", "contact_ratio", "approach_theta", "opening", "kind", "plan"])
        w.writerow(row)
    print("  [%s] 파지 기록: 접촉 %.3fm / 물체높이 %.3fm = %.0f%%, 접근각 %s도, verdict=%s" % (name, pl["contact"][2], hi - lo, 100 * ratio, row[6], v))
except Exception as e:
    print("  [%s] 파지 기록 실패: %s" % (name, e))
PYEOF
REC_PID=$!

if [ "$V" = "EXEC_OK" ] || [ "$V" = "HELD" ]; then
    echo "  [$name] pick 성공(verdict=$V) -- place 없음: 물체를 치우고 홈 복귀(병렬)"
    # 2026-09-30 속도(오버헤드 단축): (1) 그리퍼 열기는 생략 -- 다음 물체 exec 의 첫 동작이 grip-open 이라 어차피 열린다(release_here 1.1s 제거).
    # (2) 홈 복귀 모션(1.8s)을 파지 기록/물체 치우기와 병렬로 시작. PICK_RELEASE=1 이면 예전처럼 그 자리에서 열고(release_here.py) 진행.
    SKIP_HOME=0
    if [ "${NERO_CHAIN_ENABLE:-0}" = "1" ]; then
        USED_P=$(python3 -c "import json;print(json.load(open('/tmp/${name}.exec_res.json')).get('plan_used',0) or 0)" 2>/dev/null || echo 9)
        if [ "$USED_P" = "0" ]; then
            # chain 계획이 준비될 때까지만 기다린다: 준비 플래그가 생기거나, 선계산이 끝났는데(pending 플래그 없음) 준비가 안 됐으면 즉시 포기(최대 12s)
            for k in $(seq 1 60); do
                [ -f "/tmp/chain_ready_${name}" ] && break
                [ -f "/tmp/chain_pending_${name}" ] || break
                sleep 0.2
            done
            [ -f "/tmp/chain_ready_${name}" ] && SKIP_HOME=1
        fi
    fi
    [ "${NERO_RT_NO_HOME:-0}" = "1" ] && SKIP_HOME=1     # 실시간 루프: 다음 탐색 자세로 직접 이동하므로 홈 복귀를 따로 하지 않는다
    if [ "$SKIP_HOME" = "1" ]; then
        echo "  [$name] 다음 물체 chain 계획 준비됨 -- 홈 복귀 생략"
        HOME_PID=""
    else
        HOME_SETTLE_SPINS=3 python3 home_and_get_armq.py >/dev/null 2>&1 &     # arm_q 를 안 쓰는 백그라운드 호출: 정착 대기 2.0s -> 0.3s
        HOME_PID=$!
    fi
    if [ "${PICK_RELEASE:-0}" = "1" ]; then
        timeout 30 python3 release_here.py 2>&1 | sed "s/^/  [$name] release: /" || true
    fi
    # 시뮬레이션 평가 하네스: 집은 물체를 치워진 것으로 만든다(잡았던 지점 근처의 강체를 작업 영역 밖으로). SIM_REMOVE=0 이면 끔(그 자리에 남음).
    if [ "${SIM_REMOVE:-1}" = "1" ]; then
        CXYZ=$(python3 -c "
import json
used = json.load(open('/tmp/${name}.exec_res.json')).get('plan_used', 0) or 0
c = json.load(open('$HOME/grasp/${name}_step6_pick.json'))['plans'][int(used)]['contact']
print(c[0], c[1], c[2])" 2>/dev/null || true)
        # 타깃 프림은 pick 전 스냅샷에서 잡은 지점과 가장 가까운 강체로 정한다(집은 뒤엔 물체가 들려 올라가 근접 탐색이 이웃을 고를 수 있다 -- 겹침 실험 c3v2 시드 103)
        TPRIM=""
        if [ -f "/tmp/snap_${name}_a.json" ] && [ -n "$CXYZ" ]; then
            TPRIM=$(python3 -c "
import json, math, sys
a = json.load(open('/tmp/snap_${name}_a.json')); c = [float(x) for x in '$CXYZ'.split()]
print(min(a, key=lambda k: math.dist(a[k], c)) if a else '')" 2>/dev/null || true)
        fi
        [ -n "$CXYZ" ] && timeout 40 python3 sim_remove.py $CXYZ $TPRIM 2>&1 | sed "s/^/  [$name] 치움: /" || true
    fi
    if [ "${NERO_RT_PLACE:-0}" = "1" ]; then
        :        # 2026-10-01 place 이어가기: 파지 기록(ssh thor)을 기다리지 않는다(백그라운드로 마저 돈다) -> pick 끝에서 place 시작까지 지연 제거
    else
        wait $REC_PID $HOME_PID 2>/dev/null || true
    fi
else
    echo "  !! $name : pick verdict=$V"
    wait $REC_PID 2>/dev/null || true
    [ "${NERO_RT_NO_HOME:-0}" = "1" ] || timeout 90 python3 home_and_get_armq.py >/dev/null 2>&1 || echo "  !! $name : 홈 복귀 실패"
fi
exit 0
