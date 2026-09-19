#!/bin/bash
# 09-12 19:1x：把 0911i 的 3 个源文件（= 0911h 两处 + i 一处，含 hipx）装到 AGX 现行源码上并重编。作者拍板"装"。
set -e
P=$HOME/s10_logs/release/s10_runner_20260911i/s10_ws/src/S10_sdk_deploy
K="-i $HOME/.ssh/agx_s10 -o BatchMode=yes -o ConnectTimeout=8 -o HostKeyAlias=10.21.33.102"
AGX=robot@10.18.2.240
F1=state_machine/quadruped_wheel/rl_control_state.hpp
F2=state_machine/parameters/s10_control_parameters.cpp
F3=run_policy/s10_policy_runner.hpp
ssh $K $AGX 'rm -rf /tmp/stage_0911i && mkdir -p /tmp/stage_0911i/state_machine/quadruped_wheel /tmp/stage_0911i/state_machine/parameters /tmp/stage_0911i/run_policy'
( cd $P && tar -cf - $F1 $F2 $F3 ) | ssh $K $AGX 'tar -xf - -C /tmp/stage_0911i'
ssh $K $AGX 'bash -s' <<'REMOTE'
set -e
W=/home/robot/s10/s10_ws/src/S10_sdk_deploy; G=/tmp/stage_0911i
BIN=/home/robot/s10/s10_ws/install/s10_sdk_deploy/lib/s10_sdk_deploy/rl_deploy
F1=state_machine/quadruped_wheel/rl_control_state.hpp; F2=state_machine/parameters/s10_control_parameters.cpp; F3=run_policy/s10_policy_runner.hpp
m() { tr -d '\r' < "$1" | md5sum | cut -c1-8; }
echo "== 前置"
pgrep -x rl_deploy >/dev/null && { echo "!! 有 rl_deploy，停止"; exit 3; }
A=$(curl -s -m 3 http://127.0.0.1:8089/api/state | python3 -c "import sys,json; print(json.load(sys.stdin)['state']['armed'])" 2>/dev/null || echo "?")
[ "$A" = False ] || { echo "!! 控制台已接管/读不到（$A），停止"; exit 3; }
[ "$(m $G/$F1)" = bc4e1d78 ] && [ "$(m $G/$F2)" = ca7111a2 ] && [ "$(m $G/$F3)" = 03a4df58 ] || { echo "!! 暂存文件 md5 不符"; exit 4; }
echo "  现行: $F1=$(m $W/$F1) $F2=$(m $W/$F2) $F3=$(m $W/$F3) 二进制=$(md5sum $BIN | cut -c1-8)"
[ "$(md5sum $BIN | cut -c1-8)" = 29e579b2 ] || { echo "!! 现行二进制不是 0911g-hipx 的 29e579b2"; exit 6; }
echo "== 备份（*.pre_0911i，二进制 rl_deploy.0911g-hipx.bak）"
cp -a $W/$F1 $W/$F1.pre_0911i; cp -a $W/$F2 $W/$F2.pre_0911i; cp -a $W/$F3 $W/$F3.pre_0911i
cp -a $BIN /home/robot/rl_deploy.0911g-hipx.bak
echo "== 替换"
cp $G/$F1 $W/$F1; cp $G/$F2 $W/$F2; cp $G/$F3 $W/$F3
[ "$(m $W/$F1)" = bc4e1d78 ] && [ "$(m $W/$F2)" = ca7111a2 ] && [ "$(m $W/$F3)" = 03a4df58 ] || { echo "!! 替换后 md5 不符"; exit 7; }
grep -q "\-0.6109, -2.443, -2.7227" $W/$F2 && echo "  参数：hipx ±0.6109、膝 ±2.7227 ✓"
echo "== 编译 $(date +%T)"
export PATH=$(echo $PATH | tr ':' '\n' | grep -v conda | paste -sd:)
source /opt/ros/jazzy/setup.bash; source /home/robot/s10_env.sh; export ROS_DOMAIN_ID=0
cd /home/robot/s10/s10_ws; set +e; T0=$(date +%s)
nice -n 19 colcon build --packages-select s10_sdk_deploy --parallel-workers 3 --cmake-args -DBUILD_PLATFORM=arm -DCMAKE_BUILD_TYPE=Release > /home/robot/build_runner_0911i.log 2>&1
RC=$?; set -e
tail -n 2 /home/robot/build_runner_0911i.log; echo "BUILD_EXIT=$RC 用时 $(( $(date +%s) - T0 )) s"
[ $RC -eq 0 ] || { grep -n "error" /home/robot/build_runner_0911i.log | head -5; exit 8; }
echo "  二进制 $(stat -c '%y' $BIN | cut -c1-16) md5 $(md5sum $BIN | cut -c1-8)"
for s in S10_AS_WHEEL_PIT S10_AS_WHEEL_CLIMB "[RL_FB_STALE]" "0911h" "[STAND_GUARD]"; do echo "  grep -c \"$s\" = $(grep -ac -F "$s" $BIN)"; done
rm -rf /tmp/stage_0911i
REMOTE
