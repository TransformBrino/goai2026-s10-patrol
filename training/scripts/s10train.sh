#!/usr/bin/env bash
# S10 训练启动器（新机版，替代 v9train.sh）。
# 用法: s10train.sh [nenv] [niter]      环境变量: TASK（默认 PerceptS10-V0-v0）、RUN/CK（续训起点，默认热启动 seed_vendor0904/model_0.pt）
#   例: bash s10_dev/s10train.sh 1024 4000
#       TASK=PerceptS10-V1-v0 RUN=2026-09-09_xx CK=model_4000.pt bash s10_dev/s10train.sh 1024 4000
set +u
source "$(dirname "$0")/s10_env.sh"
NENV=${1:-1024}; NITER=${2:-4000}; TASK=${TASK:-PerceptS10-V0-v0}
LOGROOT=$S10_RT/logs/rsl_rl/deeprobotics_s10_percept
RUN=${RUN:-seed_vendor0904}; CK=${CK:-model_0.pt}
[ -f "$LOGROOT/$RUN/$CK" ] || { echo "!! 缺 $LOGROOT/$RUN/$CK"; exit 1; }
# 09-15 02:2x：判据从"命令行含这串字"改成"进程名真的是它"。
#   原写法 ps -eo cmd | grep 会匹配到**任何提到这些名字的 shell**——我的常驻看门狗监控命令里含
#   "rl_deploy"，于是从它启动起，所有训练都再也起不来（S14 连撞两次"评测在跑"）。
#   现在用 comm 字段过滤：训练必须是 python* 进程，仿真必须是名为 rl_deploy 的进程。
ps -eo comm,args --no-headers | grep -qE "^python[0-9.]*[[:space:]].*rsl_rl/train\.py" && { echo "!! 已有训练在跑"; exit 1; }
{ pgrep -x rl_deploy >/dev/null || ps -eo comm,args --no-headers | grep -qE "^python[0-9.]*[[:space:]].*physx_climb_eval\.py"; } && { echo "!! 评测在跑，等它结束再开训（评测不与训练并行）"; exit 1; }
STAMP=$(date '+%m%d_%H%M'); LOG=$S10_LOGS/train_s10_${STAMP}.log
echo "task=$TASK resume=$RUN/$CK nenv=$NENV niter=$NITER log=$LOG"
echo "s10 $TASK $RUN $CK $NENV $NITER $LOG $(date +%s)" > $S10_LOGS/.s10_seat
cd "$S10_RT" || exit 1
# 09-16：训练是后台起的，调用方拿不到真实退出码（Astra 首轮复核第 4 条）。
# 包一层子 shell，结束时把**本进程的**退出码写到 "$LOG.rc"，并落 pid 便于核对。
setsid bash -c '"$0" -u scripts/reinforcement_learning/rsl_rl/train.py "$@" > "'"$LOG"'" 2>&1 < /dev/null; echo $? > "'"$LOG"'.rc"' "$S10_PY" \
  --task "$TASK" --num_envs "$NENV" --max_iterations "$NITER" --headless \
  --resume --load_run "$RUN" --checkpoint "$CK" &
echo $! > "$LOG.pid"
sleep 45; ps -eo cmd | grep -qE "rsl_rl/[t]rain\.py" || { echo "!! 45 秒内退出了:"; tail -20 "$LOG"; exit 1; }
echo "已启动。看进度: grep -aoE 'Learning iteration [0-9]+/[0-9]+' $LOG | tail -1"
echo "run 目录: $(ls -td $LOGROOT/2026-* | head -1)"
