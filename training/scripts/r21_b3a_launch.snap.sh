#!/usr/bin/env bash
# R21 = B3a（前腿撑 -300 + 推起段后腿不许被压垮 -150），起点与 R20 同为 R13 ckpt。
# 先等 R20 训完（不抢 GPU），再 16env×2it 冒烟，通过才交给 one_round.sh 跑 200 点。
set -u
R=/home/robot/桌面/赵文博/S10决赛交付包-20260908
PY=$HOME/miniconda3/envs/s10_train/bin/python
NV=$HOME/miniconda3/envs/s10_train/lib/python3.12/site-packages/nvidia/cu13/lib
TRAIN='^/home/robot/miniconda3/envs/s10_train/bin/python scripts/reinforcement_learning/rsl_rl/train.py'
while pgrep -f "$TRAIN" >/dev/null 2>&1; do sleep 15; done
echo "[$(date '+%H:%M:%S')] R20 已结束，开始 B3a 冒烟"
S=$HOME/s10_logs/smoke_b3a.log
( cd $R/rl_training-main && env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" \
  $PY scripts/reinforcement_learning/rsl_rl/train.py --task PerceptS10-ClimbB3a-v0 \
  --num_envs 16 --headless --max_iterations 2 > $S 2>&1 )
# 判据三条全要：无 Traceback、banner 打了、真的走完 2 次迭代。
# （上一版写成「有 Traceback 但也有 banner 就算过」—— 崩了照样放行，白跑一轮，记此）
OK=1
grep -q "Traceback" $S && { echo "❌ 冒烟有 Traceback"; OK=0; }
grep -q "s10-climb-B3a" $S || { echo "❌ 冒烟没打 B3a banner（配置没生效）"; OK=0; }
grep -qE "Learning iteration 1/2|Learning iteration 2/2" $S || { echo "❌ 冒烟没跑完迭代"; OK=0; }
if [ $OK -eq 0 ]; then echo "--- 冒烟日志末尾 ---"; tail -30 $S; exit 1; fi
grep -m1 "s10-climb-B3a" $S; grep -m1 "s10-climb-B2z" $S
echo "✅ 冒烟通过 → 进 200 点"
exec bash $HOME/s10_logs/one_round.sh PerceptS10-ClimbB3a-v0 2026-09-18_05-31-53 model_18779.pt r21_b3a 200 2
