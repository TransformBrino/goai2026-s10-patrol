#!/usr/bin/env bash
# S16b：等 s16b.go（Y25 重跑结束）→ 冒烟 → 600 点（4096 env，从 S16 15200）→ export244 → 自激门 → 30 条验收 → 复现矩阵 → 站住探针
set -u; R=/home/robot/桌面/赵文博/S10决赛交付包-20260908; PY=$HOME/miniconda3/envs/s10_train/bin/python; NV=$HOME/miniconda3/envs/s10_train/lib/python3.12/site-packages/nvidia/cu13/lib
TRAIN='^/home/robot/miniconda3/envs/s10_train/bin/python scripts/reinforcement_learning/rsl_rl/train.py'
until [ -f $HOME/s10_logs/night/s16b.go ]; do sleep 30; done; cd $R; while pgrep -f "$TRAIN" >/dev/null 2>&1; do sleep 15; done
TAG=s16b; LR=2026-09-15_11-54-08; CK=model_15200.pt; ITERS=${ITERS:-600}
S=$HOME/s10_logs/smoke_$TAG.log; echo "[$(date +%T)] $TAG 冒烟"
( cd rl_training-main && env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" $PY scripts/reinforcement_learning/rsl_rl/train.py --task PerceptS10-StairS16b-v0 --num_envs 16 --headless --resume --load_run $LR --checkpoint $CK --max_iterations 2 > $S 2>&1 )
OK=1; grep -q Traceback $S && { echo "❌ Traceback"; OK=0; }; grep -q "stairN-S16b" $S || { echo "❌ 无 banner"; OK=0; }
[ "$(grep -ac 'Learning iteration' $S)" -ge 2 ] || { echo "❌ 没跑完迭代"; OK=0; }; [ $OK -eq 0 ] && { grep -a -A8 Traceback $S | tail -12; exit 1; }
grep -m1 "stairN-S16b" $S | cut -c1-220; echo "✅ 冒烟过 → $ITERS 点"
( cd rl_training-main && env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" $PY scripts/reinforcement_learning/rsl_rl/train.py --task PerceptS10-StairS16b-v0 --num_envs 4096 --headless --resume --load_run $LR --checkpoint $CK --max_iterations $ITERS > $HOME/s10_logs/train_$TAG.log 2>&1 )
L=$HOME/s10_logs/train_$TAG.log; echo "[$(date +%T)] 训练结束"
for k in ss_pitch_slope ss_hind_push ss_yaw_ref track_lin_vel_xy_exp; do printf "  %-22s %s\n" $k "$(tac $L | grep -a -m1 -oP "Episode_Reward/$k: *\K-?[0-9.]+")"; done
RUN=$(ls -t rl_training-main/logs/rsl_rl/deeprobotics_s10_percept | head -1); CKP=$(ls -t rl_training-main/logs/rsl_rl/deeprobotics_s10_percept/$RUN/model_*.pt | head -1)
echo "NEXT $RUN $(basename $CKP)"; O=$HOME/s10_logs/release/StairS16b_$(basename $CKP .pt); mkdir -p $O
env LD_LIBRARY_PATH="$NV:${LD_LIBRARY_PATH:-}" $PY s10_dev/export244.py "$CKP" "$O/s16b.onnx" 2>&1 | tail -1; md5sum $O/s16b.onnx | cut -c1-8
echo "######## 自激门 ########"; $PY s10_dev/selfexcite_gate.py $O/s16b.onnx --pass 7 2>/dev/null; echo "  门退出码 $?"
echo "######## 30 条验收（S8 同口径）########"; PIT=$O/s16b.onnx OUT=$O/accept30 bash s10_dev/stair_turn_accept_cap.sh 2>&1 | tail -34
echo "[$(date +%T)] s16b 训练+30 条 完成"; touch $HOME/s10_logs/night/s16b_eval.go
