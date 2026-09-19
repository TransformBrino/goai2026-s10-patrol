#!/usr/bin/env bash
# 官方栈上，从沟内直线冲出口 —— 对齐训练环境里成功那次的条件
# （固定正东、恒定速度），排除纯追踪航向动态的干扰。
set +u
D=/mnt/c/Users/86156/Desktop/机器狗项目/s10_dev
export S10_CMD_MAX_VX=3.5
export S10_START_POS="11.00,32.60,0.60"     # 沟内西端，地面≈0.15
run() {  # $1=标签 $2=vx
  bash "$D/run_bench.sh" climb S10_track "/tmp/to_$1_$2.json" \
       --speeds "$2" --climb-time 20 --risers 0.38 > "/tmp/to_$1_$2.log" 2>&1
  printf "  %-8s vx=%s : " "$1" "$2"
  grep -a "最远" "/tmp/to_$1_$2.log" | tail -1 | sed 's/.*eval_driver.\]: //'
}
export S10_POLICY_PATH="$D/train/policy_spec36.onnx"
for v in 1.8 2.0 2.2; do run 专家 "$v"; done
unset S10_POLICY_PATH
for v in 2.0 2.2; do run 基线 "$v"; done
echo
echo "对照：训练环境里同样条件下 专家vx2.0 -> x=42.09（翻出去了），基线 -> 12.80（卡住）"
