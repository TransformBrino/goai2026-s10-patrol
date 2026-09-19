# FINALS_PACK_20260920 清单（训练侧 → AGX 侧 / 作者，2026-09-19 14:03）

位置：`~/s10_logs/release/FINALS_PACK_20260920/`（总 74M）。**口径：所有验收表首行均注明MuJoCo/Isaac 仿真结论，真机验收以 AGX 侧证据表为准；已作废的数（轮阻力模型、31° 楼梯、S16 1/6）未出现；09-19 08:55 之后的候选（safe9、S16b/c/d）只归档在 release/，不入包。无新训练。**

## 结构
```
注册任务清单.txt
env
  env/版本.md
  env/conda_list.txt
  env/pip_freeze.txt
evidence
  evidence/01_自激门表.md
  evidence/02_切换成对复测表.md
  evidence/03_高台验收表.md
  evidence/04_楼梯验收表.md
  evidence/05_踏步验收表.md
  evidence/06_Isaac_MuJoCo对拍结论.md
  evidence/07_官方录包量化表.md
  evidence/R13_val_n12
  evidence/src
FINALS_PACK_清单.md
mdp模块清单.txt
policies
  policies/ClimbH_8797.md
  policies/ClimbH_8797.onnx
  policies/ClimbR13.md
  policies/ClimbR13.onnx
  policies/MD5.txt
  policies/safe8_19695.md
  policies/safe8_19695.onnx
  policies/StairN-E2_10300.md
  policies/StairN-E2_10300.onnx
  policies/StairN-E5_11200.md
  policies/StairN-E5_11200.onnx
  policies/StairN-S3_13800.md
  policies/StairN-S3_13800.onnx
  policies/StairN-S8_14800.md
  policies/StairN-S8_14800.onnx
  policies/StairS16_15200.md
  policies/StairS16_15200.onnx
  policies/T10_15197.md
  policies/T10_15197.onnx
  policies/V1H_9196.md
  policies/V1H_9196.onnx
  policies/V1HTrotH4_11500.md
  policies/V1HTrotH4_11500.onnx
  policies/V1HTrotH6_12099.md
  policies/V1HTrotH6_12099.onnx
  policies/V1HTrotH7_11800.md
  policies/V1HTrotH7_11800.onnx
  policies/V1HTrotT3_13400.md
  policies/V1HTrotT3_13400.onnx
training
  training/deeprobotics_s10_percept
  training/mdp
  training/README.md
  training/ref
  training/scripts
  evidence/src/（26 个原始汇总/验收文件，含真机 stairs_2133 逐秒表）  evidence/R13_val_n12/（n=12 原始 state.csv）
```

## 1. training/（2.2M）
- `deeprobotics_s10_percept/`（任务注册 + 全部环境配置类，去 .bak）、`mdp/`（23 模块）、`ref/`（官方参考模板 npz）、`scripts/`（30 个：模板生成/导出/门禁/验收/驱动/热启动）、`README.md`（训练命令、吞吐显存、三条线血缘与热启动检查点、纪律）。

## 2. policies/（14 个 onnx + 一页说明 + MD5.txt）
| 策略 | md5 | 槽位 | 自激门 roll 0/20/30/45 | 真机记录 |
|---|---|---|---|---|
| V1H_9196 | 7fbae6f6 | 主策略槽 S10_POLICY_PATH（AGX 上曾装、真机走过；决赛未使用） | 2.37 / 3.35 / 4.25 / 6.31 ✓（门的参照） | 09-11 t1~t3（起立/原地 RL/0.3 直行/原地转）通过；09-12 台阶 0/3；09-13/09-18 作为退回主策略走平地 |
| V1HTrotT3_13400 | d408c43b | 主策略槽（控制台平地选项 T3；09-14 真机楼梯成功那次的主策略） | 3.55 / **12.76 / 15.48 / 18.36** ✗ 越限位线（09-19 补跑） | 09-14 16:2x 作为主策略配 S8 上真机楼梯两段成功（未遇大侧倾）；同日平地退化为三轮滚 |
| V1HTrotH4_11500 | 8fd5e3b2 | 主策略槽（控制台平地选项 H4） | 1.17 / **12.92 / 15.17 / 18.45** ✗（09-19 补跑） | 无 |
| V1HTrotH6_12099 | 7a17b48a | 主策略槽（控制台平地选项 H6） | 1.29 / **10.21 / 11.91 / 14.27** ✗（09-19 补跑） | 无 |
| V1HTrotH7_11800 | 231c8437 | 主策略槽（控制台平地选项 H7） | 1.67 / **11.27 / 13.99 / 16.93** ✗（09-19 补跑） | 无 |
| T10_15197 | ebef4dd2 | 主策略槽（09-18 交接包 DELIVER_20260918 主策略；真机摔倒后退回 V1H；决赛未使用） | 1.83 / **11.39 / 13.55 / 16.16** ✗ 越限位线 | 09-18 21:24 平地摔倒（AGX 转储 0918_runs/obsdump_20260918_212306）；已退回 V1H |
| StairN-E2_10300 | e0ef6b68 | 楼梯槽 S10_POLICY_PIT（/climb_mode 2，轮缩放 3；AGX 上曾配楼梯槽 A；决赛未使用） | 3.04 / 5.55 / **8.69 / 12.74** ✗ 越限位线（AGX 复核 + 训练机复算） | 09-13 全程专家语义真机试过（未登顶记录） |
| StairN-E5_11200 | 315c942b | 楼梯槽（AGX 楼梯槽 D） | 5.50 / **8.92 / 11.31 / 13.78** ✗ 越限位线 | 09-13 21:24 切入后自走+起步冲+航向保持 → 8 s 侧翻 |
| StairN-S3_13800 | d0ea1578 | 楼梯槽（AGX 楼梯槽 E，能停） | 2.54 / 4.57 / 6.58 / **9.97** ✗（09-19 补跑；增益 6.0） | 无 |
| StairN-S8_14800 | 3a03c3af | 楼梯槽（AGX 楼梯槽 F，stop_ok，默认 0.25） | 2.34 / 1.95 / 2.98 / **6.94** ✓ 压线（增益 2.5） | **09-14 16:24 真机决赛楼梯登顶两段（Δz 3.15 m，agx_in/0914_runs/stairs_1627）**；同日 16:20 中段停 46 s 重启打转侧翻（stairs_1623） |
| StairS16_15200 | faa37c9a | 楼梯槽（09-18 交接包楼梯策略 stairs16_15200.onnx；决赛未使用） | 2.45 / 2.13 / 3.06 / 5.80 ✓ | 09-18 21:34 决赛楼梯脚下在场 10.8 s 未起步——定性：定位抖→高程图崩 + 松键退专家，非策略（见夜间报告-20260919 第一节） |
| ClimbH_8797 | 2abfc737 | 爬墙槽 S10_POLICY_CLIMB（/climb_mode 1；AGX 上曾配爬墙槽；决赛未使用） | **7.62** / 9.76 / 11.56 / 13.92 ✗ 越限位线（增益 3.6，与 T10 同病） | 09-12/14 真机双 Airy 图下上墙正常（AGX 记录） |
| ClimbR13 | c72f75ac | 爬墙槽（09-18 交接包 climb_r13.onnx；决赛未使用） | 3.30 / 2.00 / 1.60 / 2.03 ✓（最好） | 无（09-18 装机后未上石笼） |
| safe8_19695 | eebe7aaa | 候选，**不装机**（246 维，需 runner 相位钟支持 S10_GAIT_FREQ；已在 ~/s10_fork_install 编译，0911i 未合） | 2.17 / 4.71 / 5.70 / 6.73 ✓ | 无 |

## 3. evidence/（7 张表 + src/ 原始文件 + R13 n=12 原始数据）
- `evidence/01_自激门表.md` — 动作自激门（selfexcite_gate.py v2）— 全部已装机/候选策略
- `evidence/02_切换成对复测表.md` — 主策略 × 爬墙专家 成对复测（09-18）
- `evidence/03_高台验收表.md` — 33 cm 高台 / 石笼（爬墙专家 R13）
- `evidence/04_楼梯验收表.md` — 楼梯（楼梯专家）
- `evidence/05_踏步验收表.md` — 踏步 / 平地主策略（三点：0.5 / 1.0 / 1.8 m/s）
- `evidence/06_Isaac_MuJoCo对拍结论.md` — Isaac（训练）↔ MuJoCo（部署栈）对拍结论（09-14）
- `evidence/07_官方录包量化表.md` — 官方运控录包量化（作者 09-10/09-13 录制，七段）

## 4. env/
- `版本.md`（Python/torch/Isaac Sim/IsaacLab commit/rsl_rl/mujoco/onnxruntime/驱动）、`pip_freeze.txt`、`conda_list.txt`

## 5. 清单文件
- `注册任务清单.txt`（178 个 gym.register id，一行一个）、`mdp模块清单.txt`（23 个，不含 .bak）

## 6. 需要 AGX 侧 / 作者注意的口径
- **15:xx 口径更新（作者拍板）**：决赛使用 S10 内置运控；本包自研运控为研发成果、决赛未使用。policies/ 每页与 evidence/ 每表首行已加此声明，「现装/交付」改为「曾装/交接包」。
- 自激门补跑：**V1HTrotT3/H4/H6/H7、S3 全部 ✗**（与 T10 同病）；244 维过门的只有 V1H_9196 / S8 / S16 / R13。初稿 2.x.3 表里 V1HTrotT3_13400 建议标「门 ✗ 不启用」。
- V1H「上墙 98%、宽楼梯 15/16」是 09-11 宽踏面场景的数，请标日期/场景。
- T10 的平地三点数（机身高 0.342、对角同步 65/71/80%）来自 DELIVER_20260918/说明.md；原始 main_accept 目录未在 night/ 找到同名文件，表里已注明出处。
- R13 组合复测 6/6 原始目录名待补（结论在 DELIVER 说明 §③）。
- 246 维 safe8 需 runner 相位钟支持（S10_GAIT_FREQ），0911i 未合入；只作候选。
