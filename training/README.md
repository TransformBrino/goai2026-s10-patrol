# training/ — 决赛三条线（踏步 / 楼梯 / 高台）训练代码与复现说明

> 口径：本目录为训练侧源码与脚本快照（2026-09-19，去 .bak）。**决赛使用 S10 内置运控，本目录的自研运控为研发成果、决赛未使用。**所有验收数为 MuJoCo/Isaac **仿真结论**，真机验收以 AGX 侧证据表为准。

## 目录
- `deeprobotics_s10_percept/` — 任务注册（`__init__.py`，178 个 `gym.register`，见 `../注册任务清单.txt`）与环境配置 `s10_percept_env_cfg.py`（三条线全部配置类都在此文件，类文档串记录每一轮的单变量改动、判据与结果）、`agents/rsl_rl_ppo_cfg.py`。
- `mdp/` — 奖励/观测/事件/课程模块（23 个，见 `../mdp模块清单.txt`）。三条线关键模块：`climb_rewards.py`（高台相位纪录奖励、遭遇/上台面判定、爬高势函数）、`climb_ref.py` / `climb_cmd.py`（官方高台参考模板）、`gait_clock.py`（踏步相位钟、摆动跟踪、大动作罚）、`trot_ref.py`（官方踏步参考）、`heading_hold.py`（航向罚）、`stair_step.py`（楼梯跟坡/后轮顶踢面）、`percept_noise.py` + `percept_noise_core.py`（高程图腐蚀 DR：遮挡/偏移/前方沟/幻墙/整帧混沌）。
- `ref/` — 官方录包量化出的参考模板（`ref_trot.npz` 踏步、`ref_climb34*.npz` 高台、`ref_stairwalk_official.npz` 楼梯等）。运行时按绝对路径 `~/s10_logs/ref/*.npz` 读取（`trot_ref.py` / `climb_ref.py` / `climb_cmd.py`），复现时请把本目录拷到 `~/s10_logs/ref/`。
- `scripts/` — 模板生成在 `scripts/ref_builders/`（`parse_seg_bag.py` 录包→npz、`build_trot_ref.py`、`build_climb_ref.py`、`make_ref50.py`、`build_official_ref.py`、`build_stairwalk_ref2.py`、`stage_eval.py`、`pofficial.py`，对照下方「ref/ 可追溯性」表）、导出（`export244.py` / `export246.py`，权重内嵌单文件 ONNX）、门禁（`selfexcite_gate.py`）、验收（`stair_turn_accept_cap.sh` 楼梯 6 条、`stair_turn_accept.sh` 30 条、`main_accept.sh` 主策略平地/坡/粗糙、`one_round.sh` + `r21_b3a_launch.snap.sh` 高台 200 点一轮、`trotsafe_launch.sh` 246 线冒烟→训练→导出→门→平地、`s16_entry_probe.sh` 楼梯入口复现矩阵）、仿真驱动（`run_stair_switch.sh`、`switch_stair_driver.py`、`g_verify_driver.py`、`stair_outcome.py`、`mjfk.py`）、Isaac 直测 `physx_climb_eval.py`、热启动 `mkwarmstart.py` / `warmstart246.py`、`s10train.sh`。
  仿真验收还依赖 `s10_ws`（MuJoCo 桥 `mujoco_simulation_ros2_airy.py`、场景 `mjcf/bench_*.xml`）与 runner 0911i（AGX 侧包），不在本目录。

## 环境
见 `../env/版本.md`、`pip_freeze.txt`、`conda_list.txt`：conda `s10_train`（Python 3.12.13、torch 2.11+cu130、Isaac Sim 6.0.1、IsaacLab 6.1.17 源码 commit 6a7acb0、rsl-rl-lib 5.4.2、mujoco 3.12、onnxruntime 1.28），RTX 5080 16 GB、驱动 595.84。
`rl_training` 以 editable 方式安装（`pip install -e rl_training-main`），资产 `DEEPROBOTICS_S10_DEPLOY_CFG`（几何/质量/默认姿态/PD/力矩与 runner 逐位一致，hipx 外八 fl+0.05 fr−0.05 hl−0.05 hr+0.05，腿 kp80/kd2，轮 kd0.6~0.8，力矩 50/14）。

## 训练命令（通用）
```bash
cd rl_training-main
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/s10_train/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
# 冒烟（每轮必做）：16 env × 2 迭代，检查 banner 打印、无 Traceback、in_features 维数
python scripts/reinforcement_learning/rsl_rl/train.py --task <任务id> --num_envs 16 --headless --resume --load_run <run目录名> --checkpoint model_N.pt --max_iterations 2
# 正式
python scripts/reinforcement_learning/rsl_rl/train.py --task <任务id> --num_envs 8192 --headless --resume --load_run <run> --checkpoint model_N.pt --max_iterations 200
```
`--max_iterations` 为**增量**；日志里 `Learning iteration` 是累计绝对值；检查点 `logs/rsl_rl/deeprobotics_s10_percept/<时间戳>/model_*.pt`，每次 `--resume` 新建时间戳目录。每轮结束：`export244.py <pt> <onnx>` → `selfexcite_gate.py <onnx> --pass 7`（硬门）→ 对应验收脚本。**评测与训练不并行**；长任务用 `setsid nohup … &`。

## 吞吐 / 显存（RTX 5080）
| num_envs | s/迭代 | 显存 |
|---|---|---|
| 1024 | 1.7 | 5.0 GB |
| 4096 | 3.7 | 6.9 GB |
| 8192 | 5.7 | 9.3 GB |
200 迭代 @8192 ≈ 19 min；600 迭代 @4096 ≈ 37 min。

## 三条线：任务 id、血缘、迭代、热启动
### 踏步 / 平地主策略（8192 env）
`PerceptS10-V1H-v0`（V1 感知 33 cm 课程 → V1H，checkpoint 9196，AGX 上曾装的主策略，决赛未使用）→ `V1HTrotH*`（H4 model_11500 / H6 12099 / H7 11800：几何净空步态项 + 官方踏步参考跟踪）→ `V1HTrotT1~T3`（T3 model_13400，09-14 真机主策略）→ `T6/T7/T9`（执行器延迟/PD/接触随机化、左右反相、对角同步）→ `PerceptS10-V1HTrotT10-v0`（model_15197，姿态罚按坡面法向；09-18 交接包、真机自激摔倒）→ 246 维相位钟线 `PerceptS10-TrotPhaseT10-v0`（`warmstart246.py` 把 actor 244→246 / critic 247→249 补零列，lr8 model_17192）→ `PerceptS10-TrotPhaseT10Safe-v0`（safe3~safe8：腿 12 维上一动作噪声 ±1.0、大动作罚、终止罚 w=25000、倾斜 DR；safe8 model_19695，门 ✓、yaw 未达，**不装机**）。环境变量：`S10_GAIT_FREQ`、`S10_TROTP_W/YAWW/LRW`、`S10_SAFE_ACLIP/BIGW/BIGLIM/TILT/PUSHROLL/ACTNOISE/ACTNOISE_WHEEL/TERMW`。
### 楼梯专家（4096 env）
`V1Stair_12694` → `PerceptS10-StairN-v0`（窄踏面 0.30/0.35/0.40、立面 0.05~0.18、只朝前 0.2~0.6、航向锁）→ `StairNE-v0`（E2 model_10300，官方四拍步行参考）→ `StairNE3-v0`（E5 model_11200）→ E7/E8（听指令）→ S1~S5 → `PerceptS10-StairNS8-v0`（**S8 model_14800**，run 2026-09-14_12-45-52：去掉速度贴档门控、爬高 100/50、超速罚；真机 09-14 登顶）→ S9~S14_15000 → `PerceptS10-StairS16-v0`（**S16 model_15200**，run 2026-09-15_11-54-08，从 S14_15000 续 600：机身跟坡项去相位门；09-18 交接包）。`StairS16b-v0`（09-19 入口鲁棒化 DR，门未过，只归档）。
### 高台 / 石笼专家（8192 env）
`V1` → `PerceptS10-ClimbH-v0`（ClimbH model_8797，09-11 包）→ C 系（C23_13600 落平/抬头预算）→ B2 系 → `PerceptS10-ClimbB2w-v0`（爬高奖减半，`S10_B2W_PROG=150 S10_B2W_REC=100`）以 200 迭代为一轮累积（R1~R16），起点 R9 峰值 `2026-09-18_03-57-36/model_18580`，**R13 = `2026-09-18_05-31-53/model_18779`**（n=12 上台面 12/12、站住 12/12；09-18 交接包 climb_r13.onnx）。R14~R16 均劣于 R13，停止累积。
配置类文档串里逐轮记录了"改了什么、判据、读数、触发条件"（例：`DeeproboticsS10StairExpertS16EnvCfg`、`DeeproboticsS10ClimbB2wEnvCfg`、`DeeproboticsS10TrotPhaseT10SafeEnvCfg`）。

## 奖励设计纪律（节选，全部在类文档串与 `mdp/` 注释中）
均值差类项一律触地门控；瞬时相位豁免只用相位做条件；新罚项开训前先验读数并进"每局收益价目表"比价；逐条目平方和会把被罚量摊开（用线性）；每局总奖励为负 + 终止不罚 → "早摔早解脱"（IsaacLab 每步奖励 = w×value×dt，终止罚权重需 ÷dt）；n=1 排序不可信，交付前换种子 n≥12；专家验收必须按"主策略 + 专家"成对复测。

## ref/ 可追溯性（AGX 14:5x 要求：每个 npz 的来源录包与生成脚本）
| 产物 | 生成脚本（`scripts/ref_builders/`） | 输入 | 来源录包（作者录制的官方运控） | 训练里谁用 |
|---|---|---|---|---|
| `ref_trot.npz` | `build_trot_ref.py` | `实地数据/parsed/{trot_slow,trot_mid,trot_fast,trot_rough}.npz`（`parse_seg_bag.py` 解析） | 09-09/10 官方踏步录包 gftbzx1 / gftb1（200 Hz） | `mdp/trot_ref.py`（踏步参考跟踪） |
| `ref_climb34.npz` | `build_climb_ref.py parsed/climb33_2.npz [climb33_1..5] --wall_h 0.34` | `实地数据/parsed/climb33_1..5.npz`（`parse_seg_bag.py`） | 09-10 官方高台 33 cm 六次事件（s10_seg_0910_2104/23gaotai） | `mdp/climb_cmd.py` |
| `ref_climb34_50hz.npz` | `make_ref50.py`（200 Hz → 50 Hz、关节改仿真序） | `ref_climb34.npz` | 同上 | `mdp/climb_ref.py`（高台相位参考） |
| `ref_stairwalk_official.npz`、`ref_step_official*.npz` | `build_official_ref.py` / `build_stairwalk_ref2.py`（会话 scratchpad 快照） | `~/s10_logs/night/official_gait/{gfstj1,…}.csv`（官方录包解析成的逐帧 CSV；CSV 共 34 MB 未入包，留在训练机原处；`ref/official_gait/单位说明.md` 已入包，需要 CSV 可另提供） | 09-10 gfstj1 / 09-13 flat_2208（楼梯） | 不进训练；用于楼梯线与官方的动作对比表（evidence/07） |
| `ref_seg0_mirror.npz`、`official_seg0/1.state.csv` | `stage_eval.py`（scratchpad 快照） | 高台官方段 0/1 | 同 23gaotai | 不进训练；高台阶段评估/对比 |
（`ref_stair.npz`、`ref_step_merged.npz` 为更早的分析中间件，生成脚本已不可追溯，**已从本包移除**，训练与证据表均不依赖。）
录包原件：AGX `/home/robot/s10_data/`；训练机副本 `~/桌面/赵文博/实地数据/s10_bags_20260909`、`~/桌面/赵文博/s10_seg_0910_2104`、`~/s10_logs/agx_in/flat_22xx`。
