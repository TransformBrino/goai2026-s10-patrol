# S10 全地形巡逻 · 决赛技术核查代码包（草稿 2026-09-19，提交前由作者定稿）

队伍：传化具身智能 ｜ 作品：面向山猫 S10 的三维自主巡逻与跨地形导航系统 ｜ 赛题：产业园区全地形巡逻（B） ｜ 参赛模式：自主导航（系数 1.4）

## 1. 系统构成与在役标记（与 09-22 现场一致，冻结 09-21 20:00）
| 模块 | 目录 | 决赛角色 |
|---|---|---|
| 导航层工具（航点工具、/cmd_vel→/STEER 适配、协议探针；本队早期导航层，供核验） | `navigation/` | 研发成果（决赛导航由墨矩系统承担） |
| 感知与建图（双 Airy 点云 → FAST-LIO2 → 局部高程图 187 格） | `perception/`、`deploy_agx/lio/` | 在役（为自研运控与地形判读服务） |
| 运控（在役）：S10 内置运控，导航模式步态 0x3002/0x3003，由导航层经 /NAV_CMD 或 UDP 真实轴指令驱动（《软件开发指南》V1.0.1） | — | **在役，全程** |
| 自研感知驱动 RL 运控（runner 三槽 + 控制台 + 训练代码） | `deploy_agx/`、`training/` | **决赛不使用**（AGX 已停用：进程停止、crontab 注释）；随包提交供核验，研发成果 |
| 训练代码（IsaacLab + rsl_rl，决赛三条线） | `training/` | 研发成果，核验用 |
| 证据（真机逐 2 s 表、事件、门禁、说明） | `evidence/` | 核验用；大录包不入包，清单见 `evidence/清单-部署代码与策略-md5-*.md` |
| 环境依赖 | `env/` | 训练机 conda 导出 + AGX 版本表 |

## 2. 版本
- 提交 tag：`submit-20260920`；冻结 tag：`freeze-20260921`（冻结后如需修复重大缺陷，先向组委会申报）。
- AGX 现行文件 md5 与 `evidence/清单-部署代码与策略-md5-*.md` 对表；`deploy_agx/` 内文件即 AGX `/home/robot/` 现行副本（2026-09-19 14:10 已与 AGX `md5sum` 逐项核对，见清单 §6）。
- runner：官方 `S10_sdk_deploy` 二次开发版 0911i（`deploy_agx/runner/`，third_party 与官方仓库一致未复制）。

## 3. 复现（摘要，详见《档案》五）
1. 训练：conda `s10_train`（`env/`），`TASK=PerceptS10-<X>-v0 bash training/scripts/s10train.sh 4096 <iters>`；导出 `python training/scripts/export244.py <pt> <onnx>`。
2. 仿真验收：MuJoCo 部署栈（官方仓库 + `deploy_agx/runner` 的 simulation 扩展）`run_bench_linux.sh`、`training/scripts/stair_turn_accept_cap.sh`、`main_accept.sh`。
3. 门禁：`python training/scripts/selfexcite_gate.py <onnx> --pass 7`。
4. 真机：AGX `bash boot_up.sh` 起全链；控制台 `ssh -L 8089:localhost:8089`；装策略前核 md5。

## 4. 第三方组件（按原许可证引用，不复制）
云深处 S10 内置运控（闭源，决赛在役）、墨矩导航系统（闭源，决赛在役）、官方 goai_embodied_future_material（S10_sdk_deploy、dual_airy_merger、drdds）、rslidar_sdk 1.5.19、FAST-LIO2（BSD，ROS2 分支 commit a4743b0）、IsaacLab（BSD-3）、rsl_rl（BSD-3）、onnxruntime（MIT）、MuJoCo（Apache-2.0）。版本号见 `env/`。

## 5. 开源范围
自研导航、感知、运控训练与部署代码对评委开放（仓库仅评委可见）；训练检查点与大录包按需提供。
