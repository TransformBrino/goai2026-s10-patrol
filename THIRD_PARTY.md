# 第三方组件、数据与资源声明（2026-09-20）

按赛事要求登记本队材料实际使用的第三方软件、SDK、数据及资源：来源、许可证、用途、是否在决赛在役。本包只包含本队自研代码、本队二次开发的源码快照与 diff、本队产生的数据；第三方组件按原许可证引用，不复制其发行包。

| 组件 | 来源 / 许可证 | 本包中的使用方式 | 决赛在役 |
|---|---|---|---|
| S10 内置运控（rl_deploy）与本体协议 | 云深处，随机器人出厂，闭源 | 决赛在役运控；本包不含其代码 | 是 |
| 墨矩导航系统 | 第三方合作方，闭源；经合作方授权在本队参赛系统中使用 | 决赛在役导航定位；内部算法不随包，版本与接口写入赛前冻结记录并在 docs/ 二(六) 披露 | 是 |
| S10_sdk_deploy（runner 官方框架）、drdds、dual_airy_merger 0.1.0 | 组委会 / 云深处提供给参赛队的官方材料（goai_embodied_future_material） | runner 本队二次开发 +2130 行，源码与 patch 随包（deploy_agx/） | 否（自研栈决赛停用） |
| rslidar_sdk 1.5.19 | 速腾聚创，BSD-3 | 雷达驱动，仅改编译选项与配置（deploy_agx/config/） | 感知链随自研栈停用 |
| FAST-LIO2（hku-mars，ROS 2 分支 commit a4743b0） | GPL-2.0 | 本队二次开发 +96 / −18 行；源码快照与 diff 随包（perception/FAST_LIO_src），遵循 GPL-2.0 | 否 |
| Isaac Sim 6.0.1 / IsaacLab 6.1.17（commit 6a7acb0） | NVIDIA 许可 / BSD-3 | 训练环境，不随包；版本见 env/版本.md | 否 |
| rsl_rl 5.4.2 | ETH Zurich，BSD-3 | PPO 训练库，不随包 | 否 |
| MuJoCo 3.12.0 | Apache-2.0 | 自建仿真验收台（training/scripts、deploy_agx 仿真接口） | 否 |
| onnxruntime 1.28.0（训练机）/ 1.22.0（runner 内置） | MIT | 策略推理 | 否 |
| 官方仿真地图、路线、点位、赛事场地资源 | 组委会提供 | 按赛事规定使用 | 是 |
| 本队录制的 S10 内置运控实录录包（evidence/、7 段） | 本队数据 | 官方步态量化对标（evidence/sim/07） | — |
| 字体：OPPO Sans（路演 PPT，官方模板自带）、Noto Sans CJK（文档与视频） | OPPO 免费商用授权 / SIL OFL 1.1 | 排版 | — |

自研代码开放范围见 README「开源范围」；第三方依赖的版本号以 env/版本.md、env/AGX版本.md 为准。
