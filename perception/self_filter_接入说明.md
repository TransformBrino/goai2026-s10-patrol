# 雷达自遮挡滤波 · 接入说明（训练侧 2026-09-11）

## 为什么
训练里高程扫描只打地面，机器人本身不是射线目标，机身下方和四条腿周围的格子读的都是地面。
真机雷达会打到自己的腿和轮子，在高程图里变成脚边的假障碍（AGX 09-11 实测：x=+0.2、y≈−0.2~−0.3 两格读约 38 cm，是右前腿和轮子）。
这在训练分布之外，策略可能以为脚边有个坎。**滤掉之前不要用高程图跑 RL。**

## 做法
`self_filter.py`（纯 numpy，不依赖 ROS、MuJoCo）按关节角做正运动学，算出机身、每条腿的 hipx、大腿、小腿和轮子的位置。
每一节用一个包围体：默认取 S10.xml 视觉网格的有向包围盒，轮子取网格外径的圆柱，再外扩 3 cm。落在任一包围体里的点删掉。
不用固定的机身框：前轮顶到台阶时台阶立面就在 x≈+0.31，固定框会把策略最需要看的立面一起删掉。
关节角换算：`/JOINTS_DATA` 与 `/JOINTS_DATA_10HZ` 的 position 是发布值，滤波内部按 `raw = pub·DIR + OFFSET` 换成模型角，DIR 和 OFFSET 与 runner、MuJoCo 仿真桥、AGX 控制台完全一致；hipy、knee 折回 (−π, π]，等效于 runner 开机时的整圈修正。

## 训练侧验证（MuJoCo）
- 正运动学：500 组随机关节角，经发布值换算，与 MuJoCo 真值最大误差 1e-15。
- 射线实测：两台 Airy 按 09-11 修正后的装法（前雷达在 x=+0.22341 圆顶朝前，后雷达在 x=−0.22341 朝后；半球 1°×1°；盲区 0.1 m），打机器人视觉网格加场景。20 个姿态：0.42 站姿、0.35 蹲姿、4 组随机腿姿、一次真实 33 cm 上墙过程中的 14 帧（含前轮贴台阶）。

| 包围体 | 外扩 | 自身点删掉 | 世界点误删（全部 / 0.6 m 内） | 假障碍格 原始→滤后 | 丢格 | 读低 | 耗时 |
|---|---|---|---|---|---|---|---|
| 网格包围盒 | 2 cm | 100% | 0.15% / 0.67% | 241 → 0 | 2 | 0 | 3.8 ms |
| **网格包围盒（默认）** | **3 cm** | **100%** | **0.53% / 2.3%** | **241 → 0** | **11** | **2** | **3.7 ms** |
| 网格包围盒 | 4 cm | 100% | 1.15% / 5.0% | 241 → 0 | 18 | 3 | 3.7 ms |
| 碰撞体 | 3 cm | 100% | 0.26% / 1.1% | 241 → 0 | 4 | 1 | 4.2 ms |

"假障碍格"是 20 个姿态合计、比真值高出 5 cm 以上的格子；"丢格"是真值有点、滤后变空的格子，滚动窗口的累积图会从其他帧补上；"读低"是滤后比真值低 5 cm 以上的格子。耗时是本机约 2.7 万点一帧。
默认取网格包围盒加 3 cm：它包住真实外壳，多出的 1 cm 用来吸收真机雷达噪声（约 1.5 cm）和 10 Hz 关节角的滞后。漏掉一个自身点比多丢几格代价大得多。

## 接入 hmap_node.py（按 AGX 13:39 那版的结构写）
1. 把 `self_filter.py` 放到 `hmap_node.py` 同目录。先在 AGX 上跑一遍自检，四项都应"通过"：`python3 self_filter.py`
2. `__init__` 里：
```python
from self_filter import SelfFilter
self.sf = SelfFilter(margin=a.self_margin) if a.self_margin >= 0 else None
self.n_drop_self = 0
self.t_jfast = 0.0
self.create_subscription(JointsData, "/JOINTS_DATA", self.on_joints_fast, qos_profile_sensor_data)       # 200 Hz，SDK 模式才有
self.create_subscription(JointsData, "/JOINTS_DATA_10HZ", self.on_joints_slow, qos_profile_sensor_data)  # 10 Hz 镜像，一直有
```
3. 回调：200 Hz 的新鲜时优先用它，否则用 10 Hz 的：
```python
def on_joints_fast(self, m):
    if self.sf is not None:
        self.sf.set_joints_published([j.position for j in m.data.joints_data]); self.t_jfast = time.monotonic()
def on_joints_slow(self, m):
    if self.sf is not None and time.monotonic() - self.t_jfast > 0.2:
        self.sf.set_joints_published([j.position for j in m.data.joints_data])
```
4. `on_cloud` 里，`pts = cloud_xyz(msg)` 之后、"防线 1 天花板"之前插入（在机体系里滤，其余流程不变）：
```python
if self.sf is not None and len(pts):
    keep = self.sf.keep_mask(pts @ self.R_bl.T + self.t_bl)     # 雷达系 → 机体系
    self.n_drop_self += int(len(pts) - keep.sum())
    pts = pts[keep]
    if len(pts) == 0:
        return
```
5. 参数：`ap.add_argument("--self-margin", type=float, default=0.03)`，设负数就关掉滤波。周期日志里建议加上删点比例和 `self.sf.stats()`，其中 no_joint 大于 0 说明还没收到关节角，这时滤波不删任何点。
6. 关节角超过 0.5 s 没更新，滤波会自动再外扩 3 cm，并计入 `stats()["stale"]`。

## 真机验收（不接管、不起 runner 也能做）
1. 狗厂家站姿，平地：x=+0.2、y≈−0.2~−0.3 那两格应从约 38 cm 降回地面（约 −0.08），日志里删点比例几个百分点量级。
2. 狗趴下或蹲姿：脚边不应再出现高出地面的格子。
3. 前轮贴着 33 cm 台阶站住：前排格（ix≥13）仍读到台阶顶（约 −0.40），说明立面和顶面没被误删。
4. 日志里单帧耗时。AGX 是 ARM、负载高，估计比本机慢几倍，10 Hz 下仍够用。

## 已知局限
- 包围体来自 S10.xml 的 CAD 网格。真机上的线缆、护罩、支架若超出 CAD 外形 3 cm 以上，会漏删，请看验收第 1 条。
- 行走时腿动得快，用 10 Hz 镜像时关节角可能滞后几厘米。有 200 Hz 的 `/JOINTS_DATA` 时优先用它。偶发漏删的瞬时点，也会被你们现有的"上一帧也必须见过这个体素"那道防线拦下。
- 常数由 `gen_self_filter.py` 从 S10.xml 生成（训练侧，需要 MuJoCo），不要手改。
