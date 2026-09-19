"""沿途累积的局部高程图 —— 弥补前视雷达看不到脚下的物理盲区。

为什么不能单帧算：实测（s10_dev/hmapwhy.py，t=150s 那一帧）
    射线 8978 条，命中 1302
    最近回波           2.18 m       ← min_range 配的是 0.1m，但几何上到不了
    高程图框           前后 ±0.8m / 左右 ±0.5m
    落在框内的点       0 / 1302
雷达装在 base 前 0.11m、上 0.18m，离地 0.588m，垂直视场向下约 32°，
机身正下方是物理盲区。所以脚下的地形只能靠"几秒前在 2~6m 外看到的"。

累积规则：每格取**最高**的点。这不是随便选的 —— 训练侧 IsaacLab 的射线是从
机身上方 20m 垂直往下打、命中最顶上那个面（ray_caster.py:224 + observations.py:300），
"取最高"与它语义等价。取最低或取平均会把台阶读成坑，机器狗一脚踩空。

赛规允许直接用 ground truth 位姿（不要求 SLAM），所以累积不会因里程漂移而糊掉。
真机上换成里程计时，这里需要加位姿协方差加权和老化淘汰。
"""
from __future__ import annotations

import os

import numpy as np

from heightmap import (CEIL_MARGIN, CELL_XY, CLIP, NUM_RAYS, NX, NY, OFFSET,
                       _yaw_rot)


def _read_fwd_clip():
    """X2 实验开关，见 sample() 末尾。默认 None = 不生效。"""
    v = os.environ.get("S10_HMAP_FWD_CLIP")
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if f > 0 else None


_FWD_CLIP = _read_fwd_clip()


def _read_lat_clip():
    """横向裁剪，见 sample() 末尾。默认 None = 不生效。"""
    v = os.environ.get("S10_HMAP_LAT_CLIP")
    if not v:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if f > 0 else None


_LAT_CLIP = _read_lat_clip()
if _LAT_CLIP is not None:
    print(f"[hmap] 横向视野裁到 ±{_LAT_CLIP}m（默认 ±0.5m）")

# 邻域补值用最大还是中位。默认中位（见 sample() 里的实测依据）；
# S10_HMAP_FILL=max 切回旧行为，用来做 A/B。
_FILL_MAX = os.environ.get("S10_HMAP_FILL", "median").strip().lower() == "max"
print(f"[hmap] 邻域补值 = {'最大值(旧)' if _FILL_MAX else '中位数(新)'}")

_OFFSET_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _neighbor_offsets(r: int) -> tuple[np.ndarray, np.ndarray]:
    """(2r+1)² 个邻域偏移，展平成两个一维数组。建一次就够，别每帧重算。"""
    if r not in _OFFSET_CACHE:
        d = np.arange(-r, r + 1, dtype=np.int64)
        gy, gx = np.meshgrid(d, d, indexing="ij")
        _OFFSET_CACHE[r] = (gy.ravel(), gx.ravel())
    return _OFFSET_CACHE[r]


class ElevationMap:
    """世界系锚定的稠密高程栅格。赛道约 50×50m，0.05m 分辨率也才 1M 个格子。"""

    def __init__(self, resolution: float = 0.05, origin: tuple[float, float] = (-40.0, -20.0),
                 extent: tuple[float, float] = (100.0, 100.0)):
        self.res = resolution
        self.ox, self.oy = origin
        self.nx = int(round(extent[0] / resolution))
        self.ny = int(round(extent[1] / resolution))
        # -inf 表示"从没被观测过"，与"观测到地面很低"区分开
        self.z = np.full((self.ny, self.nx), -np.inf, dtype=np.float32)
        self.n_update = 0
        self.n_points = 0
        self.n_seen = 0                 # 已点亮的格子数，coverage 增量维护

    # ---------------------------------------------------------------- 累积
    def update(self, points_world: np.ndarray, base_z: float | None = None) -> int:
        """吃一帧世界系点云。返回本帧更新到的格子数。

        base_z 给了就丢掉机身以上 CEIL_MARGIN 之外的点。**必须给**，除非确定
        场景里没有头顶结构 —— 赛道楼梯段头顶约 6m 有顶棚，不滤的话 max-z
        会把天花板当地面，而且那些格子会自信地报错（不是空洞），
        策略无从分辨。实测 t=350s 那一帧 61/187 格因此错了 2.9m。
        """
        if points_world.size == 0:
            return 0
        p = points_world[np.isfinite(points_world).all(axis=1)]
        if base_z is not None:
            p = p[p[:, 2] <= base_z + CEIL_MARGIN]
        if p.size == 0:
            return 0
        ix = ((p[:, 0] - self.ox) / self.res).astype(np.int64)
        iy = ((p[:, 1] - self.oy) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        if not ok.any():
            return 0
        k = iy[ok] * self.nx + ix[ok]
        flat = self.z.reshape(-1)                   # C 连续，是视图不是拷贝
        ku = np.unique(k)
        # 覆盖率增量维护：只数**本帧新点亮**的格子，且必须在写入之前数。
        # 见 coverage 的注释 —— 这里多花一次 ~9k 元素的 gather，
        # 省掉的是每帧一次 400 万元素的全图扫描。
        self.n_seen += int(np.isneginf(flat[ku]).sum())
        np.maximum.at(flat, k, p[ok, 2].astype(np.float32))
        self.n_update += 1
        self.n_points += int(ok.sum())
        return int(ku.size)

    @property
    def coverage(self) -> float:
        """已观测格子的占比。

        **别改回 `np.isfinite(self.z).mean()`。** 那是扫 2000×2000 = 400 万
        个格子，而 sample() 每帧都要在返回字典里带上它 —— 实测这一行
        就占了 sample() 全部耗时的 95%（约 1300µs / 1400µs），
        50Hz 下白吃 6.5% 的墙钟，而 RTF 是这套栈的命门。

        （查这个 bug 时我走了三次弯路：先后用三种写法重写"补空洞循环"，
        每次都逐位对拍通过、但耗时纹丝不动 —— 因为瓶颈压根不在那儿。
        最初的 profile 把 sample() 减去一个"只算格心和取值"的片段，
        两者差了**两处**，我却把差值全记在了循环头上。）
        """
        return self.n_seen / float(self.nx * self.ny)

    # ---------------------------------------------------------------- 采样
    NAV_HALF = 3.0        # 导航图半径（米）
    NAV_STEP = 0.15       # 导航图格距（米）→ 41×41

    def nav_patch(self, base_xy: np.ndarray, base_z: float) -> np.ndarray:
        """给**导航层**的大视野局部图（世界轴对齐，41×41 覆盖 6m×6m）。

        为什么不能用 sample() 那 187 格：那张是**落脚点图**，
        覆盖只有前后 ±0.8m、左右 ±0.5m，按训练侧观测定义来的，尺寸动不得
        （动了策略要重训）。用它找路等于让人用脚下一平方米的视野找路 ——
        作者实测那一幕：机器人在 (32.37,15.32) 顶着一堵墙，
        "往左边开一点就过去了"，而那个缺口在左边 0.5m 以外，**不在图里**，
        左右都显示被挡，避障算不出"更空的一侧"，于是完全不动作。

        而这张世界系累积栅格本来就有整条赛道（0.05m 分辨率），
        抠一块 6m×6m 出来是纯 numpy 切片，几乎不花时间。
        导航吃这张找路，策略继续吃它那 187 格落脚 —— 各取所需。

        返回：41×41 的**相对脚下高度**（米），行=y 由南到北，列=x 由西到东。
        没观测到的格填 0（与 sample() 口径一致：未知按平地）。
        """
        n = int(2 * self.NAV_HALF / self.NAV_STEP) + 1
        off = (np.arange(n) - n // 2) * self.NAV_STEP
        ix = np.clip(np.floor((base_xy[0] + off - self.ox) / self.res)
                     .astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.floor((base_xy[1] + off - self.oy) / self.res)
                     .astype(np.int64), 0, self.ny - 1)
        g = self.z[np.ix_(iy, ix)]
        foot = base_z - OFFSET
        out = np.where(np.isfinite(g), g - foot, 0.0)
        # 【这一行错过两次，第三版才对，别再动】
        # v1（原始）: np.where(out > CEIL_MARGIN, 0.0, out)
        #   抄自 sample() 但**参考系差了一个 OFFSET**：sample() 里 CEIL_MARGIN
        #   是和 base_z（机身中心）比，这里 out 已经是"相对脚下"。
        #   判据实际变成"高于脚下 0.15m 一律当平地"，全图夹在 ±0.15
        #   （实测 max 0.113m），而选路阈值 0.25m 数学上永远够不到 ——
        #   nav_steer 三趟跑测一次都没触发过，管线全通、话题在发、内容是假的。
        # v2（截断）: np.clip(out, -CLIP, CLIP)
        #   把高格子留成墙。柱子看见了，但**平地上也长出一堆墙**：
        #   实测在出生点 (35.39,15.42) 和 (33.30,14.74) 这两处真值确认的
        #   平地上都报"被挡"，一趟触发 63 次、把机器人来回拨到翻车。
        #   根因：self.z 是**跨时间取 max 的世界系图**，机器人在高处平台上
        #   扫到的格子会永久留着那个高值；等它下到低处，同一格就变成
        #   "脚下 3m 高的墙"。v1 把这些抹成 0 所以看不见，v2 全变成墙。
        # v3（本版）: 按**入图口径**判"这是不是我这个高度上的真障碍"。
        #   update() 入图时滤掉 base_z+CEIL_MARGIN 以上的点，所以
        #   **在当前高度观测到的真障碍，out 最多就是 OFFSET+CEIL_MARGIN=0.65**。
        #   读数超过这个的，只可能来自更高处那次观测 → 对当前高度是过期数据，
        #   按"未知"处理（=0，与 sample() 的未知即平地口径一致），
        #   而不是当墙。0.65 本身要留住 —— 柱子正好读这个值。
        stale = OFFSET + CEIL_MARGIN + 0.05
        out = np.where(out > stale, 0.0, out)
        return np.clip(out, -CLIP, CLIP).astype(np.float32)

    def sample(self, base_xy: np.ndarray, base_z: float, yaw: float,
               search_radius_cells: int = 2) -> tuple[np.ndarray, dict]:
        """按训练侧那 187 个格心采样，返回策略能直接吃的观测。

        栅格随机身 yaw 旋转但不随 roll/pitch 倾斜（ray_alignment="yaw"）。
        格心落在地图上没被观测过的位置时，向外找 search_radius_cells 圈内
        最近的有效值；仍找不到就交给 hole 计数，由调用方决定怎么填。
        """
        cen = base_xy[None, :] + CELL_XY @ _yaw_rot(yaw).T          # (187, 2) 世界系
        # 必须和 update() 的落格方式一致 —— 都用向下取整。
        # 曾经这里写的是 np.round，而 update 是截断，**差半格 0.025m**：
        # 落格时格 k 覆盖 [ox+k*res, ox+(k+1)*res)（以左下角定位），
        # 而 round 把格 k 当成以 ox+k*res 为**中心** —— 采样出的地形整体
        # 朝世界 -x/-y 平移 0.025m。实测把 1122 格里的坏格从 117 降到 83、
        # 90分位误差从 0.112 降到 0.030（换成 floor+1 反向则升到 155，
        # 单调性排除了"碰巧"）。别改回 round。
        ix = np.floor((cen[:, 0] - self.ox) / self.res).astype(np.int64)
        iy = np.floor((cen[:, 1] - self.oy) / self.res).astype(np.int64)
        inb = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

        # 采样时再拦一道天花板。update 时用的是**当时**的 base_z ——
        # 机器狗站得高时收进来的点会永久留在地图里，等它下来之后
        # 那些点就变成假障碍。实测楼梯段有 24 格读到 base_z+0.5 的钳位值。
        ceil = base_z + CEIL_MARGIN

        hit = np.full(NUM_RAYS, np.nan, dtype=np.float64)
        v = self.z[np.clip(iy, 0, self.ny - 1), np.clip(ix, 0, self.nx - 1)]
        good = inb & np.isfinite(v) & (v <= ceil)
        hit[good] = v[good]

        # 邻域补一层：地图分辨率(0.05)比查询栅格(0.1)细，格心正好落在空格是常事。
        # 邻域取值同样要过天花板，否则等于把假障碍又捡回来。
        #
        # 这段曾是逐格 Python 循环，实测占 sample() 耗时的 99%
        # （1390/1402µs），50Hz 下吃掉 7% 墙钟 —— 而 RTF 是这套栈的命门。
        #
        # 改法上踩过一次坑：先试了"对 (2r+1)² 个偏移各做一次整批比较"，
        # 结果**慢了 3 倍**（1600µs vs 600µs）。瓶颈根本不是 Python 循环，
        # 是 numpy 的每次调用开销 —— 25 个偏移 × 5 次调用 = 125 次，
        # 每次几 µs 的固定开销就把收益吃光了。
        # 真正该压的是**调用次数**：抠出查询区那个几十见方的小窗口，
        # 一次滑窗取最大值，总共 6 次 numpy 调用。
        # 语义不变：仍是取邻域内「有值且不超过天花板」的最大值，
        # 越界按无值处理（-inf 填充，与原来把窗口裁到图内等价）。
        need = ~good & inb
        r = search_radius_cells
        if need.any() and r > 0:
            k = np.flatnonzero(need)                 # 只有 inb 的格子会进来
            dy, dx = _neighbor_offsets(r)            # (25,) 各一，缓存
            jy = np.clip(iy[k][None, :] + dy[:, None], 0, self.ny - 1)
            jx = np.clip(ix[k][None, :] + dx[:, None], 0, self.nx - 1)
            w = self.z[jy, jx]                       # (25, m) 一次取齐
            valid = np.isfinite(w) & (w <= ceil)
            # 取最大值是避障的保守做法，但对**要爬台阶**的机器人是反的：
            # 坎沿附近的空格会继承坎顶高度，等于把障碍加宽、前移，凭空造墙。
            # 实测（mapvstruth.py，ghost29 那一跑，60 帧 × 187 格）：
            #   图上 >30cm 而真值 <10cm 的**幻觉格占 14.3%**，最坏一帧 121/187。
            #   而 34cm 超过 v6 的爬升上限 25~30cm —— 它停下来是理性的。
            # S10_HMAP_FILL=max 可切回旧行为做 A/B。默认 median。
            if _FILL_MAX:
                w2 = np.where(valid, w, -np.inf)
                best = w2.max(axis=0)
                ok = np.isfinite(best)               # 全 -inf = 邻域也没值
            else:
                w2 = np.where(valid, w, np.nan)
                ok = valid.any(axis=0)
                with np.errstate(invalid="ignore"):
                    best = np.nanmedian(w2, axis=0)  # 全 NaN 那列会是 NaN，被 ok 挡掉
            hit[k[ok]] = best[ok]

        n_hole = int(np.isnan(hit).sum())
        # 没观测到的格子按"与脚下同高的平地"填 —— 中性、不制造假障碍也不制造假坑。
        # 真机上更稳妥的是填成"未知=保守障碍"，但那会让策略不敢走没扫过的地方。
        filled = np.where(np.isnan(hit), base_z - OFFSET, hit)
        obs = np.clip(base_z - filled - OFFSET, -CLIP, CLIP)

        # S10_HMAP_FWD_CLIP=<米>：把前视裁短，机体系 x 超过该值的格子
        # 一律填成"与脚下同高"。**只为 X2 那个证伪实验存在，默认不生效。**
        # 机制假设是：台阶进入 0.8m 前视框那一瞬触发减速（实测延迟 <0.05s），
        # 所以把前视裁到 L，停车点就该整体前移 (0.8-L) 米。
        # 这是个**数值**预测，比 X1「关掉高程图」那种方向预测强。
        # 裁剪**之前**的那份要留一手 —— 见下面 raw 的说明。
        raw = obs
        if _FWD_CLIP is not None:
            far = CELL_XY[:, 0] > _FWD_CLIP
            near = CELL_XY[:, 0] <= 0.1
            obs = obs.copy()
            obs[far] = float(np.median(obs[near])) if near.any() else 0.0

        # S10_HMAP_LAT_CLIP=<米>：把**横向**视野裁窄，机体系 |y| 超过该值的格子
        # 填成"与脚下同高"。默认 None = 不生效。
        #
        # 【为什么加它】FWD_CLIP 治好了"正前方远处有台阶就冻住"（#52），
        # 但 v6 还有一种死法没治：**脚下和正前方全平，侧面 0.3~0.6m 有东西，
        # 它就是不动**。2026-08-12 八趟全程实测三个点同签名（姿态正常、
        # 位置纹丝不动、烧光重试）：
        #   wp6  (−15.2, 28.28)  2/8 趟死在这，正北 0.3m 一条 +0.08~0.25 的带
        #   wp12 (−4.47, 43~45)  1/8 趟，西南 0.1m 一块 >0.25
        #   wp26 (33.35, 20.53)  r8 死在这，**去 wp26 的路整片是平的**，
        #                        但右手边 0.6m（x≥33.95）是一道墙
        # 这与任务 #50 记的「偏轴障碍面前起不来步」是同一件事。
        # 187 格图横向覆盖 ±0.5m，裁到 0.25~0.3 就把那圈侧向格子抹掉。
        #
        # ⚠ 这是**绕过缺陷不是修好缺陷**，和 FWD_CLIP 一样。而且有反向风险：
        # 横向格子本来是给"别压路肩"用的（#20 记过横向坡是摔倒主因），
        # 裁掉可能换来更多侧翻。所以必须交错 A/B，两个指标一起看：
        # 冻结次数**和**侧翻次数。
        if _LAT_CLIP is not None:
            side = np.abs(CELL_XY[:, 1]) > _LAT_CLIP
            near = CELL_XY[:, 0] <= 0.1
            obs = obs.copy()
            obs[side] = float(np.median(obs[near])) if near.any() else 0.0
        return obs, {
            # 未裁剪的原始 187 格。**裁剪是给策略打的补丁，不该连导航一起蒙住。**
            # v6 对远场障碍过度反应（台阶还在 0.8m 外就减速停住，#52），
            # 所以把 0.6m 以外抹平；但发布出去的 /height_scan 已经是裁过的，
            # 导航层接进来之后视野只有 0.6m —— 比机器人轴距 0.455m 多不了多少。
            # 等柱子进视野时已经来不及转向绕开（作者看回放："往左边走一点就过去了"，
            # 那需要提前看见）。两个消费者需求不同：
            #   策略（怎么走：抬腿、步态）0.6m 够用
            #   导航（往哪走：绕行、选路）需要 2~3m
            # 所以各取一份：策略吃 obs，导航吃 raw（走 /height_scan_raw）。
            "raw": raw,
            "n_hole": n_hole,
            "hole_rate": n_hole / NUM_RAYS,
            "map_coverage": self.coverage,
            "frames": self.n_update,
        }
