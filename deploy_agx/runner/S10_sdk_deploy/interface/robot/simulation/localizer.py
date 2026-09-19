"""已知地图下的雷达定位（在线版）。

离线验证见 s10_dev/localize.py：跟踪模式下位置中位 1mm、最差 3mm、
100% 落在 0.20m 判定圈内、120 条射线时单帧约 100ms。这里把它做成在线的。

【只用合法输入】
  · 雷达测距与方向        —— 传感器
  · 关节角                —— 编码器
  · z / roll / pitch      —— IMU 姿态 + 接触约束，常规做法
  · 上一帧估计 + 自身外推 —— 我们自己算出来的
**绝不使用真值的 x / y / yaw。** 那正是要恢复的量。

【为什么放在仿真进程里】省掉一个节点和一套消息管道。它消费的输入和一个
独立节点能拿到的完全一样，功能上等价；真要拆出去也只是搬代码。

【为什么必须异步】单帧匹配约 100ms，而仿真主循环是墙钟驱动的 ——
在主循环里同步跑会把 RTF 直接砍半，那比定位误差危害大得多。
"""
import math
import threading

import mujoco
import numpy as np
from scipy.optimize import minimize


class Localizer:
    def __init__(self, model, base_body="base_link", site="front_lidar_site",
                 nray=120, max_range=40.0):
        self.m = model
        self.nray = nray
        self.max_range = max_range
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        if self.site_id < 0 or self.base_id < 0:
            raise RuntimeError("模型里没有 front_lidar_site / base_link")

        # 地图那份 MjData：把机器人挪到 z=-1000，射线只打静态场景。
        # 这样候选位姿变化时不需要重跑运动学（腿的位置不再影响射线）。
        self.dmap = mujoco.MjData(model)
        self.dmap.qpos[0:3] = (0.0, 0.0, -1000.0)
        mujoco.mj_forward(model, self.dmap)

        # 站点相对基座的固定变换（front_lidar_link 是焊死的）
        d0 = mujoco.MjData(model)
        d0.qpos[:] = model.qpos0
        mujoco.mj_forward(model, d0)
        bp = d0.xpos[self.base_id].copy()
        bR = d0.xmat[self.base_id].reshape(3, 3).copy()
        self.site_off = bR.T @ (d0.site_xpos[self.site_id] - bp)
        self.site_rel = bR.T @ d0.site_xmat[self.site_id].reshape(3, 3)

        self.groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._gid = np.zeros(nray, dtype=np.int32)
        self._dist = np.zeros(nray, dtype=np.float64)

        # 估计状态
        self.est = None          # (x, y, yaw_deg)
        self.prev = None         # 上一帧估计，用来外推
        self.n_ok = 0
        self.n_fail = 0
        self.n_rej_res = 0        # 被残差闸挡掉
        self.n_rej_jump = 0       # 被跳变闸挡掉
        self.odo_dist = 0.0       # 里程推算累计路程，用来定跳变闸的宽度
        # 离线验证里好的匹配残差是 0.0000~0.0385，留一个量级的余量。
        self.res_max = 0.05
        self.last_ms = 0.0
        self.last_res = 0.0
        self._lock = threading.Lock()
        self._busy = False

    # ── 几何 ────────────────────────────────────────────────────────
    @staticmethod
    def _quat(r, p, y):
        r, p, y = math.radians(r), math.radians(p), math.radians(y)
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)

    def _site_pose(self, x, y, z, quat):
        w, i, j, k = quat
        R = np.array([
            [1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w)],
            [2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w)],
            [2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)]])
        return np.array([x, y, z]) + R @ self.site_off, R @ self.site_rel

    def _cast(self, pos, rot, dirs):
        wd = np.ascontiguousarray(dirs @ rot.T, dtype=np.float64)
        n = len(dirs)
        mujoco.mj_multiRay(self.m, self.dmap, pos, wd.ravel(), self.groups, True,
                           -1, self._gid[:n], self._dist[:n], None, n, self.max_range)
        return self._dist[:n].copy()

    # ── 一帧匹配 ────────────────────────────────────────────────────
    def solve(self, dirs, ranges, valid, z, roll, pitch, guess):
        """dirs/ranges 是雷达本体系的原始测量。guess=(x,y,yaw_deg)。"""
        idx = np.where(valid)[0]
        if len(idx) < 40:
            return None
        if len(idx) > self.nray:
            idx = idx[np.linspace(0, len(idx) - 1, self.nray).astype(int)]
        dd = np.ascontiguousarray(dirs[idx], dtype=np.float64)
        mm = ranges[idx]

        def cost(v):
            q = self._quat(roll, pitch, v[2])
            sp, sr = self._site_pose(v[0], v[1], z, q)
            sim = self._cast(sp, sr, dd)
            ok = (sim > 0.1) & (sim < self.max_range * 0.98)
            if ok.sum() < len(dd) * 0.2:
                return 1e3
            r = np.abs(sim[ok] - mm[ok])
            h = np.where(r < 0.3, 0.5 * r * r, 0.3 * (r - 0.15))
            return h.mean() + 0.02 * (1.0 - ok.sum() / len(dd))

        res = minimize(cost, np.asarray(guess, dtype=float), method="Nelder-Mead",
                       options=dict(xatol=2e-3, fatol=1e-5, maxiter=250))
        return (float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.fun))

    # ── 异步接口 ────────────────────────────────────────────────────
    @property
    def busy(self):
        return self._busy

    # 机器狗指令上限 3.6 m/s，加余量按 4.0 算；转向按 180°/s。
    VMAX = 4.0
    WMAX = 180.0

    def predict(self, v_fwd, yaw_rate_deg, dt):
        """里程推算：轮速给前向速度，陀螺给偏航角速度。每个控制周期都调。

        【为什么必须有这一层】没有它，匹配的初值只能是"上一帧估计"。而实测
        求解频率只有 1.5Hz（复杂地形处单帧 87~394ms），机器狗每次求解间要走
        1.3 米 —— 正确解相对上一帧就是个大跳变，被跳变闸挡掉，从此再也追不回来，
        估计冻死在出生点。删掉外推是从"跑飞"跳到了另一个极端。

        正解是标准的**里程推算 + 周期性校正**：里程负责跟住（会漂但连续），
        雷达负责把漂移拉回来。轮速与陀螺都是真机有的传感器，不是真值。
        """
        if self.est is None:
            return
        x, y, yaw = self.est
        yr = math.radians(yaw)
        self.est = (x + v_fwd * math.cos(yr) * dt,
                    y + v_fwd * math.sin(yr) * dt,
                    yaw + yaw_rate_deg * dt)
        self.odo_dist += abs(v_fwd) * dt

    def submit(self, dirs, ranges, valid, z, roll, pitch, dt):
        """交一帧给后台线程。dt 是距上一帧提交的仿真时长。"""
        if self._busy:
            return
        if self.est is None:
            return                     # 还没初始化，等 seed()
        # 初值 = 里程推算维持的当前估计。predict() 每个控制周期都在推，
        # 所以即使求解只有 1~2Hz，初值也一直跟着机器人走。
        # （第一版拿 (est−prev)/dt 自外推是正反馈，13 秒跑飞；
        #   第二版干脆不推，结果落后就被跳变闸锁死。都不对。）
        g = list(self.est)
        self._busy = True
        self._last_dt = dt
        # 跳变闸按**这次求解期间里程走了多远**来放宽，而不是按提交间隔。
        # 求解慢的时候机器人走得远，闸门必须跟着放宽，否则又会锁死。
        self._odo_at_submit = self.odo_dist
        threading.Thread(target=self._work,
                         args=(dirs.copy(), ranges.copy(), valid.copy(),
                               z, roll, pitch, g, dt), daemon=True).start()

    def _work(self, dirs, ranges, valid, z, roll, pitch, g, dt):
        import time
        t0 = time.perf_counter()
        try:
            r = self.solve(dirs, ranges, valid, z, roll, pitch, g)
        except Exception:
            r = None
        with self._lock:
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            if r is None:
                self.n_fail += 1
                self._busy = False
                return
            self.last_res = r[3]
            # ── 两道闸，缺一不可 ────────────────────────────────────
            # ① 残差门槛。第一版设 0.5，而离线验证里**好的匹配残差是
            #    0.0000~0.0385** —— 0.5 等于什么垃圾解都收，这是跑飞的直接原因。
            # ② 跳变门槛。物理上 dt 内最多移动 VMAX*dt。第一版完全没查，
            #    一个比上一帧远 5 米的解也照单全收。
            #    退化区域（比如起点那片平地，向下倾的雷达只看得见地面，
            #    平面对 x/y 没有约束）优化器本来就会乱漂，必须靠这道闸兜住。
            jump = math.hypot(r[0] - self.est[0], r[1] - self.est[1])
            dyaw = abs(((r[2] - self.est[2] + 180) % 360) - 180)
            # 闸门宽度 = 求解期间里程走过的距离 × 一个漂移系数 + 固定余量。
            # 求解慢 → 走得远 → 闸门自动放宽，不会把正确的大修正误杀。
            moved = max(0.0, self.odo_dist - getattr(self, "_odo_at_submit", 0.0))
            lim = 0.35 * moved + 0.25
            if r[3] > self.res_max:
                self.n_fail += 1
                self.n_rej_res += 1
            elif jump > lim or dyaw > 12.0:
                self.n_fail += 1
                self.n_rej_jump += 1
            else:
                self.prev = self.est
                self.est = (r[0], r[1], r[2])
                self.n_ok += 1
        self._busy = False

    def seed(self, x, y, yaw):
        """起点位姿。比赛里出生点是已知的，这不算偷看 —— 真机也是从已知位置上电。"""
        self.est = (float(x), float(y), float(yaw))
        self.prev = None
