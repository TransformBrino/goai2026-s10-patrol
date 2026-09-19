"""高程图退化观测项（09-10）：让策略见到真机链路会给出的那种图，而不是俯视真值图。

背景：训练里 height_scan 是从机身正上方 20 m 垂直打下来的 187 条射线（17 前后 × 11 左右，0.1 m 一格，
以机身为中心 ±0.8 m × ±0.5 m），每格必有值、无遮挡、无延迟、无错位。真机上这 187 个数由
雷达 → 位姿 → hmap_node 现造，09-10 实测三段包里 /height_scan 一帧都没有，runner 全程填常数 −0.10。

本项在真值图上叠四种退化，按回合随机：
  ① 几何遮挡：以机身为观察点做逐行视线遮挡（cummin 比值法），墙背后、坎背后的格子看不见；
  ② 空洞：随机丢格 + 前向截止 + 后向截止（朝前的传感器看不到身后，靠累积才有）；
  ③ 延迟：整张图取 0~3 个策略步之前的（10 Hz 点云 + 位姿变换的真实代价）；
  ④ 错位：栅格整体平移 ±1 格（位姿短期漂移）。
丢掉的格子一律填 −0.10 —— 这必须与部署侧 hmap_node 的填洞值和 runner 的兜底值三者一致。
噪声在本项内部只加给"有效格"，所以 ObsTerm 上不要再挂 noise。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from . import observations as _obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

NX, NY = 17, 11          # 前后 17 格、左右 11 格；展平顺序是 [iy, ix]（GridPattern ordering="xy"）
RES = 0.1
OFFSET = 0.5
FILL = -0.10             # 与 runner SetHeightScanFallback / hmap_node 填洞值一致


def _base_scan(env, sensor_cfg, offset):
    s = env.scene.sensors[sensor_cfg.name]
    pos = s.data.pos_w; hit = s.data.ray_hits_w
    pos = pos.torch if hasattr(pos, "torch") else pos
    hit = hit.torch if hasattr(hit, "torch") else hit
    return pos[:, 2].unsqueeze(1) - hit[..., 2] - offset


class HeightScanDegraded(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.sensor_cfg = p.get("sensor_cfg", SceneEntityCfg("height_scanner"))
        self.offset = float(p.get("offset", OFFSET))
        self.noise = float(p.get("noise", 0.1))
        self.lat_max = int(p.get("latency_max", 3))
        self.p_mode = torch.tensor(p.get("mode_probs", (0.25, 0.30, 0.25, 0.20)), device=env.device)
        N = env.num_envs
        self.buf = torch.full((N, self.lat_max + 1, NX * NY), FILL, device=env.device)
        self.mode = torch.zeros(N, dtype=torch.long, device=env.device)
        self.lat = torch.zeros(N, dtype=torch.long, device=env.device)
        self.shx = torch.zeros(N, dtype=torch.long, device=env.device)
        self.shy = torch.zeros(N, dtype=torch.long, device=env.device)
        self.pdrop = torch.zeros(N, device=env.device)
        self.fcut = torch.full((N,), NX, dtype=torch.long, device=env.device)
        self.rcut = torch.zeros(N, dtype=torch.long, device=env.device)
        self.occl = torch.zeros(N, dtype=torch.bool, device=env.device)
        self._k = 0
        self._last_step = -1
        ix = torch.arange(NX, device=env.device).float()
        self.dx = (ix - (NX - 1) / 2.0) * RES                       # 每列相对机身的前后距离
        self._resample(torch.arange(N, device=env.device))
        print(f"[hmap-degrade] 生效：模式概率 完好/轻空洞/重遮挡/全瞎 = {tuple(float(v) for v in self.p_mode)}，"
              f"延迟 0~{self.lat_max} 步，错位 ±1 格，填洞值 {FILL}", flush=True)

    def _resample(self, ids: torch.Tensor):
        n = len(ids)
        if n == 0:
            return
        d = ids.device
        self.mode[ids] = torch.multinomial(self.p_mode, n, replacement=True)
        self.lat[ids] = torch.randint(0, self.lat_max + 1, (n,), device=d)
        self.shx[ids] = torch.randint(-1, 2, (n,), device=d)
        self.shy[ids] = torch.randint(-1, 2, (n,), device=d)
        m = self.mode[ids]
        p = torch.where(m == 1, 0.05 + 0.20 * torch.rand(n, device=d), 0.20 + 0.30 * torch.rand(n, device=d))
        self.pdrop[ids] = torch.where(m == 0, torch.zeros_like(p), p)
        heavy = m == 2
        self.fcut[ids] = torch.where(heavy, torch.randint(9, NX + 1, (n,), device=d), torch.full((n,), NX, device=d, dtype=torch.long))
        self.rcut[ids] = torch.where(heavy, torch.randint(0, 5, (n,), device=d), torch.zeros(n, device=d, dtype=torch.long))
        self.occl[ids] = heavy | (m == 1) & (torch.rand(n, device=d) < 0.5)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)
        self._resample(env_ids)
        self.buf[env_ids] = FILL
        return {}

    def _shadow_mask(self, g: torch.Tensor) -> torch.Tensor:
        """g: (N, NY, NX) 深度值（= scan + offset，机身下方为正）。返回可见掩码。"""
        N = g.shape[0]
        depth = g + self.offset
        vis = torch.ones_like(depth, dtype=torch.bool)
        c = (NX - 1) // 2
        for sgn, rng in ((+1, range(c + 1, NX)), (-1, range(c - 1, -1, -1))):
            run = torch.full((N, NY), float("inf"), device=depth.device)
            for ix in rng:
                dx = abs(self.dx[ix].item())
                r = depth[:, :, ix] / max(dx, 1e-3)
                r = torch.where(depth[:, :, ix] <= 0.0, torch.full_like(r, -float("inf")), r)
                vis[:, :, ix] = r <= run + 1e-4
                run = torch.minimum(run, r)
        return vis

    def __call__(self, env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
                 offset: float = OFFSET, noise: float = 0.1, latency_max: int = 3,
                 mode_probs: tuple = (0.25, 0.30, 0.25, 0.20)) -> torch.Tensor:
        raw = _base_scan(env, sensor_cfg, offset)                      # (N, 187) 真值
        N = raw.shape[0]
        g = raw.view(N, NY, NX)
        keep = torch.ones_like(g, dtype=torch.bool)
        # ① 几何遮挡
        sm = self._shadow_mask(g)
        keep &= sm | ~self.occl.view(N, 1, 1)
        # ② 前向/后向截止 + 随机丢格
        ixv = torch.arange(NX, device=g.device).view(1, 1, NX)
        keep &= ixv < self.fcut.view(N, 1, 1)
        keep &= ixv >= self.rcut.view(N, 1, 1)
        keep &= torch.rand_like(g) >= self.pdrop.view(N, 1, 1)
        keep &= (self.mode != 3).view(N, 1, 1)                          # 全瞎回合
        out = torch.where(keep, g + noise * (2 * torch.rand_like(g) - 1), torch.full_like(g, FILL))
        # ④ 栅格错位
        sx = self.shx.view(N, 1, 1); sy = self.shy.view(N, 1, 1)
        ixs = (ixv - sx).clamp(0, NX - 1)
        iyv = torch.arange(NY, device=g.device).view(1, NY, 1)
        iys = (iyv - sy).clamp(0, NY - 1)
        out = out.gather(2, ixs.expand(N, NY, NX)).gather(1, iys.expand(N, NY, NX))
        flat = out.reshape(N, NX * NY)
        # ③ 延迟：每个策略步推一次环形缓冲
        if env.common_step_counter != self._last_step:
            self._last_step = env.common_step_counter
            self._k = (self._k + 1) % self.buf.shape[1]
            self.buf[:, self._k] = flat
        else:
            self.buf[:, self._k] = flat
        idx = (self._k - self.lat) % self.buf.shape[1]
        return self.buf[torch.arange(N, device=g.device), idx]
