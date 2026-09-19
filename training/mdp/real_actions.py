"""真机化动作项（09-11）：把仿真里缺的两样东西补进去 —— 回路延迟 与 左右不对称。

为什么：09-10 真机 bag 1 实测偏航角速度在 **1.41 Hz** 上自激振荡，占一半能量，偏航率 std 31°/s；
仿真里同一策略只有 0.07~0.11 °/s，且无此峰。厂家踏步模式的峰恰好等于其步频（1.10/1.10/1.98 Hz），
是步态节奏、有界；我们没有步频却在振荡 = 控制环打摆。而训练侧用的是隐式执行器，指令当拍生效、
观测无延迟（0 ms），真机上至少 20 ms 策略周期加 DDS 传输。零延迟训出的反馈增益接到有延迟的对象上
正是会打摆的情形。同时真机四条腿 hipx 有约 0.6° 的左右不对称（厂家控制器下也存在）。

三项，按回合随机：
  ① 动作延迟 0~N 个策略步（环形缓冲，进 process_actions 之前取旧的那一帧）；
  ② 关节零位偏置（腿）：给目标角加每台每关节的常量偏置，模拟标定/装配误差；
  ③ 轮增益：给轮速目标乘每台每轮的系数，模拟轮径差与电机增益差。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.utils import configclass


class _DelayMixin:
    """按回合随机的动作延迟。放在 process_actions 最前面，对子类其余逻辑透明。"""

    def _delay_init(self, max_steps: int):
        self._d_max = int(max_steps)
        if self._d_max > 0:
            self._d_buf = torch.zeros(self.num_envs, self._d_max + 1, self.action_dim, device=self.device)
            self._d_k = 0
            self._d_n = torch.randint(0, self._d_max + 1, (self.num_envs,), device=self.device)

    def _delay_apply(self, actions: torch.Tensor) -> torch.Tensor:
        if self._d_max <= 0:
            return actions
        self._d_k = (self._d_k + 1) % self._d_buf.shape[1]
        self._d_buf[:, self._d_k] = actions
        idx = (self._d_k - self._d_n) % self._d_buf.shape[1]
        return self._d_buf[torch.arange(self.num_envs, device=self.device), idx]

    def _delay_reset(self, env_ids):
        if self._d_max <= 0:
            return
        if env_ids is None:
            env_ids = slice(None)
            n = self.num_envs
        else:
            n = len(env_ids)
        self._d_buf[env_ids] = 0.0
        self._d_n[env_ids] = torch.randint(0, self._d_max + 1, (n,), device=self.device)


class RealJointPositionAction(_DelayMixin, JointPositionAction):
    """腿：动作延迟 + 每关节零位偏置。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._delay_init(cfg.delay_steps)
        self._bias_mag = float(cfg.zero_offset)
        self._bias = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._resample_bias(None)
        print(f"[real-act] 腿：延迟 0~{cfg.delay_steps} 步，零位偏置 ±{cfg.zero_offset} rad", flush=True)

    def _resample_bias(self, env_ids):
        sl = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        self._bias[sl] = (2 * torch.rand(n, self.action_dim, device=self.device) - 1) * self._bias_mag

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(self._delay_apply(actions))
        self._processed_actions = self._processed_actions + self._bias

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._delay_reset(env_ids)
        self._resample_bias(env_ids)


class RealJointVelocityAction(_DelayMixin, JointVelocityAction):
    """轮：动作延迟 + 每轮增益（轮径差 / 电机增益差）。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._delay_init(cfg.delay_steps)
        self._gain_mag = float(cfg.gain_range)
        self._gain = torch.ones(self.num_envs, self.action_dim, device=self.device)
        self._resample_gain(None)
        print(f"[real-act] 轮：延迟 0~{cfg.delay_steps} 步，轮增益 1±{cfg.gain_range}", flush=True)

    def _resample_gain(self, env_ids):
        sl = slice(None) if env_ids is None else env_ids
        n = self.num_envs if env_ids is None else len(env_ids)
        self._gain[sl] = 1.0 + (2 * torch.rand(n, self.action_dim, device=self.device) - 1) * self._gain_mag

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(self._delay_apply(actions))
        self._processed_actions = self._processed_actions * self._gain

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._delay_reset(env_ids)
        self._resample_gain(env_ids)


@configclass
class RealJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RealJointPositionAction
    delay_steps: int = 3
    """动作延迟上限（策略步，1 步 = 20 ms）。"""
    zero_offset: float = 0.02
    """每台每关节的零位偏置幅值（rad）。真机实测 hipx 左右不对称约 0.01 rad。"""


@configclass
class RealJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = RealJointVelocityAction
    delay_steps: int = 3
    zero_offset: float = 0.0
    gain_range: float = 0.03
    """每台每轮的增益偏差幅值。真机左右实际轮速差约 2.5%。"""
