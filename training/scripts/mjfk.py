#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**所有爬墙判据的唯一几何来源**：给一帧 qpos，用 MuJoCo 真模型算出全部判据量。

作者 09-17 16:3x：「你用 mujoco 的全部作为校验，不要看错了」。

## 为什么必须这样

此前俯仰是**直接读 CSV 的 pitch 列**，而我方 state.csv 与 `ref_climb34.npz` 的
**pitch/roll 符号相反**（见训练计划第 43 章）。后果：`wheel_fk.py` 靶写成 `pitch >= +10`，
我方 −27 长期判 ✗，我据此反复向作者报「我方低头、官方抬头」——**全错，两边都是抬头**。
还把官方轨迹倒着放给作者看，被当场指出「倒着后腿飞起来了」。

**对策**：判据量一律用**模型上的几何点**算，符号由几何决定，不由某一列的约定决定。

    抬头角  = atan2(前髋中点z − 后髋中点z, 两者水平距离)   —— 正 = 抬头，天然无歧义
    侧倾角  = atan2(左髋中点z − 右髋中点z, 两者水平距离)   —— 正 = 左高
    轮底高  = 轮心z − 轮半径
    轮前沿  = max(前两轮心x) + 轮半径
    腿长    = 髋心→轮心距离（只取决于膝角，hipx/hipy 不影响，已实测）
    机身高  = base body 的 z

CSV 的 roll/pitch/yaw 只用来**构造姿态**（这是机身朝向的唯一来源），构造后所有判据量
都从模型读，不再直接引用那三列。官方 npz 送进来前必须先取反 roll/pitch。
"""
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MJ = os.path.join(HERE, "..", "s10_ws/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/bench_top33_airy.xml")
R = 0.081
LEGS = ("fl", "fr", "hl", "hr")


def quat(roll, pitch, yaw):
    """ZYX 内旋 → wxyz。**输入必须是弧度**，且必须是本仓约定（负 pitch = 抬头）。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class FK:
    def __init__(self, xml=None):
        self.m = mujoco.MjModel.from_xml_path(xml or MJ)
        self.d = mujoco.MjData(self.m)
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.w = [bid(n + "_wheel") for n in LEGS]
        self.h = [bid(n + "_hipx") for n in LEGS]
        self.base = bid("base") if bid("base") >= 0 else 1
        for i, n in enumerate(LEGS):
            if self.w[i] < 0 or self.h[i] < 0:
                raise RuntimeError("找不到 %s 的 wheel/hipx body" % n)

    def step(self, x, y, z, roll_rad, pitch_rad, yaw_rad, q16):
        """算一帧 → dict。roll/pitch 必须已经是本仓约定（负 pitch = 抬头）。"""
        d = self.d
        d.qpos[:] = 0
        d.qpos[:3] = [x, y, z]
        d.qpos[3:7] = quat(roll_rad, pitch_rad, yaw_rad)
        d.qpos[7:23] = q16
        mujoco.mj_forward(self.m, d)
        wp = np.array([d.xpos[b] for b in self.w])       # (4,3) 轮心
        hp = np.array([d.xpos[b] for b in self.h])       # (4,3) 髋心
        fmid, rmid = hp[:2].mean(axis=0), hp[2:].mean(axis=0)
        lmid, rrmid = hp[[0, 2]].mean(axis=0), hp[[1, 3]].mean(axis=0)
        dx = np.linalg.norm((fmid - rmid)[:2])
        dy = np.linalg.norm((lmid - rrmid)[:2])
        return dict(
            # 抬头角：前髋中点比后髋中点高多少度。**正 = 抬头**，与任何列的符号约定无关。
            pitch_up=float(np.degrees(np.arctan2(fmid[2] - rmid[2], max(dx, 1e-6)))),
            # 侧倾角：左髋中点比右髋中点高多少度。正 = 左高。
            roll_left=float(np.degrees(np.arctan2(lmid[2] - rrmid[2], max(dy, 1e-6)))),
            wheel_bottom=wp[:, 2] - R,                    # (4,) 轮底高（世界）
            front_edge=float(wp[:2, 0].max() + R),        # 前轮最前沿 x
            leg_len=np.linalg.norm(wp - hp, axis=1),      # (4,) 腿长
            body_z=float(d.xpos[self.base][2]),
        )
