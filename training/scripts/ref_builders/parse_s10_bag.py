#!/usr/bin/env python3
"""解析真机 rosbag2（/JOINTS_DATA、/IMU_DATA、/IMU_103、/tf、/STEER）→ CSV/NPZ + 曲线图。

用法（需 source ROS Jazzy 与含 drdds 的 workspace，如 ~/s10_install/setup.bash 或 AGX 上的官方包）：
  /usr/bin/python3 s10_dev/parse_s10_bag.py <bag目录> [--out 输出前缀] [--plot]

关节换算（s10_interface.hpp:62-78 / dds_interface.hpp:275-277）：
  真实角 = raw_position × dir + offset(deg→rad)；真实速度 = raw_velocity × dir；真实力矩 = raw_torque × dir
  换算后的角度与 MJCF/URDF/训练 USD 同一约定（09-09 已核：USD 与 MJCF 运动学一致）。
关节顺序（robot 顺序，按数组下标）：fl/fr/hl/hr × (hipx, hipy, knee, wheel)。
时钟（AGX 端 09-09 实测）：header.stamp = 机器人钟（比 AGX 快约 41.96 s，稳定），bag 记录时间 = AGX 接收钟。
  关节/IMU/tf 都在机器人钟上 → 跨话题按 header.stamp 对齐；每行同时保留 t_hdr 与 t_rx（接收），电池类只能用 t_rx。
轮子：position 是无界累积角，不换算不使用；只用 velocity/torque（×dir）。模型约定：轮正转 = 后退（USD/MJCF 实测），
  前进时换算后的四个轮速应为负，可用 basic_mid 段核对 dir 表。
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

JOINT_NAMES = [f"{leg}_{j}" for leg in ("fl", "fr", "hl", "hr") for j in ("hipx", "hipy", "knee", "wheel")]
INIT_POS_OFFSET_DEG = [-35, -145, 156, 0, 35, -145, 156, 0, -35, 145, -156, 0, 35, 145, -156, 0]
JOINT_DIR = [1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1]
OFF = np.array(INIT_POS_OFFSET_DEG, dtype=float) * math.pi / 180.0
DIR = np.array(JOINT_DIR, dtype=float)


def _flatten_joints(msg):
    """drdds/JointsData → 16×(pos, vel, tau, temp) 原始值。兼容 data 为列表或带 .data 的嵌套。"""
    vals = getattr(msg, "data", None)
    if vals is not None and hasattr(vals, "joints_data"):      # drdds: JointsData.data.joints_data = JointData[16]
        vals = vals.joints_data
    elif vals is not None and hasattr(vals, "data"):
        vals = vals.data
    if vals is None:
        raise RuntimeError(f"JointsData 结构不认识: {[f for f in dir(msg) if not f.startswith('_')]}")
    pos, vel, tau, tmp = [], [], [], []
    for j in vals:
        pos.append(float(getattr(j, "position", math.nan)))
        vel.append(float(getattr(j, "velocity", math.nan)))
        tau.append(float(getattr(j, "torque", math.nan)))
        tmp.append(float(getattr(j, "motion_temp", getattr(j, "temperature", math.nan))))
    return pos, vel, tau, tmp


def _stamp(msg, t_bag):
    h = getattr(msg, "header", None)
    if h is not None:
        st = getattr(h, "stamp", None)
        if st is not None and hasattr(st, "sec"):
            v = float(st.sec) + float(st.nanosec) * 1e-9
            if v > 0:
                return v
        for k in ("timestamp", "time_stamp", "stamp"):
            v = getattr(h, k, None)
            if isinstance(v, (int, float)) and v > 1e9:
                return float(v) * (1e-9 if v > 1e12 else 1.0)
    return t_bag * 1e-9


def read_bag(path):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = SequentialReader()
    reader.open(StorageOptions(uri=path, storage_id=""), ConverterOptions("cdr", "cdr"))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    print("话题:", {k: v for k, v in topics.items()})
    cls = {}
    for name, typ in topics.items():
        try:
            cls[name] = get_message(typ)
        except Exception as e:
            print(f"  跳过 {name}（{typ}）: {e}")
    out = {"joints": [], "imu": [], "imu103": [], "tf": [], "steer": []}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in cls:
            continue
        msg = deserialize_message(data, cls[topic])
        if topic == "/JOINTS_DATA":
            pos, vel, tau, tmp = _flatten_joints(msg)
            out["joints"].append([_stamp(msg, t), t * 1e-9] + pos + vel + tau + tmp)
        elif topic == "/IMU_DATA":
            d = getattr(msg, "data", msg)
            out["imu"].append([_stamp(msg, t)] + [float(getattr(d, k, math.nan)) for k in
                              ("roll", "pitch", "yaw", "omega_x", "omega_y", "omega_z", "acc_x", "acc_y", "acc_z")])
        elif topic == "/IMU_103":
            q = msg.orientation; w = msg.angular_velocity; a = msg.linear_acceleration
            out["imu103"].append([_stamp(msg, t), q.x, q.y, q.z, q.w, w.x, w.y, w.z, a.x, a.y, a.z])
        elif topic == "/tf":
            for tr in msg.transforms:
                if tr.child_frame_id.endswith("base_link"):
                    tt = tr.transform.translation; rq = tr.transform.rotation
                    out["tf"].append([float(tr.header.stamp.sec) + tr.header.stamp.nanosec * 1e-9, tt.x, tt.y, tt.z, rq.x, rq.y, rq.z, rq.w])
        elif topic in ("/STEER", "/HANDLE_STEER"):
            d = getattr(msg, "data", msg)
            out["steer"].append([_stamp(msg, t)] + [float(getattr(d, k, math.nan)) for k in ("x", "y", "z", "roll", "pitch", "yaw")])
    return {k: np.array(v, dtype=float) for k, v in out.items() if len(v)}


def convert(J):
    """原始 → 真实（robot 顺序）。返回 t, q(16), qd(16), tau(16), temp(16)。"""
    t = J[:, 0]
    raw_pos, raw_vel, raw_tau, tmp = J[:, 2:18], J[:, 18:34], J[:, 34:50], J[:, 50:66]
    q = raw_pos * DIR + OFF
    q[:, 3::4] = raw_pos[:, 3::4]          # 轮子累积角不换算（无界，只作参考）
    qd = raw_vel * DIR
    tau = raw_tau * DIR
    return t, q, qd, tau, tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--raw", action="store_true", help="不做 dir/offset 换算（核对方向用）")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.bag.rstrip("/")), os.path.basename(a.bag.rstrip("/")) + "_parsed")
    D = read_bag(a.bag)
    if "joints" not in D:
        sys.exit("!! 没有 /JOINTS_DATA")
    t, q, qd, tau, tmp = convert(D["joints"])
    if a.raw:
        q, qd, tau = D["joints"][:, 2:18], D["joints"][:, 18:34], D["joints"][:, 34:50]
    dt = np.diff(t)
    print(f"/JOINTS_DATA {len(t)} 帧，时长 {t[-1]-t[0]:.2f} s，dt 中位 {np.median(dt)*1000:.2f} ms（{1/np.median(dt):.0f} Hz）")
    print("站立/起始 0.5 s 平均关节角（rad，robot 顺序）：")
    n0 = max(1, int(0.5 / max(np.median(dt), 1e-3)))
    for i, nm in enumerate(JOINT_NAMES):
        if i % 4 == 3:
            continue
        print("  %-9s q=%+.3f (%+.1f°)  |tau| 峰值 %.1f N·m  |qd| 峰值 %.1f rad/s" % (nm, q[:n0, i].mean(), math.degrees(q[:n0, i].mean()), np.abs(tau[:, i]).max(), np.abs(qd[:, i]).max()))
    for i in (3, 7, 11, 15):
        print("  %-9s |tau| 峰值 %.1f N·m  |qd| 峰值 %.1f rad/s" % (JOINT_NAMES[i], np.abs(tau[:, i]).max(), np.abs(qd[:, i]).max()))
    t_rx = D["joints"][:, 1]
    print("机器人钟 − AGX 接收钟 = %.3f s（中位）" % float(np.median(t - t_rx)))
    fwd = qd[:, 3::4].mean(0)
    print("轮速均值（换算后，前进应为负）：", np.round(fwd, 2))
    np.savez_compressed(out + ".npz", t=t, t_rx=t_rx, q=q, qd=qd, tau=tau, temp=tmp, joint_names=np.array(JOINT_NAMES),
                        imu=D.get("imu", np.zeros((0, 10))), imu103=D.get("imu103", np.zeros((0, 11))),
                        tf=D.get("tf", np.zeros((0, 8))), steer=D.get("steer", np.zeros((0, 7))))
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"q_{n}" for n in JOINT_NAMES] + [f"qd_{n}" for n in JOINT_NAMES] + [f"tau_{n}" for n in JOINT_NAMES])
        for k in range(len(t)):
            w.writerow([f"{t[k]:.4f}"] + [f"{v:.5f}" for v in q[k]] + [f"{v:.4f}" for v in qd[k]] + [f"{v:.3f}" for v in tau[k]])
    print("已存", out + ".npz / .csv")
    if "tf" in D:
        tf = D["tf"]; print(f"/tf base_link {len(tf)} 条，z 范围 {tf[:,3].min():.3f}~{tf[:,3].max():.3f}")
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        t0 = t - t[0]
        for leg in range(4):
            for j, lab in enumerate(("hipx", "hipy", "knee")):
                axes[0].plot(t0, q[:, leg * 4 + j], label=f"{JOINT_NAMES[leg*4+j]}", lw=0.8)
                axes[1].plot(t0, tau[:, leg * 4 + j], lw=0.8)
            axes[2].plot(t0, qd[:, leg * 4 + 3], label=JOINT_NAMES[leg * 4 + 3], lw=0.8)
            axes[3].plot(t0, tau[:, leg * 4 + 3], lw=0.8)
        axes[0].set_ylabel("腿关节角 rad"); axes[1].set_ylabel("腿关节力矩 N·m"); axes[2].set_ylabel("轮速 rad/s"); axes[3].set_ylabel("轮力矩 N·m")
        axes[0].legend(ncol=6, fontsize=7); axes[2].legend(ncol=4, fontsize=7); axes[3].set_xlabel("s")
        if "tf" in D:
            ax2 = axes[0].twinx(); ax2.plot(D["tf"][:, 0] - t[0], D["tf"][:, 3], "k--", lw=1.0, label="base z (tf)"); ax2.set_ylabel("base z m")
        fig.tight_layout(); fig.savefig(out + ".png", dpi=110); print("图", out + ".png")


if __name__ == "__main__":
    main()
