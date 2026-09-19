#!/usr/bin/env python3
"""解析 09-10 真机跑我方策略的 seg 包：/JOINTS_DATA /JOINTS_CMD /IMU_DATA /cmd_vel /robot_mode /rosout /RLSM_RUNNING /height_scan。
重点：把 runner 实际下发的 目标角/kp/kd 解出来（→ 默认位形、轮阻尼是否按契约），以及摔倒时刻的姿态与关节。
用法: /usr/bin/python3 s10_dev/parse_seg_bag.py <bag目录> [--out 前缀]
"""
import argparse, json, math, os, sys
import numpy as np

JN = [f"{l}_{j}" for l in ("fl", "fr", "hl", "hr") for j in ("hipx", "hipy", "knee", "wheel")]
OFF = np.array([-35,-145,156,0, 35,-145,156,0, -35,145,-156,0, 35,145,-156,0], float) * math.pi/180
DIR = np.array([1,1,-1,1, 1,-1,1,-1, -1,1,-1,1, -1,-1,1,-1], float)

def stamp(msg, tb):
    h = getattr(msg, "header", None)
    st = getattr(h, "stamp", None) if h is not None else None
    if st is not None and hasattr(st, "sec"):
        v = float(st.sec) + float(st.nanosec)*1e-9
        if v > 0: return v
    return tb*1e-9

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("bag"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(a.bag.rstrip("/"), "parsed")
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = SequentialReader(); r.open(StorageOptions(uri=a.bag, storage_id=""), ConverterOptions("cdr","cdr"))
    topics = {t.name: t.type for t in r.get_all_topics_and_types()}
    cls = {}
    for n,t in topics.items():
        try: cls[n] = get_message(t)
        except Exception: pass
    J, C, I, V, M, R, RL, HS = [], [], [], [], [], [], [], []
    while r.has_next():
        tp, data, tb = r.read_next()
        if tp not in cls: continue
        m = deserialize_message(data, cls[tp])
        if tp == "/JOINTS_DATA":
            js = m.data.joints_data
            J.append([stamp(m,tb), tb*1e-9] + [j.position for j in js] + [j.velocity for j in js] + [j.torque for j in js])
        elif tp == "/JOINTS_CMD":
            js = m.data.joints_data
            C.append([stamp(m,tb), tb*1e-9] + [j.position for j in js] + [j.velocity for j in js]
                     + [j.kp for j in js] + [j.kd for j in js] + [j.torque for j in js])
        elif tp == "/IMU_DATA":
            d = m.data
            I.append([stamp(m,tb), d.roll, d.pitch, d.yaw, d.omega_x, d.omega_y, d.omega_z, d.acc_x, d.acc_y, d.acc_z])
        elif tp == "/cmd_vel":
            V.append([tb*1e-9, m.linear.x, m.linear.y, m.angular.z])
        elif tp == "/robot_mode":
            M.append([tb*1e-9, int(m.data)])
        elif tp == "/rosout":
            R.append((tb*1e-9, getattr(m,"name",""), m.msg))
        elif tp == "/RLSM_RUNNING":
            RL.append([tb*1e-9, float(m.value)])
        elif tp in ("/height_scan", "/height_scan_raw"):
            HS.append([tb*1e-9, len(m.data), float(np.mean(m.data)) if len(m.data) else float("nan")])
    J = np.array(J, float); C = np.array(C, float); I = np.array(I, float)
    V = np.array(V, float); M = np.array(M, float); RL = np.array(RL, float); HS = np.array(HS, float)
    res = {"bag": os.path.basename(a.bag.rstrip("/"))}
    print("== %s ==" % res["bag"])
    print("话题计数: JOINTS_DATA %d  JOINTS_CMD %d  IMU %d  cmd_vel %d  robot_mode %d  height_scan %d"
          % (len(J), len(C), len(I), len(V), len(M), len(HS)))
    if len(J):
        t0 = J[0,0]; q = J[:,2:18]*DIR + OFF; q[:,3::4] = J[:,5:18:4]
        qd = J[:,18:34]*DIR; tau = J[:,34:50]*DIR
        print("JOINTS_DATA %.1f Hz（bag 接收 %.1f Hz）" % (1/np.median(np.diff(J[:,0])), 1/np.median(np.diff(J[:,1]))))
        np.savez_compressed(out+"_joints.npz", t=J[:,0], t_rx=J[:,1], q=q, qd=qd, tau=tau, names=np.array(JN))
    if len(C):
        gp_raw = C[:,2:18]; gp = gp_raw*DIR + OFF; gv = C[:,18:34]*DIR
        kp = C[:,34:50]; kd = C[:,50:66]; ff = C[:,66:82]*DIR
        dt = np.diff(C[:,1]); print("JOINTS_CMD %.1f Hz" % (1/np.median(dt)))
        print("kp（腿/轮）: %s / %s   kd: %s / %s" % (np.unique(np.round(kp[:,:3],1))[:4], np.unique(np.round(kp[:,3::4],2))[:4],
                                                     np.unique(np.round(kd[:,:3],2))[:4], np.unique(np.round(kd[:,3::4],2))[:4]))
        n0 = min(len(C), 200)
        print("前 %d 帧目标角均值（= 默认位形 + 早期动作）:" % n0)
        for leg in range(4):
            print("   %-2s hipx %+.3f  hipy %+.3f  knee %+.3f   轮目标速度 %+.2f"
                  % (("fl","fr","hl","hr")[leg], gp[:n0,leg*4].mean(), gp[:n0,leg*4+1].mean(), gp[:n0,leg*4+2].mean(), gv[:n0,leg*4+3].mean()))
        np.savez_compressed(out+"_cmd.npz", t=C[:,0], t_rx=C[:,1], gp=gp, gp_raw=gp_raw, gv=gv, kp=kp, kd=kd, ff=ff, names=np.array(JN))
    if len(I):
        t0 = I[0,0]
        roll = np.degrees(I[:,1]); pitch = np.degrees(I[:,2])
        print("IMU %.1f Hz  roll 范围 %+.1f~%+.1f°  pitch 范围 %+.1f~%+.1f°" % (1/np.median(np.diff(I[:,0])), roll.min(), roll.max(), pitch.min(), pitch.max()))
        bad = np.where((np.abs(roll) > 30) | (np.abs(pitch) > 30))[0]
        if len(bad): print("   首次 |roll|或|pitch|>30° 于 t=%.2f s（相对 IMU 起点）" % (I[bad[0],0]-t0))
        np.savez_compressed(out+"_imu.npz", t=I[:,0], roll=roll, pitch=pitch, yaw=np.degrees(I[:,3]), w=I[:,4:7], acc=I[:,7:10])
    if len(V):
        print("cmd_vel %.1f Hz  vx 范围 %.2f~%.2f  非零帧 %d/%d" % (1/np.median(np.diff(V[:,0])), V[:,1].min(), V[:,1].max(), int((np.abs(V[:,1])>1e-3).sum()), len(V)))
        np.savez_compressed(out+"_cmd_vel.npz", t=V[:,0], vx=V[:,1], vy=V[:,2], wz=V[:,3])
    if len(M): print("robot_mode 序列:", [(round(m[0]-M[0,0],2), int(m[1])) for m in M])
    if len(RL): print("RLSM_RUNNING 值:", np.unique(RL[:,1]))
    print("---- rosout（runner/节点日志）%d 条 ----" % len(R))
    for t,n,s in R: print("   [%s] %s" % (n, s.strip()[:300]))
    print()

if __name__ == "__main__":
    main()
