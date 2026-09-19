#!/usr/bin/env python3
"""仿真里让机器人站起并进入 RL 控制（真机上这一步由手柄或 steer_pub --key 完成）。
按 eval_driver.bring_up 的做法：/robot_mode=1 发 40 次 → 等 3s → /robot_mode=6 发 40 次 → 等 2s。
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8, Float32MultiArray


def main():
    rclpy.init()
    n = Node("bringup_sim")
    pub = n.create_publisher(UInt8, "/robot_mode", 10)
    state = {}
    n.create_subscription(Float32MultiArray, "/sim/state", lambda m: state.update(z=m.data[2], t=m.data[13]), 10)
    t0 = time.time()
    while "z" not in state and time.time() - t0 < 20:
        rclpy.spin_once(n, timeout_sec=0.1)
    if "z" not in state:
        print("!! 收不到 /sim/state"); return 1
    print("仿真已连上 z=%.3f" % state["z"])
    for mode, wait in ((1, 3.0), (6, 2.0)):
        for _ in range(40):
            m = UInt8(); m.data = mode; pub.publish(m); rclpy.spin_once(n, timeout_sec=0.05)
        t1 = time.time()
        while time.time() - t1 < wait:
            rclpy.spin_once(n, timeout_sec=0.05)
        print("mode=%d 后 z=%.3f" % (mode, state.get("z", -1)))
    n.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
