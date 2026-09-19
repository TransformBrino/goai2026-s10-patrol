#!/usr/bin/env python3
"""/cmd_vel(m/s) -> /STEER(drdds/msg/Steer, 归一化) 适配器：1.4 通路 A 的出口。

依据：官方 09-04 runner `dds_command_interface.hpp`（v2.0, 2026-08-27）「参考 basic_server 发出端」订阅
/STEER(data.x->前后, data.y->左右, data.yaw->偏航) 与 /GAMEPAD_KEY(String, 如 G20_KEY_L2 = RL 控制)。
机器人自带 basic_server 把手柄轴指令发布成同一话题；本节点站在手柄的位置上发同一话题——不需要 SDK 模式/授权码。
**是否真的被机器人自带 rl_deploy 接受，是 D0 第一项真机试验，纸面上定不了。**

把我方 C++ 接口的三道保护搬过来：钳位 / 斜率限制 / 500ms 看门狗归零；再加一道「未上膛不发」。
  S10_STEER_MAX_VX / VY / WZ   满杆对应的 m/s、m/s、rad/s —— **D0 要实测**（满杆 5s 走多远），默认 1.0/0.6/1.0 只是占位
  S10_STEER_SLEW_V / W         斜率限制（每秒最大变化，归一化量纲），<=0 不限
  S10_STEER_TIMEOUT_MS         /cmd_vel 超时归零，默认 500
用法：
  python3 steer_pub.py --armed            # 上膛：按 20Hz 发 /STEER
  python3 steer_pub.py --key G20_KEY_L2   # 单发一个按键（L1 站立 / L2 RL 控制 / R1 趴下 / R2 阻尼）然后退出
"""
import argparse
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from drdds.msg import Steer


def envf(k, d):
    try:
        return float(os.environ.get(k, d))
    except ValueError:
        return d


class SteerPub(Node):
    def __init__(self, armed):
        super().__init__("steer_pub")
        self.armed = armed
        self.max = (envf("S10_STEER_MAX_VX", 1.0), envf("S10_STEER_MAX_VY", 0.6), envf("S10_STEER_MAX_WZ", 1.0))
        self.slew = (envf("S10_STEER_SLEW_V", 0.0), envf("S10_STEER_SLEW_W", 0.0))
        self.timeout = envf("S10_STEER_TIMEOUT_MS", 500.0) / 1000.0
        self.tgt = [0.0, 0.0, 0.0]
        self.out = [0.0, 0.0, 0.0]
        self.last_rx = -1.0
        self.n_pub = 0
        self.n_to = 0
        self.pub = self.create_publisher(Steer, "/STEER", 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.hz = 20.0
        self.create_timer(1.0 / self.hz, self.tick)
        self.get_logger().info("armed=%s max(vx,vy,wz)=%s slew=%s timeout=%.0fms -- 满杆速度是占位值，D0 实测后再信"
                               % (armed, self.max, self.slew, self.timeout * 1000))

    def on_cmd(self, m):
        def clamp(v, lim):
            return max(-1.0, min(1.0, v / lim if lim > 0 else 0.0))
        self.tgt = [clamp(m.linear.x, self.max[0]), clamp(m.linear.y, self.max[1]), clamp(m.angular.z, self.max[2])]
        self.last_rx = time.monotonic()

    def tick(self):
        now = time.monotonic()
        dt = 1.0 / self.hz
        timed_out = self.last_rx < 0 or (now - self.last_rx) > self.timeout
        tgt = [0.0, 0.0, 0.0] if timed_out else self.tgt
        if timed_out and self.last_rx >= 0:
            self.n_to += 1
            if self.n_to in (1, 100):
                self.get_logger().warn("/cmd_vel 超时，/STEER 归零")
        else:
            self.n_to = 0
        for i in range(3):
            lim = (self.slew[0] if i < 2 else self.slew[1]) * dt
            d = tgt[i] - self.out[i]
            self.out[i] = tgt[i] if lim <= 0 or abs(d) <= lim else self.out[i] + (lim if d > 0 else -lim)
        if not self.armed:
            return                      # 未上膛：一帧都不发，避免和手柄的 /STEER 抢话题
        msg = Steer()
        msg.header.frame_id = self.n_pub
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data.x, msg.data.y, msg.data.yaw = float(self.out[0]), float(self.out[1]), float(self.out[2])
        self.pub.publish(msg)
        self.n_pub += 1
        if self.n_pub % 40 == 0:
            self.get_logger().info("/STEER x=%+.2f y=%+.2f yaw=%+.2f" % (msg.data.x, msg.data.y, msg.data.yaw))


def send_key(key):
    rclpy.init()
    n = Node("steer_key")
    p = n.create_publisher(String, "/GAMEPAD_KEY", 10)
    time.sleep(0.5)                       # 等发现
    m = String()
    m.data = key
    p.publish(m)
    time.sleep(0.2)
    n.get_logger().info("/GAMEPAD_KEY <- %s" % key)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--armed", action="store_true")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    if a.key:
        send_key(a.key)
    else:
        rclpy.init()
        node = SteerPub(a.armed)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
