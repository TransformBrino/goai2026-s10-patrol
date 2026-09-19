#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S10（09-12）：把机器人 /IMU_DATA（drdds/ImuData：roll pitch yaw omega_xyz acc_xyz，200 Hz，机体系，rad、rad/s、m/s²）
转成 sensor_msgs/Imu 发到 /s10/imu，给 FAST-LIO 用。时间戳原样透传（机器人时钟，与雷达同源）。只订阅一个、只发一个话题。"""
import math, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from drdds.msg import ImuData

OUT = sys.argv[1] if len(sys.argv) > 1 else "/s10/imu"


class Relay(Node):
    def __init__(self):
        super().__init__("s10_imu_relay")
        self.pub = self.create_publisher(Imu, OUT, QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.sub = self.create_subscription(ImuData, "/IMU_DATA", self.cb, qos_profile_sensor_data)
        self.n = 0; self.t0 = time.monotonic(); self.last_stamp = None; self.n_dup = 0
        self.create_timer(5.0, self.report)

    def cb(self, m):
        d = m.data
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and st <= self.last_stamp:   # 重复/倒退的时间戳会让 FAST-LIO 报 "imu loop back"，丢掉
            self.n_dup += 1
            return
        self.last_stamp = st
        o = Imu()
        # 不能直接 o.header = m.header：drdds 包自带的 Header 类型和 sensor_msgs 用的不是同一份 typesupport，会在 C 转换层断言崩溃
        o.header.stamp.sec = int(m.header.stamp.sec)
        o.header.stamp.nanosec = int(m.header.stamp.nanosec)
        o.header.frame_id = "body"
        cr, sr = math.cos(d.roll / 2), math.sin(d.roll / 2)
        cp, sp = math.cos(d.pitch / 2), math.sin(d.pitch / 2)
        cy, sy = math.cos(d.yaw / 2), math.sin(d.yaw / 2)
        o.orientation.w = cr * cp * cy + sr * sp * sy
        o.orientation.x = sr * cp * cy - cr * sp * sy
        o.orientation.y = cr * sp * cy + sr * cp * sy
        o.orientation.z = cr * cp * sy - sr * sp * cy
        o.angular_velocity.x = float(d.omega_x); o.angular_velocity.y = float(d.omega_y); o.angular_velocity.z = float(d.omega_z)
        o.linear_acceleration.x = float(d.acc_x); o.linear_acceleration.y = float(d.acc_y); o.linear_acceleration.z = float(d.acc_z)
        self.pub.publish(o)
        self.n += 1

    def report(self):
        dt = time.monotonic() - self.t0
        self.get_logger().info("/IMU_DATA → %s  %.0f Hz  丢重复戳 %d" % (OUT, self.n / max(dt, 1e-3), self.n_dup))
        self.n = 0; self.t0 = time.monotonic(); self.n_dup = 0


def main():
    rclpy.init()
    n = Relay()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
