# -*- coding: utf-8 -*-
"""merger 看门狗 + 定位（FAST-LIO /Odometry，09-12 起；之前是 KISS）里程计报警。

【为什么要有 —— 2026-09-11 实测】dual_airy_merger_node 会**死锁**：
进程活着、13 个线程全在 S 睡眠、utime/stime 三秒内一个 jiffy 都不动，
DDS 发布者被其他参与者老化掉（`ros2 topic info /LIDAR/POINTS_MERGED` 报 pub=0），
于是 KISS-ICP 和 hmap 一起断粮 —— **整条感知链静默失明**，
而原始 /rslidar_{front,rear}/points 仍然是好的 10 Hz。
（那次还顺带发现：重启后 /LIDAR/POINTS_MERGED 从 7.75 Hz 恢复到 10.10 Hz ——
 之前记的"23% 配对丢失"不是雷达时钟问题，是 merger 已经半死。
 实测两台雷达 header.stamp 中位错位 0.0 ms，98% 能配对。）

失明本身不会掀翻狗：runner 侧 hscan_max_age_ms_=100，超期走
SetHeightScanFallback()（按平地走）。但那样越障必然失败，而且不看日志发现不了。

【策略】只有在"原始雷达活着、合并云死了"时才动手 —— 那才是 merger 的锅。
重启之间至少隔 --cooldown 秒；连续失败 --max-fail 次就不再试、只大声记日志
（免得无限重启掩盖真问题）。只用精确进程名 dual_airy_merger_node ——
【教训】用 "dual_airy_merger" 做 pgrep 模式会连 rslidar_sdk 一起匹配到
（它的命令行里含 config_path:=.../dual_airy_merger/config/...），我 9-11 就这么
把雷达驱动一起杀了。

【09-11 追加：KISS 只报警、不重启（作者拍板）】
13:40–13:58 KISS-ICP 静默卡死 18 分钟：进程活着、照常收点云、照常耗 CPU，
但不再发 /kiss/odometry，也不写任何日志 -> hmap 的位姿闸门停发 /height_scan
-> 感知全盲，而且没人发现。自动重启会让里程计世界原点归零（将来导航若用这个
原点记航点会乱），所以作者定：这里只记日志、控制台报红，重启由人决定。
人工恢复：bash /home/robot/kiss_restart.sh（重启 KISS，再重建一次高程图）。
合并云本身死了时 KISS 当然也没输出 —— 那是 merger 的锅，不在这里报。
"""
import os
import argparse
import subprocess
import sys
import time

import rclpy
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, ReliabilityPolicy,
                       DurabilityPolicy, HistoryPolicy)
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

MERGER_PAT = "dual_airy_merger_node"      # 必须精确，见文件头
UP = "/home/robot/merger_up.sh"


def log(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale", type=float, default=3.0, help="合并云静默多久算死")
    ap.add_argument("--cooldown", type=float, default=45.0, help="两次重启的最小间隔")
    ap.add_argument("--max-fail", type=int, default=3, help="连续失败几次就不再试")
    ap.add_argument("--lio-auto", type=int, default=0,
                    help="09-12 候选：1 = 定位(FAST-LIO)长时间断流后合并云恢复、或位姿发散时，自动 lio_restart.sh（含 hmap 重建）；0 = 只报警（默认）")
    ap.add_argument("--lio-long", type=float, default=20.0, help="定位静默超过这么久算长断流（恢复后要重启才可信）")
    ap.add_argument("--lio-far", type=float, default=500.0, help="定位位置离原点超过这么多米算发散")
    ap.add_argument("--lio-cooldown", type=float, default=120.0, help="两次自动重启定位的最小间隔")
    ap.add_argument("--kiss-stale", type=float, default=2.0,
                    help="合并云正常而 /kiss/odometry 静默多久就报警（只报警不重启）")
    a = ap.parse_args()

    rclpy.init()
    n = rclpy.create_node("lidar_watch")
    t = {"m": 0.0, "f": 0.0, "r": 0.0, "k": 0.0}
    pos = {"r": 0.0}                                  # 定位位置离原点的距离（发散判据）
    n.create_subscription(PointCloud2, "/LIDAR/POINTS_MERGED",
                          lambda _: t.__setitem__("m", time.monotonic()),
                          qos_profile_sensor_data)
    n.create_subscription(PointCloud2, "/rslidar_front/points",
                          lambda _: t.__setitem__("f", time.monotonic()),
                          qos_profile_sensor_data)
    n.create_subscription(PointCloud2, "/rslidar_rear/points",
                          lambda _: t.__setitem__("r", time.monotonic()),
                          qos_profile_sensor_data)
    # KISS 发 RELIABLE；BEST_EFFORT 订阅能匹配任何发布者，且不给它回压
    n.create_subscription(Odometry, os.environ.get("S10_ODOM_TOPIC", "/Odometry"),   # 09-12：FAST-LIO
                          lambda m: (t.__setitem__("k", time.monotonic()), pos.__setitem__("r", (m.pose.pose.position.x**2 + m.pose.pose.position.y**2 + m.pose.pose.position.z**2) ** 0.5)),
                          QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                                     durability=DurabilityPolicy.VOLATILE,
                                     history=HistoryPolicy.KEEP_LAST))
    log("看门狗启动：合并云静默 >%.1fs 且原始雷达仍在 -> 重启 merger"
        "（冷却 %.0fs，连续失败 %d 次后放弃）；合并云正常而定位(FAST-LIO)静默 >%.1fs -> 只报警"
        % (a.stale, a.cooldown, a.max_fail, a.kiss_stale))
    t0 = time.monotonic()
    last_restart, fails, quiet_since = 0.0, 0, None
    kiss_quiet, kiss_nag = None, 0.0
    lio_last_restart, lio_fails, lio_long_gap = 0.0, 0, False

    def lio_restart(why):
        nonlocal lio_last_restart, lio_fails
        if subprocess.run(["pgrep", "-x", "rl_deploy"], capture_output=True).returncode == 0:
            log("!! 定位需要重启（%s），但 runner 在跑，不动；人工：bash /home/robot/lio_restart.sh" % why); return
        try:
            armed = subprocess.run(["bash", "-c", "curl -s -m 2 http://127.0.0.1:8089/api/state | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"state\"][\"armed\"])'"], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            armed = "?"
        if armed != "False":
            # lioauto2（09-14）：作者要求"没有我们的 runner 在跑就允许重启"——接管着（比如官方遥控时页面还接管着）也重启，只记一句
            log("定位需要重启（%s），控制台已接管/状态 %s 但无 runner，照样重启" % (why, armed))
        if now - lio_last_restart < a.lio_cooldown or lio_fails >= a.max_fail:
            return
        lio_last_restart, lio_fails = now, lio_fails + 1
        log("定位自动重启（%s，第 %d 次）：bash /home/robot/lio_restart.sh（含 hmap 重建）" % (why, lio_fails))
        subprocess.Popen(["setsid", "nohup", "bash", "/home/robot/lio_restart.sh"], stdout=open("/home/robot/lio_restart.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.3)
        now = time.monotonic()
        if now - t0 < 15.0:
            continue                          # 起来先看 15 秒，别抢在别人启动过程中动手
        raw_ok = (now - max(t["f"], t["r"])) < a.stale
        merged_ok = (now - t["m"]) < a.stale

        # ---- KISS：只报警。放在 merger 逻辑前面，因为下面 merged_ok 时会 continue。
        kiss_ok = (now - t["k"]) < a.kiss_stale
        if merged_ok and not kiss_ok:
            if kiss_quiet is None:
                kiss_quiet, kiss_nag = now, now
                log("!! 定位(FAST-LIO)静默 >%.1fs 而合并云正常 —— FAST-LIO 卡死或 IMU 断，hmap 已停发 "
                    "/height_scan（感知全盲）。只报警不重启；人工恢复：bash "
                    "/home/robot/lio_restart.sh" % a.kiss_stale)
            elif now - kiss_nag > 30.0:
                kiss_nag = now
                log("!! 定位仍静默，已 %.0fs" % (now - kiss_quiet))
        elif kiss_ok and kiss_quiet is not None:
            gap = now - kiss_quiet
            log("定位里程计恢复（静默了 %.1fs）" % gap)
            kiss_quiet = None
            if a.lio_auto and gap >= a.lio_long:
                lio_restart("断流 %.0f s 后恢复，FAST-LIO 长断流不自愈" % gap)
        if a.lio_auto and kiss_ok and pos["r"] > a.lio_far:
            lio_restart("位姿发散 %.0f m" % pos["r"])
        if a.lio_auto and merged_ok and kiss_quiet is not None and (now - kiss_quiet) >= a.lio_long and (now - t["m"]) < 1.0:
            lio_restart("合并云正常但定位已静默 %.0f s" % (now - kiss_quiet))

        if merged_ok:
            if quiet_since is not None:
                log("合并云恢复（静默了 %.1fs）" % (now - quiet_since))
            quiet_since, fails = None, 0
            continue
        if quiet_since is None:
            quiet_since = now
            log("合并云静默 >%.1fs（原始雷达 %s）" % (a.stale, "活着" if raw_ok else "也没了"))
        if not raw_ok:
            continue                          # 雷达本身没了，不是 merger 的锅，别乱杀
        if fails >= a.max_fail:
            continue                          # 已放弃，上面那条日志每次静默都会记
        if now - last_restart < a.cooldown:
            continue
        last_restart, fails = now, fails + 1
        pids = subprocess.run(["pgrep", "-f", MERGER_PAT], capture_output=True,
                              text=True).stdout.split()
        log("原始雷达好、合并云死 -> 判定 merger 死锁。kill -9 %s 并重启（第 %d 次）"
            % (pids or "(无进程)", fails))
        for p in pids:
            subprocess.run(["kill", "-9", p])
        time.sleep(1.5)
        subprocess.Popen(["setsid", "nohup", "bash", UP],
                         stdout=open("/home/robot/merger.log", "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        if fails >= a.max_fail:
            log("!! 连续 %d 次重启都没救回来，不再自动重启。"
                "去查 merger.log / 原始雷达配对情况（/tmp/skew.py）。" % fails)
    rclpy.shutdown()
    return 0


sys.exit(main())
