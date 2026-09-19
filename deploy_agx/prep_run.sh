#!/bin/bash
# 开跑前的一键准备：把高程图的零点重新对准当前站姿，然后验收。
#
# 为什么需要它：hmap 把地图无限累积在 KISS-ICP 的世界系里，而那个系的 z 原点
# 就是 KISS 启动那一刻狗所在的高度。狗趴→站升 31cm，旧地图里错高度的点还留着，
# 读数就漂（2026-09-10 夜实测从 -0.08 漂到 -0.5681，策略吃到假地形抽搐摔倒）。
#
# 用法：狗【站好、别动】，然后跑这个。约 50 秒。
#      结果是 GO 才能接策略；NO-GO 就别开高程图，让 runner 用平地兜底值。
set +e
source /home/robot/s10_env.sh >/dev/null 2>&1
source /home/robot/lio_ws/install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID=0
TOL=0.015

echo "════════ 1. 先确认狗是站着且不动 ════════"
python3 - <<'PY'
import math, sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from drdds.msg import ImuData, JointsData
rclpy.init(); n=rclpy.create_node('prep_pose')
be=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
              durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
box={}
n.create_subscription(ImuData,'/IMU_DATA_10HZ', lambda m: box.__setitem__('i',m), be)
n.create_subscription(JointsData,'/JOINTS_DATA_10HZ', lambda m: box.__setitem__('j',m), be)
t=time.time()
while len(box)<2 and time.time()-t<10: rclpy.spin_once(n, timeout_sec=0.05)
bad=[]
if 'i' in box:
    d=box['i'].data
    r,p=math.degrees(d.roll), math.degrees(d.pitch)
    print("  姿态  roll %+.2f°  pitch %+.2f°" % (r,p))
    if max(abs(r),abs(p))>3: bad.append("机体倾斜超过 3°，不是平地站姿")
else:
    bad.append("收不到 IMU")
if 'j' in box:
    OFF=[-35,-145,156,0, 35,-145,156,0, -35,145,-156,0, 35,145,-156,0]
    DIR=[1,1,-1,1, 1,-1,1,-1, -1,1,-1,1, -1,-1,1,-1]
    jd=box['j'].data.joints_data
    W=(3,7,11,15)
    ws=sum(abs(jd[k].velocity) for k in W)/4.0
    hy=abs(math.degrees(jd[1].position*DIR[1])+OFF[1])
    print("  轮速  %.3f rad/s     前左 hipy %.1f°" % (ws, hy))
    if ws>0.15: bad.append("轮子在转（%.2f rad/s），狗没停稳" % ws)
    if hy>45: bad.append("腿还蜷着（hipy %.0f°），狗是趴的不是站的" % hy)
else:
    bad.append("收不到关节数据")
if bad:
    print("  ✗ " + "；".join(bad))
    sys.exit(1)
print("  ✓ 站姿平稳")
rclpy.shutdown()
PY
[ $? -ne 0 ] && { echo; echo "  ==> NO-GO：先把狗放平站好再跑。"; exit 1; }

echo
echo "════════ 2. 重置里程计原点 + 重建高程图 ════════"
pkill -9 -f hmap_node.py 2>/dev/null
pkill -9 -f kiss_icp_node 2>/dev/null
sleep 3
setsid nohup bash /home/robot/kiss_up.sh > /tmp/kiss.log 2>&1 < /dev/null &
sleep 12
setsid nohup bash /home/robot/hmap_kiss_up.sh > /tmp/hmap_kiss.log 2>&1 < /dev/null &
echo "  等地图累积 25 秒…"
sleep 25
pgrep -af 'kiss_icp_node|hmap_node.py' | grep -vE 'ros2 run|bash -c' | sed 's/^/  /'

echo
echo "════════ 3. 平地基准验收 ════════"
python3 - "$TOL" <<'PY'
import math, statistics, sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
TOL=float(sys.argv[1]); TARGET=-0.08
rclpy.init(); n=rclpy.create_node('prep_chk')
rel=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
               durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
be=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
              durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
meds=[]; holes=[]; box={}
def on_hs(m):
    v=list(m.data); meds.append(statistics.median(v))
    holes.append(sum(1 for x in v if x==0.0)/float(len(v)))
n.create_subscription(Float32MultiArray,'/height_scan', on_hs, rel)
for q in (rel,be):
    n.create_subscription(Odometry,'/kiss/odometry', lambda m: box.__setitem__('o',m), q)
t=time.time()
while time.time()-t<20: rclpy.spin_once(n, timeout_sec=0.02)
if not meds:
    print("  ✗ 一帧 /height_scan 都没收到"); sys.exit(1)
m=statistics.median(meds); sd=statistics.pstdev(meds); hr=sum(holes)/len(holes)
print("  /height_scan  %d 帧   中位数 %+.4f   逐帧标准差 %.4f   空洞率 %.3f"
      % (len(meds), m, sd, hr))
if 'o' in box:
    p=box['o'].pose.pose.position
    print("  KISS 原点已复位  x %+.4f  y %+.4f  z %+.4f" % (p.x,p.y,p.z))
ok = abs(m-TARGET)<=TOL
print()
if ok:
    print("  ==> GO：基准 %+.4f 在 %.3f±%.3f 内，高程图可信，可以接策略。" % (m,TARGET,TOL))
else:
    print("  ==> NO-GO：基准 %+.4f 偏离 %.3f（容差 ±%.3f）。" % (m,TARGET,TOL))
    print("      别开高程图接策略 —— 停掉 hmap，让 runner 用平地兜底值更安全。")
    print("      需要的外参修正：z 再加 %+.4f" % (TARGET-m))
sys.exit(0 if ok else 1)
PY
RC=$?
echo
[ $RC -eq 0 ] && echo "════════ 结论：GO ════════" || echo "════════ 结论：NO-GO ════════"
exit $RC
