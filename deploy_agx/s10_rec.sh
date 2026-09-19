#!/bin/bash
# READ-ONLY recorder for S10. Subscribes only -- never publishes.
# 依赖已本地化，不再引用另一队伍的工作空间
source /home/robot/s10_env.sh
export ROS_DOMAIN_ID=0
OUT=/home/robot/s10_data
PIDF=/tmp/s10_rec.pid; NAMEF=/tmp/s10_rec.name; LOG=/tmp/s10_rec.log
CORE="/JOINTS_DATA /JOINTS_DATA_10HZ /JOINTS_CMD /IMU_DATA /BATTERY_DATA"   # 09-12 22:1x 加 10 Hz 关节流：厂家运控（SDK 关）时 /JOINTS_DATA 只有 1 Hz
# 用 SDK 跑我们自己的策略时，这几路是必需的 —— 原来的清单是为"录官方运控"
# 设计的，压根没有它们，导致录下来的段无法回答"我们下发了什么、策略看到了什么"：
#   /cmd_vel        我们下发的速度指令（对齐 command -> response 的唯一依据）
#   /robot_mode     我们下发的状态机跳转
#   /climb_mode     策略热切换标志（ClimbG 是否接管）
#   /height_scan    策略实际吃到的 187 维地形观测
#   /height_scan_raw 未裁剪版本
#   /hmap_info      高程图质量（hole_rate / coverage / frames）
#   /rosout         runner 的 [MODE]/[policy] 裁决，含它拒绝跳转时报出的真实状态
SDK="/cmd_vel /robot_mode /climb_mode /height_scan /height_scan_raw /hmap_info /rosout /Odometry /s10/imu"
# 09-12：位姿源已换 FAST-LIO（/Odometry，/tf 里也有）；/s10/imu 是喂给 FAST-LIO 的 IMU（作者要的"补洞前后高程图 + 定位"对照用）
# /kiss/odometry 是我们自己的位姿源。训练侧 AGX部署步骤.md:6 的录制清单里写的是
# /localization/pose（傅工契约），但那个话题在这台机器上从来没存在过。
EXTRA="/IMU_103 /tf /tf_static /HANDLE_STEER /STEER /REAL_STEER /GAMEPAD_KEY /RLSM_RUNNING /slope_mode /planner_mode /HEIGHT_MAP_STATUS /traversability_status_code /LIO_ODOM /LIO_ODOM_HIGH_FREQUENCY /ODOM"
# 点云极大：实测 100719 点/帧 x 16 B x 10 Hz ~= 16 MB/s（约 1 GB/分钟）。
# 只在明确要做感知对齐的短段里开，别混进常规录制。
CLOUD="/LIDAR/POINTS_MERGED"
mkdir -p $OUT

realpid () { pgrep -f "ros2 bag record -o $1( |\$)" | head -1; }

case "$1" in
  preflight) python3 /home/robot/s10_stamp.py ;;

  start)
    NAME="$2"; MODE="${3:-extra}"    # core | extra(默认) | cloud
    RUN=$(pgrep -f "ros2 bag record -o " | head -1)
    [ -n "$RUN" ] && { echo "ALREADY_RECORDING pid=$RUN"; exit 1; }
    [ -e "$OUT/$NAME" ] && { echo "NAME_EXISTS $OUT/$NAME -- pick another name"; exit 1; }
    case "$MODE" in
      core)  TOPICS="$CORE" ;;
      cloud) TOPICS="$CORE $SDK $EXTRA $CLOUD" ;;
      *)     TOPICS="$CORE $SDK $EXTRA" ;;
    esac
    WANT=$(echo $TOPICS | wc -w)
    PRE=$(python3 /home/robot/s10_stamp.py)
    cd $OUT
    setsid nohup ros2 bag record -o "$NAME" $TOPICS > $LOG 2>&1 &
    sleep 3
    PID=$(realpid "$NAME")
    echo "${PID:-0}" > $PIDF; echo "$NAME" > $NAMEF
    GOT=$(grep -c "Subscribed to topic" $LOG)
    echo "{\"phase\":\"start\",\"bag\":\"$NAME\",\"subscribed\":$GOT,\"wanted\":$WANT,\"probe\":$PRE}" > "/tmp/s10_${NAME}.start.json"
    if grep -qi "Failure in topics discovery" $LOG; then
      echo "!!! DISCOVERY_FAILED -- STOP, bag NOT trustworthy"
      grep -iE "error|could not" $LOG | sort -u | head -5
    fi
    echo "STARTED $NAME  pid=${PID:-NONE}  subscribed=$GOT/$WANT"
    for t in $TOPICS; do
      grep -q "Subscribed to topic '$t'" $LOG || echo "   [3s 快照] 尚未订阅: $t  (低频话题会稍后自动补上，非错误)"
    done
    [ -z "$PID" ] && echo "   !!! could not locate recorder pid -- STOP will fail, abort this take"
    echo "$PRE" | grep -q '"tf_alive": false' && echo "   note: /tf not publishing (no body pose)"
    ;;

  stop)
    NAME=$(cat $NAMEF 2>/dev/null)
    [ -n "$NAME" ] || { echo "NOT_RECORDING"; exit 1; }
    PID=$(cat $PIDF 2>/dev/null)
    kill -0 "$PID" 2>/dev/null || PID=$(realpid "$NAME")
    [ -n "$PID" ] || { echo "recorder for $NAME not found (already gone?)"; }
    POST=$(python3 /home/robot/s10_stamp.py)
    [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
    for i in $(seq 1 60); do
      [ -f "$OUT/$NAME/metadata.yaml" ] && break
      sleep 0.5
    done
    sleep 1
    if [ ! -f "$OUT/$NAME/metadata.yaml" ]; then
      echo "!!! metadata.yaml still missing -- forcing second SIGINT"
      P2=$(realpid "$NAME"); [ -n "$P2" ] && kill -TERM "$P2" 2>/dev/null
      for i in $(seq 1 40); do [ -f "$OUT/$NAME/metadata.yaml" ] && break; sleep 0.5; done
    fi
    rm -f $PIDF $NAMEF
    cp "/tmp/s10_${NAME}.start.json" "$OUT/$NAME/marks.json" 2>/dev/null
    INFO=$(ros2 bag info "$OUT/$NAME" 2>&1)
    DUR=$(echo "$INFO" | grep -oP "Duration:\s+\K[0-9.]+")
    echo "{\"phase\":\"stop\",\"bag\":\"$NAME\",\"duration_s\":${DUR:-0},\"probe\":$POST}" >> "$OUT/$NAME/marks.json"
    chown -R robot:robot $OUT 2>/dev/null
    echo "STOPPED $NAME  duration=${DUR:-UNKNOWN}s"
    if [ -z "$DUR" ]; then echo "!!! bag did not finalize:"; echo "$INFO" | head -5; exit 1; fi
    echo "$INFO" | grep -oP "Topic: \K[^|]+\|[^|]+\| Count: \d+" | sed 's/| Type: [^|]*//'
    echo "--- acceptance ---"
    ok=1
    check () {
      c=$(echo "$INFO" | grep -oP "Topic: $1 \| Type: [^|]+\| Count: \K\d+"); c=${c:-0}
      exp=$(awk "BEGIN{printf \"%d\", $2*$DUR}")
      lo=$(awk "BEGIN{printf \"%d\", $exp*$3}")
      if [ "$c" -ge "$lo" ] && [ "$c" -gt 0 ]; then echo "  OK   $1  $c (expect ~$exp)"
      else echo "  FAIL $1  $c (expect ~$exp)  <<<<<<"; ok=0; fi
    }
    check /JOINTS_DATA 200 0.9
    check /IMU_DATA 200 0.9
    check /IMU_103 200 0.9
    [ "$ok" = 1 ] && echo "  ==> SEGMENT ACCEPTED" || echo "  ==> SEGMENT REJECTED, re-record"
    ;;

  status)
    P=$(pgrep -f "ros2 bag record -o " | head -1)
    if [ -n "$P" ]; then echo "RECORDING $(cat $NAMEF 2>/dev/null) pid=$P size=$(du -sh $OUT/$(cat $NAMEF 2>/dev/null) 2>/dev/null | cut -f1)"
    else echo "IDLE"; fi ;;

  list)
    for d in $OUT/*/; do
      [ -d "$d" ] || continue
      i=$(ros2 bag info "$d" 2>/dev/null)
      dur=$(echo "$i" | grep -oP "Duration:\s+\K[0-9.]+")
      jc=$(echo "$i" | grep -oP "Topic: /JOINTS_DATA \| Type: [^|]+\| Count: \K\d+")
      tf=$(echo "$i" | grep -oP "Topic: /tf \| Type: [^|]+\| Count: \K\d+")
      printf "%-26s dur=%-9s JOINTS=%-7s tf=%-6s %s\n" "$(basename $d)" "${dur:-BROKEN}" "${jc:-0}" "${tf:-0}" "$(du -sh $d 2>/dev/null | cut -f1)"
    done ;;
esac
