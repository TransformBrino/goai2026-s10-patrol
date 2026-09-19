#!/bin/bash
# 09-13：把楼梯专家候选 StairN-E2_10300 装到 AGX 楼梯槽（作者拍板后才跑）。步骤：传包 → 校 md5 → 装 night4q 控制台 → 起控制台时带 S10_STAIR_POLICY=StairN-E2_10300。
# 用法：bash deploy_e2.sh            （要 AGX 在线、无 runner、未接管）
#       bash deploy_e2.sh revert     （退回 9300：去掉环境变量重启控制台，night4q 文件可留）
set -u
K="-i $HOME/.ssh/agx_s10 -o HostKeyAlias=10.21.33.102 -o ConnectTimeout=8 -o BatchMode=yes"
A=robot@10.18.2.240
D=/tmp/claude-1000/-home-robot--------S10------20260908/29d1e03d-2844-4a17-8e20-0ee6f92acf68/scratchpad
if [ "${1:-}" = revert ]; then
  ssh $K $A 'cd /home/robot; sed -i "/^export S10_STAIR_POLICY=/d" restart_console_watch.sh boot_up.sh; R=$(pgrep -x rl_deploy); A2=$(curl -s -m 3 http://127.0.0.1:8089/api/state | python3 -c "import sys,json; print(json.load(sys.stdin)[\"state\"][\"armed\"])"); [ -z "$R" ] && [ "$A2" = False ] || { echo "有 runner/已接管，不重启"; exit 2; }; (bash /home/robot/restart_console_watch.sh console > /tmp/restart_console.log 2>&1 < /dev/null); sleep 2; P=$(pgrep -f "python3 /home/robot/s10_ctrl.py" | head -1); tr "\0" "\n" < /proc/$P/environ | grep -c S10_STAIR_POLICY; echo "已退回 9300"'; exit
fi
for P in StairN-E2_10300 StairN-E4_10600 StairN-E4_11099 StairN-E5_11200 StairN-S3_13800 StairN-S8_14800; do scp -q -r $K ~/s10_logs/release/$P $A:/home/robot/policies/ || { echo "传包失败 $P"; exit 1; }; done
for P in V1HTrotH4_11500 V1HTrotH6_12099 V1HTrotH7_11800 V1HTrotT3_13400; do scp -q -r $K ~/s10_logs/release/$P $A:/home/robot/policies/ || { echo "传包失败 $P"; exit 1; }; done
scp -q $K $D/nightfix/s10_ctrl.night5c.py $A:/tmp/s10_ctrl.night4q.py || exit 1
ssh $K $A 'cd /home/robot/policies/StairN-E2_10300 && md5sum -c md5.txt | grep -c 成功 && md5sum stairne2_10300.onnx | grep -q "^e0ef6b68" && echo "E2 onnx md5 OK"
cd /home/robot; R=$(pgrep -x rl_deploy); A2=$(curl -s -m 3 http://127.0.0.1:8089/api/state | python3 -c "import sys,json; print(json.load(sys.stdin)[\"state\"][\"armed\"])"); [ -z "$R" ] && [ "$A2" = False ] || { echo "有 runner/已接管，不装"; exit 2; }
md5sum s10_ctrl.py | cut -c1-8; cp -p s10_ctrl.py s10_ctrl.py.pre_night4q; cp /tmp/s10_ctrl.night4q.py s10_ctrl.py; python3 -c "import ast;ast.parse(open(\"/home/robot/s10_ctrl.py\",encoding=\"utf-8\").read())"
grep -q "^export S10_STAIR_POLICY=" restart_console_watch.sh || sed -i "s|^export S10_STAIR_HOLD=0.*|&\nexport S10_STAIR_POLICY=StairN-E2_10300   # 09-13 作者拍板：楼梯槽装 E2 候选（未真机验收）|" restart_console_watch.sh
grep -q "^export S10_STAIR_POLICY=" boot_up.sh || sed -i "s|^export S10_STAIR_HOLD=0.*|&\nexport S10_STAIR_POLICY=StairN-E2_10300   # 09-13 作者拍板：楼梯槽装 E2 候选（未真机验收）|" boot_up.sh
bash -n restart_console_watch.sh && bash -n boot_up.sh
(bash /home/robot/restart_console_watch.sh console > /tmp/restart_console.log 2>&1 < /dev/null); sleep 2
P=$(pgrep -f "python3 /home/robot/s10_ctrl.py" | head -1); tr "\0" "\n" < /proc/$P/environ | grep "S10_STAIR_POLICY\|S10_STAIR_HOLD"; md5sum s10_ctrl.py | cut -c1-8
curl -s -m 3 http://127.0.0.1:8089/api/state | python3 -c "import sys,json; j=json.load(sys.stdin); s=j[\"state\"]; print(\"armed\",s[\"armed\"],\"runner\",s[\"runner_pid\"]); print([l[:110] for l in s.get(\"log\",[])[:2]])"'
