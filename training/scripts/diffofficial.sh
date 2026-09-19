#!/usr/bin/env bash
# 我们跑的还是不是官方那套？逐文件比对 s10_ws/src/S10_sdk_deploy 与原始素材包。
#
# 最要命的是**场景/模型**被改过 —— 那会让成绩完全不可比。
# 其次是控制器改动 —— 那是允许的（赛题就是让你做运控），但要能说清改了什么。
P=/mnt/c/Users/86156/Desktop/机器狗项目
OURS=$P/s10_ws/src/S10_sdk_deploy
ORIG=$P/goai_embodied_future_material-main/src/S10_sdk_deploy

echo "=== 0) 原始素材包在不在 ==="
[ -d "$ORIG" ] && echo "  ✓ $ORIG" || {
  echo "  ✗ 找不到，看看素材包里有什么："
  ls "$P/goai_embodied_future_material-main/src" 2>/dev/null | sed 's/^/    /'
  exit 1
}

echo
echo "=== 1) 场景与模型文件（改了就不可比）==="
for f in $(cd "$ORIG" && find S10_description -name '*.xml' 2>/dev/null | sort); do
  a="$ORIG/$f"; b="$OURS/$f"
  if [ ! -f "$b" ]; then echo "  ✗ 我们这边缺 $f"; continue; fi
  if cmp -s "$a" "$b"; then
    printf "  ✓ 一致    %s\n" "$f"
  else
    d=$(diff "$a" "$b" | grep -c '^[<>]')
    printf "  ✗ 有差异  %-52s %s 行\n" "$f" "$d"
  fi
done
echo "  --- 我们新增的 xml ---"
for f in $(cd "$OURS" && find S10_description -name '*.xml' 2>/dev/null | sort); do
  [ -f "$ORIG/$f" ] || echo "      + $f"
done

echo
echo "=== 2) 网格/资源有没有动过 ==="
for d in S10_description; do
  ha=$(cd "$ORIG/$d" 2>/dev/null && find . -name '*.stl' -o -name '*.obj' -o -name '*.STL' | sort | xargs -r md5sum 2>/dev/null | md5sum | cut -c1-12)
  hb=$(cd "$OURS/$d" 2>/dev/null && find . -name '*.stl' -o -name '*.obj' -o -name '*.STL' | sort | xargs -r md5sum 2>/dev/null | md5sum | cut -c1-12)
  na=$(cd "$ORIG/$d" 2>/dev/null && find . \( -name '*.stl' -o -name '*.obj' -o -name '*.STL' \) | wc -l)
  nb=$(cd "$OURS/$d" 2>/dev/null && find . \( -name '*.stl' -o -name '*.obj' -o -name '*.STL' \) | wc -l)
  echo "  原始 $na 个网格 hash=$ha"
  echo "  我们 $nb 个网格 hash=$hb"
  [ "$ha" = "$hb" ] && echo "  ✓ 网格完全一致" || echo "  ✗ 网格有改动"
done

echo
echo "=== 3) 代码改动全览（哪些文件、各多少行）==="
tot=0
while read -r f; do
  a="$ORIG/$f"; b="$OURS/$f"
  [ -f "$a" ] && [ -f "$b" ] || continue
  cmp -s "$a" "$b" && continue
  d=$(diff "$a" "$b" | grep -c '^[<>]')
  tot=$((tot+d))
  printf "  %-64s %5s 行\n" "$f" "$d"
done < <(cd "$ORIG" && find . -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.py' \) \
          ! -path './third_party/*' 2>/dev/null | sed 's|^\./||' | sort)
echo "  ------"
echo "  合计改动 $tot 行"

echo
echo "=== 4) 我们新增的文件 ==="
while read -r f; do
  [ -f "$ORIG/$f" ] || printf "      + %s\n" "$f"
done < <(cd "$OURS" && find . -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.py' \) \
          ! -path './third_party/*' 2>/dev/null | sed 's|^\./||' | sort)

echo
echo "=== 5) 策略文件：用的是官方的还是我们训的 ==="
ls -la "$ORIG"/../../*.onnx "$ORIG"/**/*.onnx 2>/dev/null | sed 's/^/  官方: /'
find "$P/goai_embodied_future_material-main" -name '*.onnx' 2>/dev/null | while read -r f; do
  printf "  官方 %s  %s\n" "$(md5sum "$f" | cut -c1-10)" "$(basename "$f")"
done
find "$P/s10_dev/train" "$OURS" -name '*.onnx' 2>/dev/null | while read -r f; do
  printf "  我们 %s  %s\n" "$(md5sum "$f" | cut -c1-10)" "$(basename "$f")"
done
