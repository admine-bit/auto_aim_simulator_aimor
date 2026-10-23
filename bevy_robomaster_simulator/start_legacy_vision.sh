#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SIM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SOURCE_DIR="${LEGACY_VISION_SOURCE:-/home/ad/code_game/imca_vision_26aim}"
SOURCE_COPY="${SCRIPT_DIR}/.legacy-vision-source"
BUILD_DIR="${SCRIPT_DIR}/.legacy-vision-copy-build"
RUN_DIR="$SOURCE_COPY"
SHIM="${SCRIPT_DIR}/legacy_vision/libcamera_shim.so"
JOBS="${BUILD_JOBS:-3}"
# 红方符转动方式：small 为现有小符匀速，large 为现有大符变速；切换后重启仿真生效。
RED_RUNE_MODE="${RED_RUNE_MODE:-large}"
# 蓝方符转动方式：small / large；独立于红方设置，保留各自已固定的转向。
BLUE_RUNE_MODE="${BLUE_RUNE_MODE:-large}"
# 机关本体转向：random 启动时随机一次；red_cw / red_ccw 可固定红方观测方向用于复现，蓝方相反。
RUNE_DIRECTION="${RUNE_DIRECTION:-random}"
# 云台高度补偿（米）：瞄点偏高时调小，偏低时调大；-0.5 是原来 -1 与 0 的中间值。
GIMBAL_HEIGHT_OFFSET_M="${GIMBAL_HEIGHT_OFFSET_M:--0.3}"
# 给仿真预留的物理核心数，其余核心优先留给视觉；仿真供图不足时调大。
SIMULATOR_PHYSICAL_CORES="${SIMULATOR_PHYSICAL_CORES:-2}"
# 留空按 CPU 拓扑自动分配；手动分配时填 taskset 格式的逻辑 CPU 编号。
SIMULATOR_CPU_SET="${SIMULATOR_CPU_SET:-}"
# 留空使用仿真之外的完整物理核心，包含其超线程；需要指定视觉核心时修改。
VISION_CPU_SET="${VISION_CPU_SET:-}"
# 仿真的普通调度优先级增量（0～19）；桌面受影响时调大，供图不足时调小。
SIMULATOR_NICE="${SIMULATOR_NICE:-5}"
# 仿真任务池总线程数（含加载、异步回读和计算，至少 3）；场景更大时调大。
SIMULATOR_TASK_THREADS="${SIMULATOR_TASK_THREADS:-3}"
# 仿真核心较少时用顺序执行减少短系统调度开销；大场景并行更快时改为 0。
SIMULATOR_SERIAL_SCHEDULES="${SIMULATOR_SERIAL_SCHEDULES:-1}"
# 使用 OpenCV 自带环境配置限制图像处理线程；推理仍可使用全部视觉核心。
VISION_OPENCV_THREADS="${VISION_OPENCV_THREADS:-2}"
# 用户要求保留所有调试窗口；性能调优期间也保持开启。
export IMCA_VISION_DEBUG_WINDOWS=1

for rune_variable in RED_RUNE_MODE BLUE_RUNE_MODE; do
  case "${!rune_variable}" in
    small|large) ;;
    *) printf '%s 必须是 small 或 large，实际为：%s\n' "$rune_variable" "${!rune_variable}" >&2; exit 1 ;;
  esac
done
case "$RUNE_DIRECTION" in random|red_cw|red_ccw) ;; *) printf 'RUNE_DIRECTION 必须为 random、red_cw 或 red_ccw\n' >&2; exit 1 ;; esac
export RED_RUNE_MODE BLUE_RUNE_MODE RUNE_DIRECTION
printf '[legacy-launch] rune modes: red=%s blue=%s direction=%s; initial vision follows red\n' "$RED_RUNE_MODE" "$BLUE_RUNE_MODE" "$RUNE_DIRECTION"

[[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] || {
  printf '找不到视觉源码：%s\n' "$SOURCE_DIR" >&2
  exit 1
}
source "${SIM_ROOT}/simulation_env.sh"

# Keep SMT siblings together so vision and rendering do not compete on one core.
read -r auto_simulator_cpus auto_vision_cpus < <(python3 - "$SIMULATOR_PHYSICAL_CORES" <<'PY'
import os
import sys
from pathlib import Path

reserve = int(sys.argv[1])
if reserve < 1:
    raise SystemExit("SIMULATOR_PHYSICAL_CORES 必须至少为 1")
groups = {}
for cpu in sorted(os.sched_getaffinity(0)):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    key = ((topology / "physical_package_id").read_text().strip(),
           (topology / "core_id").read_text().strip())
    groups.setdefault(key, []).append(cpu)
cores = list(groups.values())
cut = min(reserve, max(1, len(cores) - 1))
sim = [cpu for core in cores[:cut] for cpu in core]
vision = [cpu for core in cores[cut:] for cpu in core] or sim
print(",".join(map(str, sorted(sim))), ",".join(map(str, sorted(vision))))
PY
)
export SIMULATOR_CPU_SET="${SIMULATOR_CPU_SET:-$auto_simulator_cpus}"
export PROGRAM_CPU_SET="${VISION_CPU_SET:-$auto_vision_cpus}"
export SIMULATOR_NICE
export DAEDALUS_TASK_THREADS="$SIMULATOR_TASK_THREADS"
export DAEDALUS_SERIAL_SCHEDULES="$SIMULATOR_SERIAL_SCHEDULES"
export OPENCV_FOR_THREADS_NUM="$VISION_OPENCV_THREADS"
printf '[legacy-launch] simulator CPUs=%s nice=%s task_threads=%s; vision CPUs=%s OpenCV threads=%s\n' \
  "$SIMULATOR_CPU_SET" "$SIMULATOR_NICE" "$SIMULATOR_TASK_THREADS" \
  "$PROGRAM_CPU_SET" "$VISION_OPENCV_THREADS"

# Build from a local copy: some upstream __FILE__-based settings write beside
# their headers even when CMake's build directory lives elsewhere.
mkdir -p "$SOURCE_COPY"
rsync -a --exclude=.git --exclude=build --exclude=back --exclude=bin \
  --exclude=neo --exclude=Work-notes --exclude=logs --exclude=records \
  "$SOURCE_DIR/" "$SOURCE_COPY/"
cmake -S "$SOURCE_COPY" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${SIM_ROOT}/.local-deps/root/usr" \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so.3 \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so.3
cmake --build "$BUILD_DIR" --target auto_aim_debug_mpc -j "$JOBS"
c++ -std=c++20 -O3 -march=native -fPIC -shared \
  -I "${SOURCE_COPY}/io/hikrobot/include" \
  "${SCRIPT_DIR}/legacy_vision/camera_shim.cpp" -o "$SHIM"

mkdir -p "$RUN_DIR"
"${SCRIPT_DIR}/start_simulation.sh" \
  --program /usr/bin/python3 --workdir "$RUN_DIR" --clear-args -- \
  "${SCRIPT_DIR}/legacy_vision/bridge.py" \
  --source "$SOURCE_COPY" --runtime "$RUN_DIR" --vision-workdir "$SOURCE_DIR" \
  --vision "$BUILD_DIR/auto_aim_debug_mpc" --shim "$SHIM" \
  --gimbal-height-offset-m "$GIMBAL_HEIGHT_OFFSET_M"
