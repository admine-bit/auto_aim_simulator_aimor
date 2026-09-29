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
# 云台高度补偿（米）：瞄点偏高时调小，偏低时调大；-0.5 是原来 -1 与 0 的中间值。
GIMBAL_HEIGHT_OFFSET_M="${GIMBAL_HEIGHT_OFFSET_M:--0.3}"

[[ -f "${SOURCE_DIR}/CMakeLists.txt" ]] || {
  printf '找不到视觉源码：%s\n' "$SOURCE_DIR" >&2
  exit 1
}
source "${SIM_ROOT}/simulation_env.sh"

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
c++ -std=c++20 -O2 -fPIC -shared \
  -I "${SOURCE_COPY}/io/hikrobot/include" \
  "${SCRIPT_DIR}/legacy_vision/camera_shim.cpp" -o "$SHIM"

mkdir -p "$RUN_DIR"
"${SCRIPT_DIR}/start_simulation.sh" \
  --program /usr/bin/python3 --workdir "$RUN_DIR" --clear-args -- \
  "${SCRIPT_DIR}/legacy_vision/bridge.py" \
  --source "$SOURCE_COPY" --runtime "$RUN_DIR" \
  --vision "$BUILD_DIR/auto_aim_debug_mpc" --shim "$SHIM" \
  --gimbal-height-offset-m "$GIMBAL_HEIGHT_OFFSET_M"
