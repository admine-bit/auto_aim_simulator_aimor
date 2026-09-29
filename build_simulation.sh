#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/simulation_env.sh"
jobs="${BUILD_JOBS:-3}"
cargo build --manifest-path "${SIMULATION_ROOT}/bevy_robomaster_simulator/Cargo.toml" \
  --release --locked --bin daedalus -j "$jobs"
cmake -S "${SIMULATION_ROOT}/imca_vision_26aim" -B "${SIMULATION_ROOT}/imca_vision_26aim/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${SIMULATION_ROOT}/.local-deps/root/usr" \
  -DOpenVINO_DIR="${SIMULATION_ROOT}/.local-deps/openvino-2024.6/runtime/cmake" \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so.3 \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so.3
cmake --build "${SIMULATION_ROOT}/imca_vision_26aim/build" \
  --target auto_aim_debug_mpc standard_mpc -j "$jobs"
