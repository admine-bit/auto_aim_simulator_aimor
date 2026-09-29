#!/usr/bin/env bash
# Source this file to use the locally installed simulation dependencies.
SIMULATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PATH="${HOME}/.cargo/bin:${PATH}"
export CMAKE_PREFIX_PATH="${SIMULATION_ROOT}/.local-deps/root/usr${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export PKG_CONFIG_PATH="${SIMULATION_ROOT}/.local-deps/root/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export LIBRARY_PATH="${SIMULATION_ROOT}/.local-deps/root/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CPATH="${SIMULATION_ROOT}/.local-deps/root/usr/include:${SIMULATION_ROOT}/.local-deps/root/usr/include/x86_64-linux-gnu${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${SIMULATION_ROOT}/.local-deps/root/usr/lib/x86_64-linux-gnu:${SIMULATION_ROOT}/.local-deps/openvino-2024.6/runtime/lib/intel64:${SIMULATION_ROOT}/.local-deps/openvino-2024.6/runtime/3rdparty/tbb/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
