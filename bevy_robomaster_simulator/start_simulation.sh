#!/usr/bin/env bash

set -Eeuo pipefail

# 一键启动 Daedalus 仿真器和任意 Talos 客户端程序。
#
# 最常用的修改位置：
#   PROGRAM_PATH      要在仿真中运行的可执行程序，可以位于本项目之外
#   PROGRAM_WORKDIR   程序的工作目录
#   PROGRAM_ARGS      传给程序的参数数组
#
# 也可以使用命令行覆盖：
#   ./start_simulation.sh --program /opt/my_vision/bin/vision \
#       --workdir /opt/my_vision --clear-args -- config.yaml

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# ===== 可编辑配置 =====
SIMULATOR_DIR="${SIMULATOR_DIR:-${SCRIPT_DIR}}"
SIMULATOR_BIN="${SIMULATOR_BIN:-}"
SIMULATOR_LIB_DIR="${SIMULATOR_LIB_DIR:-}"
PROGRAM_PATH="${PROGRAM_PATH:-${PROJECT_ROOT}/imca_vision_26aim/build/auto_buff_debug_mpc}"
PROGRAM_WORKDIR="${PROGRAM_WORKDIR:-${PROJECT_ROOT}/imca_vision_26aim}"
PROGRAM_ARGS=("configs/sentry.yaml")
CARGO_BIN="${CARGO_BIN:-cargo}"
AUTO_AIM="${AUTO_AIM:-0}"
IMCA_VISION_DEBUG_WINDOWS="${IMCA_VISION_DEBUG_WINDOWS:-1}"
IPC_TIMEOUT_SECONDS="${IPC_TIMEOUT_SECONDS:-30}"
# =====================

META_FILE="/tmp/talos_ipc_meta"
IMAGE_POOL_FILE="/tmp/talos_ipc_image_pool"
META_SIZE=3712
IMAGE_POOL_SIZE=13996800

SIMULATOR_PID=""
PROGRAM_PID=""
SIMULATOR_STARTED=0

log() {
  printf '[start_simulation] %s\n' "$*"
}

die() {
  printf '[start_simulation] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
用法：
  ./start_simulation.sh [选项] [-- 程序参数]

默认启动：
  仿真器目录：当前脚本所在目录
  客户端程序：../imca_vision_26aim/build/auto_aim_debug_mpc
  程序工作目录：../imca_vision_26aim
  程序参数：configs/sentry.yaml
  视觉调试窗口：默认关闭（设置 IMCA_VISION_DEBUG_WINDOWS=1 可开启）

选项：
  -p, --program PATH       指定要在仿真中运行的可执行程序
  -w, --workdir DIR        指定该程序的工作目录
      --arg VALUE          追加一个程序参数，可重复使用
      --clear-args         清空默认程序参数
      --no-auto-aim         不设置 DAEDALUS_AUTO_AIM=1
      --simulator-dir DIR   指定仿真器工作目录
      --simulator-bin PATH  指定已编译的仿真器二进制文件
  -h, --help               显示帮助

示例：
  # 启动项目内默认 auto_aim_debug_mpc
  ./start_simulation.sh

  # 启动项目外的程序
  ./start_simulation.sh \
    --program /opt/my_vision/bin/vision \
    --workdir /opt/my_vision \
    --clear-args -- config.yaml

  # 启动项目外的 Python 客户端
  ./start_simulation.sh \
    --program /usr/bin/python3 \
    --workdir /opt/my_vision \
    --clear-args -- /opt/my_vision/main.py --config config.yaml
EOF
}

require_value() {
  (($# >= 2)) || die "选项 $1 缺少参数"
}

while (($# > 0)); do
  case "$1" in
    -p|--program)
      require_value "$@"
      PROGRAM_PATH="$2"
      shift 2
      ;;
    -w|--workdir)
      require_value "$@"
      PROGRAM_WORKDIR="$2"
      shift 2
      ;;
    --arg)
      require_value "$@"
      PROGRAM_ARGS+=("$2")
      shift 2
      ;;
    --clear-args)
      PROGRAM_ARGS=()
      shift
      ;;
    --no-auto-aim)
      AUTO_AIM=0
      shift
      ;;
    --simulator-dir)
      require_value "$@"
      SIMULATOR_DIR="$2"
      shift 2
      ;;
    --simulator-bin)
      require_value "$@"
      SIMULATOR_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PROGRAM_ARGS+=("$@")
      break
      ;;
    *)
      die "未知选项：$1；使用 --help 查看用法"
      ;;
  esac
done

[[ -d "$SIMULATOR_DIR" ]] || die "仿真器目录不存在：$SIMULATOR_DIR"
[[ -f "$SIMULATOR_DIR/Cargo.toml" ]] || die "仿真器目录不是有效的 Cargo 项目：$SIMULATOR_DIR"
[[ -d "$PROGRAM_WORKDIR" ]] || die "程序工作目录不存在：$PROGRAM_WORKDIR"
[[ "$IPC_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || die "IPC_TIMEOUT_SECONDS 必须是非负整数：$IPC_TIMEOUT_SECONDS"
[[ "$IMCA_VISION_DEBUG_WINDOWS" == "0" || "$IMCA_VISION_DEBUG_WINDOWS" == "1" ]] || die "IMCA_VISION_DEBUG_WINDOWS 必须是 0 或 1：$IMCA_VISION_DEBUG_WINDOWS"
SIMULATOR_DIR="$(cd -- "$SIMULATOR_DIR" && pwd -P)"

if [[ "$PROGRAM_PATH" != /* ]]; then
  PROGRAM_PATH="${PROGRAM_WORKDIR}/${PROGRAM_PATH}"
fi
[[ -f "$PROGRAM_PATH" ]] || die "程序不存在：$PROGRAM_PATH"
[[ -x "$PROGRAM_PATH" ]] || die "程序不可执行：$PROGRAM_PATH；请添加执行权限，或将 PROGRAM_PATH 指向解释器并把脚本作为参数传入"

if [[ -z "$SIMULATOR_BIN" ]]; then
  SIMULATOR_BIN="${SIMULATOR_DIR}/target/release/daedalus"
elif [[ "$SIMULATOR_BIN" != /* ]]; then
  SIMULATOR_BIN="${SIMULATOR_DIR}/${SIMULATOR_BIN}"
fi
command -v stat >/dev/null 2>&1 || die "找不到 stat 命令"
command -v od >/dev/null 2>&1 || die "找不到 od 命令"

if [[ "$AUTO_AIM" == "1" ]]; then
  export DAEDALUS_AUTO_AIM=1
else
  unset DAEDALUS_AUTO_AIM || true
fi

ipc_ready() {
  [[ -f "$META_FILE" && -f "$IMAGE_POOL_FILE" ]] || return 1

  local meta_size image_size header
  meta_size="$(stat -c '%s' -- "$META_FILE" 2>/dev/null || printf '0')"
  image_size="$(stat -c '%s' -- "$IMAGE_POOL_FILE" 2>/dev/null || printf '0')"
  ((meta_size >= META_SIZE && image_size >= IMAGE_POOL_SIZE)) || return 1

  # Talos IPC header: magic 0x54414C05, version 2，小端序为 05 4c 41 54 02 00 00 00。
  header="$(od -An -tx1 -N8 -- "$META_FILE" 2>/dev/null | tr -d '[:space:]')"
  [[ "$header" == "054c415402000000" ]]
}

find_existing_simulator() {
  command -v pgrep >/dev/null 2>&1 || return 0
  pgrep -x daedalus 2>/dev/null | head -n 1 || true
}

stop_group() {
  local pid="$1"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  if [[ -n "$PROGRAM_PID" ]] && kill -0 "$PROGRAM_PID" 2>/dev/null; then
    log "停止客户端程序（PID ${PROGRAM_PID}）"
    stop_group "$PROGRAM_PID"
  fi

  if ((SIMULATOR_STARTED == 1)) && [[ -n "$SIMULATOR_PID" ]] && kill -0 "$SIMULATOR_PID" 2>/dev/null; then
    log "停止仿真器（PID ${SIMULATOR_PID}）"
    stop_group "$SIMULATOR_PID"
  fi

  wait 2>/dev/null || true
  exit "$status"
}

trap 'exit 130' INT TERM
trap cleanup EXIT

existing_simulator_pid="$(find_existing_simulator)"
if [[ -n "$existing_simulator_pid" ]]; then
  SIMULATOR_PID="$existing_simulator_pid"
  log "检测到已运行的 Daedalus（PID ${SIMULATOR_PID}），复用现有仿真器"
  if [[ "$AUTO_AIM" == "1" ]]; then
    log "注意：已运行仿真器的环境变量不会被修改；如需强制自瞄，请先重启仿真器"
  else
    log "注意：已运行仿真器的环境变量不会被修改；如需关闭其强制自瞄，请先重启仿真器"
  fi
else
  log "启动 Daedalus：$SIMULATOR_DIR"
  # 没有正在运行的仿真器时，清理上一次运行留下的精确 IPC 文件，避免误判为已就绪。
  rm -f -- "$META_FILE" "$IMAGE_POOL_FILE"
  (
    cd -- "$SIMULATOR_DIR"
    export BEVY_ASSET_ROOT="$SIMULATOR_DIR"
    if [[ -x "$SIMULATOR_BIN" ]]; then
      log "使用已编译仿真器：$SIMULATOR_BIN"
      if [[ -z "$SIMULATOR_LIB_DIR" ]]; then
        SIMULATOR_LIB_DIR="${SIMULATOR_DIR}/target/release/deps"
      elif [[ "$SIMULATOR_LIB_DIR" != /* ]]; then
        SIMULATOR_LIB_DIR="${SIMULATOR_DIR}/${SIMULATOR_LIB_DIR}"
      fi
      if [[ -d "$SIMULATOR_LIB_DIR" ]]; then
        export LD_LIBRARY_PATH="${SIMULATOR_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      fi
      if command -v rustc >/dev/null 2>&1; then
        rust_lib_dir="$(rustc --print target-libdir 2>/dev/null || true)"
        if [[ -d "$rust_lib_dir" ]]; then
          export LD_LIBRARY_PATH="${rust_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
      fi
      simulator_command=("$SIMULATOR_BIN")
    else
      if [[ "$CARGO_BIN" == */* ]]; then
        [[ -x "$CARGO_BIN" ]] || die "找不到可执行的 cargo：$CARGO_BIN"
      else
        command -v "$CARGO_BIN" >/dev/null 2>&1 || die "找不到 cargo：$CARGO_BIN"
      fi
      log "未找到已编译仿真器，使用 cargo 构建并启动"
      simulator_command=("$CARGO_BIN" run --release)
    fi
    if command -v setsid >/dev/null 2>&1; then
      exec setsid "${simulator_command[@]}"
    else
      exec "${simulator_command[@]}"
    fi
  ) &
  SIMULATOR_PID=$!
  SIMULATOR_STARTED=1
fi

deadline=$((SECONDS + IPC_TIMEOUT_SECONDS))
while ! ipc_ready; do
  if ((SECONDS >= deadline)); then
    die "等待 Talos 共享内存超时（${IPC_TIMEOUT_SECONDS}s）：$META_FILE"
  fi
  if ((SIMULATOR_STARTED == 1)) && ! kill -0 "$SIMULATOR_PID" 2>/dev/null; then
    wait "$SIMULATOR_PID" || trueo_a
    die "Daedalus 在共享内存就绪前退出"
  fi
  sleep 0.1
done
log "Talos 共享内存已就绪"

export IMCA_TALOS_SIMULATOR=1
export IMCA_VISION_DEBUG_WINDOWS

log "启动客户端：$PROGRAM_PATH"
log "工作目录：$PROGRAM_WORKDIR"
log "参数：${PROGRAM_ARGS[*]:-(无)}"
log "视觉调试窗口：$([[ "$IMCA_VISION_DEBUG_WINDOWS" == "1" ]] && printf '开启' || printf '关闭')"
(
  cd -- "$PROGRAM_WORKDIR"
  if command -v setsid >/dev/null 2>&1; then
    exec setsid "$PROGRAM_PATH" "${PROGRAM_ARGS[@]}"
  else
    exec "$PROGRAM_PATH" "${PROGRAM_ARGS[@]}"
  fi
) &
PROGRAM_PID=$!

while kill -0 "$PROGRAM_PID" 2>/dev/null; do
  if [[ -n "$SIMULATOR_PID" ]] && ! kill -0 "$SIMULATOR_PID" 2>/dev/null; then
    SIMULATOR_STATUS=0
    wait "$SIMULATOR_PID" 2>/dev/null || SIMULATOR_STATUS=$?
    log "ERROR: Daedalus 已退出（status ${SIMULATOR_STATUS}），停止客户端程序（PID ${PROGRAM_PID}）" >&2
    stop_group "$PROGRAM_PID"
    wait "$PROGRAM_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 0.2
done

if wait "$PROGRAM_PID"; then
  PROGRAM_STATUS=0
else
  PROGRAM_STATUS=$?
fi
exit "$PROGRAM_STATUS"
