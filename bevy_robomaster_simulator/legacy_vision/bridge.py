#!/usr/bin/env python3
"""Bridge Talos poses and commands to an unmodified imca serial protocol."""

import argparse
import math
import mmap
import os
import pty
import select
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import yaml

META_SIZE = 3712
POSE_BASE = 256
GIMBAL_CMD_BASE = 1536
RX_FORMAT = "<BB4f6fBB"
TX_FORMAT = "<BB6fB"
RX_SIZE = struct.calcsize(RX_FORMAT)
TX_SIZE = struct.calcsize(TX_FORMAT)


def pose(meta, channel):
    base = POSE_BASE + channel * 256
    state = meta[base]
    index = (state & 0x03) if (state & 0x80) else meta[base + 2]
    return struct.unpack_from("<Q3f4f4xQ", meta, base + 64 + index * 64)


def camera_calibration(meta):
    info = struct.unpack_from("<Q9dII", meta, 1728)
    camera_pose = pose(meta, 3)
    if info[10:12] != (1440, 1080) or info[1] <= 0 or camera_pose[0] == 0:
        raise RuntimeError("仿真相机标定信息尚未就绪")
    w, x, y, z = camera_pose[4:8]
    rotation = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]
    return {
        "camera_matrix": [info[1], 0.0, info[3], 0.0, info[2], info[4], 0.0, 0.0, 1.0],
        "distort_coeffs": list(info[5:10]),
        "R_camera2gimbal": rotation,
        "t_camera2gimbal": list(camera_pose[1:4]),
        # Talos already reports gimbal orientation directly in the world frame.
        "R_gimbal2imubody": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    }


def receive_frame(meta, gimbal_height_offset_m):
    gimbal = pose(meta, 0)
    odom = pose(meta, 1)
    w, x, y, z = gimbal[4:8]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    # The solver uses this z as the gimbal origin's world height. Keep the
    # simulator-side calibration offset explicit so it can be tuned per scene.
    return struct.pack(RX_FORMAT, 0x5A, 1, w, x, y, z,
                       yaw, 0.0, pitch, odom[1], odom[2],
                       odom[3] + gimbal_height_offset_m,
                       1, 0xA5)


def publish_command(meta, frame):
    head, mode, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc, tail = struct.unpack(
        TX_FORMAT, frame)
    if head != 0x5A or tail != 0xA5 or mode > 2 or not all(map(math.isfinite,
                                                                    (yaw, pitch))):
        return
    write_index = meta[GIMBAL_CMD_BASE + 1]
    base = GIMBAL_CMD_BASE + 64 + write_index * 32
    struct.pack_into("<QfffB11x", meta, base, time.time_ns(),
                     math.degrees(yaw), -(90.0 + math.degrees(pitch)),
                     1.0 if mode else -1.0, 1 if mode == 2 else 0)
    previous = meta[GIMBAL_CMD_BASE]
    meta[GIMBAL_CMD_BASE] = write_index | 0x80
    meta[GIMBAL_CMD_BASE + 1] = previous & 0x03


def run(args):
    if not math.isfinite(args.gimbal_height_offset_m):
        raise RuntimeError("云台高度补偿必须是有限数字")
    source = Path(args.source).resolve(strict=True)
    runtime = Path(args.runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("assets", "model", "rm_vision_core-main"):
        target = runtime / name
        if not target.exists():
            shutil.copytree(source / name, target)
    fd = os.open("/tmp/talos_ipc_meta", os.O_RDWR)
    with os.fdopen(fd, "r+b", buffering=0) as file, mmap.mmap(file.fileno(), META_SIZE) as meta:
        if struct.unpack_from("<II", meta) != (0x54414C05, 2):
            raise RuntimeError("Talos 共享内存版本不匹配")
        master, slave = pty.openpty()
        slave_path = os.ttyname(slave)
        config = yaml.safe_load((source / "configs/sentry.yaml").read_text())
        deadline = time.monotonic() + 15.0
        while True:
            try:
                config.update(camera_calibration(meta))
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        config["com_port"] = slave_path
        # The default Daedalus view exposes blue armor to the red-side client.
        config["enemy_color"] = "blue"
        generated = runtime / "sentry-sim.yaml"
        generated.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        env = os.environ.copy()
        env["LD_PRELOAD"] = args.shim + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        env["IMCA_VISION_DEBUG_WINDOWS"] = "0"
        vision = subprocess.Popen([args.vision, str(generated)], cwd=runtime, env=env)
        os.close(slave)
        os.set_blocking(master, False)
        pending = bytearray()
        last_status = time.monotonic()
        print(f"[legacy-bridge] serial={slave_path}, vision_pid={vision.pid}, "
              f"gimbal_height_offset_m={args.gimbal_height_offset_m:+.3f}", flush=True)
        try:
            while vision.poll() is None:
                now = time.monotonic()
                if now - last_status >= 0.01:
                    try:
                        os.write(master, receive_frame(meta, args.gimbal_height_offset_m))
                    except (BlockingIOError, OSError):
                        pass
                    last_status = now
                readable, _, _ = select.select([master], [], [], 0.005)
                if readable:
                    try:
                        pending.extend(os.read(master, 4096))
                    except (BlockingIOError, OSError):
                        pass
                while pending:
                    start = pending.find(0x5A)
                    if start < 0:
                        pending.clear()
                        break
                    del pending[:start]
                    if len(pending) < TX_SIZE:
                        break
                    if pending[TX_SIZE - 1] != 0xA5:
                        del pending[0]
                        continue
                    publish_command(meta, pending[:TX_SIZE])
                    del pending[:TX_SIZE]
            return vision.returncode
        finally:
            if vision.poll() is None:
                vision.terminate()
                try:
                    vision.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    vision.kill()
                    vision.wait()
            os.close(master)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--shim", required=True)
    parser.add_argument("--gimbal-height-offset-m", type=float, default=0.0)
    args = parser.parse_args()
    try:
        sys.exit(run(args))
    except (OSError, RuntimeError, yaml.YAMLError) as error:
        print(f"[legacy-bridge] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
